"""FIFO, single-worker scheduling for responder generation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .responder import ChatRequest, Responder

LOGGER = logging.getLogger(__name__)
MAX_CONCURRENT_GENERATIONS = 1


class GenerationQueueClosedError(RuntimeError):
    """Raised when a job is submitted after queue shutdown begins."""


@dataclass(frozen=True, slots=True, repr=False)
class GenerationQueueStatus:
    """Privacy-safe read-only queue state without queued job objects."""

    worker_alive: bool
    busy: bool
    waiting_jobs: int

    def __repr__(self) -> str:
        return (
            "GenerationQueueStatus("
            f"worker_alive={self.worker_alive!r}, "
            f"busy={self.busy!r}, "
            f"waiting_jobs={self.waiting_jobs!r})"
        )


@dataclass(slots=True)
class _GenerationJob:
    request: ChatRequest
    future: asyncio.Future[str]


class SerializedGenerationQueue:
    """Own one FIFO worker and serialize all responder calls through it."""

    def __init__(self, responder: Responder, *, logger: logging.Logger = LOGGER) -> None:
        self._responder = responder
        self._logger = logger
        self._queue: asyncio.Queue[_GenerationJob | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._active_job: _GenerationJob | None = None
        self._started = False
        self._closed = False
        self._close_lock: asyncio.Lock | None = None

    @property
    def worker_task(self) -> asyncio.Task[None] | None:
        """Expose the owned worker for lifecycle tests without exposing jobs."""

        return self._worker

    @property
    def queued_count(self) -> int:
        """Return the number of jobs waiting behind the active generation."""

        return self._queue.qsize()

    def get_status(self) -> GenerationQueueStatus:
        """Return worker, active-job, and waiting-job state without I/O."""

        worker = self._worker
        return GenerationQueueStatus(
            worker_alive=worker is not None and not worker.done(),
            busy=self._active_job is not None,
            waiting_jobs=self._queue.qsize(),
        )

    async def start(self) -> None:
        """Start exactly one worker; repeated calls are harmless."""

        if self._closed:
            raise GenerationQueueClosedError("generation queue is closed")
        if self._started:
            return
        self._started = True
        self._close_lock = asyncio.Lock()
        self._worker = asyncio.create_task(self._worker_loop(), name="liuer-generation-worker")
        self._logger.debug("generation worker started")

    async def submit(self, request: ChatRequest) -> str:
        """Queue one request and await its result or the caller cancellation."""

        if self._closed:
            raise GenerationQueueClosedError("generation queue is closed")
        if not self._started:
            raise RuntimeError("generation queue has not been started")

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = _GenerationJob(request=request, future=future)
        await self._queue.put(job)
        self._logger.debug(
            "generation queued message_id=%s queued_count=%s",
            request.message_id,
            self.queued_count,
        )
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def close(self) -> None:
        """Cancel pending work, stop the worker, and make shutdown idempotent."""

        if self._closed:
            return
        self._closed = True
        if self._close_lock is None:
            self._close_lock = asyncio.Lock()
        async with self._close_lock:
            if self._worker is None:
                self._fail_queued_jobs(GenerationQueueClosedError("generation queue is closed"))
                self._logger.debug("generation queue shutdown")
                return
            self._fail_queued_jobs(GenerationQueueClosedError("generation queue is closed"))
            if self._active_job is not None and not self._active_job.future.done():
                self._active_job.future.set_exception(
                    GenerationQueueClosedError("generation queue is closed")
                )
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            finally:
                self._worker = None
            self._logger.debug("generation queue shutdown")

    def _fail_queued_jobs(self, error: Exception) -> None:
        while True:
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if job is None:
                continue
            if not job.future.done():
                job.future.set_exception(error)

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            if job is None:
                return
            if job.future.cancelled():
                self._logger.debug(
                    "generation skipped because waiter cancelled message_id=%s",
                    job.request.message_id,
                )
                continue
            self._logger.debug("generation started message_id=%s", job.request.message_id)
            self._active_job = job
            try:
                result = await self._responder.generate(job.request)
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.cancel()
                raise
            except Exception as exc:
                self._logger.error(
                    "generation failed message_id=%s exception_type=%s",
                    job.request.message_id,
                    type(exc).__name__,
                )
                if not job.future.done():
                    job.future.set_exception(exc)
            else:
                self._logger.debug("generation completed message_id=%s", job.request.message_id)
                if not job.future.done():
                    job.future.set_result(result)
            finally:
                self._active_job = None

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from functools import wraps

import pytest

from liuer_bot.generation_queue import (
    MAX_CONCURRENT_GENERATIONS,
    GenerationQueueClosedError,
    SerializedGenerationQueue,
)
from liuer_bot.responder import ChatRequest


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs) -> None:
        asyncio.run(function(*args, **kwargs))

    return wrapper


def _request(message_id: int) -> ChatRequest:
    return ChatRequest(
        message_id=message_id,
        guild_id=20,
        channel_id=123,
        author_id=10,
        content=f"private content {message_id}",
    )


@dataclass
class ControlledResponder:
    results: dict[int, str] = field(default_factory=dict)
    started: list[int] = field(default_factory=list)
    finished: list[int] = field(default_factory=list)
    active: int = 0
    max_active: int = 0
    release: dict[int, asyncio.Event] = field(default_factory=dict)
    failure_ids: set[int] = field(default_factory=set)

    async def generate(self, request: ChatRequest) -> str:
        self.started.append(request.message_id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            event = self.release.get(request.message_id)
            if event is not None:
                await event.wait()
            if request.message_id in self.failure_ids:
                raise RuntimeError(f"private failure {request.content}")
            return self.results.get(request.message_id, f"result-{request.message_id}")
        finally:
            self.active -= 1
            self.finished.append(request.message_id)


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    async def wait_for_predicate() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_predicate(), timeout)


@async_test
async def test_queue_is_fifo_and_has_one_active_generation() -> None:
    assert MAX_CONCURRENT_GENERATIONS == 1
    responder = ControlledResponder(
        release={message_id: asyncio.Event() for message_id in (1, 2, 3)},
    )
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    try:
        tasks = [
            asyncio.create_task(queue.submit(_request(message_id)))
            for message_id in (1, 2, 3)
        ]
        await _wait_until(lambda: responder.started == [1])
        assert responder.max_active == 1
        responder.release[1].set()
        await _wait_until(lambda: responder.started == [1, 2])
        responder.release[2].set()
        await _wait_until(lambda: responder.started == [1, 2, 3])
        responder.release[3].set()

        assert await asyncio.gather(*tasks) == ["result-1", "result-2", "result-3"]
        assert responder.finished == [1, 2, 3]
        assert responder.max_active == 1
    finally:
        await queue.close()


@async_test
async def test_queue_accepts_later_jobs_while_first_generation_is_waiting() -> None:
    first_release = asyncio.Event()
    responder = ControlledResponder(release={1: first_release})
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    try:
        first = asyncio.create_task(queue.submit(_request(1)))
        await _wait_until(lambda: responder.started == [1])
        second = asyncio.create_task(queue.submit(_request(2)))
        third = asyncio.create_task(queue.submit(_request(3)))
        await asyncio.sleep(0)

        assert not second.done()
        assert not third.done()
        assert responder.started == [1]

        first_release.set()
        assert await asyncio.gather(first, second, third) == ["result-1", "result-2", "result-3"]
    finally:
        await queue.close()


@async_test
async def test_generation_failure_isolated_and_worker_continues(caplog) -> None:
    responder = ControlledResponder(failure_ids={1})
    logger = logging.getLogger("test.queue.failure")
    queue = SerializedGenerationQueue(responder, logger=logger)
    await queue.start()
    try:
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            first = asyncio.create_task(queue.submit(_request(1)))
            with pytest.raises(RuntimeError):
                await first
            assert await queue.submit(_request(2)) == "result-2"

        assert responder.started == [1, 2]
        assert "private content" not in caplog.text
        assert "generation failed" in caplog.text
        assert "exception_type=RuntimeError" in caplog.text
    finally:
        await queue.close()


@async_test
async def test_cancelled_queued_waiter_is_skipped_without_generation() -> None:
    first_release = asyncio.Event()
    responder = ControlledResponder(release={1: first_release})
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    try:
        first = asyncio.create_task(queue.submit(_request(1)))
        await _wait_until(lambda: responder.started == [1])
        second = asyncio.create_task(queue.submit(_request(2)))
        third = asyncio.create_task(queue.submit(_request(3)))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second

        first_release.set()
        assert await first == "result-1"
        assert await third == "result-3"
        assert responder.started == [1, 3]
    finally:
        await queue.close()


@async_test
async def test_cancelled_active_waiter_does_not_cancel_shared_generation() -> None:
    first_release = asyncio.Event()
    responder = ControlledResponder(release={1: first_release})
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    try:
        first = asyncio.create_task(queue.submit(_request(1)))
        await _wait_until(lambda: responder.started == [1])
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert responder.active == 1

        second = asyncio.create_task(queue.submit(_request(2)))
        await asyncio.sleep(0)
        assert responder.started == [1]
        first_release.set()
        assert await second == "result-2"
        assert responder.started == [1, 2]
    finally:
        await queue.close()


@async_test
async def test_shutdown_cancels_active_and_queued_waiters_without_task_leaks() -> None:
    first_release = asyncio.Event()
    responder = ControlledResponder(release={1: first_release})
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    first = asyncio.create_task(queue.submit(_request(1)))
    await _wait_until(lambda: responder.started == [1])
    second = asyncio.create_task(queue.submit(_request(2)))
    await asyncio.sleep(0)

    await queue.close()
    await queue.close()

    with pytest.raises(GenerationQueueClosedError):
        await first
    with pytest.raises(GenerationQueueClosedError):
        await second
    assert queue.worker_task is None
    assert responder.started == [1]


@async_test
async def test_submit_before_start_and_after_close_fail_explicitly() -> None:
    responder = ControlledResponder()
    queue = SerializedGenerationQueue(responder)

    with pytest.raises(RuntimeError, match="not been started"):
        await queue.submit(_request(1))

    await queue.close()
    with pytest.raises(GenerationQueueClosedError):
        await queue.submit(_request(2))


@async_test
async def test_start_after_close_fails_explicitly() -> None:
    queue = SerializedGenerationQueue(ControlledResponder())

    await queue.close()

    with pytest.raises(GenerationQueueClosedError):
        await queue.start()


@async_test
async def test_submit_after_started_queue_is_closed_fails() -> None:
    queue = SerializedGenerationQueue(ControlledResponder())
    await queue.start()
    await queue.close()

    with pytest.raises(GenerationQueueClosedError):
        await queue.submit(_request(1))

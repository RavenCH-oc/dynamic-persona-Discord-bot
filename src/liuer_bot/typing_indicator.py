"""Best-effort, reference-counted Discord typing state for accepted requests."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _TypingState:
    channel: object
    active_count: int
    context: object | None = None


class TypingLease:
    """One idempotent lease on a channel's shared typing indicator."""

    __slots__ = ("_manager", "_channel_id", "_released")

    def __init__(self, manager: TypingIndicatorManager, channel_id: int | None) -> None:
        self._manager = manager
        self._channel_id = channel_id
        self._released = False

    async def __aenter__(self) -> TypingLease:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._channel_id is not None:
            await self._manager._release(self._channel_id)

    def __repr__(self) -> str:
        return (
            "TypingLease("
            f"channel_id={self._channel_id!r}, released={self._released!r})"
        )


class TypingIndicatorManager:
    """Own one public discord.py typing context per active channel.

    Accepted requests hold leases while they prepare, wait in the generation
    queue, generate, and persist.  The underlying ``channel.typing()`` context
    refreshes the Discord indicator at discord.py's supported cadence.
    """

    def __init__(self, *, logger: logging.Logger = LOGGER) -> None:
        self._logger = logger
        self._lock = asyncio.Lock()
        self._states: dict[int, _TypingState] = {}
        self._closed = False

    async def acquire(self, channel: object) -> TypingLease:
        """Acquire a best-effort lease keyed by the stable Discord channel ID."""

        channel_id = _channel_id(channel)
        if channel_id is None:
            self._logger.debug("typing skipped because channel ID is unavailable")
            return TypingLease(self, None)

        async with self._lock:
            if self._closed:
                return TypingLease(self, None)
            existing = self._states.get(channel_id)
            if existing is not None:
                existing.active_count += 1
                return TypingLease(self, channel_id)

            state = _TypingState(channel=channel, active_count=1)
            self._states[channel_id] = state
            try:
                typing_factory = getattr(channel, "typing")
                context = typing_factory()
                await context.__aenter__()
            except asyncio.CancelledError:
                self._states.pop(channel_id, None)
                raise
            except Exception as exc:
                self._logger.warning(
                    "typing indicator start failed channel_id=%s error_type=%s",
                    channel_id,
                    type(exc).__name__,
                )
            else:
                state.context = context
            return TypingLease(self, channel_id)

    def indicate(self, channel: object) -> TypingLease:
        """Return a lease that can be used with ``async with`` after acquire."""

        return _DeferredTypingLease(self, channel)

    def get_active_count(self, channel_id: int) -> int:
        """Return active lease count without exposing Discord objects."""

        state = self._states.get(channel_id)
        return state.active_count if state is not None else 0

    def is_typing(self, channel_id: int) -> bool:
        """Return whether the underlying typing context entered successfully."""

        state = self._states.get(channel_id)
        return state is not None and state.context is not None

    async def close(self) -> None:
        """Stop all owned typing contexts and make future leases no-ops."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            states = tuple(self._states.values())
            self._states.clear()

        for state in states:
            await self._exit_context(state, channel_id=_channel_id(state.channel))

    async def _release(self, channel_id: int) -> None:
        async with self._lock:
            state = self._states.get(channel_id)
            if state is None:
                return
            state.active_count -= 1
            if state.active_count > 0:
                return
            self._states.pop(channel_id, None)

        await self._exit_context(state, channel_id=channel_id)

    async def _exit_context(self, state: _TypingState, *, channel_id: int | None) -> None:
        context = state.context
        if context is None:
            return
        try:
            result = context.__aexit__(None, None, None)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            self._logger.debug(
                "typing indicator stop cancelled channel_id=%s",
                channel_id,
            )
        except Exception as exc:
            self._logger.warning(
                "typing indicator stop failed channel_id=%s error_type=%s",
                channel_id,
                type(exc).__name__,
            )


class _DeferredTypingLease(TypingLease):
    """Async context-manager facade that acquires on entry."""

    __slots__ = ("_channel", "_entered")

    def __init__(self, manager: TypingIndicatorManager, channel: object) -> None:
        super().__init__(manager, None)
        self._channel = channel
        self._entered: TypingLease | None = None

    async def __aenter__(self) -> TypingLease:
        self._entered = await self._manager.acquire(self._channel)
        return self._entered

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._entered is not None:
            await self._entered.release()
        self._released = True

    async def release(self) -> None:
        if self._entered is not None:
            await self._entered.release()
        self._released = True


def _channel_id(channel: object) -> int | None:
    value = getattr(channel, "id", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


__all__ = ["TypingIndicatorManager", "TypingLease"]

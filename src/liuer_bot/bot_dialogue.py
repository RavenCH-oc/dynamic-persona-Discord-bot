"""Process-local, single-peer BotChat session safety boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from uuid import uuid4

from .conversation_models import sanitize_display_name

BOT_DIALOGUE_MAX_LIUER_REPLIES = 6
BOT_DIALOGUE_IDLE_TIMEOUT_SECONDS = 300


class BotDialogueError(RuntimeError):
    """Base class for safe BotChat session policy failures."""


class BotDialogueAlreadyActiveError(BotDialogueError):
    """Raised when activation is attempted while another session is active."""


@dataclass(frozen=True, slots=True, repr=False)
class BotDialogueSession:
    """Immutable session metadata without Discord objects or message content."""

    session_id: str
    peer_bot_id: int
    peer_display_name: str
    liuer_reply_count: int
    awaiting_peer: bool
    started_monotonic: float
    last_activity_monotonic: float

    def __repr__(self) -> str:
        return (
            "BotDialogueSession("
            f"session_id={self.session_id!r}, "
            f"peer_bot_id={self.peer_bot_id!r}, "
            "peer_display_name=<redacted>, "
            f"liuer_reply_count={self.liuer_reply_count!r}, "
            f"awaiting_peer={self.awaiting_peer!r}, "
            f"started_monotonic={self.started_monotonic!r}, "
            f"last_activity_monotonic={self.last_activity_monotonic!r})"
        )


class BotDialogueService:
    """Own one lazy-expiring process-local BotChat session."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        max_replies: int = BOT_DIALOGUE_MAX_LIUER_REPLIES,
        idle_timeout_seconds: int = BOT_DIALOGUE_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        if max_replies <= 0 or idle_timeout_seconds <= 0:
            raise ValueError("BotChat limits must be positive")
        self._clock = clock
        self._max_replies = max_replies
        self._idle_timeout_seconds = idle_timeout_seconds
        self._lock = RLock()
        self._session: BotDialogueSession | None = None

    @property
    def max_replies(self) -> int:
        return self._max_replies

    @property
    def idle_timeout_seconds(self) -> int:
        return self._idle_timeout_seconds

    def activate(self, peer_bot_id: int, peer_display_name: str) -> BotDialogueSession:
        if (
            not isinstance(peer_bot_id, int)
            or isinstance(peer_bot_id, bool)
            or peer_bot_id <= 0
        ):
            raise ValueError("peer Bot ID must be positive")
        now = self._clock()
        with self._lock:
            self._expire_locked(now)
            if self._session is not None:
                raise BotDialogueAlreadyActiveError()
            self._session = BotDialogueSession(
                session_id=uuid4().hex,
                peer_bot_id=peer_bot_id,
                peer_display_name=sanitize_display_name(peer_display_name),
                liuer_reply_count=0,
                awaiting_peer=False,
                started_monotonic=now,
                last_activity_monotonic=now,
            )
            return self._session

    def get_session(self) -> BotDialogueSession | None:
        with self._lock:
            self._expire_locked(self._clock())
            return self._session

    def is_active_peer(self, author_id: int) -> bool:
        with self._lock:
            self._expire_locked(self._clock())
            return self._session is not None and self._session.peer_bot_id == author_id

    def claim_peer_turn(self, author_id: int) -> str | None:
        """Atomically claim one eligible peer message for generation."""

        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            session = self._session
            if session is None or session.peer_bot_id != author_id or not session.awaiting_peer:
                return None
            self._session = self._replace(session, awaiting_peer=False, last_activity_monotonic=now)
            return session.session_id

    def complete_turn(self, session_id: str, *, delivered: bool) -> None:
        """Advance a claimed turn or close the session on generation/delivery failure."""

        with self._lock:
            session = self._session
            if session is None or session.session_id != session_id:
                return
            if session.awaiting_peer:
                return
            if not delivered:
                self._session = None
                return
            reply_count = session.liuer_reply_count + 1
            if reply_count >= self._max_replies:
                self._session = None
                return
            self._session = self._replace(
                session,
                liuer_reply_count=reply_count,
                awaiting_peer=True,
                last_activity_monotonic=self._clock(),
            )

    def close(self) -> bool:
        with self._lock:
            had_session = self._session is not None
            self._session = None
            return had_session

    def idle_age_seconds(self) -> int | None:
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if self._session is None:
                return None
            return max(0, int(now - self._session.last_activity_monotonic))

    def remaining_timeout_seconds(self) -> int | None:
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if self._session is None:
                return None
            elapsed = now - self._session.last_activity_monotonic
            return max(0, int(self._idle_timeout_seconds - elapsed))

    def _expire_locked(self, now: float) -> None:
        if self._session is not None and now - self._session.last_activity_monotonic > (
            self._idle_timeout_seconds
        ):
            self._session = None

    @staticmethod
    def _replace(session: BotDialogueSession, **changes: object) -> BotDialogueSession:
        values = {
            "session_id": session.session_id,
            "peer_bot_id": session.peer_bot_id,
            "peer_display_name": session.peer_display_name,
            "liuer_reply_count": session.liuer_reply_count,
            "awaiting_peer": session.awaiting_peer,
            "started_monotonic": session.started_monotonic,
            "last_activity_monotonic": session.last_activity_monotonic,
        }
        values.update(changes)
        return BotDialogueSession(**values)  # type: ignore[arg-type]


__all__ = [
    "BOT_DIALOGUE_IDLE_TIMEOUT_SECONDS",
    "BOT_DIALOGUE_MAX_LIUER_REPLIES",
    "BotDialogueAlreadyActiveError",
    "BotDialogueError",
    "BotDialogueService",
    "BotDialogueSession",
]

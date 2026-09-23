"""Async global nickname policy and in-memory snapshot boundary."""

from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .addressing import PRIMARY_CALL_NAME
from .config import (
    DEFAULT_NICKNAME_MAX_CHARS,
    DEFAULT_PERSONA_DAILY_LIMIT,
    DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS,
    DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS,
    MAX_NICKNAME_MAX_CHARS,
    Config,
)
from .nickname_models import (
    NicknameDailyLimitError,
    NicknameError,
    NicknameGlobalCooldownError,
    NicknameNotInitializedError,
    NicknamePolicyStatus,
    NicknameSetResult,
    NicknameStateSnapshot,
    NicknameTooLongError,
    NicknameValidationError,
    NicknameVersion,
)
from .nickname_repository import NicknameDatabaseError, NicknameRepository
from .persona_time import Clock, SystemClock, as_utc, quota_window


def normalize_author_id(author_discord_id: int | str) -> str:
    """Store Discord snowflake identifiers as non-empty text."""

    if not isinstance(author_discord_id, (int, str)):
        raise NicknameValidationError("author ID must be text or an integer")
    normalized = str(author_discord_id).strip()
    if not normalized:
        raise NicknameValidationError("author ID must not be blank")
    return normalized


def normalize_nickname(
    nickname: str,
    max_chars: int,
    *,
    primary_name: str = PRIMARY_CALL_NAME,
) -> str:
    """Validate one nickname without logging or echoing its input."""

    if not isinstance(nickname, str):
        raise NicknameValidationError("text must be a string")
    normalized = nickname.strip()
    if not normalized:
        raise NicknameValidationError("text must not be blank")
    if any(
        character in {"@", "<", ">"}
        or unicodedata.category(character).startswith("C")
        for character in normalized
    ):
        raise NicknameValidationError("text contains a forbidden control or mention character")
    if normalized == primary_name:
        raise NicknameValidationError("text duplicates the immutable primary name")
    if len(normalized) > max_chars:
        raise NicknameTooLongError(max_chars)
    return normalized


class NicknameService:
    """Own the global Active Nickname snapshot and independent policy state."""

    def __init__(
        self,
        database_path: Path,
        *,
        nickname_max_chars: int = DEFAULT_NICKNAME_MAX_CHARS,
        global_cooldown_seconds: int = DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS,
        daily_limit: int = DEFAULT_PERSONA_DAILY_LIMIT,
        daily_reset_utc_offset_hours: int = DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS,
        clock: Clock | Callable[[], datetime] | None = None,
        repository: NicknameRepository | None = None,
    ) -> None:
        self._validate_policy_config(
            nickname_max_chars,
            global_cooldown_seconds,
            daily_limit,
            daily_reset_utc_offset_hours,
        )
        self._database_path = Path(database_path)
        self._nickname_max_chars = nickname_max_chars
        self._global_cooldown_seconds = global_cooldown_seconds
        self._daily_limit = daily_limit
        self._daily_reset_utc_offset_hours = daily_reset_utc_offset_hours
        self._clock = clock or SystemClock()
        self._repository = repository or NicknameRepository(self._database_path)
        self._snapshot: NicknameStateSnapshot | None = None

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        clock: Clock | Callable[[], datetime] | None = None,
        repository: NicknameRepository | None = None,
    ) -> NicknameService:
        return cls(
            config.database_path,
            nickname_max_chars=config.nickname_max_chars,
            global_cooldown_seconds=config.persona_global_cooldown_seconds,
            daily_limit=config.persona_daily_limit,
            daily_reset_utc_offset_hours=config.persona_daily_reset_utc_offset_hours,
            clock=clock,
            repository=repository,
        )

    async def initialize(self) -> NicknameStateSnapshot:
        """Migrate storage and load the one Active Nickname snapshot."""

        await asyncio.to_thread(self._repository.initialize)
        snapshot = await asyncio.to_thread(self._repository.load_state)
        self._snapshot = snapshot
        return snapshot

    @property
    def current_snapshot(self) -> NicknameStateSnapshot:
        if self._snapshot is None:
            raise NicknameNotInitializedError()
        return self._snapshot

    def get_active_nickname(self) -> str | None:
        """Synchronous hot-path provider; reads only memory."""

        return self.current_snapshot.active_nickname

    async def set_nickname(
        self,
        author_discord_id: int | str,
        nickname: str,
    ) -> NicknameSetResult:
        self._require_initialized()
        author_id = normalize_author_id(author_discord_id)
        normalized_nickname = normalize_nickname(nickname, self._nickname_max_chars)
        now_utc = self._now_utc()
        quota_start, quota_end = quota_window(now_utc, self._daily_reset_utc_offset_hours)
        version, snapshot = await asyncio.to_thread(
            self._repository.set_nickname,
            author_discord_id=author_id,
            nickname_text=normalized_nickname,
            now_utc=now_utc,
            cooldown_seconds=self._global_cooldown_seconds,
            quota_start_utc=quota_start,
            quota_end_utc=quota_end,
            daily_limit=self._daily_limit,
        )
        self._snapshot = snapshot
        status = await self.get_policy_status(author_id)
        return NicknameSetResult(version, snapshot, status)

    async def get_policy_status(self, author_discord_id: int | str) -> NicknamePolicyStatus:
        self._require_initialized()
        author_id = normalize_author_id(author_discord_id)
        now_utc = self._now_utc()
        quota_start, quota_end = quota_window(now_utc, self._daily_reset_utc_offset_hours)
        latest, daily_used = await asyncio.to_thread(
            self._repository.policy_metrics,
            author_id,
            quota_start,
            quota_end,
        )
        cooldown_remaining = 0
        if latest is not None:
            remaining = self._global_cooldown_seconds - (now_utc - latest).total_seconds()
            if remaining > 0:
                import math

                cooldown_remaining = math.ceil(remaining)
        return NicknamePolicyStatus(
            global_cooldown_remaining_seconds=cooldown_remaining,
            daily_used=daily_used,
            daily_limit=self._daily_limit,
            daily_remaining=max(0, self._daily_limit - daily_used),
            next_daily_reset_utc=quota_end,
        )

    def _require_initialized(self) -> None:
        if self._snapshot is None:
            raise NicknameNotInitializedError()

    def _now_utc(self) -> datetime:
        clock = self._clock
        value = clock.now_utc() if hasattr(clock, "now_utc") else clock()
        return as_utc(value)

    @staticmethod
    def _validate_policy_config(
        nickname_max_chars: int,
        global_cooldown_seconds: int,
        daily_limit: int,
        daily_reset_utc_offset_hours: int,
    ) -> None:
        if (
            not isinstance(nickname_max_chars, int)
            or isinstance(nickname_max_chars, bool)
            or not 1 <= nickname_max_chars <= MAX_NICKNAME_MAX_CHARS
        ):
            raise NicknameValidationError("maximum characters are out of range")
        if not isinstance(global_cooldown_seconds, int) or isinstance(
            global_cooldown_seconds, bool
        ):
            raise NicknameValidationError("cooldown must be an integer")
        if global_cooldown_seconds <= 0:
            raise NicknameValidationError("cooldown must be positive")
        if not isinstance(daily_limit, int) or isinstance(daily_limit, bool) or daily_limit <= 0:
            raise NicknameValidationError("daily limit must be positive")
        if (
            not isinstance(daily_reset_utc_offset_hours, int)
            or isinstance(daily_reset_utc_offset_hours, bool)
            or not -12 <= daily_reset_utc_offset_hours <= 14
        ):
            raise NicknameValidationError("daily reset offset is out of range")


__all__ = [
    "NicknameDailyLimitError",
    "NicknameDatabaseError",
    "NicknameError",
    "NicknameGlobalCooldownError",
    "NicknameNotInitializedError",
    "NicknamePolicyStatus",
    "NicknameRepository",
    "NicknameSetResult",
    "NicknameService",
    "NicknameStateSnapshot",
    "NicknameTooLongError",
    "NicknameValidationError",
    "NicknameVersion",
    "normalize_nickname",
]

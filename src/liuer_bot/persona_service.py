"""Async Persona persistence service and global policy boundary."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .config import (
    DEFAULT_PERSONA_DAILY_LIMIT,
    DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS,
    DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS,
    DEFAULT_PERSONA_MAX_CHARS,
    Config,
)
from .persona_models import (
    PersonaDailyLimitError,
    PersonaError,
    PersonaGlobalCooldownError,
    PersonaNoRollbackTargetError,
    PersonaNotInitializedError,
    PersonaPolicyStatus,
    PersonaPublishResult,
    PersonaStateSnapshot,
    PersonaTooLongError,
    PersonaValidationError,
    PersonaVersion,
)
from .persona_repository import PersonaDatabaseError, PersonaRepository
from .persona_time import Clock, SystemClock, as_utc, next_quota_reset, quota_window


def normalize_persona_text(persona_text: str, max_chars: int) -> str:
    """Validate a community Persona while preserving internal formatting."""

    if not isinstance(persona_text, str):
        raise PersonaValidationError("text must be a string")
    normalized = persona_text.strip()
    if not normalized:
        raise PersonaValidationError("text must not be blank")
    if len(normalized) > max_chars:
        raise PersonaTooLongError(max_chars)
    return normalized


def normalize_author_id(author_discord_id: int | str) -> str:
    """Store Discord snowflake identifiers as non-empty text."""

    if not isinstance(author_discord_id, (int, str)):
        raise PersonaValidationError("author ID must be text or an integer")
    normalized = str(author_discord_id).strip()
    if not normalized:
        raise PersonaValidationError("author ID must not be blank")
    return normalized


class PersonaService:
    """Own the active in-memory Persona snapshot and async domain API."""

    def __init__(
        self,
        database_path: Path,
        *,
        persona_max_chars: int = DEFAULT_PERSONA_MAX_CHARS,
        global_cooldown_seconds: int = DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS,
        daily_limit: int = DEFAULT_PERSONA_DAILY_LIMIT,
        daily_reset_utc_offset_hours: int = DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS,
        clock: Clock | Callable[[], datetime] | None = None,
        repository: PersonaRepository | None = None,
    ) -> None:
        self._validate_policy_config(
            persona_max_chars,
            global_cooldown_seconds,
            daily_limit,
            daily_reset_utc_offset_hours,
        )
        self._database_path = Path(database_path)
        self._persona_max_chars = persona_max_chars
        self._global_cooldown_seconds = global_cooldown_seconds
        self._daily_limit = daily_limit
        self._daily_reset_utc_offset_hours = daily_reset_utc_offset_hours
        self._clock = clock or SystemClock()
        self._repository = repository or PersonaRepository(self._database_path)
        self._snapshot: PersonaStateSnapshot | None = None

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        clock: Clock | Callable[[], datetime] | None = None,
        repository: PersonaRepository | None = None,
    ) -> PersonaService:
        return cls(
            config.database_path,
            persona_max_chars=config.persona_max_chars,
            global_cooldown_seconds=config.persona_global_cooldown_seconds,
            daily_limit=config.persona_daily_limit,
            daily_reset_utc_offset_hours=config.persona_daily_reset_utc_offset_hours,
            clock=clock,
            repository=repository,
        )

    async def initialize(self) -> PersonaStateSnapshot:
        """Migrate the database and load one immutable active snapshot."""

        await asyncio.to_thread(self._repository.initialize)
        snapshot = await asyncio.to_thread(self._repository.load_state)
        self._snapshot = snapshot
        return snapshot

    @property
    def current_snapshot(self) -> PersonaStateSnapshot:
        if self._snapshot is None:
            raise PersonaNotInitializedError()
        return self._snapshot

    def get_active_persona(self) -> str | None:
        """Synchronous provider API; reads only the in-memory snapshot."""

        return self.current_snapshot.active_persona

    async def get_status_message_id(self) -> int | None:
        """Read the Discord presentation pointer without reading Persona text."""

        self._require_initialized()
        return await asyncio.to_thread(self._repository.load_status_message_id)

    async def set_status_message_id(self, status_message_id: int | None) -> None:
        """Persist the Discord presentation pointer after a successful sync."""

        self._require_initialized()
        await asyncio.to_thread(self._repository.save_status_message_id, status_message_id)

    async def publish_persona(
        self,
        author_discord_id: int | str,
        persona_text: str,
    ) -> PersonaPublishResult:
        self._require_initialized()
        author_id = normalize_author_id(author_discord_id)
        normalized_text = normalize_persona_text(persona_text, self._persona_max_chars)
        now_utc = self._now_utc()
        quota_start, quota_end = quota_window(now_utc, self._daily_reset_utc_offset_hours)
        version, snapshot = await asyncio.to_thread(
            self._repository.publish,
            author_id,
            normalized_text,
            now_utc,
            self._global_cooldown_seconds,
            quota_start,
            quota_end,
            self._daily_limit,
        )
        self._snapshot = snapshot
        status = await self.get_policy_status(author_id)
        return PersonaPublishResult(version, snapshot, status)

    async def get_policy_status(self, author_discord_id: int | str) -> PersonaPolicyStatus:
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
                cooldown_remaining = math.ceil(remaining)
        return PersonaPolicyStatus(
            global_cooldown_remaining_seconds=cooldown_remaining,
            daily_used=daily_used,
            daily_limit=self._daily_limit,
            daily_remaining=max(0, self._daily_limit - daily_used),
            next_daily_reset_utc=quota_end,
        )

    async def list_history(self, limit: int = 10) -> tuple[PersonaVersion, ...]:
        self._require_initialized()
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise PersonaValidationError("history limit must be positive")
        return await asyncio.to_thread(self._repository.list_history, min(limit, 50))

    async def reset_to_default(self, actor_discord_id: int | str) -> PersonaStateSnapshot:
        self._require_initialized()
        normalize_author_id(actor_discord_id)
        snapshot = await asyncio.to_thread(self._repository.reset, self._now_utc())
        self._snapshot = snapshot
        return snapshot

    async def rollback(self, actor_discord_id: int | str) -> PersonaStateSnapshot:
        self._require_initialized()
        normalize_author_id(actor_discord_id)
        snapshot = await asyncio.to_thread(self._repository.rollback, self._now_utc())
        self._snapshot = snapshot
        return snapshot

    def _require_initialized(self) -> None:
        if self._snapshot is None:
            raise PersonaNotInitializedError()

    def _now_utc(self) -> datetime:
        clock = self._clock
        value = clock.now_utc() if hasattr(clock, "now_utc") else clock()
        return as_utc(value)

    @staticmethod
    def _validate_policy_config(
        persona_max_chars: int,
        global_cooldown_seconds: int,
        daily_limit: int,
        daily_reset_utc_offset_hours: int,
    ) -> None:
        if not isinstance(persona_max_chars, int) or isinstance(persona_max_chars, bool):
            raise PersonaValidationError("maximum characters must be an integer")
        if persona_max_chars <= 0:
            raise PersonaValidationError("maximum characters must be positive")
        if not isinstance(global_cooldown_seconds, int) or isinstance(
            global_cooldown_seconds, bool
        ):
            raise PersonaValidationError("cooldown must be an integer")
        if global_cooldown_seconds <= 0:
            raise PersonaValidationError("cooldown must be positive")
        if not isinstance(daily_limit, int) or isinstance(daily_limit, bool) or daily_limit <= 0:
            raise PersonaValidationError("daily limit must be positive")
        if (
            not isinstance(daily_reset_utc_offset_hours, int)
            or isinstance(daily_reset_utc_offset_hours, bool)
            or not -12 <= daily_reset_utc_offset_hours <= 14
        ):
            raise PersonaValidationError("daily reset offset is out of range")


__all__ = [
    "PersonaDailyLimitError",
    "PersonaDatabaseError",
    "PersonaError",
    "PersonaGlobalCooldownError",
    "PersonaNoRollbackTargetError",
    "PersonaNotInitializedError",
    "PersonaPolicyStatus",
    "PersonaPublishResult",
    "PersonaService",
    "PersonaStateSnapshot",
    "PersonaTooLongError",
    "PersonaValidationError",
    "PersonaVersion",
    "next_quota_reset",
]

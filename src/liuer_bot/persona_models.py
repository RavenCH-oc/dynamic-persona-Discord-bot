"""Immutable domain models and privacy-safe persona policy errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class PersonaError(RuntimeError):
    """Base class for safe PersonaService failures."""


class PersonaValidationError(PersonaError, ValueError):
    """Persona input or policy configuration is invalid."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"invalid persona: {reason}")


class PersonaTooLongError(PersonaValidationError):
    """The normalized persona exceeds the configured character limit."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        super().__init__(f"text exceeds maximum length of {max_chars} characters")


class PersonaGlobalCooldownError(PersonaError):
    """A successful publish is still inside the Bot-global cooldown."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = max(0, retry_after_seconds)
        super().__init__("persona publish is on global cooldown")


class PersonaDailyLimitError(PersonaError):
    """The author has reached the configured successful-publish daily limit."""

    def __init__(self, daily_limit: int, next_reset_utc: datetime) -> None:
        self.daily_limit = daily_limit
        self.next_reset_utc = next_reset_utc
        super().__init__("persona daily publish limit reached")


class PersonaNoRollbackTargetError(PersonaError):
    """No older historical persona is available for rollback."""

    def __init__(self) -> None:
        super().__init__("no persona rollback target is available")


class PersonaNotInitializedError(PersonaError):
    """A runtime service API was used before startup initialization."""

    def __init__(self) -> None:
        super().__init__("persona service is not initialized")


@dataclass(frozen=True, slots=True, repr=False)
class PersonaVersion:
    """One immutable, successfully published community persona version."""

    version_id: int
    author_discord_id: str
    persona_text: str = field(repr=False)
    created_at_utc: datetime

    def __repr__(self) -> str:
        return (
            "PersonaVersion("
            f"version_id={self.version_id!r}, "
            f"author_discord_id={self.author_discord_id!r}, "
            "persona_text=<redacted>, "
            f"created_at_utc={self.created_at_utc!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class PersonaStateSnapshot:
    """The in-memory active Persona state loaded from SQLite."""

    revision: int
    active_version_id: int | None
    active_persona: str | None = field(repr=False)
    updated_at_utc: datetime

    def __repr__(self) -> str:
        return (
            "PersonaStateSnapshot("
            f"revision={self.revision!r}, "
            f"active_version_id={self.active_version_id!r}, "
            "active_persona=<redacted>, "
            f"updated_at_utc={self.updated_at_utc!r})"
        )


@dataclass(frozen=True, slots=True)
class PersonaPolicyStatus:
    """Safe policy metadata for a future command/status layer."""

    global_cooldown_remaining_seconds: int
    daily_used: int
    daily_limit: int
    daily_remaining: int
    next_daily_reset_utc: datetime


@dataclass(frozen=True, slots=True)
class PersonaPublishResult:
    """Metadata returned after a committed Persona publish."""

    version: PersonaVersion
    state: PersonaStateSnapshot
    policy_status: PersonaPolicyStatus

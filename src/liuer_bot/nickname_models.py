"""Immutable nickname models and privacy-safe nickname policy errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class NicknameError(RuntimeError):
    """Base class for safe NicknameService failures."""


class NicknameValidationError(NicknameError, ValueError):
    """Nickname input or policy configuration is invalid."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"invalid nickname: {reason}")


class NicknameTooLongError(NicknameValidationError):
    """The normalized nickname exceeds the configured character limit."""

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        super().__init__(f"text exceeds maximum length of {max_chars} characters")


class NicknameGlobalCooldownError(NicknameError):
    """A successful nickname update is still inside the global cooldown."""

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = max(0, retry_after_seconds)
        super().__init__("nickname update is on global cooldown")


class NicknameDailyLimitError(NicknameError):
    """The author reached the successful nickname-update daily limit."""

    def __init__(self, daily_limit: int, next_reset_utc: datetime) -> None:
        self.daily_limit = daily_limit
        self.next_reset_utc = next_reset_utc
        super().__init__("nickname daily update limit reached")


class NicknameNotInitializedError(NicknameError):
    """A nickname runtime API was used before startup initialization."""

    def __init__(self) -> None:
        super().__init__("nickname service is not initialized")


@dataclass(frozen=True, slots=True, repr=False)
class NicknameVersion:
    """One immutable, successfully committed community nickname version."""

    version_id: int
    author_discord_id: str
    nickname_text: str = field(repr=False)
    created_at_utc: datetime

    def __repr__(self) -> str:
        return (
            "NicknameVersion("
            f"version_id={self.version_id!r}, "
            f"author_discord_id={self.author_discord_id!r}, "
            "nickname_text=<redacted>, "
            f"created_at_utc={self.created_at_utc!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class NicknameStateSnapshot:
    """The in-memory global Active Nickname state."""

    revision: int
    active_version_id: int | None
    active_nickname: str | None = field(repr=False)
    updated_at_utc: datetime

    def __repr__(self) -> str:
        return (
            "NicknameStateSnapshot("
            f"revision={self.revision!r}, "
            f"active_version_id={self.active_version_id!r}, "
            "active_nickname=<redacted>, "
            f"updated_at_utc={self.updated_at_utc!r})"
        )


@dataclass(frozen=True, slots=True)
class NicknamePolicyStatus:
    """Safe policy metadata for the `/set name` command."""

    global_cooldown_remaining_seconds: int
    daily_used: int
    daily_limit: int
    daily_remaining: int
    next_daily_reset_utc: datetime


@dataclass(frozen=True, slots=True)
class NicknameSetResult:
    """Metadata returned after a committed nickname update."""

    version: NicknameVersion
    state: NicknameStateSnapshot
    policy_status: NicknamePolicyStatus


__all__ = [
    "NicknameDailyLimitError",
    "NicknameError",
    "NicknameGlobalCooldownError",
    "NicknameNotInitializedError",
    "NicknamePolicyStatus",
    "NicknameSetResult",
    "NicknameStateSnapshot",
    "NicknameTooLongError",
    "NicknameValidationError",
    "NicknameVersion",
]

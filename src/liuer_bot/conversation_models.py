"""Privacy-safe immutable models for completed conversation turns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

DISPLAY_NAME_MAX_CHARS = 80
DEFAULT_DISPLAY_NAME = "Discord user"


class AuthorKind(StrEnum):
    """Explicit speaker kind persisted with every conversation turn."""

    HUMAN = "HUMAN"
    BOT = "BOT"


def sanitize_display_name(value: object) -> str:
    """Keep a bounded display-name snapshot without carrying line breaks."""

    if not isinstance(value, str):
        return DEFAULT_DISPLAY_NAME
    normalized = value.strip().replace("\r", " ").replace("\n", " ")
    normalized = " ".join(normalized.split())
    if not normalized:
        return DEFAULT_DISPLAY_NAME
    return normalized[:DISPLAY_NAME_MAX_CHARS].strip() or DEFAULT_DISPLAY_NAME


@dataclass(frozen=True, slots=True, repr=False)
class ConversationTurn:
    """One successfully generated user/assistant turn."""

    turn_id: int
    chat_channel_id: int
    user_message_id: int
    user_author_id: int
    user_display_name: str
    user_content: str
    user_image_count: int
    assistant_content: str
    created_at_utc: datetime
    context_epoch: int = 1
    author_kind: AuthorKind = AuthorKind.HUMAN

    def __repr__(self) -> str:
        return (
            "ConversationTurn("
            f"turn_id={self.turn_id!r}, "
            f"chat_channel_id={self.chat_channel_id!r}, "
            f"user_message_id={self.user_message_id!r}, "
            f"user_author_id={self.user_author_id!r}, "
            "user_display_name=<redacted>, "
            "user_content=<redacted>, "
            f"user_image_count={self.user_image_count!r}, "
            "assistant_content=<redacted>, "
            f"created_at_utc={self.created_at_utc!r}, "
            f"context_epoch={self.context_epoch!r}, "
            f"author_kind={self.author_kind.value!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ConversationState:
    """Persistent active conversation epoch metadata."""

    active_epoch: int
    revision: int
    updated_at_utc: datetime

    def __repr__(self) -> str:
        return (
            "ConversationState("
            f"active_epoch={self.active_epoch!r}, "
            f"revision={self.revision!r}, "
            f"updated_at_utc={self.updated_at_utc!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ConversationClearResult:
    """Privacy-safe result of one committed conversation epoch advance."""

    previous_epoch: int
    active_epoch: int
    revision: int
    updated_at_utc: datetime

    def __repr__(self) -> str:
        return (
            "ConversationClearResult("
            f"previous_epoch={self.previous_epoch!r}, "
            f"active_epoch={self.active_epoch!r}, "
            f"revision={self.revision!r}, "
            f"updated_at_utc={self.updated_at_utc!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ConversationRuntimeStatus:
    """Atomic count-only conversation state for local runtime observability."""

    initialized: bool
    active_epoch: int | None
    recent_turn_count: int

    def __repr__(self) -> str:
        return (
            "ConversationRuntimeStatus("
            f"initialized={self.initialized!r}, "
            f"active_epoch={self.active_epoch!r}, "
            f"recent_turn_count={self.recent_turn_count!r})"
        )


__all__ = [
    "ConversationTurn",
    "AuthorKind",
    "ConversationState",
    "ConversationClearResult",
    "ConversationRuntimeStatus",
    "DEFAULT_DISPLAY_NAME",
    "DISPLAY_NAME_MAX_CHARS",
    "sanitize_display_name",
]

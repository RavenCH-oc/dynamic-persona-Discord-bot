"""The responder boundary shared by the Discord runtime and model backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .conversation_context import ConversationGenerationContext
from .conversation_models import AuthorKind
from .image_preparation import PreparedImage
from .reply_context import ReplyContext


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    """Immutable, download-free image metadata passed to a responder."""

    attachment_id: int
    media_type: str
    size_bytes: int


@dataclass(frozen=True, slots=True, repr=False)
class ChatRequest:
    """The small message DTO passed from Discord transport to a responder."""

    message_id: int
    guild_id: int
    channel_id: int
    author_id: int
    content: str
    prepared_images: tuple[PreparedImage, ...] = ()
    author_display_name: str = ""
    reply_context: ReplyContext | None = None
    generation_context: ConversationGenerationContext | None = None
    author_kind: AuthorKind = AuthorKind.HUMAN

    def __repr__(self) -> str:
        return (
            "ChatRequest("
            f"message_id={self.message_id!r}, "
            f"guild_id={self.guild_id!r}, "
            f"channel_id={self.channel_id!r}, "
            f"author_id={self.author_id!r}, "
            "author_display_name=<redacted>, "
            "content=<redacted>, "
            "generation_context=<redacted>, "
            f"author_kind={self.author_kind.value!r})"
        )


class Responder(Protocol):
    """Async boundary for future model-backed responders."""

    async def generate(self, request: ChatRequest) -> str:
        """Generate a response for a validated chat request."""


class Phase1FakeResponder:
    """Return deterministic text without pretending to call an AI model."""

    async def generate(self, request: ChatRequest) -> str:
        return "六耳已收到訊息。"

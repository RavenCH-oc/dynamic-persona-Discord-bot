"""Safe, context-only resolution of Discord reply references."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .conversation_models import AuthorKind, sanitize_display_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, repr=False)
class ReplyContext:
    """Immutable model-safe metadata for one replied-to Discord message."""

    message_id: int
    author_display_name: str
    author_is_bot: bool
    content: str
    image_count: int
    is_current_bot: bool = False
    is_available: bool = True
    author_kind: AuthorKind = AuthorKind.HUMAN

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "author_display_name",
            sanitize_display_name(self.author_display_name),
        )
        object.__setattr__(
            self,
            "content",
            self.content.strip() if isinstance(self.content, str) else "",
        )
        if self.image_count < 0:
            raise ValueError("reply image count cannot be negative")

    def __repr__(self) -> str:
        return (
            "ReplyContext("
            f"message_id={self.message_id!r}, "
            "author_display_name=<redacted>, "
            f"author_is_bot={self.author_is_bot!r}, "
            "content=<redacted>, "
            f"image_count={self.image_count!r}, "
            f"is_current_bot={self.is_current_bot!r}, "
            f"is_available={self.is_available!r}, "
            f"author_kind={self.author_kind.value!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ReplyResolution:
    """Resolution result retained only at the current transport boundary."""

    context: ReplyContext | None
    source: str
    referenced_message_id: int | None

    def __repr__(self) -> str:
        return (
            "ReplyResolution("
            f"source={self.source!r}, "
            f"referenced_message_id={self.referenced_message_id!r}, "
            "context=<redacted>)"
        )


async def resolve_reply_context(
    message: object,
    *,
    bot_user_id: int | None,
    logger: logging.Logger = LOGGER,
) -> ReplyResolution:
    """Resolve one reply after normal routing has already accepted the message."""

    reference = getattr(message, "reference", None)
    if reference is None:
        return ReplyResolution(None, "NONE", None)

    referenced_message_id = _message_id(getattr(reference, "message_id", None))
    candidate = _valid_message(getattr(reference, "resolved", None))
    source = "RESOLVED"
    if candidate is None:
        candidate = _valid_message(getattr(reference, "cached_message", None))
        source = "CACHE"

    if candidate is None and referenced_message_id is not None:
        fetch_message = getattr(getattr(message, "channel", None), "fetch_message", None)
        if callable(fetch_message):
            try:
                fetched = await fetch_message(referenced_message_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(
                    "reply context resolution failed message_id=%s referenced_message_id=%s "
                    "source=REST error_type=%s",
                    getattr(message, "id", None),
                    referenced_message_id,
                    type(exc).__name__,
                )
            else:
                candidate = _valid_message(fetched)
                source = "REST"

    if candidate is None:
        context = _unavailable_context(referenced_message_id)
        source = "UNAVAILABLE"
    else:
        context = _build_context(candidate, bot_user_id=bot_user_id)

    logger.debug(
        "reply context resolved message_id=%s referenced_message_id=%s source=%s "
        "author_is_bot=%s image_count=%s",
        getattr(message, "id", None),
        referenced_message_id,
        source,
        context.author_is_bot,
        context.image_count,
    )
    return ReplyResolution(context, source, referenced_message_id)


def _valid_message(message: object) -> object | None:
    if message is None:
        return None
    if _message_id(getattr(message, "id", None)) is None:
        return None
    author = getattr(message, "author", None)
    if author is None or _message_id(getattr(author, "id", None)) is None:
        return None
    if not isinstance(getattr(message, "content", None), str):
        return None
    return message


def _build_context(message: object, *, bot_user_id: int | None) -> ReplyContext:
    author = getattr(message, "author")
    author_id = _message_id(getattr(author, "id", None))
    author_is_bot = bool(getattr(author, "bot", False))
    return ReplyContext(
        message_id=_message_id(getattr(message, "id")) or 0,
        author_display_name=_display_name(author),
        author_is_bot=author_is_bot,
        content=getattr(message, "content"),
        image_count=_supported_image_count(getattr(message, "attachments", ())),
        is_current_bot=(
            author_is_bot and bot_user_id is not None and author_id == bot_user_id
        ),
        author_kind=AuthorKind.BOT if author_is_bot else AuthorKind.HUMAN,
    )


def _unavailable_context(message_id: int | None) -> ReplyContext:
    return ReplyContext(
        message_id=message_id or 0,
        author_display_name="Discord user",
        author_is_bot=False,
        content="",
        image_count=0,
        is_available=False,
    )


def _display_name(author: object) -> str:
    value = getattr(author, "display_name", None)
    if not isinstance(value, str):
        value = getattr(author, "name", None)
    return sanitize_display_name(value)


def _supported_image_count(attachments: object) -> int:
    try:
        return sum(
            1
            for attachment in attachments
            if _classify_safely(attachment) is not None
        )
    except (TypeError, AttributeError):
        return 0


def _classify_safely(attachment: object) -> object | None:
    try:
        from .attachment_metadata import classify_image_attachment

        return classify_image_attachment(attachment)  # type: ignore[arg-type]
    except (AttributeError, TypeError, ValueError):
        return None


def _message_id(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


__all__ = ["ReplyContext", "ReplyResolution", "resolve_reply_context"]

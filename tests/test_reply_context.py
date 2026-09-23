from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import discord
import pytest

from liuer_bot.reply_context import ReplyContext, resolve_reply_context


def _author(
    author_id: int = 10,
    *,
    bot: bool = False,
    display_name: str = "Alice",
) -> SimpleNamespace:
    return SimpleNamespace(id=author_id, bot=bot, display_name=display_name)


def _message(
    message_id: int = 42,
    *,
    content: str = "referenced text",
    author: object | None = None,
    attachments: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        author=author or _author(),
        content=content,
        attachments=attachments or [],
    )


def _run(awaitable):
    return asyncio.run(awaitable)


def test_resolved_message_is_preferred_and_context_repr_is_redacted() -> None:
    referenced = _message(content="phase4ab-private-replied-text")
    fetch_calls: list[int] = []

    async def fetch_message(message_id: int) -> object:
        fetch_calls.append(message_id)
        raise AssertionError("resolved message must avoid REST")

    current = SimpleNamespace(
        id=1,
        reference=SimpleNamespace(message_id=42, resolved=referenced),
        channel=SimpleNamespace(fetch_message=fetch_message),
    )

    result = _run(resolve_reply_context(current, bot_user_id=999))

    assert result.source == "RESOLVED"
    assert result.referenced_message_id == 42
    assert result.context == ReplyContext(42, "Alice", False, "phase4ab-private-replied-text", 0)
    assert fetch_calls == []
    assert "phase4ab-private-replied-text" not in repr(result.context)
    assert "Alice" not in repr(result.context)


def test_cached_message_is_used_before_rest_fallback() -> None:
    cached = _message(43, content="cached reply")
    fetch_calls: list[int] = []

    async def fetch_message(message_id: int) -> object:
        fetch_calls.append(message_id)
        raise AssertionError("cached message must avoid REST")

    current = SimpleNamespace(
        id=1,
        reference=SimpleNamespace(message_id=43, resolved=None, cached_message=cached),
        channel=SimpleNamespace(fetch_message=fetch_message),
    )

    result = _run(resolve_reply_context(current, bot_user_id=999))

    assert result.source == "CACHE"
    assert result.context is not None
    assert result.context.content == "cached reply"
    assert fetch_calls == []


def test_rest_fetch_is_used_when_resolved_and_cached_are_unavailable() -> None:
    fetched = _message(44, content="rest reply")
    fetch_calls: list[int] = []

    async def fetch_message(message_id: int) -> object:
        fetch_calls.append(message_id)
        return fetched

    current = SimpleNamespace(
        id=1,
        reference=SimpleNamespace(message_id=44, resolved=None, cached_message=None),
        channel=SimpleNamespace(fetch_message=fetch_message),
    )

    result = _run(resolve_reply_context(current, bot_user_id=999))

    assert result.source == "REST"
    assert result.context is not None
    assert result.context.content == "rest reply"
    assert fetch_calls == [44]


@pytest.mark.parametrize(
    "error",
    [
        discord.NotFound(SimpleNamespace(status=404, reason="private", headers={}), "private"),
        discord.Forbidden(SimpleNamespace(status=403, reason="private", headers={}), "private"),
        discord.HTTPException(SimpleNamespace(status=500, reason="private", headers={}), "private"),
    ],
)
def test_resolution_errors_are_unavailable_and_do_not_leak_error_body(
    error: Exception, caplog
) -> None:
    async def fetch_message(message_id: int) -> object:
        raise error

    current = SimpleNamespace(
        id=1,
        reference=SimpleNamespace(message_id=45, resolved=None, cached_message=None),
        channel=SimpleNamespace(fetch_message=fetch_message),
    )
    logger = logging.getLogger(f"test.reply-resolution.{type(error).__name__}")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = _run(resolve_reply_context(current, bot_user_id=999, logger=logger))

    assert result.source == "UNAVAILABLE"
    assert result.context is not None
    assert result.context.is_available is False
    assert result.context.content == ""
    assert "private" not in caplog.text
    assert "reference" not in repr(result.context)


def test_bot_identity_and_supported_image_count_are_metadata_only() -> None:
    attachments = [
        SimpleNamespace(id=1, content_type="image/png", filename="one.png", size=10),
        SimpleNamespace(id=2, content_type="text/plain", filename="two.txt", size=10),
    ]
    referenced = _message(
        46,
        author=_author(999, bot=True, display_name="Old Nickname"),
        attachments=attachments,
    )
    current = SimpleNamespace(
        id=1,
        reference=SimpleNamespace(message_id=46, resolved=referenced),
        channel=SimpleNamespace(),
    )

    result = _run(resolve_reply_context(current, bot_user_id=999))

    assert result.context is not None
    assert result.context.author_is_bot is True
    assert result.context.is_current_bot is True
    assert result.context.image_count == 1
    assert result.context.content == "referenced text"


def test_other_bot_is_context_only_and_no_reference_is_noop() -> None:
    other_bot = _message(47, author=_author(777, bot=True, display_name="Other Bot"))
    current = SimpleNamespace(
        id=1,
        reference=SimpleNamespace(message_id=47, resolved=other_bot),
        channel=SimpleNamespace(),
    )

    result = _run(resolve_reply_context(current, bot_user_id=999))
    no_reply = _run(resolve_reply_context(SimpleNamespace(id=2, reference=None), bot_user_id=999))

    assert result.context is not None
    assert result.context.author_is_bot is True
    assert result.context.is_current_bot is False
    assert no_reply.context is None
    assert no_reply.source == "NONE"

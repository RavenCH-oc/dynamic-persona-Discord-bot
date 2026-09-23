from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from liuer_bot.conversation_models import ConversationTurn, sanitize_display_name
from liuer_bot.conversation_repository import ConversationDatabaseError
from liuer_bot.conversation_service import (
    ConversationPersistenceError,
    ConversationRecordingResponder,
    ConversationService,
)
from liuer_bot.generation_queue import SerializedGenerationQueue
from liuer_bot.image_preparation import PreparedImage
from liuer_bot.reply_context import ReplyContext
from liuer_bot.responder import ChatRequest

PRIVATE_USER = "phase4aa-private-user-content"
PRIVATE_ASSISTANT = "phase4aa-private-assistant-content"
PRIVATE_NAME = "phase4aa-private-display-name"


def _request(
    message_id: int,
    *,
    channel_id: int = 123,
    content: str = PRIVATE_USER,
    display_name: str = PRIVATE_NAME,
    images: tuple[PreparedImage, ...] = (),
) -> ChatRequest:
    return ChatRequest(
        message_id=message_id,
        guild_id=20,
        channel_id=channel_id,
        author_id=10,
        content=content,
        prepared_images=images,
        author_display_name=display_name,
    )


def _run(awaitable):
    return asyncio.run(awaitable)


def test_empty_initialization_and_successful_turn_persist_only_safe_fields(tmp_path: Path) -> None:
    path = tmp_path / "conversation.sqlite3"
    service = ConversationService(path, chat_channel_id=123, recent_turn_limit=12)

    assert _run(service.initialize()) == ()
    image = PreparedImage(77, "image/png", b"phase4aa-private-image-bytes", 1, 1)
    turn = _run(service.persist_successful_turn(_request(1, images=(image,)), PRIVATE_ASSISTANT))

    assert turn.turn_id == 1
    assert turn.chat_channel_id == 123
    assert turn.user_message_id == 1
    assert turn.user_author_id == 10
    assert turn.user_display_name == PRIVATE_NAME
    assert turn.user_content == PRIVATE_USER
    assert turn.user_image_count == 1
    assert turn.assistant_content == PRIVATE_ASSISTANT
    assert service.get_recent_turns() == (turn,)

    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT chat_channel_id, user_message_id, user_author_id, user_display_name, "
            "user_content, user_image_count, assistant_content FROM conversation_turns"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("123", "1", "10", PRIVATE_NAME, PRIVATE_USER, 1, PRIVATE_ASSISTANT)
    assert "phase4aa-private-image-bytes" not in repr(turn)


def test_successful_turn_persists_current_content_without_reply_context(tmp_path: Path) -> None:
    path = tmp_path / "conversation.sqlite3"
    service = ConversationService(path, chat_channel_id=123)
    _run(service.initialize())
    reply_body = "phase4ab-private-replied-content"
    request = ChatRequest(
        message_id=2,
        guild_id=20,
        channel_id=123,
        author_id=11,
        content="current addressed content",
        author_display_name="Bob",
        reply_context=ReplyContext(1, "Alice", False, reply_body, 0),
    )

    turn = _run(service.persist_successful_turn(request, "current answer"))

    assert turn.user_content == "current addressed content"
    assert reply_body not in repr(turn)
    assert reply_body not in turn.user_content
    assert reply_body not in str(service.get_recent_turns())


def test_display_name_sanitization_and_turn_repr_redaction() -> None:
    assert sanitize_display_name("  Raven\r\nTester  ") == "Raven Tester"
    assert sanitize_display_name("\r\n") == "Discord user"
    turn = ConversationTurn(
        1,
        123,
        1,
        10,
        PRIVATE_NAME,
        PRIVATE_USER,
        0,
        PRIVATE_ASSISTANT,
        datetime.now(UTC),
    )

    rendered = repr(turn)
    assert PRIVATE_NAME not in rendered
    assert PRIVATE_USER not in rendered
    assert PRIVATE_ASSISTANT not in rendered


def test_recent_snapshot_is_oldest_to_newest_bounded_and_restartable(tmp_path: Path) -> None:
    path = tmp_path / "conversation.sqlite3"
    service = ConversationService(path, chat_channel_id=123, recent_turn_limit=12)
    _run(service.initialize())

    for message_id in range(1, 14):
        _run(
            service.persist_successful_turn(
                _request(message_id, content=f"user-{message_id}"),
                f"assistant-{message_id}",
            )
        )

    assert [turn.user_message_id for turn in service.get_recent_turns()] == list(range(2, 14))

    service._repository.insert_turn(  # type: ignore[union-attr]
        chat_channel_id=999,
        user_message_id=900,
        user_author_id=10,
        user_display_name="old channel",
        user_content="old channel content",
        user_image_count=0,
        assistant_content="old channel answer",
        created_at_utc=datetime.now(UTC),
    )
    restarted = ConversationService(path, chat_channel_id=123, recent_turn_limit=12)
    _run(restarted.initialize())

    assert [turn.user_message_id for turn in restarted.get_recent_turns()] == list(range(2, 14))


def test_get_recent_turns_is_memory_only_after_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())

    def fail_database_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("hot path must not read SQLite")

    monkeypatch.setattr(service._repository, "load_recent_turns", fail_database_read)
    assert service.get_recent_turns() == ()


def test_generation_failure_creates_no_turn_and_worker_continues(tmp_path: Path) -> None:
    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())

    class Responder:
        async def generate(self, request: ChatRequest) -> str:
            if request.message_id == 1:
                raise RuntimeError("phase4aa-private-model-failure")
            return "second answer"

    async def run() -> None:
        queue = SerializedGenerationQueue(
            ConversationRecordingResponder(Responder(), service)
        )
        await queue.start()
        try:
            with pytest.raises(RuntimeError):
                await queue.submit(_request(1))
            assert await queue.submit(_request(2)) == "second answer"
        finally:
            await queue.close()

    _run(run())
    assert [turn.user_message_id for turn in service.get_recent_turns()] == [2]


def test_persistence_failure_fails_job_then_worker_continues_without_success_reply(
    tmp_path: Path,
    caplog,
) -> None:
    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())
    repository = service._repository
    original_insert = repository.insert_turn  # type: ignore[union-attr]
    attempts = 0

    def insert_once_fails(*args: object, **kwargs: object):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConversationDatabaseError("insert turn")
        return original_insert(*args, **kwargs)

    repository.insert_turn = insert_once_fails  # type: ignore[method-assign,union-attr]

    class Responder:
        async def generate(self, request: ChatRequest) -> str:
            return f"answer-{request.message_id}"

    async def run() -> None:
        queue = SerializedGenerationQueue(
            ConversationRecordingResponder(Responder(), service)
        )
        await queue.start()
        try:
            with pytest.raises(ConversationPersistenceError):
                await queue.submit(_request(1))
            assert await queue.submit(_request(2)) == "answer-2"
        finally:
            await queue.close()

    with caplog.at_level("ERROR"):
        _run(run())
    assert "operation=insert_turn" in caplog.text
    assert PRIVATE_USER not in caplog.text
    assert PRIVATE_ASSISTANT not in caplog.text
    assert PRIVATE_NAME not in caplog.text
    assert [turn.user_message_id for turn in service.get_recent_turns()] == [2]


def test_successful_turn_is_visible_before_next_queued_generation(tmp_path: Path) -> None:
    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())
    seen_before_generation: list[tuple[int, tuple[int, ...]]] = []

    class Responder:
        async def generate(self, request: ChatRequest) -> str:
            seen_before_generation.append(
                (
                    request.message_id,
                    tuple(turn.user_message_id for turn in service.get_recent_turns()),
                )
            )
            return f"answer-{request.message_id}"

    async def run() -> None:
        queue = SerializedGenerationQueue(
            ConversationRecordingResponder(Responder(), service)
        )
        await queue.start()
        try:
            await asyncio.gather(
                queue.submit(_request(1)),
                queue.submit(_request(2)),
                queue.submit(_request(3)),
            )
        finally:
            await queue.close()

    _run(run())
    assert seen_before_generation == [(1, ()), (2, (1,)), (3, (1, 2))]

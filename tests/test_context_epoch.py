from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from liuer_bot.conversation_context import ConversationGenerationContext
from liuer_bot.conversation_models import ConversationTurn
from liuer_bot.conversation_repository import ConversationDatabaseError
from liuer_bot.conversation_service import ConversationRecordingResponder, ConversationService
from liuer_bot.generation_queue import SerializedGenerationQueue
from liuer_bot.lm_studio_responder import LmStudioResponder
from liuer_bot.nickname_service import NicknameService
from liuer_bot.persona_service import PersonaService
from liuer_bot.reply_context import ReplyContext
from liuer_bot.responder import ChatRequest

PRIVATE_USER = "phase5b-private-user-content"
PRIVATE_ASSISTANT = "phase5b-private-assistant-content"
PRIVATE_PERSONA = "phase5b-private-persona-body"
PRIVATE_NICKNAME = "phase5b-private-nickname"


@dataclass
class FixedClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


def _run(awaitable):
    return asyncio.run(awaitable)


def _request(message_id: int, content: str = PRIVATE_USER, *, reply_context=None) -> ChatRequest:
    return ChatRequest(
        message_id=message_id,
        guild_id=20,
        channel_id=123,
        author_id=10,
        content=content,
        author_display_name="phase5b-private-display-name",
        reply_context=reply_context,
    )


def _service(path: Path, *, clock: FixedClock | None = None) -> ConversationService:
    return ConversationService(
        path,
        chat_channel_id=123,
        recent_turn_limit=12,
        clock=clock or FixedClock(datetime(2026, 1, 1, tzinfo=UTC)),
    )


def test_clear_advances_epoch_without_deleting_rows_and_clears_memory(tmp_path: Path) -> None:
    path = tmp_path / "conversation.sqlite3"
    service = _service(path)
    _run(service.initialize())
    turn = _run(service.persist_successful_turn(_request(1), PRIVATE_ASSISTANT))

    result = _run(service.clear_context())

    assert result.previous_epoch == 1
    assert result.active_epoch == 2
    assert result.revision == 1
    assert service.get_generation_context() == ConversationGenerationContext(2, ())
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT context_epoch, user_content, assistant_content FROM conversation_turns"
        ).fetchone() == (1, PRIVATE_USER, PRIVATE_ASSISTANT)
        assert connection.execute(
            "SELECT active_epoch, revision FROM conversation_state WHERE singleton_id = 1"
        ).fetchone() == (2, 1)
    finally:
        connection.close()
    assert turn.context_epoch == 1


def test_empty_and_repeated_clear_always_advance_monotonically(tmp_path: Path) -> None:
    service = _service(tmp_path / "conversation.sqlite3")
    _run(service.initialize())

    first = _run(service.clear_context())
    second = _run(service.clear_context())
    third = _run(service.clear_context())

    actual = [
        (item.previous_epoch, item.active_epoch, item.revision)
        for item in (first, second, third)
    ]
    assert actual == [
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 3),
    ]
    assert service.active_epoch == 4


def test_clear_database_failure_leaves_epoch_and_memory_unchanged(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "failed-clear.sqlite3")
    _run(service.initialize())
    _run(service.persist_successful_turn(_request(1), PRIVATE_ASSISTANT))
    repository = service._repository

    def fail_advance(*args: object, **kwargs: object):
        raise ConversationDatabaseError("advance conversation epoch")

    monkeypatch.setattr(repository, "advance_context_epoch", fail_advance)
    try:
        _run(service.clear_context())
    except ConversationDatabaseError:
        pass
    else:
        raise AssertionError("clear failure should propagate without a state update")

    assert service.active_epoch == 1
    assert [turn.user_message_id for turn in service.get_recent_turns()] == [1]


def test_new_turn_after_clear_is_active_and_restart_ignores_old_epoch(tmp_path: Path) -> None:
    path = tmp_path / "conversation.sqlite3"
    service = _service(path)
    _run(service.initialize())
    _run(service.persist_successful_turn(_request(1, "old"), "old answer"))
    _run(service.clear_context())
    new_turn = _run(service.persist_successful_turn(_request(2, "new"), "new answer"))

    assert new_turn.context_epoch == 2
    assert [turn.user_message_id for turn in service.get_recent_turns()] == [2]
    restarted = _service(path)
    _run(restarted.initialize())
    assert restarted.active_epoch == 2
    assert [turn.user_message_id for turn in restarted.get_recent_turns()] == [2]


def test_generation_captures_epoch_and_snapshot_at_worker_start_and_old_completion_stays_audit_only(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[list[ConversationGenerationContext], ConversationService]:
        service = _service(tmp_path / "race.sqlite3")
        await service.initialize()
        started = asyncio.Event()
        release = asyncio.Event()
        captured: list[ConversationGenerationContext] = []

        class Responder:
            async def generate(self, request: ChatRequest) -> str:
                assert request.generation_context is not None
                captured.append(request.generation_context)
                if request.message_id == 1:
                    started.set()
                    await release.wait()
                return f"answer-{request.message_id}"

        queue = SerializedGenerationQueue(ConversationRecordingResponder(Responder(), service))
        await queue.start()
        try:
            first = asyncio.create_task(queue.submit(_request(1)))
            await started.wait()
            cleared = await service.clear_context()
            assert cleared.active_epoch == 2
            release.set()
            assert await first == "answer-1"
            assert await queue.submit(_request(2, "new request")) == "answer-2"
        finally:
            await queue.close()
        return captured, service

    captured, service = _run(run())
    assert [context.context_epoch for context in captured] == [1, 2]
    assert captured[0].recent_turns == ()
    assert captured[1].recent_turns == ()
    assert [turn.user_message_id for turn in service.get_recent_turns()] == [2]
    connection = sqlite3.connect(tmp_path / "race.sqlite3")
    try:
        assert connection.execute(
            "SELECT user_message_id, context_epoch FROM conversation_turns ORDER BY id"
        ).fetchall() == [("1", 1), ("2", 2)]
    finally:
        connection.close()


def test_generation_context_hot_path_is_memory_only_after_startup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    _run(service.initialize())

    def fail_database_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("generation context must not read SQLite")

    monkeypatch.setattr(service._repository, "load_conversation_state", fail_database_read)
    monkeypatch.setattr(service._repository, "load_recent_turns", fail_database_read)
    assert service.get_generation_context().context_epoch == 1
    assert service.get_recent_turns() == ()


def test_reply_context_remains_explicit_current_request_context_after_clear(tmp_path: Path) -> None:
    async def run() -> tuple[ChatRequest, ConversationService]:
        service = _service(tmp_path / "reply.sqlite3")
        await service.initialize()
        await service.clear_context()
        captured: list[ChatRequest] = []

        class Responder:
            async def generate(self, request: ChatRequest) -> str:
                captured.append(request)
                return "answer"

        reply = ReplyContext(777, "Quoted user", False, "old reply body", 0)
        queue = SerializedGenerationQueue(ConversationRecordingResponder(Responder(), service))
        await queue.start()
        try:
            await queue.submit(_request(2, "new request", reply_context=reply))
        finally:
            await queue.close()
        return captured[0], service

    captured, service = _run(run())
    assert captured.reply_context is not None
    assert captured.reply_context.message_id == 777
    assert captured.generation_context is not None
    assert captured.generation_context.context_epoch == service.active_epoch == 2


def test_clear_does_not_change_persona_or_nickname_snapshots(tmp_path: Path) -> None:
    async def run() -> tuple[str | None, str | None]:
        path = tmp_path / "shared.sqlite3"
        clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
        persona = PersonaService(path, global_cooldown_seconds=1, clock=clock)
        nickname = NicknameService(path, global_cooldown_seconds=1, clock=clock)
        await persona.initialize()
        await nickname.initialize()
        await persona.publish_persona(1, PRIVATE_PERSONA)
        await nickname.set_nickname(2, PRIVATE_NICKNAME)
        conversation = _service(path, clock=clock)
        await conversation.initialize()
        await conversation.clear_context()
        return persona.get_active_persona(), nickname.get_active_nickname()

    assert _run(run()) == (PRIVATE_PERSONA, PRIVATE_NICKNAME)


def test_context_budget_transport_uses_only_post_clear_history(tmp_path: Path) -> None:
    captured_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    async def run() -> None:
        path = tmp_path / "budget.sqlite3"
        service = _service(path)
        await service.initialize()
        await service.persist_successful_turn(_request(1, "old history"), "old answer")
        await service.clear_context()
        responder = LmStudioResponder(
            _config(),
            transport=httpx.MockTransport(handler),
            conversation_context_provider=service,
        )
        request = _request(2, "current request")
        await responder.generate(request)

    _run(run())
    payload = captured_payloads[0]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "old history" not in serialized
    assert "old answer" not in serialized
    assert "current request" in serialized


def _config():
    from liuer_bot.config import Config

    return Config(
        discord_token="test-token",
        chat_channel_id=123,
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_model="configured-model",
    )


def test_generation_context_repr_redacts_turn_bodies() -> None:
    turn = ConversationTurn(
        1,
        123,
        1,
        10,
        "phase5b-private-display-name",
        PRIVATE_USER,
        0,
        PRIVATE_ASSISTANT,
        datetime.now(UTC),
    )
    rendered = repr(ConversationGenerationContext(1, (turn,)))
    assert PRIVATE_USER not in rendered
    assert PRIVATE_ASSISTANT not in rendered
    assert "phase5b-private-display-name" not in rendered

from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from PIL import Image

from liuer_bot.config import Config
from liuer_bot.conversation_context import (
    render_chat_completion_messages,
    render_current_user_text,
    render_historical_chat_messages,
    render_historical_context_text,
    render_native_chat_input,
)
from liuer_bot.conversation_models import ConversationTurn
from liuer_bot.conversation_repository import ConversationDatabaseError
from liuer_bot.conversation_service import (
    ConversationPersistenceError,
    ConversationRecordingResponder,
    ConversationService,
)
from liuer_bot.generation_queue import SerializedGenerationQueue
from liuer_bot.image_preparation import PreparedImage
from liuer_bot.lm_studio_responder import LmStudioHttpError, LmStudioResponder
from liuer_bot.reply_context import ReplyContext
from liuer_bot.responder import ChatRequest

PRIVATE_USER_A = "phase4ab-private-user-A"
PRIVATE_ASSISTANT_A = "phase4ab-private-assistant-A"
PRIVATE_USER_B = "phase4ab-private-user-B"
PRIVATE_ASSISTANT_B = "phase4ab-private-assistant-B"
PRIVATE_NAME = 'Alice "quoted"'


def _turn(
    turn_id: int,
    *,
    name: str = PRIVATE_NAME,
    user_content: str = PRIVATE_USER_A,
    image_count: int = 0,
    assistant_content: str = PRIVATE_ASSISTANT_A,
    channel_id: int = 123,
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        chat_channel_id=channel_id,
        user_message_id=turn_id,
        user_author_id=turn_id + 10,
        user_display_name=name,
        user_content=user_content,
        user_image_count=image_count,
        assistant_content=assistant_content,
        created_at_utc=datetime.now(UTC),
    )


def _request(
    message_id: int,
    content: str,
    *,
    display_name: str = "Current User",
    images: tuple[PreparedImage, ...] = (),
) -> ChatRequest:
    return ChatRequest(
        message_id=message_id,
        guild_id=20,
        channel_id=123,
        author_id=10,
        content=content,
        prepared_images=images,
        author_display_name=display_name,
    )


def _config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "discord_token": "test-token",
        "chat_channel_id": 123,
        "lm_studio_base_url": "http://localhost:1234/v1",
        "lm_studio_model": "configured-model",
        "lm_studio_timeout_seconds": 30.0,
        "lm_studio_temperature": 0.7,
        "lm_studio_max_tokens": 4096,
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def _png_image(attachment_id: int = 1) -> PreparedImage:
    image = Image.new("RGB", (1, 1), (attachment_id, 20, 30))
    output = io.BytesIO()
    try:
        image.save(output, format="PNG")
    finally:
        image.close()
    return PreparedImage(attachment_id, "image/png", output.getvalue(), 1, 1)


def _run(awaitable):
    return asyncio.run(awaitable)


def test_renderers_escape_speakers_and_preserve_multi_user_distinction() -> None:
    turns = (
        _turn(1, name='Alice "quoted"', user_content="Alice question"),
        _turn(
            2,
            name="Bob\\name",
            user_content="Bob question",
            assistant_content="Bob answer",
        ),
    )

    current = render_current_user_text('Alice "quoted"\nnot-a-new-label', "Current question")
    messages = render_historical_chat_messages(turns)
    native = render_native_chat_input(
        turns,
        current_display_name='Alice "quoted"',
        current_content="Current question",
    )

    assert current == '[使用者："Alice \\"quoted\\" not-a-new-label"]\nCurrent question'
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert '"Alice \\"quoted\\"' in messages[0]["content"]
    assert '"Bob\\\\name"' in messages[2]["content"]
    assert 'User "Alice \\"quoted\\"' in native
    assert 'User "Bob\\\\name"' in native
    assert "Current question" in native


def test_historical_image_turn_is_text_marker_only() -> None:
    turn = _turn(
        1,
        user_content="What is this?",
        image_count=2,
        assistant_content="Two hardware photos.",
    )

    messages = render_historical_chat_messages((turn,))
    user_text = messages[0]["content"]

    assert "What is this?" in user_text
    assert "歷史附件" in user_text
    assert "2 張圖片" in user_text
    assert messages[1] == {"role": "assistant", "content": "Two hardware photos."}
    assert "data:" not in user_text
    assert "https://" not in user_text


def test_historical_image_only_turn_is_not_dropped() -> None:
    turn = _turn(1, user_content="", image_count=1, assistant_content="I saw hardware.")

    rendered = render_historical_chat_messages((turn,))

    assert rendered[0]["role"] == "user"
    assert "歷史附件" in rendered[0]["content"]
    assert rendered[1]["content"] == "I saw hardware."


def test_no_history_native_input_stays_compact() -> None:
    rendered = render_native_chat_input(
        (), current_display_name="Raven", current_content="你在幹嘛？"
    )

    assert rendered == 'User "Raven":\n你在幹嘛？'
    assert "RECENT CONVERSATION" not in rendered
    assert "No previous" not in rendered


def test_native_history_uses_stable_canonical_assistant_label() -> None:
    rendered = render_native_chat_input(
        (_turn(1, name="Alice", user_content="先前問題", assistant_content="先前回答"),),
        current_display_name="Bob",
        current_content="現在問題",
    )

    assert rendered.index("先前問題") < rendered.index("先前回答")
    assert "六耳：\n先前回答" in rendered
    assert 'User "Bob":\n現在問題' in rendered
    assert "CURRENT MESSAGE" in rendered


def test_completions_history_is_oldest_to_newest_then_current() -> None:
    historical = render_historical_chat_messages(
        (
            _turn(1, user_content="oldest", assistant_content="oldest answer"),
            _turn(2, user_content="newest", assistant_content="newest answer"),
        )
    )
    messages = [
        {"role": "system", "content": "system"},
        *historical,
        {"role": "user", "content": render_current_user_text("Current", "now")},
    ]

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"] == '[使用者："Current"]\nnow'
    assert messages[-1]["content"].count("now") == 1


def test_render_chat_completion_messages_uses_current_speaker_once() -> None:
    turns = (_turn(1, user_content="history user", assistant_content="history assistant"),)

    messages = render_chat_completion_messages(
        turns,
        current_display_name="Current",
        current_content="current user",
    )

    assert messages == [
        {"role": "user", "content": '[使用者："Alice \\"quoted\\""]\nhistory user'},
        {"role": "assistant", "content": "history assistant"},
        {"role": "user", "content": '[使用者："Current"]\ncurrent user'},
    ]


def test_reply_context_is_quoted_before_current_message_in_completions() -> None:
    reply = ReplyContext(42, "Alice", False, "我的卡是4090", 0)

    messages = render_chat_completion_messages(
        (),
        current_display_name="Bob",
        current_content="他剛才說什麼？",
        reply_context=reply,
    )

    assert len(messages) == 1
    current = messages[0]["content"]
    assert current.index("[回覆上下文]") < current.index("[目前使用者：\"Bob\"]")
    assert "我的卡是4090" in current
    assert current.count("他剛才說什麼？") == 1


def test_reply_context_and_recent_history_coexist_without_historical_image_data() -> None:
    reply = ReplyContext(42, "六耳", True, "剛才那張圖看起來像顯示卡。", 1, True)
    rendered = render_native_chat_input(
        (_turn(1, user_content="看這張圖", image_count=1, assistant_content="我看到一張圖片。"),),
        current_display_name="Bob",
        current_content="我剛剛那張圖大概是什麼？",
        reply_context=reply,
    )

    assert rendered.index("RECENT CONVERSATION") < rendered.index("REPLIED MESSAGE")
    assert rendered.index("REPLIED MESSAGE") < rendered.index("CURRENT MESSAGE")
    assert "歷史附件" in rendered
    assert "圖片內容目前未重新提供" in rendered
    assert "剛才那張圖看起來像顯示卡。" in rendered
    assert rendered.count("我剛剛那張圖大概是什麼？") == 1
    assert "data:" not in rendered


def test_unavailable_reply_context_is_explicit_in_both_renderers() -> None:
    reply = ReplyContext(43, "Discord user", False, "", 0, is_available=False)

    completions = render_chat_completion_messages(
        (),
        current_display_name="Bob",
        current_content="還在嗎？",
        reply_context=reply,
    )[0]["content"]
    native = render_native_chat_input(
        (),
        current_display_name="Bob",
        current_content="還在嗎？",
        reply_context=reply,
    )

    assert "被回覆的訊息目前無法取得" in completions
    assert "被回覆的訊息目前無法取得" in native
    assert "還在嗎？" in completions
    assert "還在嗎？" in native


def test_lm_responder_injects_normal_history_and_keeps_logs_private(caplog) -> None:
    captured: list[dict[str, object]] = []
    logger = logging.getLogger("test.phase4ab.normal-history")

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    class Provider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return (_turn(1),)

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=Provider(),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        assert _run(responder.generate(_request(2, "VRAM是什麼？"))) == "answer"

    messages = captured[0]["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert PRIVATE_USER_A in messages[1]["content"]
    assert PRIVATE_ASSISTANT_A == messages[2]["content"]
    assert '"Current User"' in messages[3]["content"]
    for private_value in (PRIVATE_USER_A, PRIVATE_ASSISTANT_A, PRIVATE_NAME):
        assert private_value not in caplog.text


def test_lm_responder_injects_native_history_without_stateful_fields() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"output": [{"type": "message", "content": "answer"}]})

    class Provider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return (_turn(1),)

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=Provider(),
    )

    assert _run(responder.generate(_request(2, "hi"))) == "answer"

    payload = captured[0]
    assert "RECENT CONVERSATION" in payload["input"]
    assert PRIVATE_USER_A in payload["input"]
    assert PRIVATE_ASSISTANT_A in payload["input"]
    assert "六耳：" in payload["input"]
    assert "CURRENT MESSAGE" in payload["input"]
    assert payload["reasoning"] == "off"
    assert payload["store"] is False
    assert "previous_response_id" not in payload


def test_lm_responder_multimodal_history_has_marker_but_no_historical_image_url() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    class Provider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return (_turn(1, image_count=2),)

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=Provider(),
    )

    assert _run(
        responder.generate(_request(2, "看這張圖", images=(_png_image(),)))
    ) == "answer"

    messages = captured[0]["messages"]
    assert isinstance(messages, list)
    historical_user = messages[1]["content"]
    current_user = messages[-1]["content"]
    assert "歷史附件" in historical_user
    assert isinstance(current_user, list)
    assert current_user[0]["type"] == "text"
    assert "看這張圖" in current_user[0]["text"]
    assert all(
        message["role"] != "assistant" or "data:" not in str(message)
        for message in messages[1:-1]
    )
    assert not any("image_url" in block for block in historical_user if isinstance(block, dict))


def test_deep_history_keeps_chat_completions_and_4096_budget() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    class Provider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return (_turn(1),)

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=Provider(),
    )

    assert _run(responder.generate(_request(2, "詳細比較VRAM與系統RAM的差異"))) == "answer"

    assert captured[0]["max_tokens"] == 4096
    assert [message["role"] for message in captured[0]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_current_persona_and_nickname_apply_without_rewriting_history() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    class PersonaProvider:
        def get_active_persona(self) -> str | None:
            return "current Persona B"

    class NicknameProvider:
        def get_active_nickname(self) -> str | None:
            return "six"

    class ContextProvider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return (_turn(1, assistant_content="historical assistant text"),)

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        active_persona_provider=PersonaProvider(),
        active_nickname_provider=NicknameProvider(),
        conversation_context_provider=ContextProvider(),
    )

    assert _run(responder.generate(_request(2, "VRAM是什麼？"))) == "answer"

    system_prompt = captured[0]["messages"][0]["content"]
    assert "current Persona B" in system_prompt
    assert "目前的小名是「six」" in system_prompt
    assert captured[0]["messages"][2]["content"] == "historical assistant text"
    assert captured[0]["messages"][-1]["content"] == '[使用者："Current User"]\nVRAM是什麼？'
    assert "six" not in captured[0]["messages"][-1]["content"]


def test_restart_restored_turn_is_available_to_first_model_request(tmp_path: Path) -> None:
    service_path = tmp_path / "conversation.sqlite3"
    first = ConversationService(service_path, chat_channel_id=123)
    _run(first.initialize())
    _run(first.persist_successful_turn(_request(1, "old question"), "old answer"))

    restarted = ConversationService(service_path, chat_channel_id=123)
    _run(restarted.initialize())
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "new answer"}}]})

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=restarted,
    )
    _run(responder.generate(_request(2, "VRAM是什麼？")))

    messages = captured[0]["messages"]
    assert "old question" in messages[1]["content"]
    assert messages[2]["content"] == "old answer"


def test_restart_restores_snapshot_then_budget_trims_without_generation_db_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service_path = tmp_path / "conversation.sqlite3"
    first = ConversationService(service_path, chat_channel_id=123)
    _run(first.initialize())
    _run(first.persist_successful_turn(_request(1, "phase4b-restart-old"), "old answer"))
    _run(first.persist_successful_turn(_request(2, "phase4b-restart-new"), "new answer"))

    restarted = ConversationService(service_path, chat_channel_id=123)
    _run(restarted.initialize())
    budget = len(render_historical_context_text(restarted.get_recent_turns()[-1:]))

    def fail_database_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("generation budget planning must not read SQLite")

    monkeypatch.setattr(restarted._repository, "load_recent_turns", fail_database_read)
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    responder = LmStudioResponder(
        _config(context_history_max_chars_normal=budget),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=restarted,
    )
    _run(responder.generate(_request(3, "phase4b-restart-current")))

    messages = captured[0]["messages"]
    assert "phase4b-restart-old" not in str(messages)
    assert "phase4b-restart-new" in str(messages)
    assert [turn.user_message_id for turn in restarted.get_recent_turns()] == [1, 2]


def test_generation_context_reads_memory_only_after_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())

    def fail_database_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("generation context must not read SQLite")

    monkeypatch.setattr(service._repository, "load_recent_turns", fail_database_read)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=service,
    )
    assert _run(responder.generate(_request(1, "VRAM是什麼？"))) == "answer"


def test_queued_a_then_b_sees_a_only_after_a_is_persisted(tmp_path: Path) -> None:
    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": f"assistant-{len(payloads)}"}}]},
        )

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=service,
    )
    queue = SerializedGenerationQueue(ConversationRecordingResponder(responder, service))

    async def run() -> None:
        await queue.start()
        try:
            await asyncio.gather(
                queue.submit(_request(1, PRIVATE_USER_A, display_name="Alice")),
                queue.submit(_request(2, PRIVATE_USER_B, display_name="Bob")),
            )
        finally:
            await queue.close()

    _run(run())

    second_messages = payloads[1]["messages"]
    assert PRIVATE_USER_A in second_messages[1]["content"]
    assert PRIVATE_ASSISTANT_A not in second_messages[1]["content"]
    assert second_messages[2]["content"] == "assistant-1"
    assert '"Bob"' in second_messages[-1]["content"]
    assert [turn.user_message_id for turn in service.get_recent_turns()] == [1, 2]


def test_queued_a_then_b_applies_history_budget_after_a_is_persisted(tmp_path: Path) -> None:
    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())
    initial_request = _request(99, "phase4b-OLD", display_name="Alice")
    _run(service.persist_successful_turn(initial_request, "A-answer"))
    budget = len(render_historical_context_text(service.get_recent_turns()))
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        answer = "A-answer" if len(payloads) == 1 else "B-answer"
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

    responder = LmStudioResponder(
        _config(context_history_max_chars_normal=budget),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=service,
    )
    queue = SerializedGenerationQueue(ConversationRecordingResponder(responder, service))

    async def run() -> None:
        await queue.start()
        try:
            await asyncio.gather(
                queue.submit(_request(1, "phase4b-NEW", display_name="Alice")),
                queue.submit(_request(2, "phase4b-BBB", display_name="Alice")),
            )
        finally:
            await queue.close()

    _run(run())

    first_messages = payloads[0]["messages"]
    second_messages = payloads[1]["messages"]
    assert "phase4b-OLD" in str(first_messages)
    assert "phase4b-OLD" not in str(second_messages)
    assert "phase4b-NEW" in str(second_messages)
    assert "A-answer" in str(second_messages)
    assert [turn.user_message_id for turn in service.get_recent_turns()] == [99, 1, 2]


def test_failed_a_is_not_visible_to_b_and_worker_survives(tmp_path: Path) -> None:
    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(500, text="private server body")
        return httpx.Response(200, json={"choices": [{"message": {"content": "b answer"}}]})

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=service,
    )
    queue = SerializedGenerationQueue(ConversationRecordingResponder(responder, service))

    async def run() -> None:
        await queue.start()
        try:
            with pytest.raises(LmStudioHttpError):
                await queue.submit(_request(1, PRIVATE_USER_A))
            assert await queue.submit(_request(2, PRIVATE_USER_B)) == "b answer"
        finally:
            await queue.close()

    _run(run())
    assert len(payloads) == 2
    assert PRIVATE_USER_A not in str(payloads[1])
    assert service.get_recent_turns()[0].user_message_id == 2


def test_persistence_failure_does_not_make_a_visible_to_b(tmp_path: Path) -> None:
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
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        conversation_context_provider=service,
    )
    queue = SerializedGenerationQueue(ConversationRecordingResponder(responder, service))

    async def run() -> None:
        await queue.start()
        try:
            with pytest.raises(ConversationPersistenceError):
                await queue.submit(_request(1, PRIVATE_USER_A))
            assert await queue.submit(_request(2, PRIVATE_USER_B)) == "answer"
        finally:
            await queue.close()

    _run(run())
    assert PRIVATE_USER_A not in str(payloads[1])
    assert [turn.user_message_id for turn in service.get_recent_turns()] == [2]

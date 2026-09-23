from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from uuid import uuid4

import httpx
import pytest
from PIL import Image, PngImagePlugin

import liuer_bot.lm_studio_responder as lm_studio_responder
from liuer_bot.addressing import EMPTY_ADDRESSED_CONTENT
from liuer_bot.config import Config
from liuer_bot.conversation_context import (
    render_current_user_text,
    render_historical_context_text,
    render_native_chat_input,
)
from liuer_bot.conversation_models import ConversationTurn
from liuer_bot.generation_queue import SerializedGenerationQueue
from liuer_bot.image_preparation import PreparedImage
from liuer_bot.lm_studio_responder import (
    LmStudioHttpError,
    LmStudioImageInputError,
    LmStudioModelUnavailableError,
    LmStudioProtocolError,
    LmStudioResponder,
    LmStudioTimeoutError,
    LmStudioUnavailableError,
    derive_native_chat_url,
)
from liuer_bot.model_image_normalization import normalize_model_images
from liuer_bot.prompting import build_system_prompt
from liuer_bot.reply_context import ReplyContext
from liuer_bot.responder import ChatRequest
from liuer_bot.response_routing import (
    ResponseDecision,
    ResponseMode,
    ResponseRouter,
    ResponseRouteReason,
    response_instruction,
)


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs) -> None:
        asyncio.run(function(*args, **kwargs))

    return wrapper


def _config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "discord_token": "discord-secret",
        "chat_channel_id": 123,
        "lm_studio_base_url": "http://localhost:1234/v1",
        "lm_studio_model": "configured-model",
        "lm_studio_timeout_seconds": 300.0,
        "lm_studio_temperature": 0.7,
        "lm_studio_max_tokens": 4096,
        "lm_studio_api_token": None,
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def _request(
    message_id: int = 1,
    content: str = "phase2a-private-message-content-check",
    prepared_images: tuple[PreparedImage, ...] = (),
    reply_context: ReplyContext | None = None,
) -> ChatRequest:
    return ChatRequest(
        message_id=message_id,
        guild_id=20,
        channel_id=123,
        author_id=10,
        content=content,
        prepared_images=prepared_images,
        author_display_name="Test User",
        reply_context=reply_context,
    )


def _history_turn(
    turn_id: int,
    user_content: str,
    assistant_content: str,
    *,
    image_count: int = 0,
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        chat_channel_id=123,
        user_message_id=turn_id,
        user_author_id=turn_id + 100,
        user_display_name=f"History User {turn_id}",
        user_content=user_content,
        user_image_count=image_count,
        assistant_content=assistant_content,
        created_at_utc=datetime.now(UTC),
    )


def _current_text(content: str) -> str:
    return render_current_user_text("Test User", content)


def _current_value(content: object) -> str:
    text: object
    if isinstance(content, list):
        text = content[0]["text"]
    else:
        text = content
    assert isinstance(text, str)
    return text.rsplit("\n", 1)[-1]


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _prepared_image(
    attachment_id: int,
    media_type: str = "image/png",
    data: bytes | None = None,
    *,
    size: tuple[int, int] = (1, 1),
    marker: str | None = None,
) -> PreparedImage:
    if data is None:
        image_format = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/webp": "WEBP",
        }.get(media_type)
        if image_format is None:
            data = b"invalid-image-input"
        else:
            image = Image.new("RGB", size, (attachment_id, 40, 60))
            try:
                output = io.BytesIO()
                save_kwargs: dict[str, object] = {"format": image_format}
                if marker is not None and image_format == "PNG":
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("marker", marker)
                    save_kwargs["pnginfo"] = metadata
                image.save(output, **save_kwargs)
                data = output.getvalue()
            finally:
                image.close()
    return PreparedImage(attachment_id, media_type, data, *size)


def _data_uri_parts(block: dict[str, object]) -> tuple[str, bytes]:
    image_url = block["image_url"]
    assert isinstance(image_url, dict)
    url = image_url["url"]
    assert isinstance(url, str)
    prefix, encoded = url.split(",", 1)
    return prefix, base64.b64decode(encoded, validate=True)


@dataclass(frozen=True)
class _FixedPromptBuilder:
    prompt: str

    def build_system_prompt(self, persona_override: str | None = None) -> str:
        assert persona_override is None
        return self.prompt


@dataclass
class _NativePromptBuilder:
    prompt: str

    def build_system_prompt(
        self,
        persona_override: str | None = None,
        response_instruction: str | None = None,
    ) -> str:
        assert persona_override is None
        assert response_instruction is not None
        return f"{self.prompt}\n{response_instruction}"


@async_test
async def test_generate_sends_exact_text_only_chat_completion_payload() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  assistant text  "}}]},
        )

    responder = LmStudioResponder(_config(), transport=_transport(handler))

    result = await responder.generate(_request())

    assert result == "assistant text"
    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:1234/v1/chat/completions"
    assert json.loads(request.content) == {
        "model": "configured-model",
        "messages": [
            {
                "role": "system",
                "content": build_system_prompt(
                    response_instruction=response_instruction(ResponseMode.NORMAL)
                ),
            },
            {
                "role": "user",
                "content": _current_text("phase2a-private-message-content-check"),
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1536,
        "stream": False,
    }
    assert "authorization" not in request.headers


@pytest.mark.parametrize(
    ("content", "expected_mode", "expected_max_tokens"),
    [
        ("Please answer briefly.", ResponseMode.CHAT_SHORT, 1024),
        ("Please explain in detail.", ResponseMode.DEEP, 4096),
    ],
)
def test_response_mode_selects_payload_budget_and_prompt_layer(
    content: str,
    expected_mode: ResponseMode,
    expected_max_tokens: int,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path.endswith("/api/v1/chat"):
            return httpx.Response(
                200,
                json={"output": [{"type": "message", "content": "ok"}]},
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    asyncio.run(
        LmStudioResponder(
            _config(),
            transport=_transport(handler),
            response_router=ResponseRouter(),
        ).generate(_request(content=content))
    )

    payload = json.loads(captured[0].content)
    budget_key = "max_output_tokens" if expected_mode is ResponseMode.CHAT_SHORT else "max_tokens"
    assert payload[budget_key] == expected_max_tokens
    system_prompt = (
        payload["system_prompt"]
        if expected_mode is ResponseMode.CHAT_SHORT
        else payload["messages"][0]["content"]
    )
    expected_prompt = build_system_prompt(
        response_instruction=response_instruction(expected_mode)
    )
    assert system_prompt == expected_prompt


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "http://127.0.0.1:1234/v1",
            "http://127.0.0.1:1234/api/v1/chat",
        ),
        (
            "http://192.168.1.20:1234/v1/",
            "http://192.168.1.20:1234/api/v1/chat",
        ),
        (
            "https://lm.example.test/prefix/v1/",
            "https://lm.example.test/prefix/api/v1/chat",
        ),
    ],
)
def test_native_chat_url_is_derived_from_same_server_v1_base(
    base_url: str,
    expected: str,
) -> None:
    assert derive_native_chat_url(base_url) == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:1234/api",
        "http://127.0.0.1:1234/v2",
        "http://127.0.0.1:1234/v1?token=private",
        "not-a-url",
    ],
)
def test_native_chat_url_derivation_fails_closed(base_url: str) -> None:
    with pytest.raises(ValueError, match="native chat"):
        derive_native_chat_url(base_url)


@async_test
async def test_chat_short_uses_exact_stateless_native_payload_without_warning(caplog) -> None:
    captured: list[httpx.Request] = []
    private_input = "你在幹嘛？"
    private_system = "phase3cb-private-system-prompt"
    private_persona = "phase3cb-private-persona"
    logger = logging.getLogger("test.lm.native-payload")

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "output": [
                    {"type": "message", "content": "phase3cb-private-native-response"}
                ],
                "stats": {
                    "input_tokens": 100,
                    "total_output_tokens": 10,
                    "reasoning_output_tokens": 0,
                    "tokens_per_second": 100,
                    "time_to_first_token_seconds": 0.5,
                },
            },
        )

    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
        prompt_builder=_NativePromptBuilder(f"{private_system}\n{private_persona}"),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = await responder.generate(_request(content=private_input))

    assert result == "phase3cb-private-native-response"
    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "http://localhost:1234/api/v1/chat"
    payload = json.loads(request.content)
    assert payload == {
        "model": "configured-model",
        "input": render_native_chat_input(
            (), current_display_name="Test User", current_content=private_input
        ),
        "system_prompt": (
            f"{private_system}\n{private_persona}\n"
            f"{response_instruction(ResponseMode.CHAT_SHORT)}"
        ),
        "reasoning": "off",
        "max_output_tokens": 1024,
        "temperature": 0.7,
        "stream": False,
        "store": False,
    }
    assert "messages" not in payload
    assert "previous_response_id" not in payload
    assert "integrations" not in payload
    assert "tools" not in payload
    assert "images" not in payload
    assert "transport=NATIVE_CHAT" in caplog.text
    assert "reasoning=off" in caplog.text
    for private_value in (
        private_input,
        private_system,
        private_persona,
        "phase3cb-private-native-response",
    ):
        assert private_value not in caplog.text


def test_chat_short_reply_context_stays_native_and_uses_current_text_for_routing() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"output": [{"type": "message", "content": "answer"}]})

    reply = ReplyContext(
        77,
        "Alice",
        False,
        "phase4ab-private-large-replied-content " * 100,
        0,
    )
    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
    )

    assert (
        asyncio.run(
            responder.generate(
                _request(content="Please answer briefly.", reply_context=reply)
            )
        )
        == "answer"
    )

    payload = captured[0]
    assert str(responder._native_chat_url).endswith("/api/v1/chat")
    assert "[REPLIED MESSAGE]" in payload["input"]
    assert "phase4ab-private-large-replied-content" in payload["input"]
    assert "[CURRENT MESSAGE]" in payload["input"]
    assert payload["reasoning"] == "off"
    assert payload["store"] is False
    assert "messages" not in payload


@pytest.mark.parametrize(
    ("content", "expected_mode"),
    [
        ("Please answer briefly.", ResponseMode.CHAT_SHORT),
        ("What is this?", ResponseMode.NORMAL),
        ("Please explain in detail.", ResponseMode.DEEP),
    ],
)
def test_each_response_mode_uses_its_own_history_budget(
    content: str,
    expected_mode: ResponseMode,
) -> None:
    old = _history_turn(1, "phase4b-old-history-user", "phase4b-old-history-assistant")
    newest = _history_turn(2, "phase4b-new-history-user", "phase4b-new-history-assistant")
    all_turns = (old, newest)
    short_budget = len(render_historical_context_text((newest,)))
    full_budget = len(render_historical_context_text(all_turns))
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        if request.url.path.endswith("/api/v1/chat"):
            return httpx.Response(200, json={"output": [{"type": "message", "content": "ok"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    class Provider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return all_turns

    responder = LmStudioResponder(
        _config(
            context_history_max_chars_chat_short=short_budget,
            context_history_max_chars_normal=full_budget,
            context_history_max_chars_deep=full_budget,
        ),
        transport=_transport(handler),
        conversation_context_provider=Provider(),
    )

    assert asyncio.run(responder.generate(_request(content=content))) == "ok"

    payload = captured[0]
    if expected_mode is ResponseMode.CHAT_SHORT:
        assert "phase4b-new-history-user" in payload["input"]
        assert "phase4b-old-history-user" not in payload["input"]
        assert payload["reasoning"] == "off"
        assert payload["store"] is False
    else:
        messages = payload["messages"]
        assert isinstance(messages, list)
        rendered_messages = str(messages)
        assert "phase4b-old-history-user" in rendered_messages
        assert "phase4b-new-history-user" in rendered_messages
        assert payload["max_tokens"] == (
            1536 if expected_mode is ResponseMode.NORMAL else 4096
        )


def test_multimodal_reply_context_is_current_text_and_referenced_images_are_markers() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    reply = ReplyContext(78, "Alice", False, "她的圖卡是什麼？", 2)
    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
    )

    assert asyncio.run(
        responder.generate(
            _request(
                content="看我現在這張圖",
                prepared_images=(_prepared_image(1),),
                reply_context=reply,
            )
        )
    ) == "answer"

    messages = captured[0]["messages"]
    assert isinstance(messages, list)
    current = messages[-1]["content"]
    assert isinstance(current, list)
    text = current[0]["text"]
    assert "[回覆上下文]" in text
    assert "她的圖卡是什麼？" in text
    assert "2 張圖片" in text
    assert "圖片內容目前未重新提供" in text
    assert "看我現在這張圖" in text
    assert sum(block["type"] == "image_url" for block in current) == 1


def test_multimodal_request_uses_normal_history_budget_without_changing_images() -> None:
    old = _history_turn(1, "phase4b-multimodal-old", "old answer")
    newest = _history_turn(2, "phase4b-multimodal-new", "new answer", image_count=1)
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    class Provider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return (old, newest)

    responder = LmStudioResponder(
        _config(
            context_history_max_chars_chat_short=0,
            context_history_max_chars_normal=len(render_historical_context_text((newest,))),
            context_history_max_chars_deep=0,
        ),
        transport=_transport(handler),
        conversation_context_provider=Provider(),
    )

    assert asyncio.run(
        responder.generate(
            _request(content="看現在這張圖", prepared_images=(_prepared_image(91),))
        )
    ) == "ok"

    messages = captured[0]["messages"]
    assert isinstance(messages, list)
    assert "phase4b-multimodal-old" not in str(messages)
    assert "phase4b-multimodal-new" in str(messages)
    current = messages[-1]["content"]
    assert isinstance(current, list)
    assert current[0]["text"].endswith("看現在這張圖")
    assert sum(block["type"] == "image_url" for block in current) == 1


def test_zero_history_budget_keeps_reply_context_and_current_native_input() -> None:
    captured: list[dict[str, object]] = []
    reply = ReplyContext(900, "Alice", False, "phase4b-private-reply-context", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"output": [{"type": "message", "content": "ok"}]})

    class Provider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return (_history_turn(1, "phase4b-trimmed-history", "trimmed answer"),)

    responder = LmStudioResponder(
        _config(
            context_history_max_chars_chat_short=0,
            context_history_max_chars_normal=0,
            context_history_max_chars_deep=0,
        ),
        transport=_transport(handler),
        conversation_context_provider=Provider(),
    )

    assert asyncio.run(
        responder.generate(
            _request(content="Please answer briefly.", reply_context=reply)
        )
    ) == "ok"

    native_input = captured[0]["input"]
    assert "RECENT CONVERSATION" not in native_input
    assert "phase4b-trimmed-history" not in native_input
    assert "phase4b-private-reply-context" in native_input
    assert "Please answer briefly." in native_input


def test_zero_history_budget_keeps_reply_context_and_current_completion() -> None:
    captured: list[dict[str, object]] = []
    reply = ReplyContext(901, "Alice", False, "phase4b-private-reply-completion", 0)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    class Provider:
        def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
            return (_history_turn(1, "phase4b-trimmed-completion", "trimmed answer"),)

    responder = LmStudioResponder(
        _config(
            context_history_max_chars_chat_short=0,
            context_history_max_chars_normal=0,
            context_history_max_chars_deep=0,
        ),
        transport=_transport(handler),
        conversation_context_provider=Provider(),
    )

    assert asyncio.run(
        responder.generate(
            _request(content="What remains?", reply_context=reply)
        )
    ) == "ok"

    messages = captured[0]["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user"]
    current = messages[-1]["content"]
    assert "phase4b-trimmed-completion" not in current
    assert "phase4b-private-reply-completion" in current
    assert current.count("What remains?") == 1


def test_response_router_ignores_reply_context_size() -> None:
    reply = ReplyContext(79, "Alice", False, "x" * 10_000, 0)
    decision = ResponseRouter().route(
        _request(content="Please answer briefly.", reply_context=reply)
    )

    assert decision.mode is ResponseMode.CHAT_SHORT
    assert decision.max_tokens == 1024


def test_chat_short_with_images_fails_closed_before_any_http() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(500, text="phase3cb-private-server-body")

    class ShortRouter:
        def route(self, request: ChatRequest) -> ResponseDecision:
            return ResponseDecision(
                ResponseMode.CHAT_SHORT,
                1024,
                ResponseRouteReason.EXPLICIT_BRIEF,
            )

    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
        response_router=ShortRouter(),  # type: ignore[arg-type]
    )
    with pytest.raises(LmStudioImageInputError):
        asyncio.run(
            responder.generate(_request(content="hi", prepared_images=(_prepared_image(1),)))
        )
    assert captured == []


@async_test
async def test_native_nonzero_reasoning_stats_warn_without_surface_or_logging_content(
    caplog,
) -> None:
    private_input = "hi phase3cb-private-input"
    private_answer = "phase3cb-private-native-response"
    private_reasoning = "phase3cb-private-reasoning"
    logger = logging.getLogger("test.lm.native-reasoning-stats")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "output": [{"type": "message", "content": private_answer}],
                "stats": {"reasoning_output_tokens": 2},
                "reasoning": private_reasoning,
            },
        )

    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = await responder.generate(_request(content=private_input))

    assert result == private_answer
    assert "CHAT_SHORT requested reasoning off but server reported reasoning tokens" in caplog.text
    assert "reasoning_output_tokens=2" in caplog.text
    for private_value in (private_input, private_answer, private_reasoning):
        assert private_value not in caplog.text


@async_test
async def test_native_missing_stats_still_returns_valid_answer_without_warning(caplog) -> None:
    logger = logging.getLogger("test.lm.native-missing-stats")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"output": [{"type": "message", "content": "native ok"}]},
        )

    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        assert await responder.generate(_request(content="hi")) == "native ok"

    assert (
        "CHAT_SHORT requested reasoning off but server reported reasoning tokens"
        not in caplog.text
    )


@pytest.mark.parametrize(
    "response_json",
    [
        [],
        {},
        {"output": []},
        {"output": [{"type": "reasoning", "content": "phase3cb-private-reasoning"}]},
        {"output": [{"type": "message"}]},
        {"output": [{"type": "message", "content": None}]},
        {"output": [{"type": "message", "content": 123}]},
        {"output": [{"type": "message", "content": ""}]},
        {"output": [{"type": "message", "content": "  "}]},
    ],
)
def test_native_protocol_responses_fail_closed_without_fallback(response_json: object) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=response_json)

    with pytest.raises(LmStudioProtocolError):
        asyncio.run(
                LmStudioResponder(_config(), transport=_transport(handler)).generate(
                    _request(content="hi")
                )
        )

    assert len(calls) == 1
    assert calls[0].endswith("/api/v1/chat")
    assert all("chat/completions" not in call for call in calls)


@pytest.mark.parametrize(
    ("content", "prepared_images", "expected_suffix"),
    [
        ("hi", (), "/api/v1/chat"),
        ("VRAM是什麼？", (), "/v1/chat/completions"),
        ("詳細比較VRAM與系統RAM的差異", (), "/v1/chat/completions"),
        ("hi", (_prepared_image(1),), "/v1/chat/completions"),
    ],
)
def test_transport_matrix_keeps_native_text_only_and_completions_elsewhere(
    content: str,
    prepared_images: tuple[PreparedImage, ...],
    expected_suffix: str,
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/api/v1/chat"):
            return httpx.Response(200, json={"output": [{"type": "message", "content": "ok"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = asyncio.run(
        LmStudioResponder(_config(), transport=_transport(handler)).generate(
            _request(content=content, prepared_images=prepared_images)
        )
    )

    assert result == "ok"
    assert len(calls) == 1
    assert calls[0].endswith(expected_suffix)


def test_native_invalid_json_fails_closed_without_fallback() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"not-json")

    with pytest.raises(LmStudioProtocolError):
        asyncio.run(
            LmStudioResponder(_config(), transport=_transport(handler)).generate(
                _request(content="hi")
            )
        )

    assert len(calls) == 1
    assert calls[0].endswith("/api/v1/chat")


@pytest.mark.parametrize("status_code", [400, 401, 404, 500, 503])
def test_native_http_errors_do_not_fallback_or_log_body(status_code: int, caplog) -> None:
    private_body = "phase3cb-private-native-http-body"
    calls: list[str] = []
    logger = logging.getLogger(f"test.lm.native-http.{status_code}")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(status_code, text=private_body)

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with pytest.raises(LmStudioHttpError) as raised:
            asyncio.run(
                LmStudioResponder(
                    _config(),
                    transport=_transport(handler),
                    logger=logger,
                ).generate(_request(content="hi"))
            )

    assert raised.value.status_code == status_code
    assert len(calls) == 1
    assert calls[0].endswith("/api/v1/chat")
    assert "chat/completions" not in "".join(calls)
    assert private_body not in caplog.text


@pytest.mark.parametrize("exception_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_native_connection_and_timeout_fail_without_retry(
    exception_type: type[httpx.RequestError],
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        raise exception_type("phase3cb-private-native-request-error", request=request)

    expected_error = (
        LmStudioUnavailableError if exception_type is httpx.ConnectError else LmStudioTimeoutError
    )
    with pytest.raises(expected_error):
        asyncio.run(
            LmStudioResponder(_config(), transport=_transport(handler)).generate(
                _request(content="hi")
            )
        )

    assert len(calls) == 1
    assert calls[0].endswith("/api/v1/chat")


@async_test
async def test_injected_prompt_builder_is_sent_once_without_diagnostic_leakage(caplog) -> None:
    private_system = "phase3a-private-system-prompt-check"
    private_persona = "phase3a-private-persona-check"
    private_user_content = "phase3a-private-user-content-check"
    captured: list[httpx.Request] = []
    logger = logging.getLogger("test.lm.prompt-privacy")

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
        prompt_builder=_FixedPromptBuilder(f"{private_system}\n{private_persona}"),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        assert await responder.generate(_request(content=private_user_content)) == "ok"

    assert json.loads(captured[0].content)["messages"] == [
        {"role": "system", "content": f"{private_system}\n{private_persona}"},
        {"role": "user", "content": _current_text(private_user_content)},
    ]
    for private_value in (private_system, private_persona, private_user_content):
        assert private_value not in caplog.text


@async_test
async def test_text_only_generation_does_not_build_multimodal_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_to_thread(*args: object, **kwargs: object) -> object:
        raise AssertionError("text-only generation must not serialize image content")

    monkeypatch.setattr(lm_studio_responder.asyncio, "to_thread", unexpected_to_thread)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = await LmStudioResponder(_config(), transport=_transport(handler)).generate(_request())

    assert result == "ok"


@async_test
async def test_single_png_multimodal_payload_uses_exact_data_uri() -> None:
    original = _prepared_image(
        1,
        "image/png",
        marker="phase2bb-private-image-bytes-check",
    )
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "vision ok"}}]})

    result = await LmStudioResponder(_config(), transport=_transport(handler)).generate(
        _request(
            content="phase2bb-private-message-content-check",
            prepared_images=(original,),
        )
    )

    assert result == "vision ok"
    assert len(captured) == 1
    payload = json.loads(captured[0].content)
    assert payload["messages"] == [
        {
            "role": "system",
            "content": build_system_prompt(
                response_instruction=response_instruction(ResponseMode.NORMAL)
            ),
        },
        {"role": "user", "content": payload["messages"][1]["content"]},
    ]
    content = payload["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0] == {
        "type": "text",
        "text": _current_text("phase2bb-private-message-content-check"),
    }
    assert content[1]["type"] == "image_url"
    assert build_system_prompt() not in content[0]["text"]
    assert all(build_system_prompt() not in str(block) for block in content[1:])
    prefix, decoded = _data_uri_parts(content[1])
    assert prefix == "data:image/png;base64"
    assert decoded == original.data


@async_test
async def test_minimal_address_marker_is_valid_in_text_and_multimodal_payloads() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    responder = LmStudioResponder(_config(), transport=_transport(handler))

    assert await responder.generate(_request(content=EMPTY_ADDRESSED_CONTENT)) == "ok"
    assert (
        await responder.generate(
            _request(
                message_id=2,
                content=EMPTY_ADDRESSED_CONTENT,
                prepared_images=(_prepared_image(1),),
            )
        )
        == "ok"
    )

    assert len(captured) == 2
    text_payload = json.loads(captured[0].content)
    multimodal_payload = json.loads(captured[1].content)
    assert text_payload["messages"][1] == {
        "role": "user",
        "content": _current_text(EMPTY_ADDRESSED_CONTENT),
    }
    assert multimodal_payload["messages"][1]["content"][0] == {
        "type": "text",
        "text": _current_text(EMPTY_ADDRESSED_CONTENT),
    }


@pytest.mark.parametrize(
    "media_type",
    [
        "image/jpeg",
        "image/webp",
    ],
)
def test_jpeg_and_webp_multimodal_data_uris_preserve_original_bytes(
    media_type: str,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    original = _prepared_image(1, media_type)
    result = asyncio.run(
        LmStudioResponder(_config(), transport=_transport(handler)).generate(
            _request(prepared_images=(original,))
        )
    )

    assert result == "ok"
    content = json.loads(captured[0].content)["messages"][1]["content"]
    prefix, decoded = _data_uri_parts(content[1])
    assert prefix == f"data:{media_type};base64"
    assert decoded == original.data


@async_test
async def test_multiple_images_remain_ordered_in_one_multimodal_post() -> None:
    images = (
        _prepared_image(4, "image/png"),
        _prepared_image(2, "image/jpeg"),
        _prepared_image(8, "image/webp"),
        _prepared_image(1, "image/png"),
    )
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = await LmStudioResponder(_config(), transport=_transport(handler)).generate(
        _request(content="describe in supplied order", prepared_images=images)
    )

    assert result == "ok"
    assert len(captured) == 1
    content = json.loads(captured[0].content)["messages"][1]["content"]
    assert content[0] == {
        "type": "text",
        "text": _current_text("describe in supplied order"),
    }
    assert [
        _data_uri_parts(block)[0] for block in content[1:]
    ] == [
        "data:image/png;base64",
        "data:image/jpeg;base64",
        "data:image/webp;base64",
        "data:image/png;base64",
    ]
    assert [_data_uri_parts(block)[1] for block in content[1:]] == [image.data for image in images]


@pytest.mark.parametrize(
    "media_type",
    ["image/gif", "image/svg+xml", "application/pdf", "text/plain"],
)
def test_unsupported_model_image_media_type_fails_before_http_post(media_type: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    with pytest.raises(LmStudioImageInputError):
        asyncio.run(
            LmStudioResponder(_config(), transport=_transport(handler)).generate(
                _request(prepared_images=(_prepared_image(1, media_type),))
            )
        )

    assert calls == []


def test_empty_prepared_image_bytes_fail_before_http_post() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    with pytest.raises(LmStudioImageInputError):
        asyncio.run(
            LmStudioResponder(_config(), transport=_transport(handler)).generate(
                _request(prepared_images=(_prepared_image(1, "image/png", b""),))
            )
        )

    assert calls == []


@async_test
async def test_multimodal_content_serialization_runs_via_to_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_to_thread = asyncio.to_thread
    calls: list[object] = []

    async def recording_to_thread(function, /, *args, **kwargs):
        calls.append(function)
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(lm_studio_responder.asyncio, "to_thread", recording_to_thread)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = await LmStudioResponder(_config(), transport=_transport(handler)).generate(
        _request(prepared_images=(_prepared_image(1),))
    )

    assert result == "ok"
    assert calls == [lm_studio_responder._build_normalized_multimodal_content]


@async_test
async def test_multimodal_payload_uses_normalized_bytes_not_oversized_original() -> None:
    original = _prepared_image(1, "image/png", size=(400, 100))
    normalized = normalize_model_images(
        (original,),
        max_long_edge=100,
        max_image_pixels=10_000,
        max_total_image_pixels=10_000,
    )[0]
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = await LmStudioResponder(
        _config(
            model_image_max_long_edge=100,
            model_image_max_pixels=10_000,
            model_total_image_pixels=10_000,
        ),
        transport=_transport(handler),
    ).generate(_request(prepared_images=(original,)))

    assert result == "ok"
    content = json.loads(captured[0].content)["messages"][1]["content"]
    _, payload_bytes = _data_uri_parts(content[1])
    assert payload_bytes == normalized.data
    assert payload_bytes != original.data


@async_test
async def test_four_normalized_images_still_use_one_multimodal_post() -> None:
    originals = tuple(_prepared_image(identifier, size=(100, 100)) for identifier in range(1, 5))
    normalized = normalize_model_images(
        originals,
        max_long_edge=1_000,
        max_image_pixels=10_000,
        max_total_image_pixels=8_000,
    )
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    responder = LmStudioResponder(
        _config(
            model_image_max_long_edge=1_000,
            model_image_max_pixels=10_000,
            model_total_image_pixels=8_000,
        ),
        transport=_transport(handler),
    )
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    try:
        result = await queue.submit(_request(prepared_images=originals))
    finally:
        await queue.close()

    assert result == "ok"
    assert len(captured) == 1
    content = json.loads(captured[0].content)["messages"][1]["content"]
    assert [_data_uri_parts(block)[1] for block in content[1:]] == [
        image.data for image in normalized
    ]


@pytest.mark.parametrize("status_code", [400, 401, 500, 503])
def test_multimodal_http_errors_do_not_retry_or_leak_payload(status_code: int, caplog) -> None:
    private_message = "phase2bb-private-message-content-check"
    private_marker = (
        "phase2bb-private-image-bytes-check "
        "phase2bb-private-base64-check "
        "phase2bb-private-data-uri-check"
    )
    private_image = _prepared_image(1, "image/png", marker=private_marker)
    private_base64 = base64.b64encode(private_image.data).decode("ascii")
    private_data_uri = f"data:image/png;base64,{private_base64}"
    secret = "phase2bb-lmstudio-secret-check"
    private_server_body = "phase2bb-private-server-body-check"
    calls: list[httpx.Request] = []
    logger = logging.getLogger(f"test.lm.multimodal-http.{status_code}")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status_code, text=private_server_body)

    responder = LmStudioResponder(
        _config(lm_studio_api_token=secret),
        transport=_transport(handler),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with pytest.raises(LmStudioHttpError) as raised:
            asyncio.run(
                responder.generate(
                    _request(
                        content=private_message,
                        prepared_images=(private_image,),
                    )
                )
            )

    assert raised.value.status_code == status_code
    assert len(calls) == 1
    assert calls[0].headers["authorization"] == f"Bearer {secret}"
    assert isinstance(json.loads(calls[0].content)["messages"][1]["content"], list)
    for private_value in (
        private_message,
        private_marker,
        private_base64,
        private_data_uri,
        secret,
        private_server_body,
    ):
        assert private_value not in caplog.text
        assert private_value not in str(raised.value)


@pytest.mark.parametrize("exception_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_multimodal_connection_and_timeout_fail_without_retry(
    exception_type: type[httpx.RequestError],
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise exception_type("phase2bb-private-request-error", request=request)

    expected_error = (
        LmStudioUnavailableError if exception_type is httpx.ConnectError else LmStudioTimeoutError
    )
    with pytest.raises(expected_error) as raised:
        asyncio.run(
            LmStudioResponder(_config(), transport=_transport(handler)).generate(
                _request(prepared_images=(_prepared_image(1),))
            )
        )

    assert len(calls) == 1
    assert "phase2bb-private-request-error" not in str(raised.value)


@pytest.mark.parametrize(
    "response_json",
    [
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{}]},
    ],
)
def test_multimodal_protocol_failures_keep_phase2a_parser(response_json: object) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=response_json)

    with pytest.raises(LmStudioProtocolError):
        asyncio.run(
            LmStudioResponder(_config(), transport=_transport(handler)).generate(
                _request(prepared_images=(_prepared_image(1),))
            )
        )

    assert len(calls) == 1


@async_test
async def test_generate_sends_optional_bearer_token_without_logging_it(caplog) -> None:
    secret = "phase2a-lmstudio-secret-check"
    captured: list[httpx.Request] = []
    logger = logging.getLogger("test.lm.auth")

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    responder = LmStudioResponder(
        _config(lm_studio_api_token=secret),
        transport=_transport(handler),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        assert await responder.generate(_request()) == "ok"

    assert captured[0].headers["authorization"] == f"Bearer {secret}"
    assert secret not in caplog.text


@pytest.mark.parametrize(
    "response_json",
    [
        [],
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": 123}}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "  "}}]},
    ],
)
def test_invalid_generation_protocol_responses_fail_safely(response_json: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_json)

    with pytest.raises(LmStudioProtocolError) as raised:
        responder = LmStudioResponder(_config(), transport=_transport(handler))
        asyncio.run(responder.generate(_request()))

    assert "phase2a-private-message-content-check" not in str(raised.value)


def test_reasoning_only_length_response_is_rejected_without_leakage(caplog) -> None:
    private_reasoning = "phase2a-private-reasoning-content-check"
    logger = logging.getLogger("test.lm.reasoning-only")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "",
                            "reasoning_content": private_reasoning,
                        },
                    }
                ],
                "usage": {"completion_tokens": 256},
            },
        )

    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with pytest.raises(LmStudioProtocolError) as raised:
            asyncio.run(responder.generate(_request()))

    assert private_reasoning not in str(raised.value)
    assert private_reasoning not in caplog.text
    assert "phase2a-private-message-content-check" not in caplog.text
    assert "finish_reason=length" in caplog.text
    assert "generation reached token ceiling" in caplog.text
    assert "max_tokens=1536" in caplog.text
    assert "LM Studio generation completed" not in caplog.text


@pytest.mark.parametrize(
    ("raw_finish_reason", "diagnostic_finish_reason"),
    [("stop", "stop"), ("length", "length"), ("tool_calls", "other"), (None, "other")],
)
def test_finish_reason_diagnostics_are_safe(
    raw_finish_reason: str | None,
    diagnostic_finish_reason: str,
    caplog,
) -> None:
    private_user = "phase3c-private-finish-user-content"
    private_result = "phase3c-private-finish-assistant-content"
    private_reasoning = "phase3c-private-finish-reasoning-content"
    logger = logging.getLogger(f"test.lm.finish-reason.{diagnostic_finish_reason}")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": raw_finish_reason,
                        "message": {
                            "content": private_result,
                            "reasoning_content": private_reasoning,
                        },
                    }
                ],
                "usage": {"completion_tokens": 4096},
            },
        )

    responder = LmStudioResponder(
        _config(),
        transport=_transport(handler),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        assert asyncio.run(responder.generate(_request(content=private_user))) == private_result

    assert f"finish_reason={diagnostic_finish_reason}" in caplog.text
    assert "message_id=1" in caplog.text
    assert "response_mode=NORMAL" in caplog.text
    assert "max_tokens=1536" in caplog.text
    for private_value in (private_user, private_result, private_reasoning):
        assert private_value not in caplog.text


def test_invalid_generation_json_fails_safely() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    with pytest.raises(LmStudioProtocolError):
        responder = LmStudioResponder(_config(), transport=_transport(handler))
        asyncio.run(responder.generate(_request()))


@pytest.mark.parametrize("status_code", [400, 401, 404, 500, 503])
def test_http_errors_are_mapped_without_response_body_logging(status_code: int, caplog) -> None:
    private_body = "phase2a-private-server-error-body-check"
    logger = logging.getLogger(f"test.lm.http.{status_code}")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=private_body)

    responder = LmStudioResponder(_config(), transport=_transport(handler), logger=logger)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with pytest.raises(LmStudioHttpError) as raised:
            asyncio.run(responder.generate(_request()))

    assert raised.value.status_code == status_code
    assert str(raised.value) == f"LM Studio returned HTTP {status_code}"
    assert f"http_status={status_code}" in caplog.text
    assert private_body not in caplog.text


def test_connection_error_is_mapped_without_private_detail() -> None:
    private_detail = "phase2a-private-connection-detail"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(private_detail, request=request)

    with pytest.raises(LmStudioUnavailableError) as raised:
        responder = LmStudioResponder(_config(), transport=_transport(handler))
        asyncio.run(responder.generate(_request()))

    assert private_detail not in str(raised.value)


def test_timeout_is_mapped_without_private_detail() -> None:
    private_detail = "phase2a-private-timeout-detail"
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ReadTimeout(private_detail, request=request)

    with pytest.raises(LmStudioTimeoutError) as raised:
        responder = LmStudioResponder(_config(), transport=_transport(handler))
        asyncio.run(responder.generate(_request()))

    assert private_detail not in str(raised.value)
    assert call_count == 1


@async_test
async def test_preflight_requires_exact_configured_model_and_uses_get_only() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": [{"id": "configured-model"}]})

    await LmStudioResponder(_config(), transport=_transport(handler)).preflight()

    assert len(captured) == 1
    assert captured[0].method == "GET"
    assert str(captured[0].url) == "http://localhost:1234/v1/models"


@async_test
async def test_preflight_sends_optional_bearer_token() -> None:
    secret = "phase2a-lmstudio-secret-check"
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": [{"id": "configured-model"}]})

    responder = LmStudioResponder(
        _config(lm_studio_api_token=secret),
        transport=_transport(handler),
    )
    await responder.preflight()

    assert captured[0].headers["authorization"] == f"Bearer {secret}"


@pytest.mark.parametrize(
    "response_json",
    [
        {},
        {"data": []},
        {"data": [{"id": "CONFIGURED-MODEL"}]},
        {"data": [{"id": "other-model"}]},
    ],
)
def test_preflight_rejects_missing_or_non_exact_model(response_json: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_json)

    with pytest.raises((LmStudioModelUnavailableError, LmStudioProtocolError)):
        asyncio.run(LmStudioResponder(_config(), transport=_transport(handler)).preflight())


@pytest.mark.parametrize("status_code", [401, 500])
def test_preflight_http_error_is_safe(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="phase2a-private-server-error-body-check")

    with pytest.raises(LmStudioHttpError):
        asyncio.run(LmStudioResponder(_config(), transport=_transport(handler)).preflight())


def test_preflight_connection_timeout_and_invalid_json_fail_safely() -> None:
    def connection_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private", request=request)

    with pytest.raises(LmStudioUnavailableError):
        responder = LmStudioResponder(_config(), transport=_transport(connection_handler))
        asyncio.run(responder.preflight())

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private", request=request)

    with pytest.raises(LmStudioTimeoutError):
        asyncio.run(LmStudioResponder(_config(), transport=_transport(timeout_handler)).preflight())

    def invalid_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    with pytest.raises(LmStudioProtocolError):
        responder = LmStudioResponder(_config(), transport=_transport(invalid_json_handler))
        asyncio.run(responder.preflight())


@dataclass
class _RequestCounter:
    active: int = 0
    max_active: int = 0
    calls: list[int] | None = None


@async_test
async def test_mixed_native_and_completions_remain_fifo_with_one_active_post() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    calls: list[tuple[str, str]] = []
    active = 0
    max_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        payload = json.loads(request.content)
        if request.url.path.endswith("/api/v1/chat"):
            transport = "native"
            content = _current_value(payload["input"])
        else:
            transport = "completions"
            content = _current_value(payload["messages"][1]["content"])
        calls.append((transport, content))
        active += 1
        max_active = max(max_active, active)
        try:
            if content == "hi":
                first_started.set()
                await release_first.wait()
            if transport == "native":
                return httpx.Response(
                    200,
                    json={"output": [{"type": "message", "content": "native ok"}]},
                )
            return httpx.Response(200, json={"choices": [{"message": {"content": "normal ok"}}]})
        finally:
            active -= 1

    queue = SerializedGenerationQueue(
        LmStudioResponder(_config(), transport=httpx.MockTransport(handler))
    )
    await queue.start()
    try:
        first = asyncio.create_task(queue.submit(_request(1, "hi")))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = asyncio.create_task(queue.submit(_request(2, "VRAM是什麼？")))
        third = asyncio.create_task(queue.submit(_request(3, "hello")))
        await asyncio.sleep(0)
        assert calls == [("native", "hi")]

        release_first.set()
        assert await asyncio.gather(first, second, third) == [
            "native ok",
            "normal ok",
            "native ok",
        ]
    finally:
        await queue.close()

    assert calls == [
        ("native", "hi"),
        ("completions", "VRAM是什麼？"),
        ("native", "hello"),
    ]
    assert max_active == 1


@async_test
async def test_native_failure_does_not_terminate_mixed_queue_worker() -> None:
    calls: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path.endswith("/api/v1/chat"):
            transport = "native"
            content = _current_value(payload["input"])
            calls.append((transport, content))
            if content == "hi":
                return httpx.Response(500, text="phase3cb-private-native-failure")
            return httpx.Response(
                200,
                json={"output": [{"type": "message", "content": "native ok"}]},
            )
        transport = "completions"
        content = _current_value(payload["messages"][1]["content"])
        calls.append((transport, content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "normal ok"}}]})

    responder = LmStudioResponder(_config(), transport=httpx.MockTransport(handler))
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    try:
        with pytest.raises(LmStudioHttpError):
            await queue.submit(_request(1, "hi"))
        assert await queue.submit(_request(2, "VRAM是什麼？")) == "normal ok"
        assert await queue.submit(_request(3, "hello")) == "native ok"
    finally:
        await queue.close()

    assert calls == [
        ("native", "hi"),
        ("completions", "VRAM是什麼？"),
        ("native", "hello"),
    ]


@async_test
async def test_lm_studio_generation_remains_serialized_through_queue() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    counter = _RequestCounter(calls=[])

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        message_id = int(_current_value(payload["messages"][1]["content"]))
        counter.calls.append(message_id)
        counter.active += 1
        counter.max_active = max(counter.max_active, counter.active)
        try:
            if message_id == 1:
                first_started.set()
                await release_first.wait()
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        finally:
            counter.active -= 1

    responder = LmStudioResponder(_config(), transport=httpx.MockTransport(handler))
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    try:
        first = asyncio.create_task(queue.submit(_request(1, "1")))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = asyncio.create_task(queue.submit(_request(2, "2")))
        await asyncio.sleep(0)
        assert counter.calls == [1]

        release_first.set()
        assert await asyncio.gather(first, second) == ["ok", "ok"]
    finally:
        await queue.close()

    assert counter.calls == [1, 2]
    assert counter.max_active == 1


@async_test
async def test_multimodal_request_uses_one_queue_job_and_one_http_post() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    responder = LmStudioResponder(_config(), transport=httpx.MockTransport(handler))
    queue = SerializedGenerationQueue(responder)
    await queue.start()
    try:
        result = await queue.submit(
            _request(
                prepared_images=(
                    _prepared_image(1, "image/png"),
                    _prepared_image(2, "image/jpeg"),
                    _prepared_image(3, "image/webp"),
                )
            )
        )
    finally:
        await queue.close()

    assert result == "ok"
    assert len(captured) == 1
    content = json.loads(captured[0].content)["messages"][1]["content"]
    assert len(content) == 4


@async_test
async def test_mixed_text_and_multimodal_requests_remain_fifo_with_one_active_post() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    counter = _RequestCounter(calls=[])

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        message_id = int(_current_value(content))
        counter.calls.append(message_id)
        counter.active += 1
        counter.max_active = max(counter.max_active, counter.active)
        try:
            if message_id == 1:
                first_started.set()
                await release_first.wait()
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        finally:
            counter.active -= 1

    queue = SerializedGenerationQueue(
        LmStudioResponder(_config(), transport=httpx.MockTransport(handler))
    )
    await queue.start()
    try:
        first = asyncio.create_task(
            queue.submit(_request(1, "1", (_prepared_image(1, "image/png"),)))
        )
        await asyncio.wait_for(first_started.wait(), timeout=1)
        second = asyncio.create_task(queue.submit(_request(2, "2")))
        third = asyncio.create_task(
            queue.submit(_request(3, "3", (_prepared_image(3, "image/jpeg"),)))
        )
        await asyncio.sleep(0)
        assert counter.calls == [1]

        release_first.set()
        assert await asyncio.gather(first, second, third) == ["ok", "ok", "ok"]
    finally:
        await queue.close()

    assert counter.calls == [1, 2, 3]
    assert counter.max_active == 1


@async_test
async def test_multimodal_failure_does_not_terminate_shared_queue_worker() -> None:
    calls: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        message_id = int(_current_value(content))
        calls.append(message_id)
        if message_id == 1:
            return httpx.Response(500, text="phase2bb-private-server-body-check")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    queue = SerializedGenerationQueue(
        LmStudioResponder(_config(), transport=httpx.MockTransport(handler))
    )
    await queue.start()
    try:
        with pytest.raises(LmStudioHttpError):
            await queue.submit(_request(1, "1", (_prepared_image(1),)))
        assert await queue.submit(_request(2, "2")) == "ok"
        assert await queue.submit(_request(3, "3", (_prepared_image(3),))) == "ok"
    finally:
        await queue.close()

    assert calls == [1, 2, 3]


@async_test
async def test_normalization_failure_does_not_terminate_shared_queue_worker(caplog) -> None:
    calls: list[int] = []
    logger = logging.getLogger("test.lm.normalization-failure")

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        message_id = int(_current_value(content))
        calls.append(message_id)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    queue = SerializedGenerationQueue(
        LmStudioResponder(_config(), transport=httpx.MockTransport(handler)),
        logger=logger,
    )
    await queue.start()
    try:
        invalid = PreparedImage(1, "image/png", b"phase2bb-private-image-bytes-check", 1, 1)
        with caplog.at_level(logging.ERROR, logger=logger.name):
            with pytest.raises(LmStudioImageInputError):
                await queue.submit(_request(1, "1", (invalid,)))
        assert await queue.submit(_request(2, "2")) == "ok"
        assert await queue.submit(_request(3, "3", (_prepared_image(3),))) == "ok"
    finally:
        await queue.close()

    assert calls == [2, 3]
    assert "phase2bb-private-image-bytes-check" not in caplog.text


def test_response_metrics_do_not_log_content_or_result(caplog) -> None:
    private_result = f"phase2a-private-result-{uuid4().hex}"
    logger = logging.getLogger("test.lm.metrics")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": private_result}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
        )

    responder = LmStudioResponder(_config(), transport=_transport(handler), logger=logger)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = asyncio.run(responder.generate(_request()))

    assert result == private_result
    assert "message_id=1" in caplog.text
    assert "mode=NORMAL" in caplog.text
    assert "reason=DEFAULT_NORMAL" in caplog.text
    assert "max_tokens=1536" in caplog.text
    assert "prompt_tokens=3" in caplog.text
    assert "completion_tokens=4" in caplog.text
    assert "phase2a-private-message-content-check" not in caplog.text
    assert private_result not in caplog.text


def test_multimodal_success_diagnostics_do_not_log_payload_data(caplog) -> None:
    private_message = "phase2bb-private-message-content-check"
    private_marker = "phase2bb-private-image-bytes-check"
    private_image = _prepared_image(1, "image/png", marker=private_marker)
    private_base64 = base64.b64encode(private_image.data).decode("ascii")
    private_data_uri = f"data:image/png;base64,{private_base64}"
    secret = "phase2bb-lmstudio-secret-check"
    logger = logging.getLogger("test.lm.multimodal-privacy")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    responder = LmStudioResponder(
        _config(lm_studio_api_token=secret),
        transport=_transport(handler),
        logger=logger,
    )
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = asyncio.run(
            responder.generate(
                _request(
                    content=private_message,
                    prepared_images=(private_image,),
                )
            )
        )

    assert result == "ok"
    for private_value in (
        private_message,
        private_marker,
        private_base64,
        private_data_uri,
        secret,
    ):
        assert private_value not in caplog.text

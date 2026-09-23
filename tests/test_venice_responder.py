from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import UTC, datetime

import httpx
import pytest
from PIL import Image

from liuer_bot.config import Config
from liuer_bot.conversation_context import ConversationGenerationContext
from liuer_bot.conversation_models import AuthorKind, ConversationTurn
from liuer_bot.image_preparation import PreparedImage
from liuer_bot.llm_provider import LlmProviderKind, create_llm_provider
from liuer_bot.lm_studio_responder import LmStudioResponder
from liuer_bot.reply_context import ReplyContext
from liuer_bot.responder import ChatRequest
from liuer_bot.response_routing import (
    ResponseDecision,
    ResponseMode,
    ResponseRouteReason,
)
from liuer_bot.venice_models import VeniceModelRole, select_venice_model_role
from liuer_bot.venice_responder import (
    VeniceAuthError,
    VeniceHttpError,
    VeniceModelIncompatibleError,
    VeniceModelUnavailableError,
    VeniceProtocolError,
    VeniceResponder,
    VeniceTimeoutError,
    VeniceUnavailableError,
)

API_KEY = "phase7a-private-venice-key"
PRIVATE_REASONING = "phase7a-private-reasoning"


def _config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "discord_token": "discord-token",
        "chat_channel_id": 123,
        "persona_status_channel_id": 456,
        "llm_provider": LlmProviderKind.VENICE,
        "venice_base_url": "https://venice.test/api/v1",
        "venice_api_key": API_KEY,
        "venice_text_model_id": "exact-text-model",
        "venice_vision_model_id": "exact-vision-model",
        "lm_studio_model": "",
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def _model(*, model_id: str = "exact-vision-model", **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": model_id,
        "supportsVision": True,
        "supportsMultipleImages": True,
        "maxImages": 4,
        "offline": False,
        "deprecated": False,
    }
    value.update(overrides)
    return value


def _text_model(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "exact-text-model",
        "supportsVision": False,
        "supportsMultipleImages": False,
        "offline": False,
    }
    value.update(overrides)
    return value


def _answer(*, finish_reason: str = "stop") -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {
                    "content": "ANSWER_MARKER",
                    "reasoning_content": PRIVATE_REASONING,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


def _request(
    content: str = "你好",
    *,
    images: tuple[PreparedImage, ...] = (),
    author_kind: AuthorKind = AuthorKind.HUMAN,
    context: ConversationGenerationContext | None = None,
) -> ChatRequest:
    return ChatRequest(
        message_id=1,
        guild_id=20,
        channel_id=123,
        author_id=9,
        content=content,
        prepared_images=images,
        author_display_name="Current Speaker",
        generation_context=context,
        author_kind=author_kind,
    )


class FixedRouter:
    def __init__(self, mode: ResponseMode, max_tokens: int) -> None:
        self.decision = ResponseDecision(mode, max_tokens, ResponseRouteReason.DEFAULT_NORMAL)

    def route(self, request: ChatRequest) -> ResponseDecision:
        return self.decision


def _png(attachment_id: int, color: tuple[int, int, int]) -> PreparedImage:
    image = Image.new("RGB", (2, 2), color)
    output = io.BytesIO()
    try:
        image.save(output, format="PNG")
    finally:
        image.close()
    return PreparedImage(attachment_id, "image/png", output.getvalue(), 2, 2)


def test_provider_factory_selects_only_configured_provider() -> None:
    venice = create_llm_provider(_config())
    local = create_llm_provider(
        Config(
            discord_token="token",
            chat_channel_id=123,
            llm_provider=LlmProviderKind.LM_STUDIO,
            lm_studio_model="local-model",
        )
    )

    assert isinstance(venice, VeniceResponder)
    assert isinstance(local, LmStudioResponder)


def test_venice_preflight_requires_exact_compatible_model_and_one_get() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [_text_model(), _model()]})

    responder = VeniceResponder(_config(), transport=httpx.MockTransport(handler))
    asyncio.run(responder.preflight())

    assert [request.method for request in requests] == ["GET"]
    assert [request.url.path for request in requests] == ["/api/v1/models"]
    assert requests[0].headers["authorization"] == f"Bearer {API_KEY}"


def test_venice_preflight_accepts_explicitly_text_only_model() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [_text_model(supportsText=True), _model()]},
        )

    asyncio.run(VeniceResponder(_config(), transport=httpx.MockTransport(handler)).preflight())


@pytest.mark.parametrize(
    ("text_model", "error"),
    [
        (_text_model(offline=True), VeniceModelIncompatibleError),
        (_text_model(offline=False, status="unavailable"), VeniceModelIncompatibleError),
        (_text_model(unavailable=True), VeniceModelIncompatibleError),
        (_text_model(supportsText=False), VeniceModelIncompatibleError),
        (_text_model(inputModalities=["image"]), VeniceModelIncompatibleError),
    ],
)
def test_venice_text_preflight_fails_closed_for_explicit_incompatibility(
    text_model: dict[str, object],
    error: type[Exception],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [text_model, _model()]})

    with pytest.raises(error, match="text model"):
        asyncio.run(VeniceResponder(_config(), transport=httpx.MockTransport(handler)).preflight())


def test_venice_same_id_preflight_uses_one_catalog_request() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [_model(model_id="same-model")]})

    config = _config(venice_text_model_id="same-model", venice_vision_model_id="same-model")
    asyncio.run(VeniceResponder(config, transport=httpx.MockTransport(handler)).preflight())
    assert len(requests) == 1


def test_venice_same_id_still_routes_text_and_image_by_current_input() -> None:
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    config = _config(venice_text_model_id="same-model", venice_vision_model_id="same-model")
    responder = VeniceResponder(config, transport=httpx.MockTransport(handler))

    async def run() -> None:
        await responder.generate(_request("text"))
        await responder.generate(_request("image", images=(_png(1, (1, 1, 1)),)))

    asyncio.run(run())
    assert [payload["model"] for payload in payloads] == ["same-model", "same-model"]
    assert isinstance(payloads[0]["messages"][-1]["content"], str)  # type: ignore[index]
    assert isinstance(payloads[1]["messages"][-1]["content"], list)  # type: ignore[index]


def test_venice_preflight_reads_explicit_nested_capability_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    _text_model(),
                    {
                        "id": "exact-vision-model",
                        "status": "online",
                        "model_spec": {
                            "capabilities": {
                                "supportsVision": True,
                                "supportsMultipleImages": True,
                                "maxImages": 4,
                            }
                        },
                    }
                ]
            },
        )

    asyncio.run(
        VeniceResponder(_config(), transport=httpx.MockTransport(handler)).preflight()
    )


@pytest.mark.parametrize(
    ("models", "error"),
    [
        ([_model(model_id="other")], VeniceModelUnavailableError),
        ([_text_model(), _model(supportsVision=False)], VeniceModelIncompatibleError),
        ([_text_model(), _model(supportsMultipleImages=False)], VeniceModelIncompatibleError),
        ([_text_model(), _model(maxImages=3)], VeniceModelIncompatibleError),
        ([_text_model(), _model(offline=True)], VeniceModelIncompatibleError),
        ([_text_model(), {"id": "exact-vision-model"}], VeniceModelIncompatibleError),
    ],
)
def test_venice_preflight_fails_closed_for_missing_or_incompatible_model(
    models: list[dict[str, object]],
    error: type[Exception],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": models})

    with pytest.raises(error):
        asyncio.run(
            VeniceResponder(_config(), transport=httpx.MockTransport(handler)).preflight()
        )


def test_venice_preflight_auth_failure_redacts_secret(caplog) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"body {API_KEY}")

    with caplog.at_level(logging.DEBUG), pytest.raises(VeniceAuthError) as captured:
        asyncio.run(
            VeniceResponder(_config(), transport=httpx.MockTransport(handler)).preflight()
        )

    assert API_KEY not in str(captured.value)
    assert API_KEY not in repr(captured.value)
    assert API_KEY not in caplog.text


def test_venice_preflight_warns_safely_for_deprecated_compatible_model(caplog) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [_text_model(), _model(deprecated=True)]})

    with caplog.at_level(logging.WARNING):
        asyncio.run(
            VeniceResponder(_config(), transport=httpx.MockTransport(handler)).preflight()
        )

    assert "deprecated" in caplog.text
    assert API_KEY not in caplog.text


@pytest.mark.parametrize(
    ("mode", "max_tokens", "reasoning_expected"),
    [
        (ResponseMode.CHAT_SHORT, 1024, True),
        (ResponseMode.NORMAL, 1536, False),
        (ResponseMode.DEEP, 4096, False),
    ],
)
def test_venice_modes_use_one_stateless_chat_completion_with_safe_controls(
    mode: ResponseMode,
    max_tokens: int,
    reasoning_expected: bool,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_answer())

    responder = VeniceResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        response_router=FixedRouter(mode, max_tokens),  # type: ignore[arg-type]
    )
    result = asyncio.run(responder.generate(_request()))
    payload = json.loads(requests[0].content)

    assert result == "ANSWER_MARKER"
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v1/chat/completions"
    assert payload["model"] == "exact-text-model"
    assert payload["stream"] is False
    assert payload["max_tokens"] == max_tokens
    assert payload["venice_parameters"] == {
        "include_venice_system_prompt": False,
        "enable_web_search": "off",
        "enable_web_scraping": False,
        "enable_web_citations": False,
        "enable_x_search": False,
    }
    assert (payload.get("reasoning") == {"enabled": False}) is reasoning_expected
    assert "store" not in payload
    assert "previous_response_id" not in payload
    assert "character_slug" not in payload
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1]["role"] == "user"


def test_venice_history_preserves_explicit_human_and_bot_speaker_labels() -> None:
    now = datetime.now(UTC)
    turns = (
        ConversationTurn(1, 123, 1, 1, "Human One", "human history", 0, "a1", now),
        ConversationTurn(
            2,
            123,
            2,
            2,
            "Peer Bot",
            "bot history",
            0,
            "a2",
            now,
            author_kind=AuthorKind.BOT,
        ),
    )
    context = ConversationGenerationContext(context_epoch=1, recent_turns=turns)
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    responder = VeniceResponder(
        _config(context_history_max_chars_normal=12000),
        transport=httpx.MockTransport(handler),
        response_router=FixedRouter(ResponseMode.NORMAL, 1536),  # type: ignore[arg-type]
    )
    asyncio.run(responder.generate(_request("question", context=context)))

    serialized = json.dumps(payloads[0], ensure_ascii=False)
    assert "Human One" in serialized
    assert "Peer Bot" in serialized
    assert "human history" in serialized
    assert "bot history" in serialized


def test_venice_preserves_persona_nickname_and_reply_context_in_liuer_prompt() -> None:
    payloads: list[dict[str, object]] = []

    class Persona:
        def get_active_persona(self) -> str:
            return "CUSTOM_PERSONA_MARKER"

    class Nickname:
        def get_active_nickname(self) -> str:
            return "小六"

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    request = _request("current question")
    request = ChatRequest(
        request.message_id,
        request.guild_id,
        request.channel_id,
        request.author_id,
        request.content,
        author_display_name=request.author_display_name,
        reply_context=ReplyContext(
            message_id=99,
            author_display_name="Quoted Speaker",
            author_is_bot=False,
            content="QUOTED_REPLY_MARKER",
            image_count=0,
        ),
    )
    responder = VeniceResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        active_persona_provider=Persona(),  # type: ignore[arg-type]
        active_nickname_provider=Nickname(),  # type: ignore[arg-type]
        response_router=FixedRouter(ResponseMode.NORMAL, 1536),  # type: ignore[arg-type]
    )
    asyncio.run(responder.generate(request))

    system_prompt = payloads[0]["messages"][0]["content"]  # type: ignore[index]
    current_user = payloads[0]["messages"][-1]["content"]  # type: ignore[index]
    assert "CUSTOM_PERSONA_MARKER" in system_prompt
    assert "小六" in system_prompt
    assert "QUOTED_REPLY_MARKER" in current_user
    assert payloads[0]["venice_parameters"]["include_venice_system_prompt"] is False  # type: ignore[index]


def test_venice_text_and_vision_share_current_persona_and_nickname_prompt() -> None:
    payloads: list[dict[str, object]] = []

    class Persona:
        def get_active_persona(self) -> str:
            return "CUSTOM_PERSONA_MARKER"

    class Nickname:
        def get_active_nickname(self) -> str:
            return "小六"

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    responder = VeniceResponder(
        _config(), transport=httpx.MockTransport(handler),
        active_persona_provider=Persona(),  # type: ignore[arg-type]
        active_nickname_provider=Nickname(),  # type: ignore[arg-type]
    )

    async def run() -> None:
        await responder.generate(_request("text"))
        await responder.generate(_request("image", images=(_png(1, (1, 2, 3)),)))

    asyncio.run(run())
    assert [payload["model"] for payload in payloads] == [
        "exact-text-model", "exact-vision-model",
    ]
    for payload in payloads:
        prompt = payload["messages"][0]["content"]  # type: ignore[index]
        assert "CUSTOM_PERSONA_MARKER" in prompt
        assert "小六" in prompt
        assert payload["venice_parameters"]["include_venice_system_prompt"] is False  # type: ignore[index]


def test_venice_multimodal_keeps_one_request_and_image_order() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_answer())

    images = (_png(1, (255, 0, 0)), _png(2, (0, 255, 0)))
    responder = VeniceResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        response_router=FixedRouter(ResponseMode.NORMAL, 1536),  # type: ignore[arg-type]
    )
    asyncio.run(responder.generate(_request("inspect", images=images)))
    payload = json.loads(requests[0].content)
    content = payload["messages"][-1]["content"]

    assert len(requests) == 1
    assert [block["type"] for block in content] == ["text", "image_url", "image_url"]
    assert all(
        block["image_url"]["url"].startswith("data:image/png;base64,")
        for block in content[1:]
    )
    assert payload["venice_parameters"]["enable_web_search"] == "off"
    assert payload["model"] == "exact-vision-model"


@pytest.mark.parametrize("image_count", [2, 3, 4])
def test_venice_multiple_current_images_select_vision_once(image_count: int) -> None:
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    images = tuple(_png(index, (index, 0, 0)) for index in range(image_count))
    responder = VeniceResponder(_config(), transport=httpx.MockTransport(handler))
    asyncio.run(responder.generate(_request(images=images)))

    assert len(payloads) == 1
    assert payloads[0]["model"] == "exact-vision-model"
    content = payloads[0]["messages"][-1]["content"]  # type: ignore[index]
    assert [block["type"] for block in content] == ["text", *["image_url"] * image_count]


def test_venice_historical_image_marker_selects_text_model() -> None:
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    turn = ConversationTurn(
        1, 123, 2, 3, "Earlier Human", "prior image", 1, "answer", datetime.now(UTC)
    )
    context = ConversationGenerationContext(context_epoch=1, recent_turns=(turn,))
    responder = VeniceResponder(_config(), transport=httpx.MockTransport(handler))
    asyncio.run(responder.generate(_request("new text only", context=context)))

    assert payloads[0]["model"] == "exact-text-model"
    assert "歷史附件" in json.dumps(payloads[0]["messages"], ensure_ascii=False)


def test_venice_reply_image_marker_selects_text_model() -> None:
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    base = _request("text only")
    request = ChatRequest(
        base.message_id, base.guild_id, base.channel_id, base.author_id, base.content,
        reply_context=ReplyContext(99, "Prior Bot", True, "referenced text", 1),
    )
    responder = VeniceResponder(_config(), transport=httpx.MockTransport(handler))
    asyncio.run(responder.generate(request))

    assert payloads[0]["model"] == "exact-text-model"
    assert "referenced text" in json.dumps(payloads[0]["messages"])


@pytest.mark.parametrize("with_image", [False, True])
def test_venice_botchat_author_kind_does_not_override_image_role(with_image: bool) -> None:
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=_answer())

    images = (_png(1, (1, 2, 3)),) if with_image else ()
    responder = VeniceResponder(_config(), transport=httpx.MockTransport(handler))
    asyncio.run(responder.generate(_request(images=images, author_kind=AuthorKind.BOT)))

    assert payloads[0]["model"] == (
        "exact-vision-model" if with_image else "exact-text-model"
    )


def test_venice_model_role_selection_uses_only_current_prepared_images() -> None:
    assert select_venice_model_role(()) is VeniceModelRole.TEXT
    assert select_venice_model_role((_png(1, (0, 0, 0)),)) is VeniceModelRole.VISION


def test_venice_returns_only_final_content_and_never_logs_reasoning(caplog) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_answer(finish_reason="length"))

    responder = VeniceResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        response_router=FixedRouter(ResponseMode.NORMAL, 1536),  # type: ignore[arg-type]
    )
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(responder.generate(_request("private human content")))

    assert result == "ANSWER_MARKER"
    assert PRIVATE_REASONING not in result
    assert PRIVATE_REASONING not in caplog.text
    assert API_KEY not in caplog.text
    assert "private human content" not in caplog.text
    assert "finish_reason=length" in caplog.text


def test_venice_both_model_roles_keep_request_and_image_bodies_out_of_logs(caplog) -> None:
    private_text = "phase7a-private-human-or-bot-content"

    class Persona:
        def get_active_persona(self) -> str:
            return "phase7a-private-persona"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_answer())

    responder = VeniceResponder(
        _config(), transport=httpx.MockTransport(handler),
        active_persona_provider=Persona(),  # type: ignore[arg-type]
    )

    async def run() -> None:
        await responder.generate(_request(private_text))
        await responder.generate(_request(
            private_text, images=(_png(1, (10, 20, 30)),), author_kind=AuthorKind.BOT,
        ))

    with caplog.at_level(logging.DEBUG):
        asyncio.run(run())

    for private in (
        API_KEY, PRIVATE_REASONING, private_text, "phase7a-private-persona", "data:image/",
    ):
        assert private not in caplog.text


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{"message": {}}]}])
def test_venice_rejects_missing_final_content(payload: dict[str, object]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    responder = VeniceResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        response_router=FixedRouter(ResponseMode.NORMAL, 1536),  # type: ignore[arg-type]
    )
    with pytest.raises(VeniceProtocolError):
        asyncio.run(responder.generate(_request()))


@pytest.mark.parametrize(
    ("status_code", "error"),
    [(401, VeniceAuthError), (403, VeniceAuthError), (500, VeniceHttpError)],
)
def test_venice_generation_http_failures_are_safe_and_never_retried(
    status_code: int,
    error: type[Exception],
    caplog,
) -> None:
    requests: list[httpx.Request] = []
    private_body = "phase7a-private-http-body"

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, text=f"{private_body} {API_KEY}")

    responder = VeniceResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        response_router=FixedRouter(ResponseMode.NORMAL, 1536),  # type: ignore[arg-type]
    )
    with caplog.at_level(logging.DEBUG), pytest.raises(error) as captured:
        asyncio.run(responder.generate(_request("private request body")))

    assert len(requests) == 1
    combined = caplog.text + str(captured.value) + repr(captured.value)
    assert API_KEY not in combined
    assert private_body not in combined
    assert "private request body" not in combined


@pytest.mark.parametrize(
    ("failure_kind", "error"),
    [
        ("connection", VeniceUnavailableError),
        ("timeout", VeniceTimeoutError),
    ],
)
def test_venice_generation_classifies_connection_and_timeout_without_retry(
    failure_kind: str,
    error: type[Exception],
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if failure_kind == "timeout":
            raise httpx.ReadTimeout("private timeout marker", request=request)
        raise httpx.ConnectError("private connection marker", request=request)

    responder = VeniceResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        response_router=FixedRouter(ResponseMode.NORMAL, 1536),  # type: ignore[arg-type]
    )
    with pytest.raises(error) as captured:
        asyncio.run(responder.generate(_request()))

    assert calls == 1
    assert "private" not in str(captured.value)

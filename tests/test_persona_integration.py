from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from PIL import Image

from liuer_bot.config import Config
from liuer_bot.image_preparation import PreparedImage
from liuer_bot.lm_studio_responder import LmStudioResponder
from liuer_bot.nickname_service import NicknameService
from liuer_bot.persona_service import PersonaService
from liuer_bot.prompting import load_default_persona
from liuer_bot.responder import ChatRequest


def _config() -> Config:
    return Config(
        discord_token="test-token",
        chat_channel_id=123,
        lm_studio_base_url="http://localhost:1234/v1",
        lm_studio_model="configured-model",
        lm_studio_timeout_seconds=300.0,
        lm_studio_temperature=0.7,
        lm_studio_max_tokens=4096,
    )


def _request(content: str = "hello") -> ChatRequest:
    return ChatRequest(1, 20, 123, 10, content)


@dataclass
class MutablePersonaProvider:
    active_persona: str | None = None

    def get_active_persona(self) -> str | None:
        return self.active_persona


@dataclass
class MutableNicknameProvider:
    active_nickname: str | None = None

    def get_active_nickname(self) -> str | None:
        return self.active_nickname


def test_active_persona_switches_on_next_generation_without_db_read(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def clock() -> datetime:
        return datetime.now(UTC)

    service = PersonaService(tmp_path / "persona.sqlite3", clock=clock)
    asyncio.run(service.initialize())
    provider = service
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    def fail_if_database_is_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("generation must not query SQLite")

    monkeypatch.setattr(service._repository, "load_state", fail_if_database_is_read)
    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        active_persona_provider=provider,
    )

    asyncio.run(responder.generate(_request("default request")))
    default_prompt = captured[-1]["messages"][0]["content"]
    assert load_default_persona() in default_prompt
    assert default_prompt.count("[RESPONSE MODE]") == 1

    # Change the in-memory provider through the domain service's committed API.
    asyncio.run(service.publish_persona(55, "community active persona"))
    asyncio.run(responder.generate(_request("community request")))
    active_prompt = captured[-1]["messages"][0]["content"]
    assert "community active persona" in active_prompt
    assert load_default_persona() not in active_prompt
    assert active_prompt.index("community active persona") < active_prompt.index("[RESPONSE MODE]")

    asyncio.run(service.reset_to_default(55))
    asyncio.run(responder.generate(_request("default again")))
    assert load_default_persona() in captured[-1]["messages"][0]["content"]


def test_native_chat_uses_default_and_active_persona_snapshot(tmp_path: Path) -> None:
    def clock() -> datetime:
        return datetime.now(UTC)

    service = PersonaService(tmp_path / "native-persona.sqlite3", clock=clock)
    asyncio.run(service.initialize())
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/v1/chat")
        payload = json.loads(request.content)
        captured.append(payload)
        return httpx.Response(200, json={"output": [{"type": "message", "content": "ok"}]})

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        active_persona_provider=service,
    )

    asyncio.run(responder.generate(_request("hi")))
    default_prompt = captured[-1]["system_prompt"]
    assert load_default_persona() in default_prompt
    assert "[CORE RULES]" in default_prompt
    assert "[IDENTITY]" in default_prompt
    assert "[ACTIVE PERSONA]" in default_prompt
    assert "[RESPONSE MODE]" in default_prompt
    assert default_prompt.index("[ACTIVE PERSONA]") < default_prompt.index("[RESPONSE MODE]")

    asyncio.run(service.publish_persona(55, "community native persona"))
    asyncio.run(responder.generate(_request("hi")))
    active_prompt = captured[-1]["system_prompt"]
    assert "community native persona" in active_prompt
    assert load_default_persona() not in active_prompt

    asyncio.run(service.reset_to_default(55))
    asyncio.run(responder.generate(_request("hi")))
    assert load_default_persona() in captured[-1]["system_prompt"]


def test_nickname_snapshot_reaches_native_normal_and_multimodal_prompts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    nickname_service = NicknameService(tmp_path / "nickname.sqlite3")
    asyncio.run(nickname_service.initialize())
    asyncio.run(nickname_service.set_nickname(55, "小六"))
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        if request.url.path.endswith("/api/v1/chat"):
            return httpx.Response(200, json={"output": [{"type": "message", "content": "ok"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    responder = LmStudioResponder(
        _config(),
        transport=httpx.MockTransport(handler),
        active_nickname_provider=nickname_service,
    )
    asyncio.run(responder.generate(_request("hi")))
    assert "目前的小名是「小六」" in captured[-1]["system_prompt"]

    asyncio.run(responder.generate(_request("VRAM是什麼？")))
    assert "目前的小名是「小六」" in captured[-1]["messages"][0]["content"]

    output = io.BytesIO()
    image = Image.new("RGB", (1, 1), (10, 20, 30))
    try:
        image.save(output, format="PNG")
    finally:
        image.close()
    request = ChatRequest(
        2,
        20,
        123,
        10,
        "看這張圖",
        prepared_images=(PreparedImage(1, "image/png", output.getvalue(), 1, 1),),
    )
    asyncio.run(responder.generate(request))
    assert "目前的小名是「小六」" in captured[-1]["messages"][0]["content"]

    def fail_if_database_is_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("nickname generation must not query SQLite")

    monkeypatch.setattr(nickname_service._repository, "load_state", fail_if_database_is_read)
    assert nickname_service.get_active_nickname() == "小六"

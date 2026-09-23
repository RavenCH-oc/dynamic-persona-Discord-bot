from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import discord
import httpx
import pytest

import liuer_bot.runtime_health as runtime_health
from liuer_bot.config import Config
from liuer_bot.conversation_models import ConversationRuntimeStatus
from liuer_bot.conversation_service import ConversationService
from liuer_bot.database import initialize_database
from liuer_bot.generation_queue import GenerationQueueStatus, SerializedGenerationQueue
from liuer_bot.llm_provider import LlmProviderKind
from liuer_bot.responder import ChatRequest
from liuer_bot.runtime_commands import RuntimeCommands
from liuer_bot.runtime_health import (
    DatabaseHealthProbe,
    DatabaseHealthResult,
    DatabaseHealthState,
    HealthOverallState,
    LmStudioHealthProbe,
    LmStudioHealthResult,
    LmStudioHealthState,
    RuntimeHealthService,
    VeniceHealthProbe,
)
from liuer_bot.runtime_status import RuntimeStatusService, RuntimeStatusSnapshot

PRIVATE_MARKER = "phase5c-private-runtime-marker"
LM_TOKEN = "phase5c-private-lm-token"


def _config(*, database_path: Path | None = None) -> Config:
    return Config(
        discord_token="phase5c-discord-token",
        chat_channel_id=123,
        persona_status_channel_id=456,
        lm_studio_base_url="http://lm-studio.test/v1",
        lm_studio_model="liuer-model",
        database_path=database_path or Path("data/test.sqlite3"),
    )


def _venice_config(*, database_path: Path | None = None) -> Config:
    return Config(
        discord_token="phase7a-discord-token",
        chat_channel_id=123,
        persona_status_channel_id=456,
        llm_provider=LlmProviderKind.VENICE,
        venice_base_url="https://venice.test/api/v1",
        venice_api_key="phase7a-private-venice-key",
        venice_text_model_id="venice-text-model",
        venice_vision_model_id="venice-vision-model",
        lm_studio_model="",
        database_path=database_path or Path("data/test.sqlite3"),
    )


def _request(message_id: int) -> ChatRequest:
    return ChatRequest(
        message_id=message_id,
        guild_id=20,
        channel_id=123,
        author_id=100,
        content=f"message-{message_id}",
    )


def _run(awaitable):
    return asyncio.run(awaitable)


class BlockingResponder:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: ChatRequest) -> str:
        self.started.set()
        await self.release.wait()
        return f"result-{request.message_id}"


def test_queue_status_distinguishes_busy_and_waiting_jobs() -> None:
    async def run() -> None:
        responder = BlockingResponder()
        queue = SerializedGenerationQueue(responder)
        assert queue.get_status() == GenerationQueueStatus(False, False, 0)
        await queue.start()
        first = asyncio.create_task(queue.submit(_request(1)))
        await responder.started.wait()
        assert queue.get_status() == GenerationQueueStatus(True, True, 0)

        second = asyncio.create_task(queue.submit(_request(2)))
        third = asyncio.create_task(queue.submit(_request(3)))
        for _ in range(20):
            if queue.get_status().waiting_jobs == 2:
                break
            await asyncio.sleep(0)
        assert queue.get_status() == GenerationQueueStatus(True, True, 2)
        assert PRIVATE_MARKER not in repr(queue.get_status())

        responder.release.set()
        assert await first == "result-1"
        assert await second == "result-2"
        assert await third == "result-3"
        await queue.close()
        assert queue.get_status() == GenerationQueueStatus(False, False, 0)

    _run(run())


def test_runtime_status_is_local_and_contains_only_scalar_operational_state() -> None:
    class Queue:
        def get_status(self) -> GenerationQueueStatus:
            return GenerationQueueStatus(True, False, 0)

    class Conversation:
        def get_runtime_status(self) -> ConversationRuntimeStatus:
            return ConversationRuntimeStatus(True, 4, 7)

    class Persona:
        current_snapshot = SimpleNamespace(active_version_id=2)

    class Nickname:
        def get_active_nickname(self) -> str:
            return "six"

    clock_values = iter((100.0, 102.25, 102.25))
    service = RuntimeStatusService(
        _config(),
        generation_queue=Queue(),  # type: ignore[arg-type]
        conversation_service=Conversation(),  # type: ignore[arg-type]
        persona_service=Persona(),  # type: ignore[arg-type]
        nickname_service=Nickname(),  # type: ignore[arg-type]
        runtime_ready=lambda: True,
        discord_latency=lambda: 0.042,
        monotonic_clock=lambda: next(clock_values),
    )

    snapshot = service.get_snapshot()
    rendered = service.render()
    assert snapshot.uptime_seconds == 2
    assert snapshot.discord_latency_ms == 42
    assert snapshot.queue_status == GenerationQueueStatus(True, False, 0)
    assert snapshot.conversation_status == ConversationRuntimeStatus(True, 4, 7)
    assert snapshot.schema_version == 6
    assert snapshot.persona_label == "Custom"
    assert snapshot.nickname == "six"
    assert "schema v6" in rendered
    assert "CHAT_SHORT: 1000" in rendered
    assert "NORMAL: 12000" in rendered
    assert "DEEP: 24000" in rendered
    assert "not performed by /status" in rendered
    assert PRIVATE_MARKER not in repr(snapshot)


def test_runtime_status_is_provider_neutral_and_does_not_probe_venice() -> None:
    class Queue:
        def get_status(self) -> GenerationQueueStatus:
            return GenerationQueueStatus(True, False, 0)

    service = RuntimeStatusService(
        _venice_config(),
        generation_queue=Queue(),  # type: ignore[arg-type]
        runtime_ready=lambda: True,
        monotonic_clock=lambda: 1.0,
    )

    rendered = service.render()

    assert "LLM Provider: Venice" in rendered
    assert "文字模型: venice-text-model" in rendered
    assert "圖片模型: venice-vision-model" in rendered
    assert "Active probe: not performed by /status" in rendered
    assert "phase7a-private-venice-key" not in rendered


class FakeStatusService:
    def __init__(self, snapshot: RuntimeStatusSnapshot) -> None:
        self.snapshot = snapshot
        self.render_calls = 0
        self.snapshot_calls = 0

    def get_snapshot(self) -> RuntimeStatusSnapshot:
        self.snapshot_calls += 1
        return self.snapshot

    def render(self) -> str:
        self.render_calls += 1
        return "local status"


def _snapshot(*, busy: bool = False, waiting: int = 0, worker_alive: bool = True):
    return RuntimeStatusSnapshot(
        runtime_ready=True,
        uptime_seconds=1,
        discord_latency_ms=42,
        queue_status=GenerationQueueStatus(worker_alive, busy, waiting),
        conversation_status=ConversationRuntimeStatus(True, 1, 0),
        schema_version=6,
        persona_label="Default",
        nickname=None,
        model_id="liuer-model",
        context_history_max_chars_chat_short=1000,
        context_history_max_chars_normal=12000,
        context_history_max_chars_deep=24000,
    )


class FakeProbe:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0

    async def probe(self):
        self.calls += 1
        return self.result


def test_health_busy_queue_is_healthy_and_probes_each_dependency_once() -> None:
    status = FakeStatusService(_snapshot(busy=True, waiting=2))
    database = FakeProbe(DatabaseHealthResult(DatabaseHealthState.READY, 6))
    lm_studio = FakeProbe(LmStudioHealthResult(LmStudioHealthState.READY, 3, True))
    service = RuntimeHealthService(
        _config(),
        status_service=status,  # type: ignore[arg-type]
        database_probe=database,  # type: ignore[arg-type]
        lm_studio_probe=lm_studio,  # type: ignore[arg-type]
    )

    result = _run(service.check())
    assert result.overall is HealthOverallState.HEALTHY
    assert result.queue_status.busy is True
    assert result.queue_status.waiting_jobs == 2
    assert database.calls == 1
    assert lm_studio.calls == 1
    assert "liuer-model" not in repr(result)


def test_health_worker_dead_is_unhealthy_without_restarting_queue() -> None:
    status = FakeStatusService(_snapshot(worker_alive=False))
    database = FakeProbe(DatabaseHealthResult(DatabaseHealthState.READY, 6))
    lm_studio = FakeProbe(LmStudioHealthResult(LmStudioHealthState.READY, 3, True))
    service = RuntimeHealthService(
        _config(),
        status_service=status,  # type: ignore[arg-type]
        database_probe=database,  # type: ignore[arg-type]
        lm_studio_probe=lm_studio,  # type: ignore[arg-type]
    )

    result = _run(service.check())
    assert result.overall is HealthOverallState.UNHEALTHY
    assert database.calls == 1
    assert lm_studio.calls == 1


def test_database_health_probe_checks_schema_and_closes_connection(tmp_path: Path) -> None:
    path = tmp_path / "database.sqlite3"
    initialize_database(path)

    result = _run(DatabaseHealthProbe(path).probe())

    assert result == DatabaseHealthResult(DatabaseHealthState.READY, 6)


def test_database_health_probe_distinguishes_schema_mismatch_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "database.sqlite3"
    initialize_database(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA user_version = 4")
        connection.commit()
    finally:
        connection.close()

    mismatch = _run(DatabaseHealthProbe(path).probe())
    missing = _run(DatabaseHealthProbe(tmp_path / "missing.sqlite3").probe())
    assert mismatch == DatabaseHealthResult(DatabaseHealthState.SCHEMA_MISMATCH, 4)
    assert missing == DatabaseHealthResult(DatabaseHealthState.ERROR, None)


def test_database_health_probe_closes_connection_on_probe_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "database.sqlite3"
    path.touch()
    statements: list[str] = []

    class Cursor:
        def __init__(self, value) -> None:
            self.value = value

        def fetchone(self):
            return self.value

    class Connection:
        closed = False

        def execute(self, statement: str):
            statements.append(statement)
            if statement == "SELECT 1":
                return Cursor((1,))
            raise RuntimeError(PRIVATE_MARKER)

        def close(self) -> None:
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(runtime_health, "connect_database", lambda path: connection)

    result = _run(DatabaseHealthProbe(path).probe())

    assert result == DatabaseHealthResult(DatabaseHealthState.ERROR, None)
    assert statements == ["SELECT 1", "PRAGMA user_version"]
    assert connection.closed is True


def _health_probe(
    handler,
    *,
    token: str | None = None,
    timeout_seconds: float = 5.0,
) -> LmStudioHealthProbe:
    return LmStudioHealthProbe(
        base_url="http://lm-studio.test/v1",
        configured_model="liuer-model",
        api_token=token,
        timeout_seconds=timeout_seconds,
        transport=httpx.MockTransport(handler),
    )


def test_lm_health_probe_ready_uses_exact_model_and_one_models_request() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "other"}, {"id": "liuer-model"}]})

    result = _run(_health_probe(handler).probe())

    assert result.state is LmStudioHealthState.READY
    assert result.configured_model_present is True
    assert [request.url.path for request in requests] == ["/v1/models"]


def test_lm_health_probe_model_missing_does_not_auto_select() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "other"}]})

    result = _run(_health_probe(handler).probe())

    assert result.state is LmStudioHealthState.MODEL_MISSING
    assert result.configured_model_present is False


def test_lm_health_probe_classifies_network_timeout_auth_http_and_payload_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(PRIVATE_MARKER, request=request)

    async def timeout(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"data": []})

    async def auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=PRIVATE_MARKER)

    async def server_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=PRIVATE_MARKER)

    async def invalid(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    with caplog.at_level(logging.DEBUG):
        results = [
            _run(_health_probe(unreachable, token=LM_TOKEN).probe()),
            _run(_health_probe(timeout, timeout_seconds=0.01).probe()),
            _run(_health_probe(auth, token=LM_TOKEN).probe()),
            _run(_health_probe(server_error).probe()),
            _run(_health_probe(invalid).probe()),
        ]

    assert [result.state for result in results] == [
        LmStudioHealthState.UNREACHABLE,
        LmStudioHealthState.TIMEOUT,
        LmStudioHealthState.AUTH_ERROR,
        LmStudioHealthState.HTTP_ERROR,
        LmStudioHealthState.INVALID_RESPONSE,
    ]
    assert LM_TOKEN not in caplog.text
    assert PRIVATE_MARKER not in caplog.text
    assert all(LM_TOKEN not in repr(result) for result in results)


def test_venice_health_probe_uses_one_models_get_and_requires_compatibility() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "venice-text-model", "supportsVision": False},
                    {
                        "id": "venice-vision-model",
                        "supportsVision": True,
                        "supportsMultipleImages": True,
                        "maxImages": 4,
                        "offline": False,
                    }
                ]
            },
        )

    result = _run(
        VeniceHealthProbe.from_config(
            _venice_config(),
            transport=httpx.MockTransport(handler),
        ).probe()
    )

    assert result.state is LmStudioHealthState.READY
    assert result.text_model_state is LmStudioHealthState.READY
    assert result.vision_model_state is LmStudioHealthState.READY
    assert result.configured_model_present is True
    assert [request.method for request in requests] == ["GET"]
    assert [request.url.path for request in requests] == ["/api/v1/models"]


@pytest.mark.parametrize(
    ("models", "text_state", "vision_state"),
    [
        (
            [{"id": "venice-vision-model", "supportsVision": True,
              "supportsMultipleImages": True, "maxImages": 4}],
            LmStudioHealthState.MODEL_MISSING,
            LmStudioHealthState.READY,
        ),
        (
            [{"id": "venice-text-model", "supportsVision": False}],
            LmStudioHealthState.READY,
            LmStudioHealthState.MODEL_MISSING,
        ),
        (
            [{"id": "venice-text-model", "supportsVision": False},
             {"id": "venice-vision-model", "supportsVision": False,
              "supportsMultipleImages": True, "maxImages": 4}],
            LmStudioHealthState.READY,
            LmStudioHealthState.INCOMPATIBLE_MODEL,
        ),
    ],
)
def test_venice_health_reports_each_role_and_fails_closed(
    models: list[dict[str, object]],
    text_state: LmStudioHealthState,
    vision_state: LmStudioHealthState,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": models})

    probe = VeniceHealthProbe.from_config(_venice_config(), transport=httpx.MockTransport(handler))
    result = _run(probe.probe())

    assert result.text_model_state is text_state
    assert result.vision_model_state is vision_state
    assert result.state is not LmStudioHealthState.READY
    assert len(requests) == 1


def test_venice_health_accepts_same_model_for_both_roles() -> None:
    config = Config(
        discord_token="token", chat_channel_id=123,
        llm_provider=LlmProviderKind.VENICE,
        venice_api_key="private-key",
        venice_text_model_id="same-model", venice_vision_model_id="same-model",
    )
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{
            "id": "same-model", "supportsVision": True,
            "supportsMultipleImages": True, "maxImages": 4,
        }]})

    result = _run(VeniceHealthProbe.from_config(
        config, transport=httpx.MockTransport(handler),
    ).probe())
    assert result.state is LmStudioHealthState.READY
    assert result.text_model_state is LmStudioHealthState.READY
    assert result.vision_model_state is LmStudioHealthState.READY
    assert len(requests) == 1


def test_venice_missing_role_makes_overall_health_unhealthy() -> None:
    service = RuntimeHealthService(
        _venice_config(),
        status_service=FakeStatusService(_snapshot()),  # type: ignore[arg-type]
        database_probe=FakeProbe(DatabaseHealthResult(DatabaseHealthState.READY, 6)),  # type: ignore[arg-type]
        provider_probe=FakeProbe(LmStudioHealthResult(
            LmStudioHealthState.MODEL_MISSING, 1, False,
            text_model_state=LmStudioHealthState.READY,
            vision_model_state=LmStudioHealthState.MODEL_MISSING,
        )),  # type: ignore[arg-type]
    )

    result = _run(service.check())
    assert result.overall is HealthOverallState.UNHEALTHY
    assert result.provider.text_model_state is LmStudioHealthState.READY
    assert result.provider.vision_model_state is LmStudioHealthState.MODEL_MISSING


@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        (401, {}, LmStudioHealthState.AUTH_ERROR),
        (500, {}, LmStudioHealthState.HTTP_ERROR),
        (200, {"models": []}, LmStudioHealthState.INVALID_RESPONSE),
        (200, {"data": [{"id": "other"}]}, LmStudioHealthState.MODEL_MISSING),
        (
            200,
            {
                "data": [
                    {"id": "venice-text-model", "supportsVision": False},
                    {
                        "id": "venice-vision-model",
                        "supportsVision": False,
                        "supportsMultipleImages": True,
                        "maxImages": 4,
                    }
                ]
            },
            LmStudioHealthState.INCOMPATIBLE_MODEL,
        ),
    ],
)
def test_venice_health_probe_classifies_safe_failures(
    status_code: int,
    payload: dict[str, object],
    expected: LmStudioHealthState,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if status_code != 200:
            return httpx.Response(status_code, text=PRIVATE_MARKER)
        return httpx.Response(status_code, json=payload)

    result = _run(
        VeniceHealthProbe.from_config(
            _venice_config(),
            transport=httpx.MockTransport(handler),
        ).probe()
    )

    assert result.state is expected
    assert "phase7a-private-venice-key" not in repr(result)
    assert PRIVATE_MARKER not in repr(result)


def test_runtime_health_selects_only_active_venice_probe() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "venice-text-model", "supportsVision": False},
                    {
                        "id": "venice-vision-model",
                        "supportsVision": True,
                        "supportsMultipleImages": True,
                        "maxImages": 4,
                        "offline": False,
                    }
                ]
            },
        )

    status = FakeStatusService(_snapshot())
    service = RuntimeHealthService(
        _venice_config(),
        status_service=status,  # type: ignore[arg-type]
        database_probe=FakeProbe(DatabaseHealthResult(DatabaseHealthState.READY, 6)),  # type: ignore[arg-type]
        provider_probe=VeniceHealthProbe.from_config(
            _venice_config(),
            transport=httpx.MockTransport(handler),
        ),
    )

    result = _run(service.check())
    rendered = _run(service.check_and_render())

    assert result.overall is HealthOverallState.HEALTHY
    assert result.provider_kind is LlmProviderKind.VENICE
    assert "LLM Provider: Venice" in rendered
    assert "文字模型:\nvenice-text-model\n狀態: READY" in rendered
    assert "圖片模型:\nvenice-vision-model\n狀態: READY" in rendered
    assert "Vision: OK" in rendered
    assert "Multiple images: OK" in rendered
    assert "phase7a-private-venice-key" not in rendered
    assert all("lm-studio" not in str(request.url) for request in requests)
    assert len(requests) == 2


class FakeResponse:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool, dict[str, object]]] = []
        self._done = False

    def is_done(self) -> bool:
        return self._done

    async def send_message(
        self,
        content: str,
        *,
        ephemeral: bool = False,
        **kwargs: object,
    ) -> None:
        self.messages.append((content, ephemeral, kwargs))
        self._done = True


class FakeFollowup:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool, dict[str, object]]] = []

    async def send(self, content: str, *, ephemeral: bool = False, **kwargs: object) -> None:
        self.messages.append((content, ephemeral, kwargs))


class FakeInteraction:
    def __init__(self, *, admin: bool = True, channel_id: int = 456, is_dm: bool = False) -> None:
        self.user = SimpleNamespace(
            id=100,
            guild_permissions=SimpleNamespace(administrator=admin, manage_guild=False),
        )
        self.guild = None if is_dm else SimpleNamespace(id=20)
        self.channel_id = channel_id
        self.response = FakeResponse()
        self.followup = FakeFollowup()


def _mentions_are_none(kwargs: dict[str, object]) -> bool:
    mentions = kwargs["allowed_mentions"]
    return (
        isinstance(mentions, discord.AllowedMentions)
        and mentions.everyone is False
        and mentions.users is False
        and mentions.roles is False
        and mentions.replied_user is False
    )


def test_runtime_commands_are_ephemeral_mentions_safe_and_authorized() -> None:
    status = FakeStatusService(_snapshot())
    health = SimpleNamespace(check_and_render=lambda: _health_text())
    commands = RuntimeCommands(
        status_channel_id=456,
        status_service=status,  # type: ignore[arg-type]
        health_service=health,  # type: ignore[arg-type]
    )
    interaction = FakeInteraction()

    _run(commands.handle_status(interaction))

    content, ephemeral, kwargs = interaction.response.messages[0]
    assert content == "local status"
    assert ephemeral is True
    assert _mentions_are_none(kwargs)
    assert status.render_calls == 1


async def _health_text() -> str:
    return "health status"


def test_runtime_commands_reject_non_admin_and_wrong_channel_before_services() -> None:
    async def run() -> None:
        status = FakeStatusService(_snapshot())
        health_calls = 0

        async def health() -> str:
            nonlocal health_calls
            health_calls += 1
            return "health"

        commands = RuntimeCommands(
            status_channel_id=456,
            status_service=status,  # type: ignore[arg-type]
            health_service=SimpleNamespace(check_and_render=health),  # type: ignore[arg-type]
        )
        unauthorized = FakeInteraction(admin=False)
        wrong_channel = FakeInteraction(channel_id=999)
        await commands.handle_status(unauthorized)
        await commands.handle_health(wrong_channel)

        assert status.render_calls == 0
        assert health_calls == 0
        assert unauthorized.response.messages[0][1] is True
        assert wrong_channel.response.messages[0][1] is True
        assert _mentions_are_none(unauthorized.response.messages[0][2])
        assert _mentions_are_none(wrong_channel.response.messages[0][2])

    _run(run())


def test_runtime_commands_health_success_is_ephemeral() -> None:
    async def run() -> None:
        async def health() -> str:
            return "health status"

        commands = RuntimeCommands(
            status_channel_id=456,
            status_service=FakeStatusService(_snapshot()),  # type: ignore[arg-type]
            health_service=SimpleNamespace(check_and_render=health),  # type: ignore[arg-type]
        )
        interaction = FakeInteraction()
        await commands.handle_health(interaction)
        assert interaction.response.messages[0][0] == "health status"
        assert interaction.response.messages[0][1] is True
        assert _mentions_are_none(interaction.response.messages[0][2])

    _run(run())


def test_status_and_health_do_not_mutate_conversation_state(tmp_path: Path) -> None:
    class Queue:
        def get_status(self) -> GenerationQueueStatus:
            return GenerationQueueStatus(True, False, 0)

    service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
    _run(service.initialize())
    status = RuntimeStatusService(
        _config(database_path=tmp_path / "conversation.sqlite3"),
        generation_queue=Queue(),  # type: ignore[arg-type]
        conversation_service=service,
    )
    health = RuntimeHealthService(
        _config(database_path=tmp_path / "conversation.sqlite3"),
        status_service=status,
        database_probe=FakeProbe(DatabaseHealthResult(DatabaseHealthState.READY, 5)),  # type: ignore[arg-type]
        lm_studio_probe=FakeProbe(LmStudioHealthResult(LmStudioHealthState.READY, 1, True)),  # type: ignore[arg-type]
    )
    commands = RuntimeCommands(
        status_channel_id=456,
        status_service=status,
        health_service=health,
    )
    before = service.get_runtime_status()

    _run(commands.handle_status(FakeInteraction()))
    _run(commands.handle_health(FakeInteraction()))

    assert service.get_runtime_status() == before


def test_runtime_command_registration_is_single_guild_lifecycle() -> None:
    async def run() -> None:
        from liuer_bot.discord_runtime import DiscordRuntimeClient

        class Responder:
            async def generate(self, request: ChatRequest) -> str:
                return "unused"

        client = DiscordRuntimeClient(
            config=_config(),
            responder=Responder(),
        )

        async def fetch_channel(channel_id: int) -> object:
            return SimpleNamespace(
                id=channel_id,
                type=discord.ChannelType.text,
                guild=SimpleNamespace(id=20),
            )

        sync_calls: list[int] = []

        async def sync(*, guild: object) -> None:
            sync_calls.append(int(getattr(guild, "id")))

        client.fetch_channel = fetch_channel  # type: ignore[method-assign]
        client.tree.sync = sync  # type: ignore[method-assign]
        try:
            await client.setup_hook()
            await client.setup_hook()
            guild = discord.Object(id=20)
            assert client.tree.get_command("status", guild=guild) is not None
            assert client.tree.get_command("health", guild=guild) is not None
            assert sync_calls == [20]
        finally:
            await client.close()

    _run(run())

from __future__ import annotations

from io import StringIO
from uuid import uuid4

import pytest

import liuer_bot.cli as cli
from liuer_bot.cli import main
from liuer_bot.database import DatabaseMigrationError
from liuer_bot.lm_studio_responder import LmStudioModelUnavailableError, LmStudioUnavailableError
from liuer_bot.prompting import PromptResourceError


class _NoopPersonaService:
    @classmethod
    def from_config(cls, config: object) -> _NoopPersonaService:
        return cls()

    async def initialize(self) -> None:
        return None

    def get_active_persona(self) -> None:
        return None


class _NoopConversationService:
    @classmethod
    def from_config(cls, config: object) -> _NoopConversationService:
        return cls()

    async def initialize(self) -> None:
        return None


def test_help_is_available() -> None:
    output = StringIO()
    with pytest.raises(SystemExit) as raised:
        main(["--help"], stdout=output)

    assert raised.value.code == 0
    assert "usage:" in output.getvalue()
    assert "doctor" in output.getvalue()
    assert "run" in output.getvalue()


def test_unknown_command_fails() -> None:
    error = StringIO()
    with pytest.raises(SystemExit) as raised:
        main(["unknown-command"], stderr=error)

    assert raised.value.code == 2
    assert "invalid choice" in error.getvalue()


def test_doctor_command_is_callable() -> None:
    output = StringIO()
    error = StringIO()
    result = main(
        ["doctor"],
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
        },
        stdout=output,
        stderr=error,
    )

    assert result == 0
    assert "Configuration:" in output.getvalue()


def test_run_invalid_config_returns_2() -> None:
    output = StringIO()
    error = StringIO()

    result = main(["run"], environ={}, stdout=output, stderr=error)

    assert result == 2
    assert "Configuration: invalid" in output.getvalue()
    assert "LIUER_DISCORD_TOKEN" in error.getvalue()


def test_run_valid_config_calls_runtime_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    events: list[str] = []

    class FakePromptBuilder:
        def __init__(self) -> None:
            events.append("prompt-builder-created")

    class FakePersonaService:
        @classmethod
        def from_config(cls, config: object) -> FakePersonaService:
            events.append("persona-service-created")
            return cls()

        async def initialize(self) -> None:
            events.append("persona-initialized")

        def get_active_persona(self) -> None:
            return None

    class FakeResponder:
        def __init__(self, config: object, **kwargs: object) -> None:
            self.config = config
            events.append("responder-created")

        async def preflight(self) -> None:
            events.append("preflight")

    def fake_run(config: object, *, responder: object, **kwargs: object) -> None:
        calls.append((config, responder, kwargs))
        events.append("discord-run")

    monkeypatch.setattr(cli, "create_llm_provider", FakeResponder)
    monkeypatch.setattr(cli, "PromptBuilder", FakePromptBuilder)
    monkeypatch.setattr(cli, "PersonaService", FakePersonaService)
    monkeypatch.setattr(cli, "ConversationService", _NoopConversationService)
    monkeypatch.setattr(cli, "run_discord", fake_run)
    result = main(
        ["run"],
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
        },
    )

    assert result == 0
    assert len(calls) == 1
    assert events == [
        "prompt-builder-created",
        "persona-service-created",
        "persona-initialized",
        "responder-created",
        "preflight",
        "discord-run",
    ]


def test_run_selects_venice_and_preflights_before_discord(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeVeniceResponder:
        def __init__(self, config: object, **kwargs: object) -> None:
            events.append("venice-created")

        async def preflight(self) -> None:
            events.append("venice-preflight")

    def fake_run(config: object, *, responder: object, **kwargs: object) -> None:
        events.append("discord-run")

    monkeypatch.setattr(cli, "create_llm_provider", FakeVeniceResponder)
    monkeypatch.setattr(cli, "PersonaService", _NoopPersonaService)
    monkeypatch.setattr(cli, "NicknameService", _NoopPersonaService)
    monkeypatch.setattr(cli, "ConversationService", _NoopConversationService)
    monkeypatch.setattr(cli, "run_discord", fake_run)

    result = main(
        ["run"],
        environ={
            "LIUER_DISCORD_TOKEN": "discord-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
            "LIUER_LLM_PROVIDER": "venice",
            "LIUER_VENICE_API_KEY": "private-key",
            "LIUER_VENICE_TEXT_MODEL_ID": "exact-text-model",
            "LIUER_VENICE_VISION_MODEL_ID": "exact-vision-model",
        },
    )

    assert result == 0
    assert events == ["venice-created", "venice-preflight", "discord-run"]


def test_run_prompt_resource_failure_does_not_preflight_or_start_discord(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def unavailable_builder() -> object:
        raise PromptResourceError()

    def fake_run(config: object, *, responder: object) -> None:
        calls.append((config, responder))

    monkeypatch.setattr(cli, "PromptBuilder", unavailable_builder)
    monkeypatch.setattr(cli, "run_discord", fake_run)
    result = main(
        ["run"],
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
        },
    )

    assert result == 1
    assert calls == []


def test_run_database_failure_does_not_start_discord_or_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class FailingPersonaService:
        @classmethod
        def from_config(cls, config: object) -> FailingPersonaService:
            return cls()

        async def initialize(self) -> None:
            raise DatabaseMigrationError()

    class UnexpectedResponder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("responder-created")

    def fake_run(config: object, *, responder: object) -> None:
        calls.append((config, responder))

    monkeypatch.setattr(cli, "PersonaService", FailingPersonaService)
    monkeypatch.setattr(cli, "create_llm_provider", UnexpectedResponder)
    monkeypatch.setattr(cli, "run_discord", fake_run)
    result = main(
        ["run"],
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
        },
    )

    assert result == 1
    assert calls == []


def test_run_runtime_failure_returns_1_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = f"phase1a-{uuid4().hex}"
    output = StringIO()
    error = StringIO()

    def failing_run(config: object, *, responder: object) -> None:
        raise RuntimeError(secret)

    class FakeResponder:
        def __init__(self, config: object, **kwargs: object) -> None:
            self.config = config

        async def preflight(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_llm_provider", FakeResponder)
    monkeypatch.setattr(cli, "PersonaService", _NoopPersonaService)
    monkeypatch.setattr(cli, "ConversationService", _NoopConversationService)
    monkeypatch.setattr(cli, "run_discord", failing_run)
    result = main(
        ["run"],
        environ={
            "LIUER_DISCORD_TOKEN": secret,
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
        },
        stdout=output,
        stderr=error,
    )

    assert result == 1
    assert secret not in output.getvalue()
    assert secret not in error.getvalue()


@pytest.mark.parametrize("error_type", [LmStudioUnavailableError, LmStudioModelUnavailableError])
def test_run_preflight_failure_returns_1_without_starting_discord(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    secret = f"phase2a-{uuid4().hex}"
    output = StringIO()
    error = StringIO()
    run_calls: list[object] = []

    class FailingPreflightResponder:
        def __init__(self, config: object, **kwargs: object) -> None:
            self.config = config

        async def preflight(self) -> None:
            raise error_type()

    def fake_run(config: object, *, responder: object) -> None:
        run_calls.append((config, responder))

    monkeypatch.setattr(cli, "create_llm_provider", FailingPreflightResponder)
    monkeypatch.setattr(cli, "PersonaService", _NoopPersonaService)
    monkeypatch.setattr(cli, "ConversationService", _NoopConversationService)
    monkeypatch.setattr(cli, "run_discord", fake_run)
    result = main(
        ["run"],
        environ={
            "LIUER_DISCORD_TOKEN": secret,
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
            "LIUER_LM_STUDIO_API_TOKEN": secret,
        },
        stdout=output,
        stderr=error,
    )

    assert result == 1
    assert run_calls == []
    assert secret not in output.getvalue()
    assert secret not in error.getvalue()

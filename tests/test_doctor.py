from __future__ import annotations

import socket
from io import StringIO
from pathlib import Path
from uuid import uuid4

from liuer_bot.doctor import run_doctor


def test_missing_required_config_returns_2() -> None:
    output = StringIO()
    error = StringIO()

    result = run_doctor(environ={}, stdout=output, stderr=error)

    assert result == 2
    assert "Configuration: invalid" in output.getvalue()


def test_valid_config_returns_0() -> None:
    output = StringIO()
    error = StringIO()

    result = run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "789",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
        },
        stdout=output,
        stderr=error,
    )

    assert result == 0
    assert "Configuration: valid" in output.getvalue()
    assert "Chat channel: valid" in output.getvalue()
    assert "Persona status channel: valid" in output.getvalue()
    assert "LM Studio model: configured" in output.getvalue()
    assert "LM Studio API token: not configured" in output.getvalue()
    assert "Response chat-short max tokens: valid" in output.getvalue()
    assert "Response normal max tokens: valid" in output.getvalue()
    assert "Context history max chars chat-short: valid" in output.getvalue()
    assert "Context history max chars normal: valid" in output.getvalue()
    assert "Context history max chars deep: valid" in output.getvalue()
    assert "Image download timeout: valid" in output.getvalue()
    assert "Image limits: valid" in output.getvalue()
    assert "Model image limits: valid" in output.getvalue()
    assert "Persona max chars: valid" in output.getvalue()
    assert "Persona global cooldown: valid" in output.getvalue()
    assert "Persona daily limit: valid" in output.getvalue()
    assert "Persona daily reset offset: valid" in output.getvalue()
    assert "Nickname max chars: valid" in output.getvalue()


def test_venice_doctor_is_offline_and_redacts_api_key(monkeypatch) -> None:
    secret = "phase7a-private-venice-key"
    output = StringIO()
    error = StringIO()

    def forbidden_network(*args, **kwargs):
        raise AssertionError("doctor must remain offline")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    result = run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": "discord-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "789",
            "LIUER_LLM_PROVIDER": "venice",
            "LIUER_VENICE_API_KEY": secret,
            "LIUER_VENICE_TEXT_MODEL_ID": "exact-text-model",
            "LIUER_VENICE_VISION_MODEL_ID": "exact-vision-model",
        },
        stdout=output,
        stderr=error,
    )

    combined = output.getvalue() + error.getvalue()
    assert result == 0
    assert "LLM provider: venice" in combined
    assert "Venice text model: configured" in combined
    assert "Venice vision model: configured" in combined
    assert "Venice API key: configured" in combined
    assert secret not in combined


def test_doctor_does_not_create_database(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "liuer.sqlite3"
    result = run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "789",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
            "LIUER_DATABASE_PATH": str(database_path),
        },
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert result == 0
    assert not database_path.exists()


def test_invalid_config_returns_2() -> None:
    output = StringIO()
    error = StringIO()

    result = run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "789",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
            "LIUER_LM_STUDIO_BASE_URL": "not-a-url",
        },
        stdout=output,
        stderr=error,
    )

    assert result == 2
    assert "Configuration: invalid" in output.getvalue()


def test_invalid_model_image_config_returns_2() -> None:
    output = StringIO()
    error = StringIO()

    result = run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "789",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
            "LIUER_MODEL_IMAGE_MAX_PIXELS": "0",
        },
        stdout=output,
        stderr=error,
    )

    assert result == 2
    assert "LIUER_MODEL_IMAGE_MAX_PIXELS" in error.getvalue()


def test_invalid_nickname_max_chars_returns_2() -> None:
    output = StringIO()
    error = StringIO()

    result = run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "789",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
            "LIUER_NICKNAME_MAX_CHARS": "65",
        },
        stdout=output,
        stderr=error,
    )

    assert result == 2
    assert "LIUER_NICKNAME_MAX_CHARS" in error.getvalue()


def test_secret_is_not_written_to_doctor_output() -> None:
    secret = f"phase0-{uuid4().hex}"
    output = StringIO()
    error = StringIO()

    run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": secret,
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "789",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
            "LIUER_LM_STUDIO_API_TOKEN": secret,
        },
        stdout=output,
        stderr=error,
    )

    assert secret not in output.getvalue()
    assert secret not in error.getvalue()


def test_missing_lm_studio_model_returns_2() -> None:
    output = StringIO()
    error = StringIO()

    result = run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
        },
        stdout=output,
        stderr=error,
    )

    assert result == 2
    assert "LIUER_LM_STUDIO_MODEL" in error.getvalue()


def test_doctor_remains_offline(monkeypatch) -> None:
    def network_access(*args, **kwargs) -> None:
        raise AssertionError("doctor must not access the network")

    monkeypatch.setattr(socket, "create_connection", network_access)
    result = run_doctor(
        environ={
            "LIUER_DISCORD_TOKEN": "test-token",
            "LIUER_CHAT_CHANNEL_ID": "123",
            "LIUER_PERSONA_STATUS_CHANNEL_ID": "789",
            "LIUER_LM_STUDIO_MODEL": "configured-model",
        },
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert result == 0

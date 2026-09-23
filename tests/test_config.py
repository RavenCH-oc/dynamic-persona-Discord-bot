from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from liuer_bot.config import (
    DEFAULT_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT,
    DEFAULT_CONTEXT_HISTORY_MAX_CHARS_DEEP,
    DEFAULT_CONTEXT_HISTORY_MAX_CHARS_NORMAL,
    DEFAULT_DATABASE_PATH,
    DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_LM_STUDIO_MAX_TOKENS,
    DEFAULT_LM_STUDIO_TEMPERATURE,
    DEFAULT_LM_STUDIO_TIMEOUT_SECONDS,
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_IMAGES_PER_MESSAGE,
    DEFAULT_MAX_TOTAL_IMAGE_BYTES,
    DEFAULT_MODEL_IMAGE_MAX_LONG_EDGE,
    DEFAULT_MODEL_IMAGE_MAX_PIXELS,
    DEFAULT_MODEL_TOTAL_IMAGE_PIXELS,
    DEFAULT_NICKNAME_MAX_CHARS,
    DEFAULT_PERSONA_DAILY_LIMIT,
    DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS,
    DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS,
    DEFAULT_PERSONA_MAX_CHARS,
    DEFAULT_RESPONSE_CHAT_SHORT_MAX_TOKENS,
    DEFAULT_RESPONSE_NORMAL_MAX_TOKENS,
    DEFAULT_VENICE_BASE_URL,
    MAX_CONTEXT_HISTORY_MAX_CHARS,
    MAX_NICKNAME_MAX_CHARS,
    Config,
    ConfigError,
    load_config,
)
from liuer_bot.llm_provider import LlmProviderKind


def _env(**overrides: str) -> dict[str, str]:
    values = {
        "LIUER_DISCORD_TOKEN": "test-token",
        "LIUER_CHAT_CHANNEL_ID": "123",
        "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
        "LIUER_LM_STUDIO_MODEL": "configured-model",
    }
    values.update(overrides)
    return values


def test_required_token_missing() -> None:
    with pytest.raises(ConfigError, match="LIUER_DISCORD_TOKEN is required"):
        load_config(environ={})


def test_valid_token_is_loaded() -> None:
    config = load_config(environ=_env())

    assert config.discord_token == "test-token"
    assert config.persona_status_channel_id == 456


def test_missing_persona_status_channel_is_invalid() -> None:
    values = _env()
    values.pop("LIUER_PERSONA_STATUS_CHANNEL_ID")

    with pytest.raises(ConfigError, match="LIUER_PERSONA_STATUS_CHANNEL_ID is required"):
        load_config(environ=values)


def test_missing_chat_channel_is_invalid() -> None:
    values = _env()
    values.pop("LIUER_CHAT_CHANNEL_ID")

    with pytest.raises(ConfigError, match="LIUER_CHAT_CHANNEL_ID is required"):
        load_config(environ=values)


def test_chat_and_persona_status_channels_must_differ() -> None:
    with pytest.raises(ConfigError, match="must differ"):
        load_config(environ=_env(LIUER_PERSONA_STATUS_CHANNEL_ID="123"))


@pytest.mark.parametrize("value", ["0", "-1", "not-a-snowflake"])
def test_invalid_persona_status_channel_is_invalid(value: str) -> None:
    with pytest.raises(ConfigError, match="LIUER_PERSONA_STATUS_CHANNEL_ID"):
        load_config(environ=_env(LIUER_PERSONA_STATUS_CHANNEL_ID=value))


def test_default_lm_studio_url() -> None:
    config = load_config(environ=_env())

    assert config.lm_studio_base_url == DEFAULT_LM_STUDIO_BASE_URL
    assert config.llm_provider is LlmProviderKind.LM_STUDIO


def test_venice_active_provider_requires_only_venice_identity_fields() -> None:
    values = _env(
        LIUER_LLM_PROVIDER="venice",
        LIUER_VENICE_API_KEY="private-venice-key",
        LIUER_VENICE_TEXT_MODEL_ID="exact-text-model",
        LIUER_VENICE_VISION_MODEL_ID="exact-vision-model",
        LIUER_LM_STUDIO_MODEL="",
        LIUER_LM_STUDIO_BASE_URL="inactive-invalid-url",
    )

    config = load_config(environ=values)

    assert config.llm_provider is LlmProviderKind.VENICE
    assert config.venice_base_url == DEFAULT_VENICE_BASE_URL
    assert config.venice_api_key == "private-venice-key"
    assert config.venice_text_model_id == "exact-text-model"
    assert config.venice_vision_model_id == "exact-vision-model"


def test_venice_text_and_vision_model_ids_may_match() -> None:
    config = load_config(environ=_env(
        LIUER_LLM_PROVIDER="venice",
        LIUER_VENICE_API_KEY="private-key",
        LIUER_VENICE_TEXT_MODEL_ID="same-model",
        LIUER_VENICE_VISION_MODEL_ID="same-model",
    ))

    assert config.venice_text_model_id == config.venice_vision_model_id == "same-model"


@pytest.mark.parametrize(
    "missing",
    ["LIUER_VENICE_API_KEY", "LIUER_VENICE_TEXT_MODEL_ID", "LIUER_VENICE_VISION_MODEL_ID"],
)
def test_venice_active_provider_requires_key_and_both_exact_models(missing: str) -> None:
    values = _env(
        LIUER_LLM_PROVIDER="venice",
        LIUER_VENICE_API_KEY="private-key",
        LIUER_VENICE_TEXT_MODEL_ID="exact-text-model",
        LIUER_VENICE_VISION_MODEL_ID="exact-vision-model",
    )
    values.pop(missing)

    with pytest.raises(ConfigError, match=f"{missing} is required"):
        load_config(environ=values)


def test_venice_active_provider_requires_valid_http_base_url() -> None:
    with pytest.raises(ConfigError, match="LIUER_VENICE_BASE_URL"):
        load_config(
            environ=_env(
                LIUER_LLM_PROVIDER="venice",
                LIUER_VENICE_API_KEY="private-key",
                LIUER_VENICE_TEXT_MODEL_ID="exact-text-model",
                LIUER_VENICE_VISION_MODEL_ID="exact-vision-model",
                LIUER_VENICE_BASE_URL="not-a-url",
            )
        )


def test_lm_studio_does_not_require_venice_fields() -> None:
    config = load_config(environ=_env(
        LIUER_VENICE_API_KEY="",
        LIUER_VENICE_TEXT_MODEL_ID="",
        LIUER_VENICE_VISION_MODEL_ID="",
    ))

    assert config.llm_provider is LlmProviderKind.LM_STUDIO
    assert config.venice_api_key is None
    assert config.venice_text_model_id == ""
    assert config.venice_vision_model_id == ""


def test_invalid_llm_provider_fails_closed() -> None:
    with pytest.raises(ConfigError, match="LIUER_LLM_PROVIDER"):
        load_config(environ=_env(LIUER_LLM_PROVIDER="unknown"))


def test_config_repr_redacts_venice_api_key() -> None:
    secret = "phase7a-private-venice-key"
    config = load_config(
        environ=_env(
            LIUER_LLM_PROVIDER="venice",
            LIUER_VENICE_API_KEY=secret,
            LIUER_VENICE_TEXT_MODEL_ID="exact-text-model",
            LIUER_VENICE_VISION_MODEL_ID="exact-vision-model",
        )
    )

    assert secret not in repr(config)


def test_custom_lm_studio_url() -> None:
    config = load_config(
        environ={
            **_env(),
            "LIUER_LM_STUDIO_BASE_URL": "https://localhost:4321/custom",
        }
    )

    assert config.lm_studio_base_url == "https://localhost:4321/custom"


def test_lm_studio_base_url_trailing_slash_is_normalized() -> None:
    config = load_config(environ=_env(LIUER_LM_STUDIO_BASE_URL="http://localhost:1234/v1/"))

    assert config.lm_studio_base_url == "http://localhost:1234/v1"


def test_lm_studio_model_is_required() -> None:
    with pytest.raises(ConfigError, match="LIUER_LM_STUDIO_MODEL is required"):
        load_config(
            environ={
                "LIUER_DISCORD_TOKEN": "test-token",
                "LIUER_CHAT_CHANNEL_ID": "123",
                "LIUER_PERSONA_STATUS_CHANNEL_ID": "456",
            }
        )


def test_lm_studio_model_is_loaded() -> None:
    config = load_config(environ=_env(LIUER_LM_STUDIO_MODEL="my-local-model"))

    assert config.lm_studio_model == "my-local-model"


def test_lm_studio_generation_defaults() -> None:
    config = load_config(environ=_env())

    assert config.lm_studio_timeout_seconds == DEFAULT_LM_STUDIO_TIMEOUT_SECONDS
    assert config.lm_studio_temperature == DEFAULT_LM_STUDIO_TEMPERATURE
    assert config.lm_studio_max_tokens == DEFAULT_LM_STUDIO_MAX_TOKENS
    assert DEFAULT_LM_STUDIO_MAX_TOKENS == 4096
    assert config.response_chat_short_max_tokens == DEFAULT_RESPONSE_CHAT_SHORT_MAX_TOKENS == 1024
    assert config.response_normal_max_tokens == DEFAULT_RESPONSE_NORMAL_MAX_TOKENS == 1536
    assert (
        1
        <= config.response_chat_short_max_tokens
        <= config.response_normal_max_tokens
        <= config.lm_studio_max_tokens
        == 4096
    )
    assert config.lm_studio_api_token is None


def test_custom_response_token_budgets_are_loaded() -> None:
    config = load_config(
        environ=_env(
            LIUER_RESPONSE_CHAT_SHORT_MAX_TOKENS="256",
            LIUER_RESPONSE_NORMAL_MAX_TOKENS="768",
            LIUER_LM_STUDIO_MAX_TOKENS="1024",
        )
    )

    assert config.response_chat_short_max_tokens == 256
    assert config.response_normal_max_tokens == 768


def test_config_schema_has_no_persona_prompt_or_address_alias_fields() -> None:
    field_names = set(Config.__dataclass_fields__)

    assert {
        "persona",
        "system_prompt",
        "prompt_file",
        "nickname",
        "active_nickname",
        "call_name",
        "call_names",
        "alias",
        "aliases",
    }.isdisjoint(field_names)


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("image_download_timeout_seconds", DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS),
        ("max_images_per_message", DEFAULT_MAX_IMAGES_PER_MESSAGE),
        ("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES),
        ("max_total_image_bytes", DEFAULT_MAX_TOTAL_IMAGE_BYTES),
        ("max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS),
    ],
)
def test_image_acquisition_defaults(attribute: str, expected: int | float) -> None:
    config = load_config(environ=_env())

    assert getattr(config, attribute) == expected


def test_custom_image_acquisition_values_are_loaded() -> None:
    config = load_config(
        environ=_env(
            LIUER_IMAGE_DOWNLOAD_TIMEOUT_SECONDS="12.5",
            LIUER_MAX_IMAGES_PER_MESSAGE="2",
            LIUER_MAX_IMAGE_BYTES="1024",
            LIUER_MAX_TOTAL_IMAGE_BYTES="512",
            LIUER_MAX_IMAGE_PIXELS="100",
        )
    )

    assert config.image_download_timeout_seconds == 12.5
    assert config.max_images_per_message == 2
    assert config.max_image_bytes == 1024
    assert config.max_total_image_bytes == 512
    assert config.max_image_pixels == 100


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LIUER_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "0"),
        ("LIUER_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "-1"),
        ("LIUER_IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "invalid"),
        ("LIUER_MAX_IMAGES_PER_MESSAGE", "0"),
        ("LIUER_MAX_IMAGES_PER_MESSAGE", "-1"),
        ("LIUER_MAX_IMAGES_PER_MESSAGE", "12.5"),
        ("LIUER_MAX_IMAGES_PER_MESSAGE", "invalid"),
        ("LIUER_MAX_IMAGE_BYTES", "0"),
        ("LIUER_MAX_IMAGE_BYTES", "-1"),
        ("LIUER_MAX_IMAGE_BYTES", "12.5"),
        ("LIUER_MAX_IMAGE_BYTES", "invalid"),
        ("LIUER_MAX_TOTAL_IMAGE_BYTES", "0"),
        ("LIUER_MAX_TOTAL_IMAGE_BYTES", "-1"),
        ("LIUER_MAX_TOTAL_IMAGE_BYTES", "12.5"),
        ("LIUER_MAX_TOTAL_IMAGE_BYTES", "invalid"),
        ("LIUER_MAX_IMAGE_PIXELS", "0"),
        ("LIUER_MAX_IMAGE_PIXELS", "-1"),
        ("LIUER_MAX_IMAGE_PIXELS", "12.5"),
        ("LIUER_MAX_IMAGE_PIXELS", "invalid"),
    ],
)
def test_invalid_image_acquisition_values_fail_closed(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        load_config(environ=_env(**{name: value}))


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("model_image_max_long_edge", DEFAULT_MODEL_IMAGE_MAX_LONG_EDGE),
        ("model_image_max_pixels", DEFAULT_MODEL_IMAGE_MAX_PIXELS),
        ("model_total_image_pixels", DEFAULT_MODEL_TOTAL_IMAGE_PIXELS),
    ],
)
def test_model_image_normalization_defaults(attribute: str, expected: int) -> None:
    config = load_config(environ=_env())

    assert getattr(config, attribute) == expected


def test_custom_model_image_normalization_values_are_loaded() -> None:
    config = load_config(
        environ=_env(
            LIUER_MODEL_IMAGE_MAX_LONG_EDGE="1024",
            LIUER_MODEL_IMAGE_MAX_PIXELS="1000000",
            LIUER_MODEL_TOTAL_IMAGE_PIXELS="2000000",
        )
    )

    assert config.model_image_max_long_edge == 1024
    assert config.model_image_max_pixels == 1_000_000
    assert config.model_total_image_pixels == 2_000_000


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LIUER_MODEL_IMAGE_MAX_LONG_EDGE", "0"),
        ("LIUER_MODEL_IMAGE_MAX_LONG_EDGE", "-1"),
        ("LIUER_MODEL_IMAGE_MAX_LONG_EDGE", "12.5"),
        ("LIUER_MODEL_IMAGE_MAX_LONG_EDGE", "invalid"),
        ("LIUER_MODEL_IMAGE_MAX_PIXELS", "0"),
        ("LIUER_MODEL_IMAGE_MAX_PIXELS", "-1"),
        ("LIUER_MODEL_IMAGE_MAX_PIXELS", "12.5"),
        ("LIUER_MODEL_IMAGE_MAX_PIXELS", "invalid"),
        ("LIUER_MODEL_TOTAL_IMAGE_PIXELS", "0"),
        ("LIUER_MODEL_TOTAL_IMAGE_PIXELS", "-1"),
        ("LIUER_MODEL_TOTAL_IMAGE_PIXELS", "12.5"),
        ("LIUER_MODEL_TOTAL_IMAGE_PIXELS", "invalid"),
    ],
)
def test_invalid_model_image_normalization_values_fail_closed(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        load_config(environ=_env(**{name: value}))


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_lm_studio_timeout(value: str) -> None:
    with pytest.raises(ConfigError, match="LIUER_LM_STUDIO_TIMEOUT_SECONDS"):
        load_config(environ=_env(LIUER_LM_STUDIO_TIMEOUT_SECONDS=value))


def test_custom_lm_studio_timeout_is_loaded() -> None:
    config = load_config(environ=_env(LIUER_LM_STUDIO_TIMEOUT_SECONDS="12.5"))

    assert config.lm_studio_timeout_seconds == 12.5


@pytest.mark.parametrize("value", ["-0.1", "2.1", "invalid"])
def test_invalid_lm_studio_temperature(value: str) -> None:
    with pytest.raises(ConfigError, match="LIUER_LM_STUDIO_TEMPERATURE"):
        load_config(environ=_env(LIUER_LM_STUDIO_TEMPERATURE=value))


@pytest.mark.parametrize("value", ["0", "0.7", "2"])
def test_valid_lm_studio_temperature(value: str) -> None:
    config = load_config(environ=_env(LIUER_LM_STUDIO_TEMPERATURE=value))

    assert config.lm_studio_temperature == float(value)


@pytest.mark.parametrize("value", ["0", "-1", "12.5", "invalid"])
def test_invalid_lm_studio_max_tokens(value: str) -> None:
    with pytest.raises(ConfigError, match="LIUER_LM_STUDIO_MAX_TOKENS"):
        load_config(environ=_env(LIUER_LM_STUDIO_MAX_TOKENS=value))


def test_custom_lm_studio_max_tokens_is_loaded() -> None:
    config = load_config(
        environ=_env(
            LIUER_LM_STUDIO_MAX_TOKENS="512",
            LIUER_RESPONSE_CHAT_SHORT_MAX_TOKENS="128",
            LIUER_RESPONSE_NORMAL_MAX_TOKENS="256",
        )
    )

    assert config.lm_studio_max_tokens == 512


@pytest.mark.parametrize(
    "overrides",
    [
        {"LIUER_RESPONSE_CHAT_SHORT_MAX_TOKENS": "0"},
        {"LIUER_RESPONSE_NORMAL_MAX_TOKENS": "invalid"},
        {
            "LIUER_RESPONSE_CHAT_SHORT_MAX_TOKENS": "1536",
            "LIUER_RESPONSE_NORMAL_MAX_TOKENS": "1024",
        },
        {"LIUER_RESPONSE_NORMAL_MAX_TOKENS": "4097"},
        {"LIUER_LM_STUDIO_MAX_TOKENS": "1024"},
    ],
)
def test_response_token_budgets_fail_closed(overrides: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match="response token budgets|LIUER_RESPONSE|LIUER_LM_STUDIO"):
        load_config(environ=_env(**overrides))


def test_lm_studio_api_token_is_redacted() -> None:
    secret = f"phase2a-{uuid4().hex}"
    config = load_config(environ=_env(LIUER_LM_STUDIO_API_TOKEN=secret))

    assert config.lm_studio_api_token == secret
    assert secret not in repr(config)


@pytest.mark.parametrize("url", ["localhost:1234/v1", "ftp://localhost/v1", "http://"])
def test_invalid_lm_studio_url(url: str) -> None:
    with pytest.raises(ConfigError, match="LIUER_LM_STUDIO_BASE_URL"):
        load_config(environ=_env(LIUER_LM_STUDIO_BASE_URL=url))


def test_default_database_path() -> None:
    config = load_config(environ=_env())

    assert config.database_path == DEFAULT_DATABASE_PATH


def test_persona_policy_defaults() -> None:
    config = load_config(environ=_env())

    assert config.persona_max_chars == DEFAULT_PERSONA_MAX_CHARS == 4000
    assert (
        config.persona_global_cooldown_seconds
        == DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS
        == 1200
    )
    assert config.persona_daily_limit == DEFAULT_PERSONA_DAILY_LIMIT == 3
    assert (
        config.persona_daily_reset_utc_offset_hours
        == DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS
        == 8
    )


def test_nickname_max_length_default_and_custom_value() -> None:
    assert load_config(environ=_env()).nickname_max_chars == DEFAULT_NICKNAME_MAX_CHARS == 32
    assert load_config(environ=_env(LIUER_NICKNAME_MAX_CHARS="64")).nickname_max_chars == 64


@pytest.mark.parametrize("value", ["", "0", "-1", "65", "1.5", "invalid"])
def test_invalid_nickname_max_length_fails_closed(value: str) -> None:
    with pytest.raises(ConfigError, match="LIUER_NICKNAME_MAX_CHARS"):
        load_config(environ=_env(LIUER_NICKNAME_MAX_CHARS=value))


def test_nickname_max_length_hard_limit_is_explicit() -> None:
    assert MAX_NICKNAME_MAX_CHARS == 64


def test_custom_persona_policy_values_are_loaded() -> None:
    config = load_config(
        environ=_env(
            LIUER_PERSONA_MAX_CHARS="123",
            LIUER_PERSONA_GLOBAL_COOLDOWN_SECONDS="45",
            LIUER_PERSONA_DAILY_LIMIT="7",
            LIUER_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS="-5",
        )
    )

    assert config.persona_max_chars == 123
    assert config.persona_global_cooldown_seconds == 45
    assert config.persona_daily_limit == 7
    assert config.persona_daily_reset_utc_offset_hours == -5


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LIUER_PERSONA_MAX_CHARS", "0"),
        ("LIUER_PERSONA_MAX_CHARS", "-1"),
        ("LIUER_PERSONA_MAX_CHARS", "invalid"),
        ("LIUER_PERSONA_GLOBAL_COOLDOWN_SECONDS", "0"),
        ("LIUER_PERSONA_GLOBAL_COOLDOWN_SECONDS", "-1"),
        ("LIUER_PERSONA_GLOBAL_COOLDOWN_SECONDS", "invalid"),
        ("LIUER_PERSONA_DAILY_LIMIT", "0"),
        ("LIUER_PERSONA_DAILY_LIMIT", "-1"),
        ("LIUER_PERSONA_DAILY_LIMIT", "invalid"),
        ("LIUER_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS", "-13"),
        ("LIUER_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS", "15"),
        ("LIUER_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS", "1.5"),
        ("LIUER_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS", "invalid"),
    ],
)
def test_invalid_persona_policy_values_fail_closed(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        load_config(environ=_env(**{name: value}))


def test_custom_database_path() -> None:
    config = load_config(
        environ={
            **_env(),
            "LIUER_DATABASE_PATH": "tmp/custom.sqlite3",
        }
    )

    assert config.database_path == Path("tmp/custom.sqlite3")


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "warning"])
def test_valid_log_levels(level: str) -> None:
    config = load_config(environ=_env(LIUER_LOG_LEVEL=level))

    assert config.log_level == level.upper()


def test_invalid_log_level() -> None:
    with pytest.raises(ConfigError, match="LIUER_LOG_LEVEL"):
        load_config(environ=_env(LIUER_LOG_LEVEL="TRACE"))


def test_config_repr_does_not_include_secret() -> None:
    secret = f"phase0-{uuid4().hex}"
    config = load_config(environ=_env(LIUER_DISCORD_TOKEN=secret))

    assert secret not in repr(config)
    assert "configured" in repr(config)


def test_dotenv_is_supported_and_environment_wins(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "LIUER_DISCORD_TOKEN=dotenv-token\n"
        "LIUER_CHAT_CHANNEL_ID=123\n"
        "LIUER_PERSONA_STATUS_CHANNEL_ID=456\n"
        "LIUER_LM_STUDIO_MODEL=configured-model\n"
        "LIUER_LOG_LEVEL=WARNING\n",
        encoding="utf-8",
    )

    config = load_config(
        environ={"LIUER_DISCORD_TOKEN": "environment-token"},
        dotenv_path=dotenv,
    )

    assert config.discord_token == "environment-token"
    assert config.log_level == "WARNING"


def test_one_chat_channel_id() -> None:
    config = load_config(environ=_env(LIUER_CHAT_CHANNEL_ID="123"))

    assert config.chat_channel_id == 123


def test_chat_channel_id_is_not_plural_or_comma_separated() -> None:
    with pytest.raises(ConfigError, match="LIUER_CHAT_CHANNEL_ID"):
        load_config(environ=_env(LIUER_CHAT_CHANNEL_ID="123,456"))



@pytest.mark.parametrize("value", ["", "   ", "123,456", "abc", "0", "-1"])
def test_invalid_chat_channel_id(value: str) -> None:
    with pytest.raises(ConfigError, match="LIUER_CHAT_CHANNEL_ID"):
        load_config(environ=_env(LIUER_CHAT_CHANNEL_ID=value))


def test_context_recent_turns_default_and_custom_value() -> None:
    assert load_config(environ=_env()).context_recent_turns == 12
    assert load_config(environ=_env(LIUER_CONTEXT_RECENT_TURNS="25")).context_recent_turns == 25


def test_context_history_character_budgets_default_and_independent() -> None:
    config = load_config(environ=_env())

    assert (
        config.context_history_max_chars_chat_short
        == DEFAULT_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT
        == 1000
    )
    assert (
        config.context_history_max_chars_normal == DEFAULT_CONTEXT_HISTORY_MAX_CHARS_NORMAL == 12000
    )
    assert (
        config.context_history_max_chars_deep == DEFAULT_CONTEXT_HISTORY_MAX_CHARS_DEEP == 24000
    )

    unusual = load_config(
        environ=_env(
            LIUER_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT="0",
            LIUER_CONTEXT_HISTORY_MAX_CHARS_NORMAL="3",
            LIUER_CONTEXT_HISTORY_MAX_CHARS_DEEP="1",
        )
    )
    assert unusual.context_history_max_chars_chat_short == 0
    assert unusual.context_history_max_chars_normal == 3
    assert unusual.context_history_max_chars_deep == 1


@pytest.mark.parametrize(
    "name",
    [
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT",
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_NORMAL",
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_DEEP",
    ],
)
@pytest.mark.parametrize("value", ["-1", str(MAX_CONTEXT_HISTORY_MAX_CHARS + 1), "invalid"])
def test_invalid_context_history_character_budget(name: str, value: str) -> None:
    with pytest.raises(ConfigError, match=name):
        load_config(environ=_env(**{name: value}))


@pytest.mark.parametrize("value", ["", "0", "-1", "51", "invalid"])
def test_invalid_context_recent_turns(value: str) -> None:
    with pytest.raises(ConfigError, match="LIUER_CONTEXT_RECENT_TURNS"):
        load_config(environ=_env(LIUER_CONTEXT_RECENT_TURNS=value))

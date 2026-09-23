"""Loading and validating Discord and active LLM-provider configuration."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

from .llm_provider import LlmProviderKind

DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_VENICE_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_LM_STUDIO_TIMEOUT_SECONDS = 300.0
DEFAULT_LM_STUDIO_TEMPERATURE = 0.7
DEFAULT_LM_STUDIO_MAX_TOKENS = 4096
DEFAULT_RESPONSE_CHAT_SHORT_MAX_TOKENS = 1024
DEFAULT_RESPONSE_NORMAL_MAX_TOKENS = 1536
DEFAULT_CONTEXT_RECENT_TURNS = 12
MAX_CONTEXT_RECENT_TURNS = 50
DEFAULT_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT = 1_000
DEFAULT_CONTEXT_HISTORY_MAX_CHARS_NORMAL = 12_000
DEFAULT_CONTEXT_HISTORY_MAX_CHARS_DEEP = 24_000
MAX_CONTEXT_HISTORY_MAX_CHARS = 200_000
DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_IMAGES_PER_MESSAGE = 4
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_MODEL_IMAGE_MAX_LONG_EDGE = 2048
DEFAULT_MODEL_IMAGE_MAX_PIXELS = 4_000_000
DEFAULT_MODEL_TOTAL_IMAGE_PIXELS = 8_000_000
DEFAULT_DATABASE_PATH = Path("data/liuer.sqlite3")
DEFAULT_PERSONA_MAX_CHARS = 4_000
DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS = 1_200
DEFAULT_PERSONA_DAILY_LIMIT = 3
DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS = 8
DEFAULT_NICKNAME_MAX_CHARS = 32
MAX_NICKNAME_MAX_CHARS = 64
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_CHANNEL_ID = re.compile(r"^[0-9]+$")
_SIGNED_INTEGER = re.compile(r"^-?[0-9]+$")

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Config:
    """Validated configuration, with the Discord token excluded from repr."""

    discord_token: str | None = field(repr=False)
    chat_channel_id: int
    llm_provider: LlmProviderKind = LlmProviderKind.LM_STUDIO
    lm_studio_base_url: str = DEFAULT_LM_STUDIO_BASE_URL
    lm_studio_model: str = ""
    lm_studio_timeout_seconds: float = DEFAULT_LM_STUDIO_TIMEOUT_SECONDS
    lm_studio_temperature: float = DEFAULT_LM_STUDIO_TEMPERATURE
    lm_studio_max_tokens: int = DEFAULT_LM_STUDIO_MAX_TOKENS
    response_chat_short_max_tokens: int = DEFAULT_RESPONSE_CHAT_SHORT_MAX_TOKENS
    response_normal_max_tokens: int = DEFAULT_RESPONSE_NORMAL_MAX_TOKENS
    context_recent_turns: int = DEFAULT_CONTEXT_RECENT_TURNS
    context_history_max_chars_chat_short: int = DEFAULT_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT
    context_history_max_chars_normal: int = DEFAULT_CONTEXT_HISTORY_MAX_CHARS_NORMAL
    context_history_max_chars_deep: int = DEFAULT_CONTEXT_HISTORY_MAX_CHARS_DEEP
    lm_studio_api_token: str | None = field(default=None, repr=False)
    venice_base_url: str = DEFAULT_VENICE_BASE_URL
    venice_api_key: str | None = field(default=None, repr=False)
    venice_text_model_id: str = ""
    venice_vision_model_id: str = ""
    image_download_timeout_seconds: float = DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS
    max_images_per_message: int = DEFAULT_MAX_IMAGES_PER_MESSAGE
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_total_image_bytes: int = DEFAULT_MAX_TOTAL_IMAGE_BYTES
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS
    model_image_max_long_edge: int = DEFAULT_MODEL_IMAGE_MAX_LONG_EDGE
    model_image_max_pixels: int = DEFAULT_MODEL_IMAGE_MAX_PIXELS
    model_total_image_pixels: int = DEFAULT_MODEL_TOTAL_IMAGE_PIXELS
    database_path: Path = DEFAULT_DATABASE_PATH
    persona_max_chars: int = DEFAULT_PERSONA_MAX_CHARS
    persona_global_cooldown_seconds: int = DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS
    persona_daily_limit: int = DEFAULT_PERSONA_DAILY_LIMIT
    persona_daily_reset_utc_offset_hours: int = DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS
    persona_status_channel_id: int | None = None
    nickname_max_chars: int = DEFAULT_NICKNAME_MAX_CHARS
    log_level: str = DEFAULT_LOG_LEVEL

    def __post_init__(self) -> None:
        if not isinstance(self.llm_provider, LlmProviderKind):
            raise ConfigError("LIUER_LLM_PROVIDER must be one of: lm_studio, venice")
        if self.llm_provider is LlmProviderKind.VENICE:
            if not self.venice_api_key:
                raise ConfigError("LIUER_VENICE_API_KEY is required")
            if not self.venice_text_model_id.strip():
                raise ConfigError("LIUER_VENICE_TEXT_MODEL_ID is required")
            if not self.venice_vision_model_id.strip():
                raise ConfigError("LIUER_VENICE_VISION_MODEL_ID is required")
        if not isinstance(self.chat_channel_id, int) or self.chat_channel_id <= 0:
            raise ConfigError("LIUER_CHAT_CHANNEL_ID must be a positive integer")
        if self.persona_status_channel_id is not None:
            if self.persona_status_channel_id <= 0:
                raise ConfigError("LIUER_PERSONA_STATUS_CHANNEL_ID must be a positive integer")
            if self.chat_channel_id == self.persona_status_channel_id:
                raise ConfigError(
                    "LIUER_CHAT_CHANNEL_ID must differ from LIUER_PERSONA_STATUS_CHANNEL_ID"
                )
        if not 1 <= self.context_recent_turns <= MAX_CONTEXT_RECENT_TURNS:
            raise ConfigError(
                f"LIUER_CONTEXT_RECENT_TURNS must be at most {MAX_CONTEXT_RECENT_TURNS}"
            )
        for name, value in (
            (
                "LIUER_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT",
                self.context_history_max_chars_chat_short,
            ),
            ("LIUER_CONTEXT_HISTORY_MAX_CHARS_NORMAL", self.context_history_max_chars_normal),
            ("LIUER_CONTEXT_HISTORY_MAX_CHARS_DEEP", self.context_history_max_chars_deep),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= MAX_CONTEXT_HISTORY_MAX_CHARS
            ):
                raise ConfigError(
                    f"{name} must be between 0 and {MAX_CONTEXT_HISTORY_MAX_CHARS}"
                )
        if (
            not isinstance(self.nickname_max_chars, int)
            or isinstance(self.nickname_max_chars, bool)
            or not 1 <= self.nickname_max_chars <= MAX_NICKNAME_MAX_CHARS
        ):
            raise ConfigError(
                f"LIUER_NICKNAME_MAX_CHARS must be between 1 and {MAX_NICKNAME_MAX_CHARS}"
            )

    def __repr__(self) -> str:
        token_status = "configured" if self.discord_token else "missing"
        return (
            "Config("
            f"discord_token=<{token_status}>, "
            f"chat_channel_id={self.chat_channel_id!r}, "
            f"llm_provider={self.llm_provider.value!r}, "
            f"lm_studio_base_url={self.lm_studio_base_url!r}, "
            "lm_studio_model="
            f"<{'configured' if self.lm_studio_model else 'missing'}>, "
            "lm_studio_api_token="
            f"<{'configured' if self.lm_studio_api_token else 'not configured'}>, "
            f"venice_base_url={self.venice_base_url!r}, "
            "venice_api_key="
            f"<{'configured' if self.venice_api_key else 'not configured'}>, "
            "venice_text_model_id="
            f"<{'configured' if self.venice_text_model_id else 'missing'}>, "
            "venice_vision_model_id="
            f"<{'configured' if self.venice_vision_model_id else 'missing'}>, "
            f"response_chat_short_max_tokens={self.response_chat_short_max_tokens!r}, "
            f"response_normal_max_tokens={self.response_normal_max_tokens!r}, "
            f"context_recent_turns={self.context_recent_turns!r}, "
            f"context_history_max_chars_chat_short={self.context_history_max_chars_chat_short!r}, "
            f"context_history_max_chars_normal={self.context_history_max_chars_normal!r}, "
            f"context_history_max_chars_deep={self.context_history_max_chars_deep!r}, "
            f"database_path={str(self.database_path)!r}, "
            f"persona_status_channel_id={self.persona_status_channel_id!r}, "
            f"nickname_max_chars={self.nickname_max_chars!r}, "
            f"log_level={self.log_level!r})"
        )


def _unquote_dotenv_value(value: str) -> str:
    """Remove one matching pair of simple dotenv quotes."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _validate_positive_number(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a positive number") from exc
    if not isfinite(parsed) or parsed <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return parsed


def _validate_temperature(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigError("LIUER_LM_STUDIO_TEMPERATURE must be between 0.0 and 2.0") from exc
    if not isfinite(parsed) or not 0.0 <= parsed <= 2.0:
        raise ConfigError("LIUER_LM_STUDIO_TEMPERATURE must be between 0.0 and 2.0")
    return parsed


def _validate_positive_integer(value: str, name: str) -> int:
    if not _CHANNEL_ID.fullmatch(value):
        raise ConfigError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return parsed


def _validate_signed_integer(value: str, name: str, *, minimum: int, maximum: int) -> int:
    if not _SIGNED_INTEGER.fullmatch(value):
        raise ConfigError(f"{name} must be an integer")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read a deliberately small, non-interpolating dotenv format."""

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError("could not read .env") from exc

    values: dict[str, str] = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            raise ConfigError(f"invalid .env entry on line {line_number}")
        name, value = (part.strip() for part in stripped.split("=", 1))
        if not _ENV_NAME.fullmatch(name):
            raise ConfigError(f"invalid .env variable name on line {line_number}")
        values[name] = _unquote_dotenv_value(value)
    return values


def _load_values(
    environ: Mapping[str, str] | None,
    dotenv_path: str | Path | None,
) -> dict[str, str]:
    """Merge optional dotenv values with the process environment."""

    process_values = dict(os.environ if environ is None else environ)
    values: dict[str, str] = {}

    # An explicitly supplied mapping is generally used by tests or callers
    # that want isolation, so it does not implicitly read the current folder's
    # .env. Normal process loading does read the current working folder.
    if environ is None or dotenv_path is not None:
        path = Path(dotenv_path) if dotenv_path is not None else Path.cwd() / ".env"
        values.update(_read_dotenv(path))
    values.update(process_values)
    return values


def _required_value(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is required")
    return value


def _validate_url(value: str, name: str = "LIUER_LM_STUDIO_BASE_URL") -> str:
    if not value or any(character.isspace() for character in value):
        raise ConfigError(f"{name} must be a valid HTTP/HTTPS URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ConfigError(f"{name} must be a valid HTTP/HTTPS URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ConfigError(f"{name} must be a valid HTTP/HTTPS URL")
    return value.rstrip("/")


def _validate_database_path(value: str) -> Path:
    if not value or "\x00" in value:
        raise ConfigError("LIUER_DATABASE_PATH must be a valid file path")
    try:
        path = Path(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("LIUER_DATABASE_PATH must be a valid file path") from exc
    if not path.name or path.name in {".", ".."}:
        raise ConfigError("LIUER_DATABASE_PATH must be a valid file path")
    return path


def _validate_log_level(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigError(f"LIUER_LOG_LEVEL must be one of: {allowed}")
    return normalized


def load_config(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | Path | None = None,
) -> Config:
    """Load and validate configuration from dotenv values and environment.

    Real process environment values take precedence over values in `.env`.
    Passing an explicit mapping skips implicit `.env` loading unless a
    ``dotenv_path`` is also supplied, which keeps tests deterministic.
    """

    values = _load_values(environ, dotenv_path)
    discord_token = _required_value(values, "LIUER_DISCORD_TOKEN")
    chat_channel_id = _validate_positive_integer(
        _required_value(values, "LIUER_CHAT_CHANNEL_ID"),
        "LIUER_CHAT_CHANNEL_ID",
    )
    provider_value = values.get("LIUER_LLM_PROVIDER", LlmProviderKind.LM_STUDIO.value).strip()
    try:
        llm_provider = LlmProviderKind(provider_value)
    except ValueError as exc:
        raise ConfigError("LIUER_LLM_PROVIDER must be one of: lm_studio, venice") from exc
    lm_studio_base_url = values.get("LIUER_LM_STUDIO_BASE_URL", DEFAULT_LM_STUDIO_BASE_URL).strip()
    lm_studio_model = values.get("LIUER_LM_STUDIO_MODEL", "").strip()
    venice_base_url = values.get("LIUER_VENICE_BASE_URL", DEFAULT_VENICE_BASE_URL).strip()
    venice_api_key = values.get("LIUER_VENICE_API_KEY", "").strip() or None
    venice_text_model_id = values.get("LIUER_VENICE_TEXT_MODEL_ID", "").strip()
    venice_vision_model_id = values.get("LIUER_VENICE_VISION_MODEL_ID", "").strip()
    if llm_provider is LlmProviderKind.LM_STUDIO:
        lm_studio_model = _required_value(values, "LIUER_LM_STUDIO_MODEL")
        lm_studio_base_url = _validate_url(lm_studio_base_url)
    else:
        venice_api_key = _required_value(values, "LIUER_VENICE_API_KEY")
        venice_text_model_id = _required_value(values, "LIUER_VENICE_TEXT_MODEL_ID")
        venice_vision_model_id = _required_value(values, "LIUER_VENICE_VISION_MODEL_ID")
        venice_base_url = _validate_url(venice_base_url, "LIUER_VENICE_BASE_URL")
    timeout_value = values.get(
        "LIUER_LM_STUDIO_TIMEOUT_SECONDS",
        str(DEFAULT_LM_STUDIO_TIMEOUT_SECONDS),
    ).strip()
    temperature_value = values.get(
        "LIUER_LM_STUDIO_TEMPERATURE",
        str(DEFAULT_LM_STUDIO_TEMPERATURE),
    ).strip()
    max_tokens_value = values.get(
        "LIUER_LM_STUDIO_MAX_TOKENS",
        str(DEFAULT_LM_STUDIO_MAX_TOKENS),
    ).strip()
    response_chat_short_value = values.get(
        "LIUER_RESPONSE_CHAT_SHORT_MAX_TOKENS",
        str(DEFAULT_RESPONSE_CHAT_SHORT_MAX_TOKENS),
    ).strip()
    response_normal_value = values.get(
        "LIUER_RESPONSE_NORMAL_MAX_TOKENS",
        str(DEFAULT_RESPONSE_NORMAL_MAX_TOKENS),
    ).strip()
    context_recent_turns_value = values.get(
        "LIUER_CONTEXT_RECENT_TURNS",
        str(DEFAULT_CONTEXT_RECENT_TURNS),
    ).strip()
    context_history_max_chars_chat_short_value = values.get(
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT",
        str(DEFAULT_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT),
    ).strip()
    context_history_max_chars_normal_value = values.get(
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_NORMAL",
        str(DEFAULT_CONTEXT_HISTORY_MAX_CHARS_NORMAL),
    ).strip()
    context_history_max_chars_deep_value = values.get(
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_DEEP",
        str(DEFAULT_CONTEXT_HISTORY_MAX_CHARS_DEEP),
    ).strip()
    api_token = values.get("LIUER_LM_STUDIO_API_TOKEN", "").strip() or None
    image_download_timeout_value = values.get(
        "LIUER_IMAGE_DOWNLOAD_TIMEOUT_SECONDS",
        str(DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS),
    ).strip()
    max_images_value = values.get(
        "LIUER_MAX_IMAGES_PER_MESSAGE",
        str(DEFAULT_MAX_IMAGES_PER_MESSAGE),
    ).strip()
    max_image_bytes_value = values.get(
        "LIUER_MAX_IMAGE_BYTES",
        str(DEFAULT_MAX_IMAGE_BYTES),
    ).strip()
    max_total_image_bytes_value = values.get(
        "LIUER_MAX_TOTAL_IMAGE_BYTES",
        str(DEFAULT_MAX_TOTAL_IMAGE_BYTES),
    ).strip()
    max_image_pixels_value = values.get(
        "LIUER_MAX_IMAGE_PIXELS",
        str(DEFAULT_MAX_IMAGE_PIXELS),
    ).strip()
    model_image_max_long_edge_value = values.get(
        "LIUER_MODEL_IMAGE_MAX_LONG_EDGE",
        str(DEFAULT_MODEL_IMAGE_MAX_LONG_EDGE),
    ).strip()
    model_image_max_pixels_value = values.get(
        "LIUER_MODEL_IMAGE_MAX_PIXELS",
        str(DEFAULT_MODEL_IMAGE_MAX_PIXELS),
    ).strip()
    model_total_image_pixels_value = values.get(
        "LIUER_MODEL_TOTAL_IMAGE_PIXELS",
        str(DEFAULT_MODEL_TOTAL_IMAGE_PIXELS),
    ).strip()
    database_value = values.get("LIUER_DATABASE_PATH", str(DEFAULT_DATABASE_PATH)).strip()
    persona_max_chars_value = values.get(
        "LIUER_PERSONA_MAX_CHARS",
        str(DEFAULT_PERSONA_MAX_CHARS),
    ).strip()
    persona_cooldown_value = values.get(
        "LIUER_PERSONA_GLOBAL_COOLDOWN_SECONDS",
        str(DEFAULT_PERSONA_GLOBAL_COOLDOWN_SECONDS),
    ).strip()
    persona_daily_limit_value = values.get(
        "LIUER_PERSONA_DAILY_LIMIT",
        str(DEFAULT_PERSONA_DAILY_LIMIT),
    ).strip()
    persona_offset_value = values.get(
        "LIUER_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS",
        str(DEFAULT_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS),
    ).strip()
    nickname_max_chars_value = values.get(
        "LIUER_NICKNAME_MAX_CHARS",
        str(DEFAULT_NICKNAME_MAX_CHARS),
    ).strip()
    persona_status_channel_value = _required_value(
        values,
        "LIUER_PERSONA_STATUS_CHANNEL_ID",
    )
    persona_status_channel_id = _validate_positive_integer(
        persona_status_channel_value,
        "LIUER_PERSONA_STATUS_CHANNEL_ID",
    )
    if chat_channel_id == persona_status_channel_id:
        raise ConfigError(
            "LIUER_CHAT_CHANNEL_ID must differ from LIUER_PERSONA_STATUS_CHANNEL_ID"
        )
    log_level = values.get("LIUER_LOG_LEVEL", DEFAULT_LOG_LEVEL)

    lm_studio_max_tokens = _validate_positive_integer(
        max_tokens_value,
        "LIUER_LM_STUDIO_MAX_TOKENS",
    )
    response_chat_short_max_tokens = _validate_positive_integer(
        response_chat_short_value,
        "LIUER_RESPONSE_CHAT_SHORT_MAX_TOKENS",
    )
    response_normal_max_tokens = _validate_positive_integer(
        response_normal_value,
        "LIUER_RESPONSE_NORMAL_MAX_TOKENS",
    )
    context_recent_turns = _validate_positive_integer(
        context_recent_turns_value,
        "LIUER_CONTEXT_RECENT_TURNS",
    )
    if context_recent_turns > MAX_CONTEXT_RECENT_TURNS:
        raise ConfigError(
            f"LIUER_CONTEXT_RECENT_TURNS must be at most {MAX_CONTEXT_RECENT_TURNS}"
        )
    context_history_max_chars_chat_short = _validate_signed_integer(
        context_history_max_chars_chat_short_value,
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_CHAT_SHORT",
        minimum=0,
        maximum=MAX_CONTEXT_HISTORY_MAX_CHARS,
    )
    context_history_max_chars_normal = _validate_signed_integer(
        context_history_max_chars_normal_value,
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_NORMAL",
        minimum=0,
        maximum=MAX_CONTEXT_HISTORY_MAX_CHARS,
    )
    context_history_max_chars_deep = _validate_signed_integer(
        context_history_max_chars_deep_value,
        "LIUER_CONTEXT_HISTORY_MAX_CHARS_DEEP",
        minimum=0,
        maximum=MAX_CONTEXT_HISTORY_MAX_CHARS,
    )
    if not response_chat_short_max_tokens <= response_normal_max_tokens <= lm_studio_max_tokens:
        raise ConfigError(
            "response token budgets must satisfy "
            "LIUER_RESPONSE_CHAT_SHORT_MAX_TOKENS <= "
            "LIUER_RESPONSE_NORMAL_MAX_TOKENS <= LIUER_LM_STUDIO_MAX_TOKENS"
        )

    return Config(
        discord_token=discord_token,
        chat_channel_id=chat_channel_id,
        llm_provider=llm_provider,
        lm_studio_base_url=lm_studio_base_url,
        lm_studio_model=lm_studio_model,
        lm_studio_timeout_seconds=_validate_positive_number(
            timeout_value,
            "LIUER_LM_STUDIO_TIMEOUT_SECONDS",
        ),
        lm_studio_temperature=_validate_temperature(temperature_value),
        lm_studio_max_tokens=lm_studio_max_tokens,
        response_chat_short_max_tokens=response_chat_short_max_tokens,
        response_normal_max_tokens=response_normal_max_tokens,
        context_recent_turns=context_recent_turns,
        context_history_max_chars_chat_short=context_history_max_chars_chat_short,
        context_history_max_chars_normal=context_history_max_chars_normal,
        context_history_max_chars_deep=context_history_max_chars_deep,
        lm_studio_api_token=api_token,
        venice_base_url=venice_base_url,
        venice_api_key=venice_api_key,
        venice_text_model_id=venice_text_model_id,
        venice_vision_model_id=venice_vision_model_id,
        image_download_timeout_seconds=_validate_positive_number(
            image_download_timeout_value,
            "LIUER_IMAGE_DOWNLOAD_TIMEOUT_SECONDS",
        ),
        max_images_per_message=_validate_positive_integer(
            max_images_value,
            "LIUER_MAX_IMAGES_PER_MESSAGE",
        ),
        max_image_bytes=_validate_positive_integer(
            max_image_bytes_value,
            "LIUER_MAX_IMAGE_BYTES",
        ),
        max_total_image_bytes=_validate_positive_integer(
            max_total_image_bytes_value,
            "LIUER_MAX_TOTAL_IMAGE_BYTES",
        ),
        max_image_pixels=_validate_positive_integer(
            max_image_pixels_value,
            "LIUER_MAX_IMAGE_PIXELS",
        ),
        model_image_max_long_edge=_validate_positive_integer(
            model_image_max_long_edge_value,
            "LIUER_MODEL_IMAGE_MAX_LONG_EDGE",
        ),
        model_image_max_pixels=_validate_positive_integer(
            model_image_max_pixels_value,
            "LIUER_MODEL_IMAGE_MAX_PIXELS",
        ),
        model_total_image_pixels=_validate_positive_integer(
            model_total_image_pixels_value,
            "LIUER_MODEL_TOTAL_IMAGE_PIXELS",
        ),
        database_path=_validate_database_path(database_value),
        persona_max_chars=_validate_positive_integer(
            persona_max_chars_value,
            "LIUER_PERSONA_MAX_CHARS",
        ),
        persona_global_cooldown_seconds=_validate_positive_integer(
            persona_cooldown_value,
            "LIUER_PERSONA_GLOBAL_COOLDOWN_SECONDS",
        ),
        persona_daily_limit=_validate_positive_integer(
            persona_daily_limit_value,
            "LIUER_PERSONA_DAILY_LIMIT",
        ),
        persona_daily_reset_utc_offset_hours=_validate_signed_integer(
            persona_offset_value,
            "LIUER_PERSONA_DAILY_RESET_UTC_OFFSET_HOURS",
            minimum=-12,
            maximum=14,
        ),
        persona_status_channel_id=persona_status_channel_id,
        nickname_max_chars=_validate_positive_integer(
            nickname_max_chars_value,
            "LIUER_NICKNAME_MAX_CHARS",
        ),
        log_level=_validate_log_level(log_level),
    )

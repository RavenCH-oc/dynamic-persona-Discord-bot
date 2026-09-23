"""Read-only database and active-provider health probes."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from .config import Config
from .database import SCHEMA_VERSION, connect_database
from .generation_queue import GenerationQueueStatus
from .llm_provider import LlmProviderKind, active_model_id
from .runtime_status import RuntimeStatusService, RuntimeStatusSnapshot
from .venice_models import VeniceModelState, assess_venice_models

LOGGER = logging.getLogger(__name__)
LM_STUDIO_HEALTH_TIMEOUT_SECONDS = 5.0


class ProviderHealthState(StrEnum):
    READY = "READY"
    UNREACHABLE = "UNREACHABLE"
    TIMEOUT = "TIMEOUT"
    AUTH_ERROR = "AUTH_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    MODEL_MISSING = "MODEL_MISSING"
    INCOMPATIBLE_MODEL = "INCOMPATIBLE_MODEL"


@dataclass(frozen=True, slots=True, repr=False)
class ProviderHealthResult:
    """Safe active-provider probe result without response bodies or headers."""

    state: ProviderHealthState
    latency_ms: int | None
    configured_model_present: bool | None
    text_model_state: ProviderHealthState | None = None
    vision_model_state: ProviderHealthState | None = None

    def __repr__(self) -> str:
        return (
            "ProviderHealthResult("
            f"state={self.state.value!r}, "
            f"latency_ms={self.latency_ms!r}, "
            f"configured_model_present={self.configured_model_present!r}, "
            f"text_model_state={self.text_model_state!r}, "
            f"vision_model_state={self.vision_model_state!r})"
        )


# Phase 5C compatibility aliases for callers importing the previous names.
LmStudioHealthState = ProviderHealthState
LmStudioHealthResult = ProviderHealthResult


class DatabaseHealthState(StrEnum):
    READY = "READY"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True, repr=False)
class DatabaseHealthResult:
    """Safe structural database probe result."""

    state: DatabaseHealthState
    schema_version: int | None

    def __repr__(self) -> str:
        return (
            "DatabaseHealthResult("
            f"state={self.state.value!r}, schema_version={self.schema_version!r})"
        )


class DatabaseHealthProbe:
    """Run a lightweight SELECT 1 and schema-version check without content reads."""

    def __init__(
        self,
        database_path: Path,
        *,
        expected_schema_version: int = SCHEMA_VERSION,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._database_path = Path(database_path)
        self._expected_schema_version = expected_schema_version
        self._logger = logger

    async def probe(self) -> DatabaseHealthResult:
        return await asyncio.to_thread(self._probe_sync)

    def _probe_sync(self) -> DatabaseHealthResult:
        if not self._database_path.exists():
            return DatabaseHealthResult(DatabaseHealthState.ERROR, None)
        connection = None
        try:
            connection = connect_database(self._database_path)
            connection.execute("SELECT 1").fetchone()
            row = connection.execute("PRAGMA user_version").fetchone()
            version = int(row[0]) if row else None
            if version != self._expected_schema_version:
                return DatabaseHealthResult(DatabaseHealthState.SCHEMA_MISMATCH, version)
            return DatabaseHealthResult(DatabaseHealthState.READY, version)
        except Exception as exc:
            self._logger.debug(
                "database health probe failed error_type=%s",
                type(exc).__name__,
            )
            return DatabaseHealthResult(DatabaseHealthState.ERROR, None)
        finally:
            if connection is not None:
                connection.close()


class LmStudioHealthProbe:
    """Probe exactly one LM Studio model-list endpoint with a short timeout."""

    def __init__(
        self,
        *,
        base_url: str,
        configured_model: str,
        api_token: str | None = None,
        timeout_seconds: float = LM_STUDIO_HEALTH_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._models_url = f"{base_url.rstrip('/')}/models"
        self._configured_model = configured_model
        self._api_token = api_token
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._logger = logger

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        logger: logging.Logger = LOGGER,
    ) -> LmStudioHealthProbe:
        return cls(
            base_url=config.lm_studio_base_url,
            configured_model=config.lm_studio_model,
            api_token=config.lm_studio_api_token,
            transport=transport,
            logger=logger,
        )

    async def probe(self) -> ProviderHealthResult:
        started = monotonic()
        headers = (
            {"Authorization": f"Bearer {self._api_token}"}
            if self._api_token is not None
            else None
        )
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                headers=headers,
                transport=self._transport,
            ) as client:
                async with asyncio.timeout(self._timeout_seconds):
                    response = await client.get(self._models_url)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            return self._result(LmStudioHealthState.TIMEOUT, started, None)
        except httpx.RequestError as exc:
            self._logger.debug(
                "LM Studio health probe unavailable error_type=%s",
                type(exc).__name__,
            )
            return self._result(LmStudioHealthState.UNREACHABLE, started, None)
        except Exception as exc:
            self._logger.debug(
                "LM Studio health probe failed error_type=%s",
                type(exc).__name__,
            )
            return self._result(LmStudioHealthState.INVALID_RESPONSE, started, None)

        if response.status_code in {401, 403}:
            return self._result(LmStudioHealthState.AUTH_ERROR, started, None)
        if response.is_error:
            return self._result(LmStudioHealthState.HTTP_ERROR, started, None)
        try:
            payload = response.json()
            model_ids = _model_ids(payload)
        except (TypeError, ValueError):
            return self._result(LmStudioHealthState.INVALID_RESPONSE, started, None)
        if self._configured_model in model_ids:
            return self._result(LmStudioHealthState.READY, started, True)
        return self._result(LmStudioHealthState.MODEL_MISSING, started, False)

    @staticmethod
    def _latency(started: float) -> int:
        return max(0, round((monotonic() - started) * 1000))

    def _result(
        self,
        state: LmStudioHealthState,
        started: float,
        configured_model_present: bool | None,
    ) -> ProviderHealthResult:
        return ProviderHealthResult(
            state=state,
            latency_ms=self._latency(started),
            configured_model_present=configured_model_present,
        )


class VeniceHealthProbe:
    """Probe both Venice model roles through one catalog request."""

    def __init__(
        self,
        *,
        base_url: str,
        text_model_id: str,
        vision_model_id: str,
        api_key: str,
        required_max_images: int,
        timeout_seconds: float = LM_STUDIO_HEALTH_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._models_url = f"{base_url.rstrip('/')}/models"
        self._text_model_id = text_model_id
        self._vision_model_id = vision_model_id
        self._api_key = api_key
        self._required_max_images = required_max_images
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._logger = logger

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        logger: logging.Logger = LOGGER,
    ) -> VeniceHealthProbe:
        return cls(
            base_url=config.venice_base_url,
            text_model_id=config.venice_text_model_id,
            vision_model_id=config.venice_vision_model_id,
            api_key=config.venice_api_key or "",
            required_max_images=config.max_images_per_message,
            transport=transport,
            logger=logger,
        )

    async def probe(self) -> ProviderHealthResult:
        started = monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                headers={"Authorization": f"Bearer {self._api_key}"},
                transport=self._transport,
            ) as client:
                async with asyncio.timeout(self._timeout_seconds):
                    response = await client.get(self._models_url)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.TimeoutException):
            return self._result(LmStudioHealthState.TIMEOUT, started, None)
        except httpx.RequestError as exc:
            self._logger.debug(
                "Venice health probe unavailable error_type=%s",
                type(exc).__name__,
            )
            return self._result(LmStudioHealthState.UNREACHABLE, started, None)
        except Exception as exc:
            self._logger.debug(
                "Venice health probe failed error_type=%s",
                type(exc).__name__,
            )
            return self._result(LmStudioHealthState.INVALID_RESPONSE, started, None)
        if response.status_code in {401, 403}:
            return self._result(LmStudioHealthState.AUTH_ERROR, started, None)
        if response.is_error:
            return self._result(LmStudioHealthState.HTTP_ERROR, started, None)
        try:
            assessment = assess_venice_models(
                response.json(),
                text_model_id=self._text_model_id,
                vision_model_id=self._vision_model_id,
                required_max_images=self._required_max_images,
            )
        except (TypeError, ValueError):
            return self._result(LmStudioHealthState.INVALID_RESPONSE, started, None)
        text_state = ProviderHealthState(assessment.text_state.value)
        vision_state = ProviderHealthState(assessment.vision_state.value)
        overall_state = (
            ProviderHealthState.MODEL_MISSING
            if VeniceModelState.MODEL_MISSING in (
                assessment.text_state, assessment.vision_state
            )
            else ProviderHealthState.INCOMPATIBLE_MODEL
            if VeniceModelState.INCOMPATIBLE_MODEL in (
                assessment.text_state, assessment.vision_state
            )
            else ProviderHealthState.READY
        )
        return self._result(
            overall_state,
            started,
            assessment.text_model is not None and assessment.vision_model is not None,
            text_model_state=text_state,
            vision_model_state=vision_state,
        )

    @staticmethod
    def _result(
        state: LmStudioHealthState,
        started: float,
        configured_model_present: bool | None,
        *,
        text_model_state: ProviderHealthState | None = None,
        vision_model_state: ProviderHealthState | None = None,
    ) -> ProviderHealthResult:
        return ProviderHealthResult(
            state=state,
            latency_ms=max(0, round((monotonic() - started) * 1000)),
            configured_model_present=configured_model_present,
            text_model_state=text_model_state,
            vision_model_state=vision_model_state,
        )


def _model_ids(payload: Any) -> set[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("invalid model payload")
    model_ids: set[str] = set()
    for item in payload["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("invalid model entry")
        model_ids.add(item["id"])
    return model_ids


class HealthOverallState(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeHealthResult:
    """Aggregated read-only health state with safe scalar fields only."""

    overall: HealthOverallState
    runtime_ready: bool
    conversation_initialized: bool
    queue_status: GenerationQueueStatus
    database: DatabaseHealthResult
    provider: ProviderHealthResult
    configured_model: str
    provider_kind: LlmProviderKind = LlmProviderKind.LM_STUDIO
    vision_model: str | None = None

    def __repr__(self) -> str:
        return (
            "RuntimeHealthResult("
            f"overall={self.overall.value!r}, "
            f"runtime_ready={self.runtime_ready!r}, "
            f"conversation_initialized={self.conversation_initialized!r}, "
            f"queue_status={self.queue_status!r}, "
            f"database={self.database!r}, "
            f"provider={self.provider!r}, "
            f"provider_kind={self.provider_kind.value!r}, "
            "configured_model=<configured>, vision_model=<redacted>)"
        )

    @property
    def lm_studio(self) -> ProviderHealthResult:
        """Compatibility view for Phase 5C callers."""

        return self.provider


class RuntimeHealthService:
    """Coordinate independent DB and active-provider probes outside generation."""

    def __init__(
        self,
        config: Config,
        *,
        status_service: RuntimeStatusService,
        database_probe: DatabaseHealthProbe | None = None,
        lm_studio_probe: LmStudioHealthProbe | None = None,
        provider_probe: LmStudioHealthProbe | VeniceHealthProbe | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._config = config
        self._status_service = status_service
        self._logger = logger
        self._database_probe = database_probe or DatabaseHealthProbe(
            config.database_path,
            logger=logger,
        )
        selected_probe = provider_probe or lm_studio_probe
        if selected_probe is None:
            selected_probe = (
                VeniceHealthProbe.from_config(config)
                if config.llm_provider is LlmProviderKind.VENICE
                else LmStudioHealthProbe.from_config(config)
            )
        self._provider_probe = selected_probe

    async def check(self) -> RuntimeHealthResult:
        snapshot = self._status_service.get_snapshot()
        database_probe, provider_probe = await asyncio.gather(
            self._database_probe.probe(),
            self._provider_probe.probe(),
            return_exceptions=True,
        )
        if isinstance(database_probe, asyncio.CancelledError):
            raise database_probe
        if isinstance(provider_probe, asyncio.CancelledError):
            raise provider_probe
        if isinstance(database_probe, BaseException):
            self._logger.error(
                "database health probe raised error_type=%s",
                type(database_probe).__name__,
            )
        if isinstance(provider_probe, BaseException):
            self._logger.error(
                "LLM provider health probe raised provider=%s error_type=%s",
                self._config.llm_provider.value,
                type(provider_probe).__name__,
            )
        database = (
            database_probe
            if isinstance(database_probe, DatabaseHealthResult)
            else DatabaseHealthResult(DatabaseHealthState.ERROR, None)
        )
        lm_studio = (
            provider_probe
            if isinstance(provider_probe, ProviderHealthResult)
            else ProviderHealthResult(ProviderHealthState.INVALID_RESPONSE, None, None)
        )
        overall = self._overall(snapshot, database, lm_studio)
        return RuntimeHealthResult(
            overall=overall,
            runtime_ready=snapshot.runtime_ready,
            conversation_initialized=snapshot.conversation_status.initialized,
            queue_status=snapshot.queue_status,
            database=database,
            provider=lm_studio,
            configured_model=active_model_id(self._config),
            provider_kind=self._config.llm_provider,
            vision_model=(
                self._config.venice_vision_model_id
                if self._config.llm_provider is LlmProviderKind.VENICE else None
            ),
        )

    @staticmethod
    def _overall(
        snapshot: RuntimeStatusSnapshot,
        database: DatabaseHealthResult,
        lm_studio: ProviderHealthResult,
    ) -> HealthOverallState:
        if (
            not snapshot.runtime_ready
            or not snapshot.conversation_status.initialized
            or not snapshot.queue_status.worker_alive
            or database.state is not DatabaseHealthState.READY
            or lm_studio.state is not ProviderHealthState.READY
        ):
            return HealthOverallState.UNHEALTHY
        return HealthOverallState.HEALTHY

    async def check_and_render(self) -> str:
        result = await self.check()
        return _render_health(result)


def _render_health(result: RuntimeHealthResult) -> str:
    runtime = "OK" if result.runtime_ready else "Not ready"
    worker = "OK" if result.queue_status.worker_alive else "Dead"
    database = result.database.state.value
    provider_state = result.provider.state.value
    loaded = (
        "Yes"
        if result.provider.configured_model_present is True
        else "No"
        if result.provider.configured_model_present is False
        else "Unknown"
    )
    latency = (
        f"{result.provider.latency_ms} ms"
        if result.provider.latency_ms is not None
        else "Unknown"
    )
    text_state = result.provider.text_model_state
    vision_state = result.provider.vision_model_state
    vision_ready = vision_state is ProviderHealthState.READY
    model_lines = (
        (
            f"API: {'READY' if text_state is not None else provider_state}",
            "文字模型:",
            str(result.configured_model),
            f"狀態: {text_state.value if text_state is not None else 'Unknown'}",
            "圖片模型:",
            str(result.vision_model),
            f"狀態: {vision_state.value if vision_state is not None else 'Unknown'}",
            f"Vision: {'OK' if vision_ready else 'Not ready'}",
            f"Multiple images: {'OK' if vision_ready else 'Not ready'}",
        )
        if result.provider_kind is LlmProviderKind.VENICE
        else (
            f"Configured model: {result.configured_model}",
            f"Model loaded: {loaded}",
        )
    )
    return "\n".join(
        (
            "六耳 Health",
            f"Overall: {result.overall.value}",
            f"Discord runtime: {runtime}",
            f"Generation worker: {worker}",
            f"Database: {database}",
            f"LLM Provider: {result.provider_kind.display_name}",
            f"Provider state: {provider_state}",
            *model_lines,
            f"Provider probe: {latency}",
            "",
            f"Queue busy: {'Yes' if result.queue_status.busy else 'No'}",
            f"Queue waiting: {result.queue_status.waiting_jobs}",
        )
    )


__all__ = [
    "DatabaseHealthProbe",
    "DatabaseHealthResult",
    "DatabaseHealthState",
    "HealthOverallState",
    "LM_STUDIO_HEALTH_TIMEOUT_SECONDS",
    "LmStudioHealthProbe",
    "LmStudioHealthResult",
    "LmStudioHealthState",
    "ProviderHealthResult",
    "ProviderHealthState",
    "RuntimeHealthResult",
    "RuntimeHealthService",
    "VeniceHealthProbe",
]

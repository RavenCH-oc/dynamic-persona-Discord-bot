"""Read-only local runtime status aggregation for the Phase 5C control plane."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from .config import Config
from .conversation_models import ConversationRuntimeStatus
from .conversation_service import ConversationService
from .database import SCHEMA_VERSION
from .generation_queue import GenerationQueueStatus, SerializedGenerationQueue
from .llm_provider import LlmProviderKind, active_model_id
from .nickname_service import NicknameService
from .persona_service import PersonaService


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeStatusSnapshot:
    """Immutable status values with no prompts, turns, or external objects."""

    runtime_ready: bool
    uptime_seconds: int
    discord_latency_ms: int | None
    queue_status: GenerationQueueStatus
    conversation_status: ConversationRuntimeStatus
    schema_version: int
    persona_label: str
    nickname: str | None
    model_id: str
    context_history_max_chars_chat_short: int
    context_history_max_chars_normal: int
    context_history_max_chars_deep: int
    vision_model_id: str | None = None

    def __repr__(self) -> str:
        return (
            "RuntimeStatusSnapshot("
            f"runtime_ready={self.runtime_ready!r}, "
            f"uptime_seconds={self.uptime_seconds!r}, "
            f"discord_latency_ms={self.discord_latency_ms!r}, "
            f"queue_status={self.queue_status!r}, "
            f"conversation_status={self.conversation_status!r}, "
            f"schema_version={self.schema_version!r}, "
            f"persona_label={self.persona_label!r}, "
            f"nickname={self.nickname!r}, "
            "model_id=<configured>, "
            "vision_model_id=<redacted>, "
            f"context_history_max_chars_chat_short="
            f"{self.context_history_max_chars_chat_short!r}, "
            f"context_history_max_chars_normal={self.context_history_max_chars_normal!r}, "
            f"context_history_max_chars_deep={self.context_history_max_chars_deep!r})"
        )


def format_uptime(seconds: int) -> str:
    """Render monotonic uptime as a compact human-readable duration."""

    bounded = max(0, int(seconds))
    days, remainder = divmod(bounded, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds:02d}s")
    return " ".join(parts)


class RuntimeStatusService:
    """Aggregate already-loaded runtime state without HTTP or database I/O."""

    def __init__(
        self,
        config: Config,
        *,
        generation_queue: SerializedGenerationQueue,
        conversation_service: ConversationService | None = None,
        persona_service: PersonaService | None = None,
        nickname_service: NicknameService | None = None,
        runtime_ready: Callable[[], bool] | None = None,
        discord_latency: Callable[[], float] | None = None,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        self._config = config
        self._generation_queue = generation_queue
        self._conversation_service = conversation_service
        self._persona_service = persona_service
        self._nickname_service = nickname_service
        self._runtime_ready = runtime_ready or (lambda: False)
        self._discord_latency = discord_latency or (lambda: float("nan"))
        self._monotonic = monotonic_clock
        self._started_at = monotonic_clock()

    def get_snapshot(self) -> RuntimeStatusSnapshot:
        """Return a coherent local snapshot; this method never performs I/O."""

        conversation_status = self._conversation_status()
        return RuntimeStatusSnapshot(
            runtime_ready=self._safe_ready(),
            uptime_seconds=max(0, int(self._monotonic() - self._started_at)),
            discord_latency_ms=self._latency_ms(),
            queue_status=self._generation_queue.get_status(),
            conversation_status=conversation_status,
            schema_version=SCHEMA_VERSION,
            persona_label=self._persona_label(),
            nickname=self._nickname_value(),
            model_id=active_model_id(self._config),
            vision_model_id=(
                self._config.venice_vision_model_id
                if self._config.llm_provider is LlmProviderKind.VENICE else None
            ),
            context_history_max_chars_chat_short=(
                self._config.context_history_max_chars_chat_short
            ),
            context_history_max_chars_normal=self._config.context_history_max_chars_normal,
            context_history_max_chars_deep=self._config.context_history_max_chars_deep,
        )

    def render(self) -> str:
        """Render one bounded, deterministic ephemeral status response."""

        snapshot = self.get_snapshot()
        ws = (
            f"{snapshot.discord_latency_ms} ms"
            if snapshot.discord_latency_ms is not None
            else "Unknown"
        )
        conversation = snapshot.conversation_status
        epoch = (
            str(conversation.active_epoch)
            if conversation.active_epoch is not None
            else "Unknown"
        )
        initialized = "Yes" if conversation.initialized else "No"
        worker = "Alive" if snapshot.queue_status.worker_alive else "Dead"
        busy = "Yes" if snapshot.queue_status.busy else "No"
        nickname = snapshot.nickname if snapshot.nickname is not None else "未設定"
        runtime = "Ready" if snapshot.runtime_ready else "Not ready"
        return "\n".join(
            (
                "六耳 Runtime Status",
                f"Bot: {runtime}",
                f"Uptime: {format_uptime(snapshot.uptime_seconds)}",
                f"Discord WS: {ws}",
                "",
                "Generation:",
                f"Worker: {worker}",
                f"Busy: {busy}",
                f"Waiting: {snapshot.queue_status.waiting_jobs}",
                "",
                "Conversation:",
                f"DB: {'Initialized' if initialized == 'Yes' else 'Not initialized'} "
                f"(schema v{snapshot.schema_version})",
                f"Epoch: {epoch}",
                f"Recent turns: {conversation.recent_turn_count}",
                "",
                "Identity:",
                f"Persona: {snapshot.persona_label}",
                f"Nickname: {nickname}",
                "",
                "History budgets (chars):",
                f"CHAT_SHORT: {snapshot.context_history_max_chars_chat_short}",
                f"NORMAL: {snapshot.context_history_max_chars_normal}",
                f"DEEP: {snapshot.context_history_max_chars_deep}",
                "",
                f"LLM Provider: {self._config.llm_provider.display_name}",
                *(
                    (
                        f"文字模型: {snapshot.model_id}",
                        f"圖片模型: {snapshot.vision_model_id}",
                    )
                    if self._config.llm_provider is LlmProviderKind.VENICE
                    else (f"Model: {snapshot.model_id}",)
                ),
                "Active probe: not performed by /status",
            )
        )

    def _conversation_status(self) -> ConversationRuntimeStatus:
        if self._conversation_service is None:
            return ConversationRuntimeStatus(False, None, 0)
        return self._conversation_service.get_runtime_status()

    def _safe_ready(self) -> bool:
        try:
            return bool(self._runtime_ready())
        except Exception:
            return False

    def _latency_ms(self) -> int | None:
        try:
            value = self._discord_latency()
        except Exception:
            return None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        if not math.isfinite(float(value)) or value < 0:
            return None
        return round(float(value) * 1000)

    def _persona_label(self) -> str:
        if self._persona_service is None:
            return "Unknown"
        try:
            return (
                "Custom"
                if self._persona_service.current_snapshot.active_version_id is not None
                else "Default"
            )
        except Exception:
            return "Unknown"

    def _nickname_value(self) -> str | None:
        if self._nickname_service is None:
            return None
        try:
            return self._nickname_service.get_active_nickname()
        except Exception:
            return None


__all__ = ["RuntimeStatusService", "RuntimeStatusSnapshot", "format_uptime"]

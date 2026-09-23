"""Async Venice Chat Completions provider with fail-closed model preflight."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Mapping
from time import monotonic
from typing import Any

import httpx

from .config import Config
from .conversation_context import (
    ContextSelection,
    ConversationContextProvider,
    ConversationGenerationContext,
    render_chat_completion_messages,
    render_current_user_text,
    render_historical_chat_messages,
    select_recent_turns_with_budget,
)
from .image_preparation import PreparedImage
from .llm_payload import build_system_prompt, safe_finish_reason, usage_value
from .llm_provider import LlmProviderError
from .model_image_normalization import (
    SUPPORTED_MODEL_IMAGE_MEDIA_TYPES,
    ModelImageNormalizationError,
    normalize_model_images,
)
from .prompting import (
    ActiveNicknameProvider,
    ActivePersonaProvider,
    PromptBuilder,
    SystemPromptBuilder,
)
from .responder import ChatRequest
from .response_routing import ResponseDecision, ResponseMode, ResponseRouter, response_instruction
from .venice_models import (
    VeniceModelRole,
    VeniceModelState,
    assess_venice_models,
    select_venice_model_role,
)

LOGGER = logging.getLogger(__name__)

VENICE_PARAMETERS: dict[str, object] = {
    "include_venice_system_prompt": False,
    "enable_web_search": "off",
    "enable_web_scraping": False,
    "enable_web_citations": False,
    "enable_x_search": False,
}


class VeniceError(LlmProviderError):
    """Base class for privacy-safe Venice failures."""


class VeniceUnavailableError(VeniceError):
    def __init__(self) -> None:
        super().__init__("Venice is unavailable")


class VeniceTimeoutError(VeniceError):
    def __init__(self) -> None:
        super().__init__("Venice request timed out")


class VeniceAuthError(VeniceError):
    def __init__(self) -> None:
        super().__init__("Venice authentication failed")


class VeniceHttpError(VeniceError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"Venice returned HTTP {status_code}")


class VeniceProtocolError(VeniceError):
    def __init__(self) -> None:
        super().__init__("Venice returned an invalid response")


class VeniceModelUnavailableError(VeniceError):
    def __init__(self, role: VeniceModelRole | None = None) -> None:
        label = role.value.lower() if role is not None else "configured"
        super().__init__(f"Venice {label} model is unavailable")


class VeniceModelIncompatibleError(VeniceError):
    def __init__(self, role: VeniceModelRole | None = None) -> None:
        label = role.value.lower() if role is not None else "configured"
        super().__init__(f"Venice {label} model is incompatible")


class VeniceImageInputError(VeniceError):
    def __init__(self) -> None:
        super().__init__("Venice received an invalid image input")


class VeniceResponder:
    """Generate through configured text and vision roles in one Venice provider."""

    def __init__(
        self,
        config: Config,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        prompt_builder: SystemPromptBuilder | None = None,
        active_persona_provider: ActivePersonaProvider | None = None,
        active_nickname_provider: ActiveNicknameProvider | None = None,
        conversation_context_provider: ConversationContextProvider | None = None,
        response_router: ResponseRouter | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._base_url = config.venice_base_url
        self._model_ids = {
            VeniceModelRole.TEXT: config.venice_text_model_id,
            VeniceModelRole.VISION: config.venice_vision_model_id,
        }
        self._api_key = config.venice_api_key or ""
        self._timeout_seconds = config.lm_studio_timeout_seconds
        self._temperature = config.lm_studio_temperature
        self._required_max_images = config.max_images_per_message
        self._model_image_max_long_edge = config.model_image_max_long_edge
        self._model_image_max_pixels = config.model_image_max_pixels
        self._model_total_image_pixels = config.model_total_image_pixels
        self._transport = transport
        self._prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        self._active_persona_provider = active_persona_provider
        self._active_nickname_provider = active_nickname_provider
        self._conversation_context_provider = conversation_context_provider
        self._context_history_max_chars = {
            ResponseMode.CHAT_SHORT: config.context_history_max_chars_chat_short,
            ResponseMode.NORMAL: config.context_history_max_chars_normal,
            ResponseMode.DEEP: config.context_history_max_chars_deep,
        }
        self._response_router = response_router or ResponseRouter.from_config(config)
        self._logger = logger

    async def preflight(self) -> None:
        """Validate both exact model roles from one non-generation catalog request."""

        response = await self._send("GET", "/models")
        payload = _response_object(response)
        try:
            assessment = assess_venice_models(
                payload,
                text_model_id=self._model_ids[VeniceModelRole.TEXT],
                vision_model_id=self._model_ids[VeniceModelRole.VISION],
                required_max_images=self._required_max_images,
            )
        except ValueError as exc:
            raise VeniceProtocolError() from exc
        for role, state, model in (
            (VeniceModelRole.TEXT, assessment.text_state, assessment.text_model),
            (VeniceModelRole.VISION, assessment.vision_state, assessment.vision_model),
        ):
            if state is VeniceModelState.MODEL_MISSING:
                raise VeniceModelUnavailableError(role)
            if state is VeniceModelState.INCOMPATIBLE_MODEL:
                raise VeniceModelIncompatibleError(role)
            if model is not None and model.deprecated is True:
                self._logger.warning(
                    "configured Venice model is deprecated role=%s model_id=%s",
                    role.value,
                    self._model_ids[role],
                )

    async def generate(self, request: ChatRequest) -> str:
        """Send exactly one non-streaming Venice Chat Completions request."""

        started_at = monotonic()
        decision = self._response_router.route(request)
        model_role = select_venice_model_role(request.prepared_images)
        self._logger.debug(
            "response routed message_id=%s mode=%s reason=%s max_tokens=%s provider=venice",
            request.message_id,
            decision.mode.name,
            decision.reason.name,
            decision.max_tokens,
        )
        selection = self._select_context(decision, request)
        self._log_context_selection(request, decision, selection)
        active_persona = (
            self._active_persona_provider.get_active_persona()
            if self._active_persona_provider is not None
            else None
        )
        active_nickname = (
            self._active_nickname_provider.get_active_nickname()
            if self._active_nickname_provider is not None
            else None
        )
        system_prompt = build_system_prompt(
            self._prompt_builder,
            active_persona,
            response_instruction(decision.mode),
            active_nickname,
        )
        current_text = render_current_user_text(
            request.author_display_name,
            request.content,
            request.reply_context,
            author_kind=request.author_kind,
        )
        if request.prepared_images:
            try:
                current_content: str | list[dict[str, object]] = await asyncio.to_thread(
                    _build_multimodal_content,
                    current_text,
                    request.prepared_images,
                    self._model_image_max_long_edge,
                    self._model_image_max_pixels,
                    self._model_total_image_pixels,
                )
            except (ModelImageNormalizationError, ValueError) as exc:
                raise VeniceImageInputError() from exc
            messages = [
                *render_historical_chat_messages(selection.turns),
                {"role": "user", "content": current_content},
            ]
        else:
            messages = render_chat_completion_messages(
                selection.turns,
                current_display_name=request.author_display_name,
                current_content=request.content,
                reply_context=request.reply_context,
                author_kind=request.author_kind,
            )
        body: dict[str, object] = {
            "model": self._model_ids[model_role],
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": self._temperature,
            "max_tokens": decision.max_tokens,
            "stream": False,
            "venice_parameters": dict(VENICE_PARAMETERS),
        }
        if decision.mode is ResponseMode.CHAT_SHORT:
            body["reasoning"] = {"enabled": False}
        response = await self._send("POST", "/chat/completions", json=body)
        payload = _response_object(response)
        finish_reason = safe_finish_reason(payload)
        usage = payload.get("usage")
        completion_tokens = usage_value(usage, "completion_tokens")
        self._logger.debug(
            "Venice generation result message_id=%s response_mode=%s finish_reason=%s "
            "max_tokens=%s completion_tokens=%s",
            request.message_id,
            decision.mode.name,
            finish_reason,
            decision.max_tokens,
            completion_tokens,
        )
        if finish_reason == "length":
            self._logger.warning(
                "generation reached token ceiling message_id=%s response_mode=%s max_tokens=%s",
                request.message_id,
                decision.mode.name,
                decision.max_tokens,
            )
        content = _assistant_content(payload)
        self._logger.debug(
            "Venice generation completed message_id=%s elapsed_ms=%s http_status=%s "
            "prompt_tokens=%s completion_tokens=%s",
            request.message_id,
            round((monotonic() - started_at) * 1000),
            response.status_code,
            usage_value(usage, "prompt_tokens"),
            completion_tokens,
        )
        return content

    def _generation_context(self, request: ChatRequest) -> ConversationGenerationContext:
        if request.generation_context is not None:
            return request.generation_context
        if self._conversation_context_provider is None:
            return ConversationGenerationContext(context_epoch=1, recent_turns=())
        provider = self._conversation_context_provider
        method = getattr(provider, "get_generation_context", None)
        if callable(method):
            return method()
        return ConversationGenerationContext(
            context_epoch=1,
            recent_turns=provider.get_recent_turns(),
        )

    def _select_context(
        self,
        decision: ResponseDecision,
        request: ChatRequest,
    ) -> ContextSelection:
        return select_recent_turns_with_budget(
            self._generation_context(request).recent_turns,
            max_chars=self._context_history_max_chars[decision.mode],
        )

    def _log_context_selection(
        self,
        request: ChatRequest,
        decision: ResponseDecision,
        selection: ContextSelection,
    ) -> None:
        self._logger.debug(
            "conversation context selected message_id=%s response_mode=%s "
            "available_turn_count=%s included_turn_count=%s trimmed_turn_count=%s "
            "rendered_history_chars=%s configured_history_max_chars=%s provider=venice",
            request.message_id,
            decision.mode.name,
            selection.available_turn_count,
            selection.included_turn_count,
            selection.trimmed_turn_count,
            selection.rendered_history_chars,
            self._context_history_max_chars[decision.mode],
        )

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_seconds,
                headers={"Authorization": f"Bearer {self._api_key}"},
                transport=self._transport,
            ) as client:
                response = await client.request(method, f"{self._base_url}{path}", **kwargs)
        except httpx.TimeoutException as exc:
            raise VeniceTimeoutError() from exc
        except httpx.RequestError as exc:
            raise VeniceUnavailableError() from exc
        if response.status_code in {401, 403}:
            raise VeniceAuthError()
        if response.is_error:
            raise VeniceHttpError(response.status_code)
        return response


def _response_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise VeniceProtocolError() from exc
    if not isinstance(payload, dict):
        raise VeniceProtocolError()
    return payload


def _assistant_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise VeniceProtocolError()
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise VeniceProtocolError()
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise VeniceProtocolError()
    return content.strip()


def _build_multimodal_content(
    text: str,
    prepared_images: tuple[PreparedImage, ...],
    max_long_edge: int,
    max_image_pixels: int,
    max_total_image_pixels: int,
) -> list[dict[str, object]]:
    normalized = normalize_model_images(
        prepared_images,
        max_long_edge=max_long_edge,
        max_image_pixels=max_image_pixels,
        max_total_image_pixels=max_total_image_pixels,
    )
    content: list[dict[str, object]] = [{"type": "text", "text": text}]
    for image in normalized:
        if image.media_type not in SUPPORTED_MODEL_IMAGE_MEDIA_TYPES or not image.data:
            raise ValueError("invalid prepared image")
        encoded = base64.b64encode(image.data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.media_type};base64,{encoded}"},
            }
        )
    return content


__all__ = [
    "VENICE_PARAMETERS",
    "VeniceAuthError",
    "VeniceError",
    "VeniceHttpError",
    "VeniceImageInputError",
    "VeniceModelIncompatibleError",
    "VeniceModelUnavailableError",
    "VeniceProtocolError",
    "VeniceResponder",
    "VeniceTimeoutError",
    "VeniceUnavailableError",
]

"""Async, stateless LM Studio Chat Completions responder."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Mapping
from time import monotonic
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import Config
from .conversation_context import (
    ContextSelection,
    ConversationContextProvider,
    ConversationGenerationContext,
    render_chat_completion_messages,
    render_current_user_text,
    render_historical_chat_messages,
    render_native_chat_input,
    select_recent_turns_with_budget,
)
from .conversation_models import ConversationTurn
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

LOGGER = logging.getLogger(__name__)


class LmStudioError(LlmProviderError):
    """Base class for privacy-safe LM Studio backend failures."""


class LmStudioUnavailableError(LmStudioError):
    """The configured LM Studio server could not be reached."""

    def __init__(self) -> None:
        super().__init__("LM Studio is unavailable")


class LmStudioTimeoutError(LmStudioError):
    """The configured LM Studio server did not respond in time."""

    def __init__(self) -> None:
        super().__init__("LM Studio request timed out")


class LmStudioHttpError(LmStudioError):
    """LM Studio returned an HTTP error without exposing its response body."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"LM Studio returned HTTP {status_code}")


class LmStudioProtocolError(LmStudioError):
    """LM Studio returned a response that does not meet the backend contract."""

    def __init__(self) -> None:
        super().__init__("LM Studio returned an invalid response")


class LmStudioImageInputError(LmStudioError):
    """A prepared image cannot safely be serialized for the model boundary."""

    def __init__(self) -> None:
        super().__init__("LM Studio received an invalid image input")


class LmStudioModelUnavailableError(LmStudioError):
    """The configured exact model ID was absent from the local server."""

    def __init__(self) -> None:
        super().__init__("configured LM Studio model is unavailable")


def derive_native_chat_url(base_url: str) -> str:
    """Derive the same-server LM Studio native Chat endpoint safely."""

    try:
        parsed = urlsplit(base_url)
    except ValueError as exc:
        raise ValueError("LM Studio base URL cannot derive native chat endpoint") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("LM Studio base URL cannot derive native chat endpoint")

    path = parsed.path.rstrip("/")
    if not path or not path.split("/")[-1].lower() == "v1":
        raise ValueError("LM Studio base URL must end in /v1 for native chat")
    prefix = path[: -len("/v1")]
    native_path = f"{prefix}/api/v1/chat" if prefix else "/api/v1/chat"
    return urlunsplit((parsed.scheme, parsed.netloc, native_path, "", ""))


class LmStudioResponder:
    """Generate text through LM Studio's OpenAI-compatible local API."""

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
        self._base_url = config.lm_studio_base_url
        self._model = config.lm_studio_model
        self._timeout_seconds = config.lm_studio_timeout_seconds
        self._temperature = config.lm_studio_temperature
        self._api_token = config.lm_studio_api_token
        self._native_chat_url = derive_native_chat_url(config.lm_studio_base_url)
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
        """Verify the configured exact model is available without inference."""

        response = await self._send("GET", "/models")
        payload = _response_object(response)
        data = payload.get("data")
        if not isinstance(data, list):
            raise LmStudioProtocolError()
        model_ids = {
            item.get("id")
            for item in data
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        if self._model not in model_ids:
            raise LmStudioModelUnavailableError()

    async def generate(self, request: ChatRequest) -> str:
        """Submit one non-streaming text or prepared-image completion request."""

        started_at = monotonic()
        decision = self._response_router.route(request)
        self._logger.debug(
            "response routed message_id=%s mode=%s reason=%s max_tokens=%s",
            request.message_id,
            decision.mode.name,
            decision.reason.name,
            decision.max_tokens,
        )
        if decision.mode is ResponseMode.CHAT_SHORT:
            if request.prepared_images:
                raise LmStudioImageInputError()
            context_selection = self._select_context(decision, request)
            self._log_context_selection(request, decision, context_selection)
            return await self._generate_native_chat(
                request,
                decision,
                started_at,
                context_selection.turns,
            )

        context_selection = self._select_context(decision, request)
        self._log_context_selection(request, decision, context_selection)
        recent_turns = context_selection.turns

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
        user_content: str | list[dict[str, object]] = current_text
        if request.prepared_images:
            try:
                user_content = await asyncio.to_thread(
                    _build_normalized_multimodal_content,
                    current_text,
                    request.prepared_images,
                    self._model_image_max_long_edge,
                    self._model_image_max_pixels,
                    self._model_total_image_pixels,
                )
            except ModelImageNormalizationError as exc:
                raise LmStudioImageInputError() from exc
        if isinstance(user_content, str):
            conversation_messages = render_chat_completion_messages(
                recent_turns,
                current_display_name=request.author_display_name,
                current_content=request.content,
                reply_context=request.reply_context,
                author_kind=request.author_kind,
            )
        else:
            conversation_messages = [
                *render_historical_chat_messages(recent_turns),
                {"role": "user", "content": user_content},
            ]
        try:
            response = await self._send(
                "POST",
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *conversation_messages,
                    ],
                    "temperature": self._temperature,
                    "max_tokens": decision.max_tokens,
                    "stream": False,
                },
            )
        except LmStudioHttpError as exc:
            self._logger.debug(
                "LM Studio generation failed message_id=%s http_status=%s exception_type=%s",
                request.message_id,
                exc.status_code,
                type(exc).__name__,
            )
            raise
        payload = _response_object(response)
        finish_reason = safe_finish_reason(payload)
        usage = payload.get("usage")
        completion_tokens = usage_value(usage, "completion_tokens")
        self._logger.debug(
            "LM Studio generation result message_id=%s response_mode=%s "
            "finish_reason=%s max_tokens=%s completion_tokens=%s",
            request.message_id,
            decision.mode.name,
            finish_reason,
            decision.max_tokens,
            completion_tokens,
        )
        if finish_reason == "length":
            self._logger.warning(
                "generation reached token ceiling message_id=%s response_mode=%s "
                "max_tokens=%s",
                request.message_id,
                decision.mode.name,
                decision.max_tokens,
            )
        content = _assistant_content(payload)
        prompt_tokens = usage_value(usage, "prompt_tokens")
        self._logger.debug(
            "LM Studio generation completed message_id=%s elapsed_ms=%s http_status=%s "
            "prompt_tokens=%s completion_tokens=%s",
            request.message_id,
            round((monotonic() - started_at) * 1000),
            response.status_code,
            prompt_tokens,
            completion_tokens,
        )
        return content

    async def _generate_native_chat(
        self,
        request: ChatRequest,
        decision: ResponseDecision,
        started_at: float,
        recent_turns: tuple[ConversationTurn, ...],
    ) -> str:
        """Generate one stateless text-only CHAT_SHORT native completion."""

        mode = decision.mode
        max_tokens = decision.max_tokens
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
            response_instruction(mode),
            active_nickname,
        )
        native_input = render_native_chat_input(
            recent_turns,
            current_display_name=request.author_display_name,
            current_content=request.content,
            reply_context=request.reply_context,
            current_author_kind=request.author_kind,
        )
        try:
            response = await self._send_url(
                "POST",
                self._native_chat_url,
                json={
                    "model": self._model,
                    "input": native_input,
                    "system_prompt": system_prompt,
                    "reasoning": "off",
                    "max_output_tokens": max_tokens,
                    "temperature": self._temperature,
                    "stream": False,
                    "store": False,
                },
            )
        except LmStudioHttpError as exc:
            self._logger.debug(
                "LM Studio native generation failed message_id=%s transport=NATIVE_CHAT "
                "http_status=%s exception_type=%s",
                request.message_id,
                exc.status_code,
                type(exc).__name__,
            )
            raise

        payload = _response_object(response)
        content = _native_content(payload)
        stats = _native_stats(payload)
        reasoning_output_tokens = stats["reasoning_output_tokens"]
        self._logger.debug(
            "LM Studio native generation completed message_id=%s mode=%s "
            "transport=NATIVE_CHAT reasoning=off elapsed_ms=%s input_tokens=%s "
            "total_output_tokens=%s reasoning_output_tokens=%s tokens_per_second=%s "
            "time_to_first_token_seconds=%s",
            request.message_id,
            mode.name,
            round((monotonic() - started_at) * 1000),
            stats["input_tokens"],
            stats["total_output_tokens"],
            reasoning_output_tokens,
            stats["tokens_per_second"],
            stats["time_to_first_token_seconds"],
        )
        if isinstance(reasoning_output_tokens, (int, float)) and reasoning_output_tokens > 0:
            self._logger.warning(
                "CHAT_SHORT requested reasoning off but server reported reasoning tokens "
                "message_id=%s reasoning_output_tokens=%s",
                request.message_id,
                reasoning_output_tokens,
            )
        return content

    def _generation_context(self, request: ChatRequest) -> ConversationGenerationContext:
        if request.generation_context is not None:
            return request.generation_context
        if self._conversation_context_provider is None:
            return ConversationGenerationContext(context_epoch=1, recent_turns=())
        provider = self._conversation_context_provider
        get_generation_context = getattr(provider, "get_generation_context", None)
        if callable(get_generation_context):
            return get_generation_context()
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
        historical_image_turn_count = sum(
            1 for turn in selection.turns if getattr(turn, "user_image_count", 0) > 0
        )
        self._logger.debug(
            "conversation context selected message_id=%s response_mode=%s "
            "available_turn_count=%s included_turn_count=%s trimmed_turn_count=%s "
            "rendered_history_chars=%s configured_history_max_chars=%s "
            "historical_image_turn_count=%s",
            request.message_id,
            decision.mode.name,
            selection.available_turn_count,
            selection.included_turn_count,
            selection.trimmed_turn_count,
            selection.rendered_history_chars,
            self._context_history_max_chars[decision.mode],
            historical_image_turn_count,
        )

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        return await self._send_url(method, f"{self._base_url}{path}", **kwargs)

    async def _send_url(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
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
                response = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise LmStudioTimeoutError() from exc
        except httpx.RequestError as exc:
            raise LmStudioUnavailableError() from exc
        if response.is_error:
            raise LmStudioHttpError(response.status_code)
        return response


def _response_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise LmStudioProtocolError() from exc
    if not isinstance(payload, dict):
        raise LmStudioProtocolError()
    return payload


def _native_content(payload: Mapping[str, Any]) -> str:
    """Extract only non-empty final message content from native output."""

    output = payload.get("output")
    if not isinstance(output, list):
        raise LmStudioProtocolError()
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    raise LmStudioProtocolError()


def _native_stats(payload: Mapping[str, Any]) -> dict[str, int | float | None]:
    stats = payload.get("stats")
    if not isinstance(stats, Mapping):
        stats = {}

    def value(name: str) -> int | float | None:
        candidate = stats.get(name)
        if isinstance(candidate, bool):
            return None
        return candidate if isinstance(candidate, (int, float)) else None

    return {
        "input_tokens": value("input_tokens"),
        "total_output_tokens": value("total_output_tokens"),
        "reasoning_output_tokens": value("reasoning_output_tokens"),
        "tokens_per_second": value("tokens_per_second"),
        "time_to_first_token_seconds": value("time_to_first_token_seconds"),
    }


def build_multimodal_content(
    text: str,
    prepared_images: tuple[PreparedImage, ...],
) -> list[dict[str, object]]:
    """Build an ordered OpenAI-compatible user content array in memory only."""

    content: list[dict[str, object]] = [{"type": "text", "text": text}]
    for image in prepared_images:
        if image.media_type not in SUPPORTED_MODEL_IMAGE_MEDIA_TYPES or not image.data:
            raise LmStudioImageInputError()
        encoded = base64.b64encode(image.data).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image.media_type};base64,{encoded}"},
            }
        )
    return content


def _build_normalized_multimodal_content(
    text: str,
    prepared_images: tuple[PreparedImage, ...],
    max_long_edge: int,
    max_image_pixels: int,
    max_total_image_pixels: int,
) -> list[dict[str, object]]:
    normalized_images = normalize_model_images(
        prepared_images,
        max_long_edge=max_long_edge,
        max_image_pixels=max_image_pixels,
        max_total_image_pixels=max_total_image_pixels,
    )
    return build_multimodal_content(text, normalized_images)


def _assistant_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LmStudioProtocolError()
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise LmStudioProtocolError()
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise LmStudioProtocolError()
    content = message.get("content")
    if not isinstance(content, str):
        raise LmStudioProtocolError()
    normalized = content.strip()
    if not normalized:
        raise LmStudioProtocolError()
    return normalized

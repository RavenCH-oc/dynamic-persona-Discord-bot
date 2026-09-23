"""Provider-neutral model backend identity, contract, and composition."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from .responder import ChatRequest

if TYPE_CHECKING:
    import logging

    import httpx

    from .config import Config
    from .conversation_context import ConversationContextProvider
    from .prompting import ActiveNicknameProvider, ActivePersonaProvider, SystemPromptBuilder
    from .response_routing import ResponseRouter


class LlmProviderKind(StrEnum):
    """Supported model providers selected once during runtime composition."""

    LM_STUDIO = "lm_studio"
    VENICE = "venice"

    @property
    def display_name(self) -> str:
        return "LM Studio" if self is self.LM_STUDIO else "Venice"


class LlmProviderError(RuntimeError):
    """Base class for privacy-safe active-provider failures."""


class LlmProvider(Protocol):
    """Minimal generation boundary shared by production providers."""

    async def preflight(self) -> None: ...

    async def generate(self, request: ChatRequest) -> str: ...


def active_model_id(config: Config) -> str:
    """Return only the configured exact model ID for the active provider."""

    if config.llm_provider is LlmProviderKind.VENICE:
        return config.venice_text_model_id
    return config.lm_studio_model


def create_llm_provider(
    config: Config,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    prompt_builder: SystemPromptBuilder | None = None,
    active_persona_provider: ActivePersonaProvider | None = None,
    active_nickname_provider: ActiveNicknameProvider | None = None,
    conversation_context_provider: ConversationContextProvider | None = None,
    response_router: ResponseRouter | None = None,
    logger: logging.Logger | None = None,
) -> LlmProvider:
    """Construct exactly one configured provider for the process lifetime."""

    common = {
        "transport": transport,
        "prompt_builder": prompt_builder,
        "active_persona_provider": active_persona_provider,
        "active_nickname_provider": active_nickname_provider,
        "conversation_context_provider": conversation_context_provider,
        "response_router": response_router,
    }
    if logger is not None:
        common["logger"] = logger
    if config.llm_provider is LlmProviderKind.LM_STUDIO:
        from .lm_studio_responder import LmStudioResponder

        return LmStudioResponder(config, **common)
    if config.llm_provider is LlmProviderKind.VENICE:
        from .venice_responder import VeniceResponder

        return VeniceResponder(config, **common)
    raise ValueError("unsupported LLM provider")


__all__ = [
    "LlmProvider",
    "LlmProviderError",
    "LlmProviderKind",
    "active_model_id",
    "create_llm_provider",
]

"""Provider-neutral prompt and response-metadata helpers."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

from .prompting import SystemPromptBuilder


def build_system_prompt(
    builder: SystemPromptBuilder,
    persona_override: str | None,
    instruction: str,
    active_nickname: str | None,
) -> str:
    """Call current and legacy injected prompt builders safely."""

    method = builder.build_system_prompt
    try:
        parameters = tuple(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        parameters = ()
    supports_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    supports_instruction = supports_var_kwargs or any(
        parameter.name == "response_instruction" for parameter in parameters
    )
    supports_nickname = supports_var_kwargs or any(
        parameter.name == "active_nickname" for parameter in parameters
    )
    kwargs: dict[str, str | None] = {}
    if supports_nickname:
        kwargs["active_nickname"] = active_nickname
    if supports_instruction:
        kwargs["response_instruction"] = instruction
    return method(persona_override, **kwargs)


def safe_finish_reason(payload: Mapping[str, object]) -> str:
    """Return a small allowlisted finish-reason category for diagnostics."""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        finish_reason = choices[0].get("finish_reason")
        if finish_reason in {"stop", "length"}:
            return str(finish_reason)
    return "other"


def usage_value(usage: object, field: str) -> int | None:
    """Read one non-sensitive integer usage value."""

    if not isinstance(usage, Mapping):
        return None
    value = usage.get(field)
    return value if isinstance(value, int) else None


__all__ = ["build_system_prompt", "safe_finish_reason", "usage_value"]

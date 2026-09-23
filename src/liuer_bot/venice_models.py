"""Strict, privacy-safe parsing of Venice model capability metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class VeniceModelRole(StrEnum):
    """The two semantic model roles within one Venice provider."""

    TEXT = "TEXT"
    VISION = "VISION"


class VeniceModelState(StrEnum):
    READY = "READY"
    MODEL_MISSING = "MODEL_MISSING"
    INCOMPATIBLE_MODEL = "INCOMPATIBLE_MODEL"


def select_venice_model_role(prepared_images: tuple[object, ...]) -> VeniceModelRole:
    """Only current prepared images choose the vision model."""

    return VeniceModelRole.VISION if prepared_images else VeniceModelRole.TEXT


@dataclass(frozen=True, slots=True, repr=False)
class VeniceModelCapabilities:
    """Only capability values required by liuer-bot's current product contract."""

    model_id: str
    supports_vision: bool | None
    supports_multiple_images: bool | None
    max_images: int | None
    offline: bool | None
    deprecated: bool | None
    text_compatible: bool

    def __repr__(self) -> str:
        return (
            "VeniceModelCapabilities("
            "model_id=<configured>, "
            f"supports_vision={self.supports_vision!r}, "
            f"supports_multiple_images={self.supports_multiple_images!r}, "
            f"max_images={self.max_images!r}, "
            f"offline={self.offline!r}, "
            f"deprecated={self.deprecated!r})"
        )


def find_venice_model(payload: Any, configured_model: str) -> VeniceModelCapabilities | None:
    """Find an exact model ID and extract explicit capability metadata only."""

    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise ValueError("invalid Venice model payload")
    for item in payload["data"]:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError("invalid Venice model entry")
        if item["id"] != configured_model:
            continue
        metadata = _merged_metadata(item)
        return VeniceModelCapabilities(
            model_id=configured_model,
            supports_vision=_optional_bool(metadata, "supportsVision", "supports_vision"),
            supports_multiple_images=_optional_bool(
                metadata,
                "supportsMultipleImages",
                "supports_multiple_images",
            ),
            max_images=_optional_positive_int(metadata, "maxImages", "max_images"),
            offline=_offline(item, metadata),
            deprecated=_optional_bool(metadata, "deprecated", "isDeprecated"),
            text_compatible=_text_compatible(metadata),
        )
    return None


def model_supports_required_images(
    model: VeniceModelCapabilities,
    *,
    required_max_images: int,
) -> bool:
    """Fail closed unless vision and current multi-image behavior are explicit."""

    return (
        model.supports_vision is True
        and model.supports_multiple_images is True
        and model.max_images is not None
        and model.max_images >= required_max_images
        and model.offline is not True
    )


def assess_venice_models(
    payload: Any,
    *,
    text_model_id: str,
    vision_model_id: str,
    required_max_images: int,
) -> VeniceModelAssessment:
    """Evaluate both exact roles from one already-received model catalog."""

    text_model = find_venice_model(payload, text_model_id)
    vision_model = (
        text_model if text_model_id == vision_model_id
        else find_venice_model(payload, vision_model_id)
    )
    text_state = (
        VeniceModelState.MODEL_MISSING if text_model is None
        else VeniceModelState.READY
        if text_model.offline is not True and text_model.text_compatible
        else VeniceModelState.INCOMPATIBLE_MODEL
    )
    vision_state = (
        VeniceModelState.MODEL_MISSING if vision_model is None
        else VeniceModelState.READY
        if model_supports_required_images(
            vision_model, required_max_images=required_max_images
        )
        else VeniceModelState.INCOMPATIBLE_MODEL
    )
    return VeniceModelAssessment(text_state, vision_state, text_model, vision_model)


@dataclass(frozen=True, slots=True, repr=False)
class VeniceModelAssessment:
    """Safe role results; model IDs and catalog bodies are absent from repr."""

    text_state: VeniceModelState
    vision_state: VeniceModelState
    text_model: VeniceModelCapabilities | None
    vision_model: VeniceModelCapabilities | None

    def __repr__(self) -> str:
        return (
            "VeniceModelAssessment("
            f"text_state={self.text_state.value!r}, "
            f"vision_state={self.vision_state.value!r})"
        )


def _merged_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    def visit(value: Mapping[str, Any], depth: int) -> None:
        metadata.update(value)
        if depth >= 3:
            return
        for nested in value.values():
            if isinstance(nested, Mapping):
                visit(nested, depth + 1)

    visit(item, 0)
    return metadata


def _optional_bool(metadata: Mapping[str, Any], *names: str) -> bool | None:
    for name in names:
        value = metadata.get(name)
        if isinstance(value, bool):
            return value
    return None


def _optional_positive_int(metadata: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = metadata.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _text_compatible(metadata: Mapping[str, Any]) -> bool:
    for name in (
        "supportsText", "supportsTextInput", "supportsTextOutput",
        "supportsChatCompletions", "supportsChat",
    ):
        if metadata.get(name) is False:
            return False
    for name in ("inputModalities", "outputModalities"):
        modalities = metadata.get(name)
        if isinstance(modalities, list) and all(isinstance(item, str) for item in modalities):
            if "text" not in {item.lower() for item in modalities}:
                return False
    return True


def _offline(item: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool | None:
    unavailable = _optional_bool(metadata, "unavailable", "isUnavailable")
    if unavailable is True:
        return True
    explicit = _optional_bool(metadata, "offline", "isOffline")
    if explicit is True:
        return True
    status = item.get("status")
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {"offline", "unavailable"}:
            return True
        if normalized in {"online", "available", "ready"}:
            return False
    if explicit is not None:
        return explicit
    return None


__all__ = [
    "VeniceModelAssessment",
    "VeniceModelCapabilities",
    "VeniceModelRole",
    "VeniceModelState",
    "assess_venice_models",
    "find_venice_model",
    "model_supports_required_images",
    "select_venice_model_role",
]

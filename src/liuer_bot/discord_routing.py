"""Pure single-chat-channel and addressing routing policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True, slots=True)
class RoutingInput:
    """Minimal transport metadata consumed by the Phase 4A-A policy."""

    guild_id: int | None
    channel_id: int
    author_id: int
    author_is_bot: bool
    has_text_content: bool
    has_supported_image: bool
    is_thread: bool = False
    is_addressed: bool = False


class RoutingDecision(Enum):
    """The complete single-channel routing decision set."""

    RESPOND = auto()
    IGNORE_AUTHOR_BOT = auto()
    IGNORE_DM = auto()
    IGNORE_THREAD = auto()
    IGNORE_CHANNEL = auto()
    IGNORE_NOT_ADDRESSED = auto()


def route_input(
    routing_input: RoutingInput,
    *,
    chat_channel_id: int,
) -> RoutingDecision:
    """Route only explicitly addressed messages in the configured chat stream."""

    if routing_input.author_is_bot:
        return RoutingDecision.IGNORE_AUTHOR_BOT
    if routing_input.guild_id is None:
        return RoutingDecision.IGNORE_DM
    if routing_input.is_thread:
        return RoutingDecision.IGNORE_THREAD
    if routing_input.channel_id != chat_channel_id:
        return RoutingDecision.IGNORE_CHANNEL
    if not routing_input.is_addressed:
        return RoutingDecision.IGNORE_NOT_ADDRESSED
    return RoutingDecision.RESPOND

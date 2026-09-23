from __future__ import annotations

from liuer_bot.discord_routing import RoutingDecision, RoutingInput, route_input


def _routing_input(**overrides: object) -> RoutingInput:
    values: dict[str, object] = {
        "guild_id": 20,
        "channel_id": 123,
        "author_id": 10,
        "author_is_bot": False,
        "has_text_content": True,
        "has_supported_image": False,
        "is_thread": False,
        "is_addressed": True,
    }
    values.update(overrides)
    return RoutingInput(**values)  # type: ignore[arg-type]


def _route(**overrides: object) -> RoutingDecision:
    return route_input(_routing_input(**overrides), chat_channel_id=123)


def test_addressed_human_text_in_chat_channel_responds() -> None:
    assert _route() is RoutingDecision.RESPOND


def test_addressed_empty_remainder_is_allowed() -> None:
    assert _route(has_text_content=False) is RoutingDecision.RESPOND


def test_unaddressed_human_text_in_chat_channel_is_ignored() -> None:
    assert _route(is_addressed=False) is RoutingDecision.IGNORE_NOT_ADDRESSED


def test_unaddressed_supported_image_only_is_ignored() -> None:
    assert (
        _route(is_addressed=False, has_text_content=False, has_supported_image=True)
        is RoutingDecision.IGNORE_NOT_ADDRESSED
    )


def test_addressed_text_and_supported_image_responds_once() -> None:
    assert _route(has_supported_image=True) is RoutingDecision.RESPOND


def test_other_channel_is_ignored_even_when_addressed() -> None:
    assert _route(channel_id=456) is RoutingDecision.IGNORE_CHANNEL


def test_dm_is_ignored() -> None:
    assert _route(guild_id=None) is RoutingDecision.IGNORE_DM


def test_thread_under_chat_channel_is_ignored() -> None:
    assert _route(is_thread=True) is RoutingDecision.IGNORE_THREAD


def test_bot_author_is_ignored_even_when_addressed() -> None:
    assert _route(author_is_bot=True) is RoutingDecision.IGNORE_AUTHOR_BOT

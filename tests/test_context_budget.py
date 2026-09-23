from __future__ import annotations

from datetime import UTC, datetime

import pytest

from liuer_bot.conversation_context import (
    render_historical_context_text,
    select_recent_turns_with_budget,
)
from liuer_bot.conversation_models import ConversationTurn


def _turn(
    turn_id: int,
    user_content: str,
    assistant_content: str,
    *,
    image_count: int = 0,
    display_name: str = "User",
) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        chat_channel_id=123,
        user_message_id=turn_id,
        user_author_id=turn_id + 100,
        user_display_name=display_name,
        user_content=user_content,
        user_image_count=image_count,
        assistant_content=assistant_content,
        created_at_utc=datetime.now(UTC),
    )


def test_all_turns_fit_and_rendered_character_cost_is_exact() -> None:
    turns = (
        _turn(1, "A-user", "A-assistant"),
        _turn(2, "B-user", "B-assistant"),
        _turn(3, "C-user", "C-assistant"),
    )
    expected_chars = len(render_historical_context_text(turns))

    selection = select_recent_turns_with_budget(turns, max_chars=expected_chars)

    assert selection.turns == turns
    assert selection.available_turn_count == 3
    assert selection.included_turn_count == 3
    assert selection.trimmed_turn_count == 0
    assert selection.rendered_history_chars == expected_chars


def test_exact_boundary_includes_newest_complete_turns() -> None:
    turns = (
        _turn(1, "old", "old answer"),
        _turn(2, "middle", "middle answer"),
        _turn(3, "new", "new answer"),
    )
    expected = turns[1:]
    budget = len(render_historical_context_text(expected))

    selection = select_recent_turns_with_budget(turns, max_chars=budget)

    assert selection.turns == expected
    assert selection.rendered_history_chars == budget


def test_budget_trims_oldest_complete_turns_only() -> None:
    turns = (
        _turn(1, "A", "A-answer"),
        _turn(2, "B", "B-answer"),
        _turn(3, "C", "C-answer"),
        _turn(4, "D", "D-answer"),
    )
    expected = turns[2:]

    selection = select_recent_turns_with_budget(
        turns,
        max_chars=len(render_historical_context_text(expected)),
    )

    assert selection.turns == expected
    assert selection.included_turn_count == 2
    assert selection.trimmed_turn_count == 2


def test_budget_stops_at_first_older_turn_that_does_not_fit() -> None:
    turns = (
        _turn(1, "tiny old", "tiny answer"),
        _turn(2, "C" * 100, "C answer"),
        _turn(3, "D", "D answer"),
    )
    newest = turns[-1:]

    selection = select_recent_turns_with_budget(
        turns,
        max_chars=len(render_historical_context_text(newest)),
    )

    assert selection.turns == newest
    assert "tiny old" not in repr(selection)


def test_oversized_newest_turn_is_excluded_without_internal_truncation() -> None:
    newest = _turn(2, "new user", "N" * 200)
    turns = (_turn(1, "old", "old answer"), newest)

    selection = select_recent_turns_with_budget(
        turns,
        max_chars=len(render_historical_context_text((newest,))) - 1,
    )

    assert selection.turns == ()
    assert selection.included_turn_count == 0
    assert selection.trimmed_turn_count == 2
    assert selection.rendered_history_chars == 0


def test_zero_budget_keeps_no_completed_history() -> None:
    selection = select_recent_turns_with_budget(
        (_turn(1, "user", "assistant"),),
        max_chars=0,
    )

    assert selection.turns == ()
    assert selection.rendered_history_chars == 0


def test_historical_image_marker_is_part_of_character_cost() -> None:
    text_only = _turn(1, "same user", "same assistant")
    with_images = _turn(1, "same user", "same assistant", image_count=2)

    text_cost = len(render_historical_context_text((text_only,)))
    image_cost = len(render_historical_context_text((with_images,)))

    assert image_cost > text_cost
    assert "2 張圖片" in render_historical_context_text((with_images,))


@pytest.mark.parametrize("value", [-1, True, 1.5, "100"])
def test_budget_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        select_recent_turns_with_budget((), max_chars=value)  # type: ignore[arg-type]


def test_selection_repr_redacts_selected_turn_content_and_names() -> None:
    private_turn = _turn(
        1,
        "phase4b-private-history-user",
        "phase4b-private-history-assistant",
        display_name="phase4b-private-display-name",
    )

    rendered = repr(
        select_recent_turns_with_budget(
            (private_turn,),
            max_chars=len(render_historical_context_text((private_turn,))),
        )
    )

    assert "phase4b-private-history-user" not in rendered
    assert "phase4b-private-history-assistant" not in rendered
    assert "phase4b-private-display-name" not in rendered
    assert "turns=<redacted>" in rendered

from __future__ import annotations

import pytest

from liuer_bot.addressing import (
    EMPTY_ADDRESSED_CONTENT,
    AddressKind,
    parse_addressed_message,
)


@pytest.mark.parametrize(
    ("content", "expected_content"),
    [
        ("六耳 你好", "你好"),
        ("  六耳你好", "你好"),
        ("六耳，你在幹嘛？", "你在幹嘛？"),
        ("六耳, 你好", "你好"),
        ("六耳：幫我看看", "幫我看看"),
        ("六耳: 幫我看看", "幫我看看"),
    ],
)
def test_primary_name_accepts_leading_whitespace_and_common_separators(
    content: str,
    expected_content: str,
) -> None:
    result = parse_addressed_message(content)

    assert result.is_addressed is True
    assert result.address_kind is AddressKind.PRIMARY_NAME
    assert result.conversational_content == expected_content


def test_only_one_leading_primary_name_is_removed() -> None:
    result = parse_addressed_message("六耳 你覺得六耳這名字如何？")

    assert result.conversational_content == "你覺得六耳這名字如何？"


def test_later_primary_name_does_not_trigger() -> None:
    result = parse_addressed_message("我覺得六耳很有趣")

    assert result.is_addressed is False
    assert result.address_kind is AddressKind.NONE


def test_empty_address_uses_minimal_nonempty_content() -> None:
    primary = parse_addressed_message("六耳")
    mention = parse_addressed_message("<@999>", bot_user_id=999)

    assert EMPTY_ADDRESSED_CONTENT.strip()
    for result in (primary, mention):
        assert result.is_addressed is True
        assert result.conversational_content == EMPTY_ADDRESSED_CONTENT


def test_current_bot_user_mention_is_a_supported_leading_form() -> None:
    result = parse_addressed_message("<@!999> 你好", bot_user_id=999)

    assert result.is_addressed is True
    assert result.address_kind is AddressKind.BOT_MENTION
    assert result.conversational_content == "你好"


def test_other_user_mention_and_late_bot_mention_do_not_address() -> None:
    other = parse_addressed_message("<@998> 你好", bot_user_id=999)
    later = parse_addressed_message("你問 <@999> 看看", bot_user_id=999)

    assert other.is_addressed is False
    assert later.is_addressed is False


def test_addressing_result_repr_redacts_conversational_content() -> None:
    private = "phase4aa-addressing-private-content"

    assert private not in repr(parse_addressed_message(f"六耳 {private}"))


@pytest.mark.parametrize(
    ("nickname", "content", "expected_kind", "expected_cleaned"),
    [
        ("小六", "小六 你好", AddressKind.ACTIVE_NICKNAME, "你好"),
        ("六耳醬", "六耳醬，你好", AddressKind.ACTIVE_NICKNAME, "你好"),
        ("六耳醬", "六耳 你好", AddressKind.PRIMARY_NAME, "你好"),
        ("六", "六耳 你好", AddressKind.PRIMARY_NAME, "你好"),
        ("六", "六 你好", AddressKind.ACTIVE_NICKNAME, "你好"),
    ],
)
def test_active_nickname_uses_longest_literal_leading_match(
    nickname: str,
    content: str,
    expected_kind: AddressKind,
    expected_cleaned: str,
) -> None:
    result = parse_addressed_message(content, active_nickname=nickname)

    assert result.is_addressed is True
    assert result.address_kind is expected_kind
    assert result.conversational_content == expected_cleaned


def test_nickname_is_exact_leading_only_and_case_sensitive() -> None:
    later = parse_addressed_message("我覺得小六很可愛", active_nickname="小六")
    wrong_case = parse_addressed_message("liuer hello", active_nickname="Liuer")
    exact = parse_addressed_message("Liuer hello", active_nickname="Liuer")

    assert later.is_addressed is False
    assert wrong_case.is_addressed is False
    assert exact.is_addressed is True
    assert exact.address_kind is AddressKind.ACTIVE_NICKNAME

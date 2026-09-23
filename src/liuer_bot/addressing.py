"""Pure parsing boundary for explicit liuer-bot message addressing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

PRIMARY_CALL_NAME = "六耳"
EMPTY_ADDRESSED_CONTENT = "…"
_DISCORD_USER_MENTION = re.compile(r"<@!?(?P<user_id>[0-9]+)>")


class AddressKind(StrEnum):
    """The supported addressing forms, ready for a future nickname alias."""

    NONE = "none"
    NOT_ADDRESSED = "none"
    PRIMARY_NAME = "primary_name"
    ACTIVE_NICKNAME = "active_nickname"
    BOT_MENTION = "bot_mention"


@dataclass(frozen=True, slots=True, repr=False)
class AddressingResult:
    """Immutable addressing result with redacted representation."""

    is_addressed: bool
    conversational_content: str
    address_kind: AddressKind

    def __repr__(self) -> str:
        return (
            "AddressingResult("
            f"is_addressed={self.is_addressed!r}, "
            f"address_kind={self.address_kind.value!r}, "
            "conversational_content=<redacted>)"
        )


def parse_addressed_message(
    content: str,
    *,
    bot_user_id: int | None = None,
    primary_name: str = PRIMARY_CALL_NAME,
    active_nickname: str | None = None,
) -> AddressingResult:
    """Parse one leading primary name, nickname, or current-Bot mention.

    Only one leading address is removed. The remainder is otherwise preserved;
    separator whitespace and punctuation immediately following the address are
    discarded so the model sees the conversational message, not transport
    addressing metadata. A fully stripped remainder uses one minimal marker so
    text-only and multimodal provider payloads are never empty.
    """

    if not isinstance(content, str):
        return _not_addressed()

    normalized = content.lstrip()
    name_candidates: list[tuple[str, AddressKind, int]] = []
    if isinstance(primary_name, str) and primary_name:
        name_candidates.append((primary_name, AddressKind.PRIMARY_NAME, 0))
    if isinstance(active_nickname, str) and active_nickname:
        name_candidates.append((active_nickname, AddressKind.ACTIVE_NICKNAME, 1))
    name_candidates.sort(key=lambda item: (-len(item[0]), item[2]))
    for call_name, address_kind, _priority in name_candidates:
        if normalized.startswith(call_name):
            return _addressed(normalized[len(call_name) :], address_kind)

    if bot_user_id is not None and isinstance(bot_user_id, int) and bot_user_id > 0:
        mention = _DISCORD_USER_MENTION.match(normalized)
        if mention is not None and int(mention.group("user_id")) == bot_user_id:
            return _addressed(
                normalized[mention.end() :],
                AddressKind.BOT_MENTION,
            )

    return _not_addressed()


def _addressed(remainder: str, address_kind: AddressKind) -> AddressingResult:
    conversational_content = _strip_adjacent_separators(remainder)
    return AddressingResult(
        is_addressed=True,
        conversational_content=conversational_content or EMPTY_ADDRESSED_CONTENT,
        address_kind=address_kind,
    )


def _not_addressed() -> AddressingResult:
    return AddressingResult(
        is_addressed=False,
        conversational_content="",
        address_kind=AddressKind.NONE,
    )


def _strip_adjacent_separators(value: str) -> str:
    index = 0
    while index < len(value):
        character = value[index]
        if character.isspace() or unicodedata.category(character).startswith("P"):
            index += 1
            continue
        break
    return value[index:]


__all__ = [
    "AddressKind",
    "AddressingResult",
    "EMPTY_ADDRESSED_CONTENT",
    "PRIMARY_CALL_NAME",
    "parse_addressed_message",
]

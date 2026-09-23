"""Pure, privacy-aware rendering of recent conversation context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .conversation_models import AuthorKind, ConversationTurn, sanitize_display_name
from .reply_context import ReplyContext


class ConversationContextProvider(Protocol):
    """Synchronous in-memory boundary for recent completed turns."""

    def get_generation_context(self) -> ConversationGenerationContext:
        """Return one atomic active-epoch and recent-turn snapshot."""

    def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
        """Return the current bounded snapshot without external I/O."""


@dataclass(frozen=True, slots=True, repr=False)
class ConversationGenerationContext:
    """Immutable epoch and history captured at serialized generation start."""

    context_epoch: int
    recent_turns: tuple[ConversationTurn, ...]

    def __repr__(self) -> str:
        return (
            "ConversationGenerationContext("
            f"context_epoch={self.context_epoch!r}, recent_turns=<redacted>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ContextSelection:
    """A privacy-safe result of deterministic recent-history selection."""

    turns: tuple[ConversationTurn, ...]
    available_turn_count: int
    included_turn_count: int
    trimmed_turn_count: int
    rendered_history_chars: int

    def __repr__(self) -> str:
        return (
            "ContextSelection("
            "turns=<redacted>, "
            f"available_turn_count={self.available_turn_count!r}, "
            f"included_turn_count={self.included_turn_count!r}, "
            f"trimmed_turn_count={self.trimmed_turn_count!r}, "
            f"rendered_history_chars={self.rendered_history_chars!r})"
        )


def render_current_user_text(
    display_name: str,
    content: str,
    reply_context: ReplyContext | None = None,
    *,
    author_kind: AuthorKind = AuthorKind.HUMAN,
) -> str:
    """Render the current speaker label and already-cleaned user content."""

    if reply_context is not None:
        current_label = (
            _current_user_label(display_name)
            if author_kind is AuthorKind.HUMAN
            else _current_speaker_label(display_name, author_kind)
        )
        return "\n".join(
            (
                render_reply_context_text(reply_context),
                current_label,
                content,
            )
        )
    label = _current_speaker_label(display_name, author_kind)
    return f"{label}\n{content}" if content else label


def render_reply_context_text(reply_context: ReplyContext) -> str:
    """Render one quoted replied-to message without promoting it to policy."""

    if not reply_context.is_available:
        return "[回覆上下文]\n[被回覆的訊息目前無法取得]"
    lines = ["[回覆上下文]", _reply_label(reply_context)]
    if reply_context.content:
        lines.append(reply_context.content)
    if reply_context.image_count > 0:
        lines.append(
            f"[被回覆的訊息曾附帶 {reply_context.image_count} 張圖片；"
            "圖片內容目前未重新提供]"
        )
    if not reply_context.content and reply_context.image_count == 0:
        lines.append("[被回覆的訊息沒有可用文字內容]")
    return "\n".join(lines)


def render_historical_user_text(turn: ConversationTurn) -> str:
    """Render one historical user turn without exposing transport metadata."""

    lines = [_historical_speaker_label(turn.user_display_name, turn.author_kind)]
    if turn.user_content:
        lines.append(turn.user_content)
    if turn.user_image_count > 0:
        lines.append(render_historical_image_marker(turn.user_image_count))
    return "\n".join(lines)


def render_historical_turn_text(turn: ConversationTurn) -> str:
    """Render one complete historical turn for canonical character costing."""

    return "\n".join((render_historical_user_text(turn), "六耳：", turn.assistant_content))


def render_historical_context_text(
    turns: tuple[ConversationTurn, ...],
) -> str:
    """Render selected complete turns with deterministic inter-turn separators."""

    return "\n".join(render_historical_turn_text(turn) for turn in turns)


def render_historical_image_marker(image_count: int) -> str:
    """Describe unavailable historical images without retaining their contents."""

    return f"[歷史附件：此輪曾附帶 {image_count} 張圖片；圖片內容目前未重新提供]"


def render_historical_chat_messages(
    recent_turns: tuple[ConversationTurn, ...],
) -> list[dict[str, str]]:
    """Build oldest-to-newest alternating Chat Completions history messages."""

    messages: list[dict[str, str]] = []
    for turn in recent_turns:
        messages.append({"role": "user", "content": render_historical_user_text(turn)})
        messages.append({"role": "assistant", "content": turn.assistant_content})
    return messages


def select_recent_turns_with_budget(
    turns: tuple[ConversationTurn, ...],
    *,
    max_chars: int,
) -> ContextSelection:
    """Select the newest contiguous complete suffix that fits ``max_chars``."""

    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0:
        raise ValueError("history character budget must be a non-negative integer")

    available_turns = tuple(turns)
    selected_newest_first: list[ConversationTurn] = []
    rendered_history_chars = 0
    for turn in reversed(available_turns):
        candidate_newest_first = [*selected_newest_first, turn]
        candidate = tuple(reversed(candidate_newest_first))
        candidate_chars = len(render_historical_context_text(candidate))
        if candidate_chars > max_chars:
            break
        selected_newest_first = candidate_newest_first
        rendered_history_chars = candidate_chars

    selected = tuple(reversed(selected_newest_first))
    return ContextSelection(
        turns=selected,
        available_turn_count=len(available_turns),
        included_turn_count=len(selected),
        trimmed_turn_count=len(available_turns) - len(selected),
        rendered_history_chars=rendered_history_chars,
    )


def render_chat_completion_messages(
    recent_turns: tuple[ConversationTurn, ...],
    *,
    current_display_name: str,
    current_content: str,
    reply_context: ReplyContext | None = None,
    author_kind: AuthorKind = AuthorKind.HUMAN,
) -> list[dict[str, str]]:
    """Build history followed by exactly one current user message."""

    return [
        *render_historical_chat_messages(recent_turns),
        {
            "role": "user",
            "content": render_current_user_text(
                current_display_name,
                current_content,
                reply_context,
                author_kind=author_kind,
            ),
        },
    ]


def render_native_chat_input(
    recent_turns: tuple[ConversationTurn, ...],
    *,
    current_display_name: str,
    current_content: str,
    reply_context: ReplyContext | None = None,
    current_author_kind: AuthorKind = AuthorKind.HUMAN,
) -> str:
    """Build the compact stateless native CHAT_SHORT transcript."""

    current_label = _native_speaker_label(current_display_name, current_author_kind)
    current_lines = [current_label]
    if current_content:
        current_lines.append(current_content)
    if not recent_turns and reply_context is None:
        return "\n".join(current_lines)

    lines: list[str] = []
    if recent_turns:
        lines.append("[RECENT CONVERSATION]")
        for turn in recent_turns:
            lines.append(_render_native_historical_turn_text(turn))
    if reply_context is not None:
        lines.extend(("[REPLIED MESSAGE]", _native_reply_text(reply_context)))
    lines.extend(("[CURRENT MESSAGE]", *current_lines))
    return "\n".join(lines)


def _quoted_display_name(value: object) -> str:
    return json.dumps(sanitize_display_name(value), ensure_ascii=False)


def _discord_user_label(display_name: object) -> str:
    return f"[使用者：{_quoted_display_name(display_name)}]"


def _current_user_label(display_name: object) -> str:
    return f"[目前使用者：{_quoted_display_name(display_name)}]"


def _historical_speaker_label(display_name: object, author_kind: AuthorKind) -> str:
    if author_kind is AuthorKind.BOT:
        return f"[Bot：{_quoted_display_name(display_name)}]"
    return _discord_user_label(display_name)


def _current_speaker_label(display_name: object, author_kind: AuthorKind) -> str:
    if author_kind is AuthorKind.BOT:
        return f"[Bot：{_quoted_display_name(display_name)}]"
    return _discord_user_label(display_name)


def _reply_label(reply_context: ReplyContext) -> str:
    if reply_context.is_current_bot:
        return "[六耳]"
    if reply_context.author_kind is AuthorKind.BOT or reply_context.author_is_bot:
        return f"[機器人：{_quoted_display_name(reply_context.author_display_name)}]"
    return _discord_user_label(reply_context.author_display_name)


def _native_reply_text(reply_context: ReplyContext) -> str:
    if not reply_context.is_available:
        return "[被回覆的訊息目前無法取得]"
    lines = []
    if reply_context.is_current_bot:
        lines.append("六耳：")
    elif reply_context.author_kind is AuthorKind.BOT or reply_context.author_is_bot:
        lines.append(f"Bot {_quoted_display_name(reply_context.author_display_name)}:")
    else:
        lines.append(_native_user_label(reply_context.author_display_name))
    if reply_context.content:
        lines.append(reply_context.content)
    if reply_context.image_count > 0:
        lines.append(
            f"[被回覆的訊息曾附帶 {reply_context.image_count} 張圖片；"
            "圖片內容目前未重新提供]"
        )
    if not reply_context.content and reply_context.image_count == 0:
        lines.append("[被回覆的訊息沒有可用文字內容]")
    return "\n".join(lines)


def _native_user_label(display_name: object) -> str:
    return f"User {_quoted_display_name(display_name)}:"


def _native_speaker_label(display_name: object, author_kind: AuthorKind) -> str:
    if author_kind is AuthorKind.BOT:
        return f"Bot {_quoted_display_name(display_name)}:"
    return _native_user_label(display_name)


def _render_native_historical_turn_text(turn: ConversationTurn) -> str:
    lines = [_native_speaker_label(turn.user_display_name, turn.author_kind)]
    if turn.user_content:
        lines.append(turn.user_content)
    if turn.user_image_count > 0:
        lines.append(render_historical_image_marker(turn.user_image_count))
    lines.extend(("六耳：", turn.assistant_content))
    return "\n".join(lines)


__all__ = [
    "ContextSelection",
    "ConversationContextProvider",
    "ConversationGenerationContext",
    "render_chat_completion_messages",
    "render_historical_context_text",
    "render_current_user_text",
    "render_historical_chat_messages",
    "render_historical_image_marker",
    "render_historical_turn_text",
    "render_historical_user_text",
    "render_native_chat_input",
    "render_reply_context_text",
    "select_recent_turns_with_budget",
]

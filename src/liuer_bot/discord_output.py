"""Pure helpers for delivering Discord-safe generated text."""

from __future__ import annotations

import re
from dataclasses import dataclass

DISCORD_MESSAGE_MAX_CHARS = 2000

_PARAGRAPH_BOUNDARY_RE = re.compile(r"\r?\n[ \t]*\r?\n")
_NEWLINE_BOUNDARY_RE = re.compile(r"\r?\n")
_WHITESPACE_BOUNDARY_RE = re.compile(r"\s+", re.UNICODE)
_FENCE_LINE_RE = re.compile(r"^[ \t]{0,3}`{3,}(?P<info>[^\r\n]*)$")
_SENTENCE_ENDINGS = frozenset("。！？!?；;")
_CODE_FENCE_CLOSE = "\n```"


@dataclass(frozen=True, slots=True)
class _FenceState:
    inside: bool
    info: str = ""


def split_discord_response(
    text: str,
    *,
    max_chars: int = DISCORD_MESSAGE_MAX_CHARS,
) -> tuple[str, ...]:
    """Split completed model output into deterministic Discord-safe chunks.

    Boundary selection prefers semantic boundaries that are close to the usable
    limit. Standard backtick fences are reopened and closed around chunks when
    a split occurs inside a fenced block. The input is otherwise kept intact;
    only outer whitespace is normalized to avoid whitespace-only chunks.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    normalized = text
    if not normalized.strip():
        return ()
    if len(normalized) > max_chars:
        if not normalized[:max_chars].strip():
            normalized = normalized.lstrip()
        if not normalized[-max_chars:].strip():
            normalized = normalized.rstrip()
    if len(normalized) <= max_chars:
        return (normalized,)

    chunks: list[str] = []
    position = 0
    while position < len(normalized):
        start_state = _fence_state_at(normalized, position)
        prefix = _continuation_prefix(start_state.info, max_chars) if start_state.inside else ""
        end = _payload_end(normalized, position, max_chars, len(prefix))
        end = _choose_boundary(normalized, position, end)
        if end <= position:
            end = min(len(normalized), position + max(1, max_chars - len(prefix)))

        end_state = _fence_state_at(normalized, end)
        suffix = _CODE_FENCE_CLOSE if end_state.inside else ""
        if len(prefix) + (end - position) + len(suffix) > max_chars:
            end = _shrink_for_fence(normalized, position, end, max_chars, len(prefix))
            end_state = _fence_state_at(normalized, end)
            suffix = _CODE_FENCE_CLOSE if end_state.inside else ""

        payload = normalized[position:end]
        chunk = f"{prefix}{payload}{suffix}"
        if not chunk or not chunk.strip():
            end = min(len(normalized), position + max_chars)
            payload = normalized[position:end]
            chunk = payload
        if len(chunk) > max_chars:
            raise AssertionError("Discord response splitter produced an oversized chunk")

        chunks.append(chunk)
        position = end

    return tuple(chunks)


def _payload_end(text: str, start: int, max_chars: int, prefix_length: int) -> int:
    """Find a payload limit while reserving room for a continuation fence."""

    if prefix_length >= max_chars:
        prefix_length = 0
    end = min(len(text), start + max(1, max_chars - prefix_length))
    for _ in range(8):
        state = _fence_state_at(text, end)
        suffix_length = len(_CODE_FENCE_CLOSE) if state.inside else 0
        available = max_chars - prefix_length - suffix_length
        if available <= 0:
            available = max_chars - prefix_length
        next_end = min(len(text), start + max(1, available))
        if next_end == end:
            break
        end = next_end
    return end


def _shrink_for_fence(
    text: str,
    start: int,
    end: int,
    max_chars: int,
    prefix_length: int,
) -> int:
    """Make a selected boundary fit after accounting for a closing fence."""

    while end > start:
        state = _fence_state_at(text, end)
        suffix_length = len(_CODE_FENCE_CLOSE) if state.inside else 0
        if prefix_length + end - start + suffix_length <= max_chars:
            return end
        end -= 1
    return min(len(text), start + max(1, max_chars - prefix_length))


def _choose_boundary(text: str, start: int, maximum: int) -> int:
    if maximum >= len(text):
        return len(text)
    fragment = text[start:maximum]
    if not fragment:
        return maximum

    near_start = start + max(1, (maximum - start) * 3 // 4)
    boundary_groups = (
        [start + match.end() for match in _PARAGRAPH_BOUNDARY_RE.finditer(fragment)],
        [start + match.end() for match in _NEWLINE_BOUNDARY_RE.finditer(fragment)],
        [
            start + index + 1
            for index, character in enumerate(fragment)
            if character in _SENTENCE_ENDINGS
        ],
        [start + match.end() for match in _WHITESPACE_BOUNDARY_RE.finditer(fragment)],
    )
    for candidates in boundary_groups:
        valid = [
            candidate
            for candidate in candidates
            if start < candidate <= maximum
            and candidate >= near_start
            and text[start:candidate].strip()
        ]
        if valid:
            return max(valid)

    candidates = [
        candidate
        for group in boundary_groups
        for candidate in group
        if start < candidate <= maximum and text[start:candidate].strip()
    ]
    return max(candidates, default=maximum)


def _continuation_prefix(info: str, max_chars: int) -> str:
    prefix = f"```{info}\n"
    return prefix if len(prefix) < max_chars else ""


def _fence_state_at(text: str, position: int) -> _FenceState:
    inside = False
    info = ""
    for line in text[:position].splitlines():
        match = _FENCE_LINE_RE.fullmatch(line.rstrip("\r"))
        if match is None:
            continue
        line_info = match.group("info").strip()
        if inside:
            if not line_info:
                inside = False
                info = ""
        else:
            inside = True
            info = line_info
    return _FenceState(inside=inside, info=info)


__all__ = ["DISCORD_MESSAGE_MAX_CHARS", "split_discord_response"]

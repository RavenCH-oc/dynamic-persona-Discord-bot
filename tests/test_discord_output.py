import pytest

from liuer_bot.discord_output import DISCORD_MESSAGE_MAX_CHARS, split_discord_response


def test_splitter_prefers_a_nearby_paragraph_boundary() -> None:
    text = "a" * 1800 + "\n\n" + "b" * 300

    chunks = split_discord_response(text)

    assert chunks[0].endswith("\n\n")
    assert len(chunks[0]) == 1802
    assert "".join(chunks) == text
    assert all(len(chunk) <= DISCORD_MESSAGE_MAX_CHARS for chunk in chunks)


@pytest.mark.parametrize(
    ("text", "boundary"),
    [
        ("a" * 1800 + "\n" + "b" * 300, "\n"),
        ("a" * 1800 + "。" + "b" * 300, "。"),
        ("a" * 1800 + " " + "b" * 300, " "),
    ],
)
def test_splitter_uses_progressively_less_semantic_boundaries(text: str, boundary: str) -> None:
    chunks = split_discord_response(text)

    assert chunks[0].endswith(boundary)
    assert "".join(chunks) == text
    assert all(len(chunk) <= DISCORD_MESSAGE_MAX_CHARS for chunk in chunks)


def test_splitter_hard_splits_only_when_no_boundary_exists() -> None:
    text = "x" * 2200

    chunks = split_discord_response(text)

    assert chunks == ("x" * 2000, "x" * 200)


def test_splitter_does_not_emit_whitespace_only_chunks() -> None:
    assert split_discord_response(" \n\t ") == ()


def test_splitter_preserves_fenced_code_blocks_across_chunks() -> None:
    text = "```python\n" + "\n".join(f"print({index})" for index in range(30)) + "\n```\n完成"

    chunks = split_discord_response(text, max_chars=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks)
    assert chunks[0].startswith("```python\n")
    assert any(chunk.startswith("```python\n") for chunk in chunks[1:])
    assert "print(0)" in "".join(chunks)
    assert chunks[-1].endswith("完成")


def test_splitter_keeps_short_fenced_code_block_unchanged() -> None:
    text = "```python\nprint('ok')\n```"

    assert split_discord_response(text) == (text,)


@pytest.mark.parametrize("max_chars", [0, -1, True])
def test_splitter_rejects_invalid_limits(max_chars: int) -> None:
    with pytest.raises(ValueError, match="max_chars"):
        split_discord_response("text", max_chars=max_chars)

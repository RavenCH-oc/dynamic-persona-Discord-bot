from __future__ import annotations

from dataclasses import dataclass

import pytest

from liuer_bot.attachment_metadata import (
    classify_image_attachment,
    classify_image_attachments,
)
from liuer_bot.responder import ImageAttachment


@dataclass
class FakeAttachment:
    id: int = 1
    content_type: str | None = "image/png"
    filename: str = "image.bin"
    size: int = 100


@pytest.mark.parametrize(
    ("content_type", "filename", "expected_media_type"),
    [
        ("image/png", "image.bin", "image/png"),
        ("IMAGE/JPEG", "image.bin", "image/jpeg"),
        ("image/jpg; charset=binary", "image.bin", "image/jpeg"),
        ("image/webp; version=1", "image.bin", "image/webp"),
        ("image/gif", "image.bin", "image/gif"),
    ],
)
def test_supported_content_types_are_normalized(
    content_type: str,
    filename: str,
    expected_media_type: str,
) -> None:
    result = classify_image_attachment(FakeAttachment(content_type=content_type, filename=filename))

    assert result == ImageAttachment(1, expected_media_type, 100)


@pytest.mark.parametrize(
    ("filename", "expected_media_type"),
    [
        ("image.PNG", "image/png"),
        ("image.jpg", "image/jpeg"),
        ("image.jpeg", "image/jpeg"),
        ("image.webp", "image/webp"),
        ("image.GIF", "image/gif"),
    ],
)
def test_filename_extension_is_fallback_when_content_type_is_missing(
    filename: str,
    expected_media_type: str,
) -> None:
    result = classify_image_attachment(FakeAttachment(content_type=None, filename=filename))

    assert result == ImageAttachment(1, expected_media_type, 100)


@pytest.mark.parametrize(
    "attachment",
    [
        FakeAttachment(content_type="application/pdf", filename="report.png"),
        FakeAttachment(content_type="text/plain", filename="notes.txt"),
        FakeAttachment(content_type="application/zip", filename="archive.zip"),
        FakeAttachment(content_type="video/mp4", filename="clip.mp4"),
    ],
)
def test_unsupported_or_explicit_non_image_types_are_ignored(attachment: FakeAttachment) -> None:
    assert classify_image_attachment(attachment) is None


def test_filename_fallback_does_not_use_filename_when_content_type_is_blank_image_unknown() -> None:
    result = classify_image_attachment(FakeAttachment(content_type="   ", filename="image.PNG"))

    assert result == ImageAttachment(1, "image/png", 100)


def test_multiple_supported_attachments_preserve_order_and_drop_unsupported() -> None:
    attachments = [
        FakeAttachment(id=1, content_type="image/png", filename="one.png", size=1),
        FakeAttachment(id=2, content_type="application/pdf", filename="two.png", size=2),
        FakeAttachment(id=3, content_type=None, filename="three.GIF", size=3),
    ]

    assert classify_image_attachments(attachments) == (
        ImageAttachment(1, "image/png", 1),
        ImageAttachment(3, "image/gif", 3),
    )


def test_image_metadata_dto_contains_no_filename_url_or_bytes() -> None:
    result = classify_image_attachment(
        FakeAttachment(
            id=7,
            content_type="image/png",
            filename="private-name.png",
            size=777,
        )
    )

    assert result is not None
    rendered = repr(result)
    assert "private-name.png" not in rendered
    assert "https://" not in rendered
    assert not hasattr(result, "data")
    assert not hasattr(result, "url")

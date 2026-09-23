from __future__ import annotations

import io

import pytest
from PIL import Image

from liuer_bot.image_preparation import (
    ImagePixelLimitError,
    ImagePreparationError,
    ImageTooLargeError,
    PreparedImage,
    UnsupportedImageFormatError,
    prepare_image,
)


def _image_bytes(
    image_format: str,
    *,
    size: tuple[int, int] = (3, 2),
    color: tuple[int, int, int] = (20, 40, 60),
) -> bytes:
    image = Image.new("RGB", size, color)
    try:
        output = io.BytesIO()
        image.save(output, format=image_format)
        return output.getvalue()
    finally:
        image.close()


def _animated_gif_bytes() -> bytes:
    first = Image.new("RGB", (2, 1), (255, 0, 0))
    second = Image.new("RGB", (2, 1), (0, 0, 255))
    try:
        output = io.BytesIO()
        first.save(output, format="GIF", save_all=True, append_images=[second], loop=0)
        return output.getvalue()
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    ("image_format", "declared_media_type", "expected_media_type"),
    [
        ("PNG", "image/png", "image/png"),
        ("JPEG", "image/jpeg", "image/jpeg"),
        ("WEBP", "image/webp", "image/webp"),
    ],
)
def test_static_supported_images_keep_original_bytes_after_validation(
    image_format: str,
    declared_media_type: str,
    expected_media_type: str,
) -> None:
    data = _image_bytes(image_format)

    prepared = prepare_image(
        7,
        declared_media_type,
        data,
        max_image_bytes=10_000,
        max_image_pixels=6,
    )

    assert prepared.attachment_id == 7
    assert prepared.media_type == expected_media_type
    assert prepared.data == data
    assert (prepared.width, prepared.height) == (3, 2)


def test_actual_decoded_format_wins_over_declared_media_type() -> None:
    data = _image_bytes("PNG")

    prepared = prepare_image(
        8,
        "image/jpeg",
        data,
        max_image_bytes=10_000,
        max_image_pixels=6,
    )

    assert prepared.media_type == "image/png"
    assert prepared.data == data


def test_animated_gif_normalizes_only_first_frame_to_png() -> None:
    prepared = prepare_image(
        9,
        "image/gif",
        _animated_gif_bytes(),
        max_image_bytes=10_000,
        max_image_pixels=2,
    )

    assert prepared.media_type == "image/png"
    with Image.open(io.BytesIO(prepared.data)) as output:
        assert output.format == "PNG"
        assert output.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"phase2ba-private-image-bytes-check",
        b"%PDF-1.7\nphase2ba-private-image-bytes-check",
        _image_bytes("PNG")[:8],
        _image_bytes("JPEG")[:20],
    ],
)
def test_corrupt_or_unsupported_image_data_is_rejected_without_data_leakage(data: bytes) -> None:
    with pytest.raises(ImagePreparationError) as raised:
        prepare_image(
            10,
            "image/png",
            data,
            max_image_bytes=10_000,
            max_image_pixels=100,
        )

    assert "phase2ba-private-image-bytes-check" not in str(raised.value)


def test_fake_image_mime_with_pdf_bytes_is_unsupported() -> None:
    with pytest.raises(UnsupportedImageFormatError):
        prepare_image(
            11,
            "image/png",
            b"%PDF-1.7\nnot an image",
            max_image_bytes=10_000,
            max_image_pixels=100,
        )


def test_per_image_byte_limit_rejects_before_decode() -> None:
    with pytest.raises(ImageTooLargeError):
        prepare_image(
            12,
            "image/png",
            _image_bytes("PNG"),
            max_image_bytes=1,
            max_image_pixels=100,
        )


def test_pixel_limit_allows_exact_limit_and_rejects_over_limit() -> None:
    data = _image_bytes("PNG", size=(3, 2))

    allowed = prepare_image(
        13,
        "image/png",
        data,
        max_image_bytes=10_000,
        max_image_pixels=6,
    )
    assert (allowed.width, allowed.height) == (3, 2)

    with pytest.raises(ImagePixelLimitError):
        prepare_image(
            14,
            "image/png",
            data,
            max_image_bytes=10_000,
            max_image_pixels=5,
        )


def test_prepared_image_repr_is_immutable_and_redacts_bytes() -> None:
    data = b"phase2ba-private-image-bytes-check"
    prepared = PreparedImage(15, "image/png", data, 1, 1)

    assert "phase2ba-private-image-bytes-check" not in repr(prepared)
    with pytest.raises(AttributeError):
        prepared.width = 2  # type: ignore[misc]

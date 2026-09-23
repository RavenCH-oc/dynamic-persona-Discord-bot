from __future__ import annotations

import io

import pytest
from PIL import Image

from liuer_bot.image_preparation import PreparedImage
from liuer_bot.model_image_normalization import (
    ModelImageNormalizationError,
    normalize_model_images,
)

_MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def _prepared_image(
    attachment_id: int,
    image_format: str = "PNG",
    size: tuple[int, int] = (40, 20),
) -> PreparedImage:
    image = Image.new("RGB", size, (20, 40, 60))
    try:
        output = io.BytesIO()
        image.save(output, format=image_format)
        return PreparedImage(
            attachment_id,
            _MEDIA_TYPES[image_format],
            output.getvalue(),
            *size,
        )
    finally:
        image.close()


def _normalize(
    images: tuple[PreparedImage, ...],
    *,
    max_long_edge: int = 200,
    max_image_pixels: int = 40_000,
    max_total_image_pixels: int = 80_000,
) -> tuple[PreparedImage, ...]:
    return normalize_model_images(
        images,
        max_long_edge=max_long_edge,
        max_image_pixels=max_image_pixels,
        max_total_image_pixels=max_total_image_pixels,
    )


def test_compliant_image_retains_exact_bytes_and_is_never_upscaled() -> None:
    original = _prepared_image(1, size=(40, 20))

    normalized = _normalize((original,))

    assert normalized == (original,)
    assert normalized[0] is original
    assert normalized[0].data == original.data
    assert (normalized[0].width, normalized[0].height) == (40, 20)


def test_long_edge_limit_resizes_proportionally() -> None:
    original = _prepared_image(1, size=(400, 100))

    normalized = _normalize((original,), max_long_edge=200, max_image_pixels=40_000)

    assert (normalized[0].width, normalized[0].height) == (200, 50)
    assert normalized[0].data != original.data


def test_pixel_limit_resizes_proportionally() -> None:
    original = _prepared_image(1, size=(400, 400))

    normalized = _normalize((original,), max_long_edge=1000, max_image_pixels=10_000)

    assert normalized[0].width * normalized[0].height <= 10_000
    assert normalized[0].width == normalized[0].height


@pytest.mark.parametrize(
    ("max_long_edge", "max_image_pixels", "expected_size"),
    [
        (200, 100_000, (200, 50)),
        (1_000, 10_000, (200, 50)),
    ],
)
def test_long_edge_and_pixel_constraints_each_can_be_effective_limit(
    max_long_edge: int,
    max_image_pixels: int,
    expected_size: tuple[int, int],
) -> None:
    original = _prepared_image(1, size=(400, 100))

    normalized = _normalize(
        (original,),
        max_long_edge=max_long_edge,
        max_image_pixels=max_image_pixels,
    )

    assert (normalized[0].width, normalized[0].height) == expected_size


def test_multi_image_total_budget_uses_equal_effective_pixel_limit_in_original_order() -> None:
    originals = tuple(_prepared_image(identifier, size=(100, 100)) for identifier in (4, 3, 2, 1))

    normalized = _normalize(
        originals,
        max_long_edge=1_000,
        max_image_pixels=10_000,
        max_total_image_pixels=8_000,
    )

    assert [image.attachment_id for image in normalized] == [4, 3, 2, 1]
    assert all(image.width * image.height <= 2_000 for image in normalized)
    assert sum(image.width * image.height for image in normalized) <= 8_000


def test_small_and_large_images_preserve_small_bytes_and_resize_large_image() -> None:
    small = _prepared_image(1, size=(20, 10))
    large = _prepared_image(2, size=(400, 100))

    normalized = _normalize((small, large), max_long_edge=100, max_image_pixels=10_000)

    assert [image.attachment_id for image in normalized] == [1, 2]
    assert normalized[0] is small
    assert normalized[0].data == small.data
    assert (normalized[1].width, normalized[1].height) == (100, 25)
    assert normalized[1].data != large.data


@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "WEBP"])
def test_resized_supported_media_remains_decodable_with_same_media_type(image_format: str) -> None:
    original = _prepared_image(1, image_format, size=(400, 100))

    normalized = _normalize((original,), max_long_edge=100, max_image_pixels=10_000)[0]

    assert normalized.media_type == original.media_type
    assert normalized.width * normalized.height <= 10_000
    with Image.open(io.BytesIO(normalized.data)) as image:
        assert image.format == image_format
        assert image.size == (normalized.width, normalized.height)


@pytest.mark.parametrize(
    "prepared_image",
    [
        PreparedImage(1, "image/gif", b"gif", 1, 1),
        PreparedImage(1, "image/png", b"", 1, 1),
        PreparedImage(1, "image/png", b"corrupt", 1, 1),
        PreparedImage(1, "image/png", b"corrupt", 0, 1),
    ],
)
def test_invalid_model_image_inputs_fail_closed(prepared_image: PreparedImage) -> None:
    with pytest.raises(ModelImageNormalizationError):
        _normalize((prepared_image,))


def test_effective_pixel_budget_below_one_fails_closed() -> None:
    images = (_prepared_image(1), _prepared_image(2))

    with pytest.raises(ModelImageNormalizationError):
        _normalize(images, max_image_pixels=1, max_total_image_pixels=1)

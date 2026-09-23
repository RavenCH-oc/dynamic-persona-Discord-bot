"""Pure, in-memory normalization of prepared images for VLM input."""

from __future__ import annotations

import io
import warnings
from math import sqrt

from PIL import Image, UnidentifiedImageError

from .image_preparation import PreparedImage

SUPPORTED_MODEL_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
_FORMAT_MEDIA_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}
_MEDIA_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
}


class ModelImageNormalizationError(RuntimeError):
    """Prepared image data cannot safely be normalized for model input."""

    def __init__(self) -> None:
        super().__init__("model image normalization failed")


def normalize_model_images(
    prepared_images: tuple[PreparedImage, ...],
    *,
    max_long_edge: int,
    max_image_pixels: int,
    max_total_image_pixels: int,
) -> tuple[PreparedImage, ...]:
    """Return ordered, proportional VLM-input images without mutating inputs."""

    if not prepared_images:
        return ()
    if min(max_long_edge, max_image_pixels, max_total_image_pixels) < 1:
        raise ModelImageNormalizationError()

    effective_pixels = min(max_image_pixels, max_total_image_pixels // len(prepared_images))
    if effective_pixels < 1:
        raise ModelImageNormalizationError()

    return tuple(
        _normalize_image(
            image,
            max_long_edge=max_long_edge,
            max_pixels=effective_pixels,
        )
        for image in prepared_images
    )


def _normalize_image(
    prepared_image: PreparedImage,
    *,
    max_long_edge: int,
    max_pixels: int,
) -> PreparedImage:
    if (
        prepared_image.media_type not in SUPPORTED_MODEL_IMAGE_MEDIA_TYPES
        or not prepared_image.data
        or prepared_image.width < 1
        or prepared_image.height < 1
    ):
        raise ModelImageNormalizationError()

    actual_width, actual_height = _verified_dimensions(prepared_image)
    if (actual_width, actual_height) != (prepared_image.width, prepared_image.height):
        raise ModelImageNormalizationError()

    target_width, target_height = _target_dimensions(
        actual_width,
        actual_height,
        max_long_edge=max_long_edge,
        max_pixels=max_pixels,
    )
    if (target_width, target_height) == (actual_width, actual_height):
        return prepared_image

    return _resize_image(prepared_image, target_width, target_height)


def _verified_dimensions(prepared_image: PreparedImage) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(prepared_image.data)) as image:
                if _FORMAT_MEDIA_TYPES.get(image.format or "") != prepared_image.media_type:
                    raise ModelImageNormalizationError()
                dimensions = image.size
                image.verify()
                return dimensions
    except ModelImageNormalizationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ModelImageNormalizationError() from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ModelImageNormalizationError() from exc


def _target_dimensions(
    width: int,
    height: int,
    *,
    max_long_edge: int,
    max_pixels: int,
) -> tuple[int, int]:
    longest_edge = max(width, height)
    long_edge_scale = min(1.0, max_long_edge / longest_edge)
    pixel_scale = min(1.0, sqrt(max_pixels / (width * height)))
    scale = min(long_edge_scale, pixel_scale)
    return max(1, int(width * scale)), max(1, int(height * scale))


def _resize_image(
    prepared_image: PreparedImage,
    width: int,
    height: int,
) -> PreparedImage:
    try:
        with Image.open(io.BytesIO(prepared_image.data)) as image:
            image.load()
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
            converted: Image.Image | None = None
            try:
                output = io.BytesIO()
                image_format = _MEDIA_FORMATS[prepared_image.media_type]
                image_to_save = resized
                save_kwargs: dict[str, object] = {"format": image_format}
                if prepared_image.media_type == "image/jpeg":
                    if resized.mode not in {"RGB", "L"}:
                        converted = resized.convert("RGB")
                        image_to_save = converted
                    save_kwargs["quality"] = 90
                elif prepared_image.media_type == "image/webp":
                    save_kwargs["quality"] = 90
                image_to_save.save(output, **save_kwargs)
                normalized_data = output.getvalue()
            finally:
                if converted is not None:
                    converted.close()
                resized.close()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ModelImageNormalizationError() from exc
    except (OSError, SyntaxError, ValueError, KeyError) as exc:
        raise ModelImageNormalizationError() from exc

    if not normalized_data:
        raise ModelImageNormalizationError()
    return PreparedImage(
        attachment_id=prepared_image.attachment_id,
        media_type=prepared_image.media_type,
        data=normalized_data,
        width=width,
        height=height,
    )

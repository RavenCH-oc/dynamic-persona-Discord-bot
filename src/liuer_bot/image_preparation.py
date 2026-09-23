"""Synchronous, byte-based validation and normalization of Discord images."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field

from PIL import Image, UnidentifiedImageError

_FORMAT_MEDIA_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


class ImagePreparationError(RuntimeError):
    """Base class for privacy-safe selected-image failures."""


class ImageDownloadError(ImagePreparationError):
    """Discord could not supply selected attachment bytes."""

    def __init__(self, message: str = "image download failed") -> None:
        super().__init__(message)


class ImageDownloadTimeoutError(ImageDownloadError):
    """Discord attachment download exceeded its configured timeout."""

    def __init__(self) -> None:
        super().__init__("image download timed out")


class ImageTooLargeError(ImagePreparationError):
    """A selected image exceeds a configured byte limit."""

    def __init__(self) -> None:
        super().__init__("image exceeds configured size limit")


class ImagePixelLimitError(ImagePreparationError):
    """A selected image exceeds the configured decoded pixel limit."""

    def __init__(self) -> None:
        super().__init__("image exceeds configured pixel limit")


class TooManyImagesError(ImagePreparationError):
    """A message has more selected image candidates than allowed."""

    def __init__(self) -> None:
        super().__init__("too many image attachments")


class UnsupportedImageFormatError(ImagePreparationError):
    """Selected bytes are not one of the supported image formats."""

    def __init__(self) -> None:
        super().__init__("unsupported image format")


@dataclass(frozen=True, slots=True, repr=False)
class PreparedImage:
    """Validated image data without Discord metadata or byte representation."""

    attachment_id: int
    media_type: str
    data: bytes = field(repr=False)
    width: int = 0
    height: int = 0

    def __repr__(self) -> str:
        return (
            "PreparedImage("
            f"attachment_id={self.attachment_id!r}, "
            f"media_type={self.media_type!r}, "
            "data=<redacted>, "
            f"width={self.width!r}, "
            f"height={self.height!r})"
        )


def prepare_image(
    attachment_id: int,
    declared_media_type: str,
    data: bytes,
    *,
    max_image_bytes: int,
    max_image_pixels: int,
) -> PreparedImage:
    """Decode actual bytes and return a validated immutable image DTO.

    ``declared_media_type`` is intentionally not trusted: Pillow's decoded
    format is the only input-format authority at this boundary.
    """

    _ = declared_media_type
    if not isinstance(data, bytes):
        raise ImagePreparationError("image preparation failed")
    if len(data) > max_image_bytes:
        raise ImageTooLargeError()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                actual_format = image.format
                media_type = _FORMAT_MEDIA_TYPES.get(actual_format or "")
                if media_type is None:
                    raise UnsupportedImageFormatError()
                image.load()
                width, height = image.size
                _validate_dimensions(width, height, max_image_pixels)
                is_animated = bool(getattr(image, "is_animated", False))
                is_animated = is_animated or getattr(image, "n_frames", 1) > 1
                if actual_format == "GIF" or is_animated:
                    normalized = _first_frame_png(image)
                    return PreparedImage(
                        attachment_id=attachment_id,
                        media_type="image/png",
                        data=normalized,
                        width=width,
                        height=height,
                    )
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImagePixelLimitError() from exc
    except ImagePreparationError:
        raise
    except UnidentifiedImageError as exc:
        raise UnsupportedImageFormatError() from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImagePreparationError("image preparation failed") from exc

    return PreparedImage(
        attachment_id=attachment_id,
        media_type=media_type,
        data=data,
        width=width,
        height=height,
    )


def _validate_dimensions(width: int, height: int, max_image_pixels: int) -> None:
    if width <= 0 or height <= 0 or width * height > max_image_pixels:
        raise ImagePixelLimitError()


def _first_frame_png(image: Image.Image) -> bytes:
    image.seek(0)
    first_frame = image.copy()
    try:
        output = io.BytesIO()
        first_frame.save(output, format="PNG")
        return output.getvalue()
    finally:
        first_frame.close()

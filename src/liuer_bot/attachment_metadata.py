"""Download-free Discord image attachment classification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Protocol

from .responder import ImageAttachment

SUPPORTED_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_EXTENSION_MEDIA_TYPES: Mapping[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class AttachmentLike(Protocol):
    id: int
    content_type: str | None
    filename: str
    size: int


def _normalize_content_type(content_type: str) -> str | None:
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized == "image/jpg":
        normalized = "image/jpeg"
    return normalized if normalized in SUPPORTED_IMAGE_MEDIA_TYPES else None


def classify_image_attachment(attachment: AttachmentLike) -> ImageAttachment | None:
    """Classify one attachment without reading bytes or retaining its URL."""

    raw_content_type = attachment.content_type
    if raw_content_type and raw_content_type.strip():
        media_type = _normalize_content_type(raw_content_type)
        if media_type is None:
            return None
    else:
        extension = PurePosixPath(attachment.filename).suffix.lower()
        media_type = _EXTENSION_MEDIA_TYPES.get(extension)
        if media_type is None:
            return None

    return ImageAttachment(
        attachment_id=int(attachment.id),
        media_type=media_type,
        size_bytes=int(attachment.size),
    )


def classify_image_attachments(
    attachments: tuple[AttachmentLike, ...] | list[AttachmentLike],
) -> tuple[ImageAttachment, ...]:
    """Classify supported image attachments while preserving message order."""

    return tuple(
        metadata
        for attachment in attachments
        if (metadata := classify_image_attachment(attachment)) is not None
    )

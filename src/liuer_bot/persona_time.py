"""Stdlib clock and UTC policy-window helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import Protocol


class Clock(Protocol):
    """Clock boundary used by PersonaService policy calculations."""

    def now_utc(self) -> datetime:
        """Return an aware UTC timestamp."""


class SystemClock:
    """Production wall clock."""

    def now_utc(self) -> datetime:
        return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    """Normalize an aware timestamp to UTC without using local timezone state."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def format_utc(value: datetime) -> str:
    """Serialize a timestamp in one deterministic UTC representation."""

    return as_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    """Parse the database's deterministic UTC representation."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return as_utc(parsed)


def quota_window(now_utc: datetime, offset_hours: int) -> tuple[datetime, datetime]:
    """Return the current fixed-offset calendar day as UTC bounds."""

    local_timezone = timezone(timedelta(hours=offset_hours))
    local_now = as_utc(now_utc).astimezone(local_timezone)
    local_start = datetime.combine(local_now.date(), time.min, tzinfo=local_timezone)
    local_end = local_start + timedelta(days=1)
    return local_start.astimezone(UTC), local_end.astimezone(UTC)


def next_quota_reset(now_utc: datetime, offset_hours: int) -> datetime:
    """Return the next configured local midnight represented in UTC."""

    _start, end = quota_window(now_utc, offset_hours)
    return end


def utc_date(value: datetime) -> date:
    """Expose a small helper for deterministic tests and diagnostics."""

    return as_utc(value).date()

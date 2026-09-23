from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from liuer_bot.persona_models import (
    PersonaDailyLimitError,
    PersonaGlobalCooldownError,
    PersonaNoRollbackTargetError,
    PersonaTooLongError,
    PersonaValidationError,
)
from liuer_bot.persona_service import PersonaService
from liuer_bot.prompting import load_default_persona


@dataclass
class FixedClock:
    current: datetime

    def now_utc(self) -> datetime:
        return self.current


def _run(awaitable):
    return asyncio.run(awaitable)


def _service(
    path: Path,
    clock: FixedClock,
    *,
    cooldown: int = 1,
    daily_limit: int = 10,
    offset: int = 8,
) -> PersonaService:
    return PersonaService(
        path,
        global_cooldown_seconds=cooldown,
        daily_limit=daily_limit,
        daily_reset_utc_offset_hours=offset,
        clock=clock,
    )


def test_initialize_creates_parent_and_default_snapshot_without_db_persona(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(path, clock)

    snapshot = _run(service.initialize())

    assert path.exists()
    assert snapshot.revision == 0
    assert snapshot.active_version_id is None
    assert snapshot.active_persona is None
    assert _run(service.list_history()) == ()
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM persona_versions").fetchone() == (0,)
    finally:
        connection.close()
    assert load_default_persona() not in repr(snapshot)


def test_publish_restart_reset_and_rollback_persist_domain_state(tmp_path: Path) -> None:
    path = tmp_path / "persona.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, 12, tzinfo=UTC))
    service = _service(path, clock)
    _run(service.initialize())

    first = _run(service.publish_persona(100, "  first persona\nwith formatting  "))
    clock.current += timedelta(seconds=1)
    second = _run(service.publish_persona("200", "second persona"))

    assert first.version.version_id == 1
    assert first.version.persona_text == "first persona\nwith formatting"
    assert second.state.active_version_id == 2
    assert second.state.revision == 2
    assert [item.version_id for item in _run(service.list_history())] == [2, 1]
    assert all(
        load_default_persona() not in item.persona_text
        for item in _run(service.list_history())
    )

    restarted = _service(path, clock)
    restarted_snapshot = _run(restarted.initialize())
    assert restarted_snapshot.active_version_id == 2
    assert restarted.get_active_persona() == "second persona"

    clock.current += timedelta(seconds=1)
    reset_snapshot = _run(restarted.reset_to_default(999))
    assert reset_snapshot.active_version_id is None
    assert reset_snapshot.revision == 3
    assert [item.version_id for item in _run(restarted.list_history())] == [2, 1]

    rollback_snapshot = _run(restarted.rollback(999))
    assert rollback_snapshot.active_version_id == 2
    assert rollback_snapshot.active_persona == "second persona"
    assert rollback_snapshot.revision == 4

    first_rollback = _run(restarted.rollback(999))
    assert first_rollback.active_version_id == 1
    assert first_rollback.active_persona == "first persona\nwith formatting"
    assert first_rollback.revision == 5
    with pytest.raises(PersonaNoRollbackTargetError):
        _run(restarted.rollback(999))
    assert restarted.current_snapshot.revision == 5

    default_again = _run(restarted.reset_to_default(999))
    assert default_again.revision == 6
    recreated = _service(path, clock)
    assert _run(recreated.initialize()).active_version_id is None


def test_history_is_newest_first_and_limit_is_capped(tmp_path: Path) -> None:
    path = tmp_path / "persona.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(path, clock)
    _run(service.initialize())
    for index in range(3):
        _run(service.publish_persona(index + 1, f"persona {index}"))
        clock.current += timedelta(seconds=1)

    assert [item.version_id for item in _run(service.list_history(2))] == [3, 2]
    assert len(_run(service.list_history(500))) == 3
    with pytest.raises(PersonaValidationError):
        _run(service.list_history(0))


def test_global_cooldown_is_cross_user_and_allows_exact_boundary(tmp_path: Path) -> None:
    path = tmp_path / "persona.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(path, clock, cooldown=1200)
    _run(service.initialize())
    _run(service.publish_persona("A", "first"))

    clock.current += timedelta(seconds=1199)
    with pytest.raises(PersonaGlobalCooldownError) as raised:
        _run(service.publish_persona("B", "blocked"))
    assert raised.value.retry_after_seconds == 1
    assert _run(service.get_policy_status("B")).daily_used == 0

    clock.current += timedelta(seconds=1)
    result = _run(service.publish_persona("B", "allowed at boundary"))
    assert result.version.author_discord_id == "B"


def test_daily_quota_uses_configured_utc_plus_eight_midnight(tmp_path: Path) -> None:
    path = tmp_path / "persona.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, 15, 59, 55, tzinfo=UTC))
    service = _service(path, clock, daily_limit=3, offset=8)
    _run(service.initialize())
    for index in range(3):
        _run(service.publish_persona("A", f"day one {index}"))
        clock.current += timedelta(seconds=1)

    with pytest.raises(PersonaDailyLimitError) as raised:
        _run(service.publish_persona("A", "fourth day one"))
    assert raised.value.daily_limit == 3
    assert _run(service.get_policy_status("A")).daily_used == 3

    clock.current = datetime(2026, 1, 1, 16, tzinfo=UTC)
    status = _run(service.get_policy_status("A"))
    assert status.daily_used == 0
    assert status.daily_remaining == 3
    assert status.next_daily_reset_utc == datetime(2026, 1, 2, 16, tzinfo=UTC)
    assert _run(service.publish_persona("A", "next local day"))


def test_reset_does_not_bypass_global_cooldown_or_consume_quota(tmp_path: Path) -> None:
    path = tmp_path / "persona.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(path, clock, cooldown=1200, daily_limit=3)
    _run(service.initialize())
    _run(service.publish_persona("A", "first"))
    clock.current += timedelta(seconds=10)
    _run(service.reset_to_default("maintenance"))

    with pytest.raises(PersonaGlobalCooldownError):
        _run(service.publish_persona("B", "still blocked"))
    status = _run(service.get_policy_status("A"))
    assert status.daily_used == 1
    assert status.daily_remaining == 2


def test_concurrent_publish_has_one_sqlite_transaction_winner(tmp_path: Path) -> None:
    path = tmp_path / "race.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(path, clock, cooldown=1200, daily_limit=3)
    _run(service.initialize())

    async def publish_both():
        return await asyncio.gather(
            service.publish_persona("A", "race A"),
            service.publish_persona("B", "race B"),
            return_exceptions=True,
        )

    results = _run(publish_both())

    successes = [result for result in results if not isinstance(result, Exception)]
    failures = [result for result in results if isinstance(result, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], PersonaGlobalCooldownError)
    assert len(_run(service.list_history())) == 1
    assert service.current_snapshot.active_persona in {"race A", "race B"}


def test_failed_validation_and_daily_rejection_do_not_add_versions(tmp_path: Path) -> None:
    path = tmp_path / "persona.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = PersonaService(path, persona_max_chars=4, daily_limit=1, clock=clock)
    _run(service.initialize())
    with pytest.raises(PersonaValidationError):
        _run(service.publish_persona("A", " \n\t"))
    with pytest.raises(PersonaTooLongError):
        _run(service.publish_persona("A", "too long"))

    _run(service.publish_persona("A", "ok"))
    clock.current += timedelta(seconds=1200)
    with pytest.raises(PersonaDailyLimitError):
        _run(service.publish_persona("A", "new"))
    assert len(_run(service.list_history())) == 1


def test_persona_dtos_and_errors_redact_persona_text(tmp_path: Path) -> None:
    secret = "phase3b-private-persona-text"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = PersonaService(tmp_path / "persona.sqlite3", clock=clock)
    _run(service.initialize())
    result = _run(service.publish_persona("A", secret))

    assert secret not in repr(result.version)
    assert secret not in repr(result.state)
    assert secret not in repr(result)
    short_service = PersonaService(
        tmp_path / "short-persona.sqlite3",
        persona_max_chars=1,
        clock=clock,
    )
    _run(short_service.initialize())
    with pytest.raises(PersonaTooLongError) as raised:
        _run(short_service.publish_persona("A", secret))
    assert secret not in repr(raised.value)

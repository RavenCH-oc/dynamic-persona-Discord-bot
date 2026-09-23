import asyncio
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from liuer_bot.addressing import AddressKind, parse_addressed_message
from liuer_bot.nickname_models import (
    NicknameDailyLimitError,
    NicknameGlobalCooldownError,
    NicknameTooLongError,
    NicknameValidationError,
)
from liuer_bot.nickname_service import NicknameService, normalize_nickname
from liuer_bot.persona_models import PersonaGlobalCooldownError
from liuer_bot.persona_service import PersonaService


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
    daily_limit: int = 3,
    offset: int = 8,
    max_chars: int = 32,
) -> NicknameService:
    return NicknameService(
        path,
        nickname_max_chars=max_chars,
        global_cooldown_seconds=cooldown,
        daily_limit=daily_limit,
        daily_reset_utc_offset_hours=offset,
        clock=clock,
    )


def test_initialize_has_one_empty_global_snapshot_and_v4_tables(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "nickname.sqlite3"
    service = _service(path, FixedClock(datetime(2026, 1, 1, tzinfo=UTC)))

    snapshot = _run(service.initialize())

    assert snapshot.active_version_id is None
    assert service.get_active_nickname() is None
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM nickname_versions").fetchone() == (0,)
        assert connection.execute(
            "SELECT active_version_id, revision FROM nickname_state WHERE singleton_id = 1"
        ).fetchone() == (None, 0)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "value",
    ["", "  \n\t  ", "has\nnewline", "has\ttab", "@everyone", "<@123>", "六耳"],
)
def test_nickname_validation_rejects_unsafe_or_redundant_values(value: str) -> None:
    with pytest.raises(NicknameValidationError):
        normalize_nickname(value, 32)


def test_nickname_normalization_preserves_unicode_case_and_internal_spaces() -> None:
    assert normalize_nickname("  Little Liu  ", 32) == "Little Liu"
    assert normalize_nickname("Liuer", 32) == "Liuer"
    assert normalize_nickname("六耳醬", 32) == "六耳醬"
    assert normalize_nickname("小六", 32) == "小六"
    with pytest.raises(NicknameTooLongError):
        normalize_nickname("x" * 33, 32)


def test_set_restart_and_memory_hot_path_restore_active_nickname(tmp_path: Path) -> None:
    path = tmp_path / "nickname.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(path, clock)
    _run(service.initialize())

    result = _run(service.set_nickname(100, "  小六  "))

    assert result.version.nickname_text == "小六"
    assert service.get_active_nickname() == "小六"
    service._repository.load_state = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("hot path queried SQLite")
    )
    assert service.get_active_nickname() == "小六"

    restarted = _service(path, clock)
    _run(restarted.initialize())
    assert restarted.get_active_nickname() == "小六"


def test_changing_nickname_invalidates_old_address_but_keeps_primary_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "address-change.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(path, clock, cooldown=1)
    _run(service.initialize())
    _run(service.set_nickname(100, "小六"))
    clock.current += timedelta(seconds=1)
    _run(service.set_nickname(100, "阿六"))

    old = parse_addressed_message("小六 你好", active_nickname=service.get_active_nickname())
    new = parse_addressed_message("阿六 你好", active_nickname=service.get_active_nickname())
    primary = parse_addressed_message("六耳 你好", active_nickname=service.get_active_nickname())

    assert old.is_addressed is False
    assert new.address_kind is AddressKind.ACTIVE_NICKNAME
    assert primary.address_kind is AddressKind.PRIMARY_NAME


def test_nickname_policy_is_independent_from_persona_policy(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    persona = PersonaService(path, global_cooldown_seconds=1, daily_limit=3, clock=clock)
    nickname = _service(path, clock, cooldown=1, daily_limit=3)
    _run(persona.initialize())
    _run(nickname.initialize())

    _run(persona.publish_persona("persona-author", "community"))
    _run(nickname.set_nickname("nickname-author", "小六"))
    clock.current += timedelta(milliseconds=500)

    with pytest.raises(NicknameGlobalCooldownError):
        _run(nickname.set_nickname("nickname-author", "阿六"))
    with pytest.raises(PersonaGlobalCooldownError):
        _run(persona.publish_persona("persona-author", "second"))

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM persona_versions").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM nickname_versions").fetchone() == (1,)
    finally:
        connection.close()


def test_daily_quota_and_utc_plus_eight_reset_are_nickname_only(tmp_path: Path) -> None:
    path = tmp_path / "quota.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, 15, 59, 55, tzinfo=UTC))
    service = _service(path, clock, cooldown=1, daily_limit=3, offset=8)
    _run(service.initialize())
    for index in range(3):
        _run(service.set_nickname("A", f"name {index}"))
        clock.current += timedelta(seconds=1)

    with pytest.raises(NicknameDailyLimitError) as raised:
        _run(service.set_nickname("A", "fourth"))
    assert raised.value.daily_limit == 3
    assert _run(service.get_policy_status("A")).daily_used == 3

    clock.current = datetime(2026, 1, 1, 16, tzinfo=UTC)
    assert _run(service.get_policy_status("A")).daily_used == 0
    _run(service.set_nickname("A", "next day"))


def test_concurrent_nickname_set_has_one_cooldown_winner(tmp_path: Path) -> None:
    path = tmp_path / "race.sqlite3"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(path, clock, cooldown=1_200)
    _run(service.initialize())

    async def set_both():
        return await asyncio.gather(
            service.set_nickname("A", "小六"),
            service.set_nickname("B", "阿六"),
            return_exceptions=True,
        )

    results = _run(set_both())
    successes = [result for result in results if not isinstance(result, Exception)]
    failures = [result for result in results if isinstance(result, Exception)]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], NicknameGlobalCooldownError)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM nickname_versions").fetchone() == (1,)
    finally:
        connection.close()


def test_nickname_dtos_and_errors_redact_submitted_text(tmp_path: Path) -> None:
    private = "phase4aa2-private-nickname"
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = _service(tmp_path / "privacy.sqlite3", clock, cooldown=1)
    _run(service.initialize())
    result = _run(service.set_nickname("A", private))

    assert private not in repr(result.version)
    assert private not in repr(result.state)
    assert private not in repr(result)
    with pytest.raises(NicknameValidationError) as raised:
        _run(service.set_nickname("A", "@" + private))
    assert private not in repr(raised.value)

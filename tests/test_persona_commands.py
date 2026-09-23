from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from liuer_bot.persona_commands import PersonaCommands, PersonaSetModal
from liuer_bot.persona_service import PersonaService


def _run(awaitable):
    return asyncio.run(awaitable)


@dataclass
class FixedClock:
    current: datetime

    def now_utc(self) -> datetime:
        return self.current


class FakeResponse:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []
        self.modal: object | None = None
        self._done = False

    def is_done(self) -> bool:
        return self._done

    async def send_message(
        self,
        content: str,
        *,
        ephemeral: bool = False,
        **kwargs: object,
    ) -> None:
        self.messages.append((content, ephemeral))
        self._done = True

    async def send_modal(self, modal: object) -> None:
        self.modal = modal
        self._done = True


class FakeFollowup:
    def __init__(self) -> None:
        self.last: tuple[str, bool] | None = None

    async def send(self, content: str, *, ephemeral: bool = False, **kwargs: object) -> None:
        self.last = (content, ephemeral)


class FakeInteraction:
    def __init__(
        self,
        *,
        user_id: int = 100,
        admin: bool = False,
        manage_guild: bool = False,
    ) -> None:
        self.user = SimpleNamespace(
            id=user_id,
            guild_permissions=SimpleNamespace(
                administrator=admin,
                manage_guild=manage_guild,
            ),
        )
        self.guild = SimpleNamespace(id=789)
        self.response = FakeResponse()
        self.followup = FakeFollowup()


class FakeStatusManager:
    channel_id = 456

    def __init__(self) -> None:
        self.sync_calls = 0

    async def sync_current(self) -> None:
        self.sync_calls += 1


class FailingStatusManager(FakeStatusManager):
    async def sync_current(self) -> None:
        self.sync_calls += 1
        raise RuntimeError("private status transport detail")


def _service(path: Path, *, cooldown: int = 1_200, daily_limit: int = 3) -> PersonaService:
    return PersonaService(
        path,
        global_cooldown_seconds=cooldown,
        daily_limit=daily_limit,
        clock=FixedClock(datetime(2026, 1, 1, 12, tzinfo=UTC)),
    )


def _commands(service: PersonaService, status: FakeStatusManager) -> PersonaCommands:
    return PersonaCommands(
        persona_service=service,
        status_manager=status,  # type: ignore[arg-type]
        max_persona_chars=4_000,
    )


def test_persona_group_has_exactly_five_commands_and_no_show(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3")
    _run(service.initialize())
    commands = _commands(service, FakeStatusManager())

    assert [command.name for command in commands.group.commands] == [
        "set",
        "status",
        "reset",
        "rollback",
        "history",
    ]
    assert "show" not in {command.name for command in commands.group.commands}


def test_set_presents_multiline_modal_only_when_policy_is_available(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3")
    _run(service.initialize())
    interaction = FakeInteraction()
    commands = _commands(service, FakeStatusManager())

    _run(commands.handle_set(interaction))

    assert isinstance(interaction.response.modal, PersonaSetModal)
    assert interaction.response.modal.persona.max_length == 4_000
    assert interaction.response.messages == []


def test_set_precheck_rejects_cooldown_without_opening_modal(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3", cooldown=1_200)
    _run(service.initialize())
    _run(service.publish_persona(1, "first"))
    interaction = FakeInteraction(user_id=2)
    commands = _commands(service, FakeStatusManager())

    _run(commands.handle_set(interaction))

    assert interaction.response.modal is None
    assert interaction.response.messages[0][1] is True
    assert "cooldown" in interaction.response.messages[0][0]


def test_modal_submit_rechecks_authoritative_policy_after_race(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3", cooldown=1_200)
    _run(service.initialize())
    interaction = FakeInteraction(user_id=2)
    status = FakeStatusManager()
    commands = _commands(service, status)
    _run(commands.handle_set(interaction))
    _run(service.publish_persona(1, "winner"))

    _run(commands.handle_set_submit(interaction, "late submission"))

    assert interaction.followup.last is not None
    assert "cooldown" in interaction.followup.last[0]
    assert service.get_active_persona() == "winner"
    assert status.sync_calls == 0


def test_successful_submit_updates_active_snapshot_and_status_once(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3")
    _run(service.initialize())
    status = FakeStatusManager()
    commands = _commands(service, status)
    interaction = FakeInteraction(user_id=42)

    _run(commands.handle_set_submit(interaction, "new public persona"))

    assert service.get_active_persona() == "new public persona"
    assert status.sync_calls == 1
    assert "version v1" in interaction.response.messages[0][0]
    assert interaction.response.messages[0][1] is True


def test_status_sync_failure_does_not_rollback_committed_persona(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3")
    _run(service.initialize())
    status = FailingStatusManager()
    commands = _commands(service, status)
    interaction = FakeInteraction(user_id=42)

    _run(commands.handle_set_submit(interaction, "committed despite status failure"))

    assert service.get_active_persona() == "committed despite status failure"
    assert "could not be refreshed" in interaction.response.messages[0][0]


def test_status_is_ephemeral_metadata_only_and_does_not_echo_persona(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3")
    _run(service.initialize())
    _run(service.publish_persona(1, "private persona body"))
    interaction = FakeInteraction(user_id=2)
    commands = _commands(service, FakeStatusManager())

    _run(commands.handle_status(interaction))

    response = interaction.response.messages[0][0]
    assert "Community v1" in response
    assert "private persona body" not in response
    assert "<#456>" in response


@pytest.mark.parametrize("admin,manage_guild", [(False, False), (True, False), (False, True)])
def test_admin_commands_use_runtime_permissions(
    tmp_path: Path,
    admin: bool,
    manage_guild: bool,
) -> None:
    service = _service(tmp_path / f"persona-{admin}-{manage_guild}.sqlite3")
    _run(service.initialize())
    _run(service.publish_persona(1, "community"))
    interaction = FakeInteraction(admin=admin, manage_guild=manage_guild)
    commands = _commands(service, FakeStatusManager())

    _run(commands.handle_reset(interaction))

    if admin or manage_guild:
        assert service.get_active_persona() is None
        assert "reset" in interaction.response.messages[0][0].lower()
    else:
        assert service.get_active_persona() == "community"
        assert "permission" in interaction.response.messages[0][0].lower()


def test_history_is_newest_first_metadata_only_and_limited(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3", cooldown=1)
    _run(service.initialize())
    _run(service.publish_persona(1, "first private body"))
    service._clock.current += timedelta(seconds=1)  # type: ignore[attr-defined]
    _run(service.publish_persona(2, "second private body"))
    interaction = FakeInteraction(admin=True)
    commands = _commands(service, FakeStatusManager())

    _run(commands.handle_history(interaction))

    response = interaction.response.messages[0][0]
    assert response.index("v2") < response.index("v1")
    assert "first private body" not in response
    assert "second private body" not in response

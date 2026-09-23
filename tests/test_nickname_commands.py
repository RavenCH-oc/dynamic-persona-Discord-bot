import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import discord
import pytest

from liuer_bot.nickname_commands import NicknameCommands
from liuer_bot.nickname_service import NicknameService


@dataclass
class FixedClock:
    current: datetime

    def now_utc(self) -> datetime:
        return self.current


class FakeResponse:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool, dict[str, object]]] = []
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
        self.messages.append((content, ephemeral, kwargs))
        self._done = True


class FakeFollowup:
    async def send(self, content: str, *, ephemeral: bool = False, **kwargs: object) -> None:
        raise AssertionError("unexpected followup")


class FakeInteraction:
    def __init__(self, user_id: int = 100) -> None:
        self.user = SimpleNamespace(id=user_id)
        self.guild = SimpleNamespace(id=789)
        self.response = FakeResponse()
        self.followup = FakeFollowup()


class FakeStatusManager:
    def __init__(self) -> None:
        self.sync_calls = 0

    async def sync_current(self) -> None:
        self.sync_calls += 1


class FailingStatusManager(FakeStatusManager):
    async def sync_current(self) -> None:
        self.sync_calls += 1
        raise RuntimeError("private status transport detail")


def _run(awaitable):
    return asyncio.run(awaitable)


def _service(path: Path, *, cooldown: int = 1_200) -> NicknameService:
    service = NicknameService(
        path,
        global_cooldown_seconds=cooldown,
        clock=FixedClock(datetime(2026, 1, 1, tzinfo=UTC)),
    )
    _run(service.initialize())
    return service


def test_set_group_registers_name_for_all_guild_members_without_queue(tmp_path: Path) -> None:
    service = _service(tmp_path / "commands.sqlite3")
    status = FakeStatusManager()
    commands = NicknameCommands(nickname_service=service, status_manager=status)  # type: ignore[arg-type]

    assert commands.group.name == "set"
    assert [command.name for command in commands.group.commands] == ["name"]
    interaction = FakeInteraction()
    _run(commands.handle_set_name(interaction, "小六"))

    assert service.get_active_nickname() == "小六"
    assert status.sync_calls == 1
    assert interaction.response.messages[0][1] is False
    allowed_mentions = interaction.response.messages[0][2]["allowed_mentions"]
    assert isinstance(allowed_mentions, discord.AllowedMentions)
    assert allowed_mentions.everyone is False
    assert allowed_mentions.roles is False
    assert allowed_mentions.users is False
    assert allowed_mentions.replied_user is False


@pytest.mark.parametrize("nickname", ["", "@everyone", "六耳", "x" * 33])
def test_set_name_validation_is_ephemeral_and_does_not_change_state(
    tmp_path: Path,
    nickname: str,
) -> None:
    service = _service(tmp_path / "validation.sqlite3")
    status = FakeStatusManager()
    commands = NicknameCommands(nickname_service=service, status_manager=status)  # type: ignore[arg-type]
    interaction = FakeInteraction()

    _run(commands.handle_set_name(interaction, nickname))

    assert service.get_active_nickname() is None
    assert status.sync_calls == 0
    assert interaction.response.messages[0][1] is True


def test_set_name_cooldown_is_domain_authoritative_and_does_not_use_generation_queue(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "cooldown.sqlite3")
    _run(service.set_nickname(1, "first"))
    status = FakeStatusManager()
    commands = NicknameCommands(nickname_service=service, status_manager=status)  # type: ignore[arg-type]
    interaction = FakeInteraction(user_id=2)

    _run(commands.handle_set_name(interaction, "second"))

    assert service.get_active_nickname() == "first"
    assert status.sync_calls == 0
    assert "冷卻" in interaction.response.messages[0][0]
    assert interaction.response.messages[0][1] is True


def test_set_name_daily_limit_rejection_is_ephemeral(tmp_path: Path) -> None:
    clock = FixedClock(datetime(2026, 1, 1, tzinfo=UTC))
    service = NicknameService(
        tmp_path / "daily-limit.sqlite3",
        global_cooldown_seconds=1,
        daily_limit=1,
        clock=clock,
    )
    _run(service.initialize())
    _run(service.set_nickname(1, "first"))
    clock.current += timedelta(seconds=1)
    commands = NicknameCommands(
        nickname_service=service,
        status_manager=FakeStatusManager(),  # type: ignore[arg-type]
    )
    interaction = FakeInteraction(user_id=1)

    _run(commands.handle_set_name(interaction, "second"))

    assert interaction.response.messages[0][1] is True
    assert "上限" in interaction.response.messages[0][0]


def test_status_sync_failure_does_not_rollback_committed_nickname(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path / "status-failure.sqlite3")
    status = FailingStatusManager()
    commands = NicknameCommands(nickname_service=service, status_manager=status)  # type: ignore[arg-type]
    interaction = FakeInteraction()

    _run(commands.handle_set_name(interaction, "小六"))

    assert service.get_active_nickname() == "小六"
    assert status.sync_calls == 1
    assert interaction.response.messages[0][1] is True
    assert "稍後才會更新" in interaction.response.messages[0][0]


def test_nickname_text_is_not_written_to_command_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private = "phase4aa2-command-private-nickname"
    service = _service(tmp_path / "privacy.sqlite3", cooldown=1)
    commands = NicknameCommands(nickname_service=service, status_manager=FakeStatusManager())  # type: ignore[arg-type]

    with caplog.at_level("DEBUG"):
        _run(commands.handle_set_name(FakeInteraction(), private))

    assert private not in caplog.text

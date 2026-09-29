"""Offline Discord server-nickname presentation regressions."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import discord
import pytest

from liuer_bot.config import Config
from liuer_bot.discord_nickname import DiscordNicknameSynchronizer
from liuer_bot.discord_runtime import DiscordRuntimeClient


@dataclass
class FakeMember:
    nick: str | None = None
    error: Exception | None = None
    edits: list[str | None] = field(default_factory=list)

    async def edit(self, *, nick: str | None) -> None:
        self.edits.append(nick)
        if self.error is not None:
            raise self.error
        self.nick = nick


@dataclass
class FakeGuild:
    me: FakeMember | None
    id: int = 789
    fetched_member: FakeMember | None = None
    fetch_calls: list[int] = field(default_factory=list)

    async def fetch_member(self, member_id: int) -> FakeMember:
        self.fetch_calls.append(member_id)
        if self.fetched_member is None:
            raise RuntimeError("member unavailable")
        return self.fetched_member


class FakeNicknameService:
    def __init__(self, nickname: str | None) -> None:
        self.nickname = nickname
        self.calls = 0

    def get_active_nickname(self) -> str | None:
        self.calls += 1
        return self.nickname


class NoopResponder:
    async def generate(self, request: object) -> str:
        return "ok"


def _synchronizer(guild: FakeGuild) -> DiscordNicknameSynchronizer:
    return DiscordNicknameSynchronizer(
        guild_provider=lambda: guild,  # type: ignore[arg-type]
        bot_user_id_provider=lambda: 42,
    )


def _http_error(kind: type[discord.HTTPException], body: str) -> discord.HTTPException:
    status = 403 if kind is discord.Forbidden else 500
    response = SimpleNamespace(status=status, reason="private reason", headers={})
    return kind(response, body)


def test_helper_edits_only_the_configured_guild_member_when_nickname_differs() -> None:
    member = FakeMember(nick="old")
    result = asyncio.run(_synchronizer(FakeGuild(me=member)).synchronize("伊利亞"))

    assert result.succeeded is True
    assert result.changed is True
    assert member.edits == ["伊利亞"]
    assert member.nick == "伊利亞"


def test_helper_skips_matching_nickname_and_can_clear_to_account_name() -> None:
    member = FakeMember(nick="伊利亞")
    synchronizer = _synchronizer(FakeGuild(me=member))

    async def run() -> None:
        same = await synchronizer.synchronize("伊利亞")
        cleared = await synchronizer.synchronize(None)
        same_default = await synchronizer.synchronize(None)
        assert (same.succeeded, same.changed) == (True, False)
        assert (cleared.succeeded, cleared.changed) == (True, True)
        assert (same_default.succeeded, same_default.changed) == (True, False)

    asyncio.run(run())
    assert member.edits == [None]


def test_helper_fetches_own_member_once_when_guild_me_is_not_cached() -> None:
    member = FakeMember(nick=None)
    guild = FakeGuild(me=None, fetched_member=member)
    result = asyncio.run(_synchronizer(guild).synchronize("伊利亞"))

    assert result.succeeded is True
    assert guild.fetch_calls == [42]
    assert member.edits == ["伊利亞"]


@pytest.mark.parametrize("kind", [discord.Forbidden, discord.HTTPException])
def test_helper_http_failures_are_safe_and_do_not_retain_nickname(
    kind: type[discord.HTTPException],
    caplog: pytest.LogCaptureFixture,
) -> None:
    nickname = "phase-nickname-private-marker"
    body = "phase-discord-private-http-body"
    member = FakeMember(nick="old", error=_http_error(kind, body))

    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(_synchronizer(FakeGuild(me=member)).synchronize(nickname))

    assert result.succeeded is False
    assert member.nick == "old"
    assert member.edits == [nickname]
    assert nickname not in caplog.text + repr(result)
    assert body not in caplog.text + repr(result)
    assert kind.__name__ in caplog.text


@pytest.mark.parametrize(
    ("active_nickname", "current_nick", "expected_edits"),
    [
        ("伊利亞", "old", ["伊利亞"]),
        ("伊利亞", "伊利亞", []),
        (None, "old", [None]),
    ],
)
def test_runtime_reconciles_once_at_ready(
    monkeypatch: pytest.MonkeyPatch,
    active_nickname: str | None,
    current_nick: str | None,
    expected_edits: list[str | None],
) -> None:
    monkeypatch.setattr(
        DiscordRuntimeClient, "user", SimpleNamespace(id=42), raising=False,
    )
    service = FakeNicknameService(active_nickname)
    member = FakeMember(nick=current_nick)
    client = DiscordRuntimeClient(
        config=Config(discord_token="fake-token", chat_channel_id=123),
        responder=NoopResponder(),  # type: ignore[arg-type]
        nickname_service=service,  # type: ignore[arg-type]
    )
    client._chat_channel = SimpleNamespace(guild=FakeGuild(me=member))

    async def run() -> None:
        await client.on_ready()
        await client.on_ready()
        await client.close()

    asyncio.run(run())
    assert member.edits == expected_edits
    assert service.calls == 1


def test_runtime_startup_sync_failure_does_not_prevent_readiness(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        DiscordRuntimeClient, "user", SimpleNamespace(id=42), raising=False,
    )
    service = FakeNicknameService("phase-startup-private-nickname")
    body = "phase-startup-private-http-body"
    member = FakeMember(nick="old", error=_http_error(discord.Forbidden, body))
    client = DiscordRuntimeClient(
        config=Config(discord_token="fake-token", chat_channel_id=123),
        responder=NoopResponder(),  # type: ignore[arg-type]
        nickname_service=service,  # type: ignore[arg-type]
    )
    client._chat_channel = SimpleNamespace(guild=FakeGuild(me=member))

    async def run() -> None:
        await client.on_ready()
        await client.close()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(run())

    assert member.edits == [service.nickname]
    assert service.calls == 1
    assert "Forbidden" in caplog.text
    assert service.nickname not in caplog.text
    assert body not in caplog.text

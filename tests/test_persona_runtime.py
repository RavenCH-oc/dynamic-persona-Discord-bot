from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import discord
import pytest

import liuer_bot.discord_runtime as discord_runtime
from liuer_bot.config import Config
from liuer_bot.nickname_service import NicknameService
from liuer_bot.persona_service import PersonaService
from liuer_bot.prompting import PromptBuilder


def _run(awaitable):
    return asyncio.run(awaitable)


class NoopResponder:
    async def generate(self, request):
        return "ok"


class FakeMessage:
    def __init__(self, message_id: int, author_id: int) -> None:
        self.id = message_id
        self.author = SimpleNamespace(id=author_id, bot=True)

    async def edit(self, **kwargs: object) -> None:
        return None


class FakeChannel:
    type = discord.ChannelType.text

    def __init__(self, channel_id: int = 456, guild_id: int = 789) -> None:
        self.id = channel_id
        self.guild = SimpleNamespace(id=guild_id)
        self.message: FakeMessage | None = None
        self.send_count = 0

    async def fetch_message(self, message_id: int) -> FakeMessage:
        if self.message is None or self.message.id != message_id:
            raise discord.NotFound(
                SimpleNamespace(status=404, reason="not found", headers={}),
                "not found",
            )
        return self.message

    async def send(self, **kwargs: object) -> FakeMessage:
        self.send_count += 1
        self.message = FakeMessage(900 + self.send_count, 42)
        return self.message


def test_setup_hook_registers_guild_commands_and_status_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = PersonaService(tmp_path / "persona.sqlite3")
    _run(service.initialize())
    nickname_service = NicknameService(tmp_path / "persona.sqlite3")
    _run(nickname_service.initialize())
    channel = FakeChannel(456)
    chat_channel = FakeChannel(123)
    config = Config(
        discord_token="private-token",
        chat_channel_id=123,
        persona_status_channel_id=456,
    )
    monkeypatch.setattr(
        discord_runtime.DiscordRuntimeClient,
        "user",
        SimpleNamespace(id=42),
        raising=False,
    )
    client = discord_runtime.DiscordRuntimeClient(
        config=config,
        responder=NoopResponder(),
        persona_service=service,
        nickname_service=nickname_service,
        prompt_builder=PromptBuilder(),
    )
    sync_guilds: list[int] = []

    async def fetch_channel(channel_id: int) -> FakeChannel:
        return chat_channel if channel_id == 123 else channel

    async def sync(*, guild: discord.Object):
        sync_guilds.append(guild.id)
        return []

    monkeypatch.setattr(client, "fetch_channel", fetch_channel)
    monkeypatch.setattr(client.tree, "sync", sync)

    async def run() -> None:
        await client.setup_hook()
        await client.setup_hook()
        assert sync_guilds == [789]
        assert channel.send_count == 1
        assert client._persona_commands is not None
        assert [command.name for command in client._persona_commands.group.commands] == [
            "set",
            "status",
            "reset",
            "rollback",
            "history",
        ]
        assert client._nickname_commands is not None
        assert [command.name for command in client._nickname_commands.group.commands] == ["name"]
        await client._generation_queue.close()

    _run(run())


def test_setup_hook_rejects_non_text_status_channel(monkeypatch, tmp_path: Path) -> None:
    service = PersonaService(tmp_path / "persona.sqlite3")
    _run(service.initialize())
    channel = FakeChannel(456)
    chat_channel = FakeChannel(123)
    channel.type = discord.ChannelType.voice
    config = Config(
        discord_token="private-token",
        chat_channel_id=123,
        persona_status_channel_id=456,
    )
    monkeypatch.setattr(
        discord_runtime.DiscordRuntimeClient,
        "user",
        SimpleNamespace(id=42),
        raising=False,
    )
    client = discord_runtime.DiscordRuntimeClient(
        config=config,
        responder=NoopResponder(),
        persona_service=service,
        prompt_builder=PromptBuilder(),
    )

    async def fetch_channel(channel_id: int) -> FakeChannel:
        return chat_channel if channel_id == 123 else channel

    monkeypatch.setattr(client, "fetch_channel", fetch_channel)

    async def run() -> None:
        try:
            await client.setup_hook()
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsupported status channel must fail startup")
        await client._generation_queue.close()

    _run(run())


def test_setup_hook_rejects_chat_and_status_channels_from_different_guilds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    service = PersonaService(tmp_path / "persona.sqlite3")
    _run(service.initialize())
    status_channel = FakeChannel(456, guild_id=789)
    chat_channel = FakeChannel(123, guild_id=790)
    config = Config(
        discord_token="private-token",
        chat_channel_id=123,
        persona_status_channel_id=456,
    )
    client = discord_runtime.DiscordRuntimeClient(
        config=config,
        responder=NoopResponder(),
        persona_service=service,
        prompt_builder=PromptBuilder(),
    )

    async def fetch_channel(channel_id: int) -> FakeChannel:
        return chat_channel if channel_id == 123 else status_channel

    monkeypatch.setattr(client, "fetch_channel", fetch_channel)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="same a guild|share a guild"):
            await client.setup_hook()
        await client._generation_queue.close()

    _run(run())

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import discord

from liuer_bot.config import Config
from liuer_bot.context_commands import ContextCommands
from liuer_bot.conversation_service import ConversationError, ConversationService
from liuer_bot.discord_runtime import DiscordRuntimeClient
from liuer_bot.responder import ChatRequest

PRIVATE_BODY = "phase5b-private-command-body"


class FakeResponse:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.messages: list[tuple[str, bool, dict[str, object]]] = []
        self._done = False
        self.error = error

    def is_done(self) -> bool:
        return self._done

    async def send_message(
        self,
        content: str,
        *,
        ephemeral: bool = False,
        **kwargs: object,
    ) -> None:
        if self.error is not None:
            raise self.error
        self.messages.append((content, ephemeral, kwargs))
        self._done = True


class FakeFollowup:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool, dict[str, object]]] = []

    async def send(self, content: str, *, ephemeral: bool = False, **kwargs: object) -> None:
        self.messages.append((content, ephemeral, kwargs))


class FakeInteraction:
    def __init__(
        self,
        *,
        admin: bool = True,
        manage_guild: bool = False,
        channel_id: int = 123,
        is_dm: bool = False,
        response_error: Exception | None = None,
    ) -> None:
        self.user = SimpleNamespace(
            id=100,
            guild_permissions=SimpleNamespace(
                administrator=admin,
                manage_guild=manage_guild,
            ),
        )
        self.guild = None if is_dm else SimpleNamespace(id=20)
        self.channel_id = channel_id
        self.response = FakeResponse(error=response_error)
        self.followup = FakeFollowup()


def _run(awaitable):
    return asyncio.run(awaitable)


def _service(path: Path) -> ConversationService:
    service = ConversationService(path, chat_channel_id=123)
    _run(service.initialize())
    return service


def _allowed_mentions_are_none(kwargs: dict[str, object]) -> bool:
    mentions = kwargs["allowed_mentions"]
    return (
        isinstance(mentions, discord.AllowedMentions)
        and mentions.everyone is False
        and mentions.users is False
        and mentions.roles is False
        and mentions.replied_user is False
    )


def test_context_group_has_only_clear_command(tmp_path: Path) -> None:
    commands = ContextCommands(
        conversation_service=_service(tmp_path / "commands.sqlite3"),
        chat_channel_id=123,
    )

    assert [command.name for command in commands.group.commands] == ["clear"]


def test_context_command_registers_once_across_repeated_setup_hook(tmp_path: Path) -> None:
    async def run() -> tuple[list[int], list[str]]:
        service = ConversationService(tmp_path / "runtime.sqlite3", chat_channel_id=123)

        class Responder:
            async def generate(self, request: ChatRequest) -> str:
                return "unused"

        client = DiscordRuntimeClient(
            config=Config(discord_token="token", chat_channel_id=123),
            responder=Responder(),
            conversation_service=service,
        )

        async def fetch_channel(channel_id: int) -> object:
            return SimpleNamespace(
                id=channel_id,
                type=discord.ChannelType.text,
                guild=SimpleNamespace(id=20),
            )

        sync_calls: list[int] = []

        async def sync(*, guild: object) -> None:
            sync_calls.append(int(getattr(guild, "id")))

        client.fetch_channel = fetch_channel  # type: ignore[method-assign]
        client.tree.sync = sync  # type: ignore[method-assign]
        try:
            await client.setup_hook()
            await client.setup_hook()
            group = client.tree.get_command("context", guild=discord.Object(id=20))
            child_names = [command.name for command in group.commands] if group else []
            return sync_calls, child_names
        finally:
            await client.close()

    sync_calls, child_names = _run(run())
    assert sync_calls == [20]
    assert child_names == ["clear"]


def test_admin_clear_is_public_mentions_safe_and_does_not_generate(tmp_path: Path) -> None:
    service = _service(tmp_path / "commands.sqlite3")
    calls = 0
    original_clear = service.clear_context

    async def clear_context():
        nonlocal calls
        calls += 1
        return await original_clear()

    service.clear_context = clear_context  # type: ignore[method-assign]
    interaction = FakeInteraction(admin=True)
    commands = ContextCommands(conversation_service=service, chat_channel_id=123)

    _run(commands.handle_clear(interaction))

    assert calls == 1
    assert len(interaction.response.messages) == 1
    content, ephemeral, kwargs = interaction.response.messages[0]
    assert ephemeral is False
    assert "上下文" in content
    assert "新的對話" in content
    assert _allowed_mentions_are_none(kwargs)


def test_manage_guild_clear_uses_same_runtime_permission_boundary(tmp_path: Path) -> None:
    service = _service(tmp_path / "commands.sqlite3")
    interaction = FakeInteraction(admin=False, manage_guild=True)

    commands = ContextCommands(conversation_service=service, chat_channel_id=123)
    _run(commands.handle_clear(interaction))

    assert interaction.response.messages[0][1] is False
    assert service.active_epoch == 2


def test_unauthorized_clear_is_ephemeral_and_does_not_mutate(tmp_path: Path, caplog) -> None:
    service = _service(tmp_path / "commands.sqlite3")
    interaction = FakeInteraction(admin=False, manage_guild=False)
    commands = ContextCommands(conversation_service=service, chat_channel_id=123)

    with caplog.at_level(logging.WARNING):
        _run(commands.handle_clear(interaction))

    assert service.active_epoch == 1
    assert interaction.response.messages[0][1] is True
    assert "permission" in interaction.response.messages[0][0].lower()
    assert PRIVATE_BODY not in caplog.text


def test_wrong_channel_and_dm_are_ephemeral_and_do_not_mutate(tmp_path: Path) -> None:
    service = _service(tmp_path / "commands.sqlite3")
    commands = ContextCommands(conversation_service=service, chat_channel_id=123)
    wrong_channel = FakeInteraction(channel_id=999)
    dm = FakeInteraction(is_dm=True)

    _run(commands.handle_clear(wrong_channel))
    _run(commands.handle_clear(dm))

    assert service.active_epoch == 1
    assert wrong_channel.response.messages[0][1] is True
    assert dm.response.messages[0][1] is True


def test_clear_failure_is_ephemeral_and_logs_only_exception_type(tmp_path: Path, caplog) -> None:
    service = _service(tmp_path / "commands.sqlite3")

    async def fail_clear():
        raise ConversationError(PRIVATE_BODY)

    service.clear_context = fail_clear  # type: ignore[method-assign]
    interaction = FakeInteraction()
    commands = ContextCommands(conversation_service=service, chat_channel_id=123)

    with caplog.at_level(logging.ERROR):
        _run(commands.handle_clear(interaction))

    assert interaction.response.messages[0][1] is True
    assert "could not be cleared" in interaction.response.messages[0][0]
    assert PRIVATE_BODY not in caplog.text


def test_public_response_failure_does_not_rollback_committed_clear(tmp_path: Path, caplog) -> None:
    service = _service(tmp_path / "commands.sqlite3")
    interaction = FakeInteraction(response_error=RuntimeError(PRIVATE_BODY))
    commands = ContextCommands(conversation_service=service, chat_channel_id=123)

    with caplog.at_level(logging.ERROR):
        _run(commands.handle_clear(interaction))

    assert service.active_epoch == 2
    assert "public response failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert PRIVATE_BODY not in caplog.text

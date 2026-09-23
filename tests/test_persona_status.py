from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import discord

from liuer_bot.persona_models import PersonaStateSnapshot
from liuer_bot.persona_service import PersonaService
from liuer_bot.persona_status import (
    STATUS_EMBED_DESCRIPTION_LIMIT,
    PersonaStatusManager,
    excerpt_persona_text,
    render_persona_status,
)
from liuer_bot.prompting import PromptBuilder


def _run(awaitable):
    return asyncio.run(awaitable)


class FakeStatusMessage:
    def __init__(self, message_id: int, *, author_id: int, bot: bool = True) -> None:
        self.id = message_id
        self.author = SimpleNamespace(id=author_id, bot=bot)
        self.edits: list[dict[str, object]] = []

    async def edit(self, **kwargs: object) -> None:
        self.edits.append(kwargs)


class FakeStatusChannel:
    def __init__(self, *, message: FakeStatusMessage | None = None) -> None:
        self.id = 456
        self.guild = SimpleNamespace(id=789)
        self.type = discord.ChannelType.text
        self.message = message
        self.sent: list[dict[str, object]] = []
        self._next_id = 900

    async def fetch_message(self, message_id: int) -> FakeStatusMessage:
        if self.message is None or self.message.id != message_id:
            raise discord.NotFound(
                SimpleNamespace(status=404, reason="not found", headers={}),
                "not found",
            )
        return self.message

    async def send(self, **kwargs: object) -> FakeStatusMessage:
        self.sent.append(kwargs)
        self.message = FakeStatusMessage(self._next_id, author_id=42)
        self._next_id += 1
        return self.message


class FakeStatusClient:
    def __init__(self, channel: FakeStatusChannel) -> None:
        self.user = SimpleNamespace(id=42)
        self.channel = channel

    async def fetch_channel(self, channel_id: int) -> FakeStatusChannel:
        assert channel_id == self.channel.id
        return self.channel


def _service(path: Path) -> PersonaService:
    service = PersonaService(path)
    _run(service.initialize())
    return service


def test_excerpt_is_deterministic_and_respects_embed_budget() -> None:
    value = excerpt_persona_text("x" * 5_000)

    assert len(value) <= STATUS_EMBED_DESCRIPTION_LIMIT
    assert value.startswith("x")
    assert "excerpt" in value


def test_default_and_community_status_render_without_prompt_fallback_markers() -> None:
    builder = PromptBuilder()
    default_snapshot = PersonaStateSnapshot(0, None, None, datetime.now(UTC))
    community_snapshot = PersonaStateSnapshot(1, 12, "public community persona", datetime.now(UTC))

    default_rendered = render_persona_status(default_snapshot, builder)
    community_rendered = render_persona_status(community_snapshot, builder)

    assert "Persona" in default_rendered.embed.title
    assert "預設" in default_rendered.embed.title
    assert default_rendered.embed.description
    assert "Persona" in community_rendered.embed.title
    assert "自訂" in community_rendered.embed.title
    assert community_rendered.embed.description == "public community persona"


def test_status_render_shows_fixed_name_nickname_and_set_command() -> None:
    builder = PromptBuilder()
    snapshot = PersonaStateSnapshot(0, None, None, datetime.now(UTC))

    rendered = render_persona_status(snapshot, builder, active_nickname="小六")

    assert "固定名稱" in {field.name for field in rendered.embed.fields}
    assert "六耳" in {field.value for field in rendered.embed.fields}
    assert "小六" in {field.value for field in rendered.embed.fields}
    assert "/set name" in rendered.content


class FakeNicknameProvider:
    def __init__(self, nickname: str | None) -> None:
        self.nickname = nickname

    def get_active_nickname(self) -> str | None:
        return self.nickname


def test_status_manager_creates_then_edits_one_bot_owned_canonical_message(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3")
    channel = FakeStatusChannel()
    manager = PersonaStatusManager(
        client=FakeStatusClient(channel),
        status_channel_id=456,
        persona_service=service,
        prompt_builder=PromptBuilder(),
    )

    _run(manager.initialize())
    assert len(channel.sent) == 1
    payload = channel.sent[0]
    allowed_mentions = payload["allowed_mentions"]
    assert isinstance(allowed_mentions, discord.AllowedMentions)
    assert allowed_mentions.everyone is False
    assert allowed_mentions.roles is False
    assert allowed_mentions.users is False
    assert _run(service.get_status_message_id()) == channel.message.id

    _run(manager.sync_current())
    assert len(channel.sent) == 1
    assert len(channel.message.edits) == 1


def test_status_manager_edits_same_board_with_current_nickname(tmp_path: Path) -> None:
    service = _service(tmp_path / "nickname-status.sqlite3")
    channel = FakeStatusChannel()
    manager = PersonaStatusManager(
        client=FakeStatusClient(channel),
        status_channel_id=456,
        persona_service=service,
        prompt_builder=PromptBuilder(),
        nickname_service=FakeNicknameProvider("小六"),  # type: ignore[arg-type]
    )

    _run(manager.initialize())
    assert len(channel.sent) == 1
    assert len(channel.sent[0]["embed"].fields) == 2
    assert "小六" in {field.value for field in channel.sent[0]["embed"].fields}

    _run(manager.sync_current())
    assert len(channel.sent) == 1
    assert len(channel.message.edits) == 1


def test_status_manager_replaces_missing_or_non_bot_pointer(tmp_path: Path) -> None:
    service = _service(tmp_path / "persona.sqlite3")
    channel = FakeStatusChannel(message=FakeStatusMessage(901, author_id=77, bot=False))
    _run(service.set_status_message_id(901))
    manager = PersonaStatusManager(
        client=FakeStatusClient(channel),
        status_channel_id=456,
        persona_service=service,
        prompt_builder=PromptBuilder(),
    )

    _run(manager.initialize())

    assert len(channel.sent) == 1
    assert _run(service.get_status_message_id()) == 900

"""Public Persona status-channel rendering and canonical message lifecycle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import discord

from .addressing import PRIMARY_CALL_NAME
from .nickname_service import NicknameService
from .persona_models import PersonaStateSnapshot
from .persona_service import PersonaService
from .prompting import PromptBuilder, load_default_persona

LOGGER = logging.getLogger(__name__)
STATUS_EMBED_DESCRIPTION_LIMIT = 4_096

STATUS_INSTRUCTIONS = (
    "`/persona set`：設定新的全域 Persona\n"
    "`/persona status`：查看目前 Persona 狀態與使用限制\n\n"
    "`/set name`：修改六耳目前的全域小名\n\n"
    "管理員：\n"
    "`/persona reset`：恢復預設 Persona\n"
    "`/persona rollback`：回復上一版 Persona\n"
    "`/persona history`：查看 Persona 版本紀錄\n"
    "`/context clear`：清除目前對話上下文\n"
    "`/status`：查看六耳執行狀態\n"
    "`/health`：執行系統健康檢查\n"
    "`/botchat stop`：停止目前 Bot 對話\n"
    "`/botchat status`：查看 Bot 對話狀態\n\n"
    "對 Bot 訊息使用「讓六耳回覆此 Bot」可開始 Bot 對話。"
)


class PersonaStatusError(RuntimeError):
    """A status-channel operation failed without exposing Discord details."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"persona status operation failed: {operation}")


@dataclass(frozen=True, slots=True)
class RenderedPersonaStatus:
    """Safe presentation payload for one canonical Discord status message."""

    content: str
    embed: discord.Embed


def _default_persona(prompt_builder: PromptBuilder) -> str:
    getter = getattr(prompt_builder, "get_default_persona", None)
    if callable(getter):
        return str(getter())
    return load_default_persona()


def _display_persona(snapshot: PersonaStateSnapshot, prompt_builder: PromptBuilder) -> str:
    active = snapshot.active_persona
    return active if active is not None else _default_persona(prompt_builder)


def excerpt_persona_text(persona_text: str, *, limit: int = STATUS_EMBED_DESCRIPTION_LIMIT) -> str:
    """Keep public Persona rendering deterministic and within an embed budget."""

    if limit <= 0:
        raise ValueError("status excerpt limit must be positive")
    if len(persona_text) <= limit:
        return persona_text
    marker_template = "\n\n[excerpt: first {shown} of {total} characters]"
    marker = marker_template.format(shown=0, total=len(persona_text))
    shown = max(0, limit - len(marker))
    marker = marker_template.format(shown=shown, total=len(persona_text))
    shown = max(0, limit - len(marker))
    marker = marker_template.format(shown=shown, total=len(persona_text))
    return persona_text[:shown] + marker


def render_persona_status(
    snapshot: PersonaStateSnapshot,
    prompt_builder: PromptBuilder,
    *,
    active_nickname: str | None = None,
    description_limit: int = STATUS_EMBED_DESCRIPTION_LIMIT,
) -> RenderedPersonaStatus:
    """Build the public status message from the in-memory Persona snapshot."""

    persona_text = _display_persona(snapshot, prompt_builder)
    if snapshot.active_version_id is None:
        title = "六耳 Persona — 預設 Persona"
    else:
        title = f"六耳 Persona — 自訂 Persona v{snapshot.active_version_id}"
    embed = discord.Embed(
        title=title,
        description=excerpt_persona_text(persona_text, limit=description_limit),
    )
    embed.add_field(name="固定名稱", value=PRIMARY_CALL_NAME, inline=True)
    embed.add_field(
        name="目前小名",
        value=active_nickname.strip() if active_nickname and active_nickname.strip() else "未設定",
        inline=True,
    )
    return RenderedPersonaStatus(content=STATUS_INSTRUCTIONS, embed=embed)


def _is_supported_status_channel(channel: Any) -> bool:
    channel_type = getattr(channel, "type", None)
    supported_types = {
        getattr(discord.ChannelType, "text", None),
        getattr(discord.ChannelType, "news", None),
    }
    if channel_type is not None:
        return channel_type in supported_types
    channel_classes = tuple(
        channel_class
        for channel_class in (
            getattr(discord, "TextChannel", None),
            getattr(discord, "NewsChannel", None),
        )
        if isinstance(channel_class, type)
    )
    return isinstance(channel, channel_classes) or (
        getattr(channel, "guild", None) is not None and callable(getattr(channel, "send", None))
    )


def _allowed_mentions() -> discord.AllowedMentions:
    return discord.AllowedMentions.none()


class PersonaStatusManager:
    """Own one Bot-canonical status message in the configured guild channel."""

    def __init__(
        self,
        *,
        client: discord.Client,
        status_channel_id: int,
        persona_service: PersonaService,
        prompt_builder: PromptBuilder,
        nickname_service: NicknameService | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._client = client
        self._status_channel_id = status_channel_id
        self._persona_service = persona_service
        self._prompt_builder = prompt_builder
        self._nickname_service = nickname_service
        self._logger = logger
        self._channel: Any | None = None
        self._guild_id: int | None = None

    @property
    def guild_id(self) -> int | None:
        return self._guild_id

    @property
    def channel_id(self) -> int:
        return self._status_channel_id

    async def initialize(self) -> int:
        """Fetch, validate, and synchronize the channel once per client lifecycle."""

        guild_id = await self.initialize_channel()
        await self.sync_current()
        return guild_id

    async def initialize_channel(self) -> int:
        """Fetch and validate the configured channel without presentation side effects."""

        if self._channel is not None and self._guild_id is not None:
            return self._guild_id
        channel = await self._fetch_channel()
        self._validate_channel(channel)
        self._channel = channel
        self._guild_id = int(channel.guild.id)
        return self._guild_id

    async def sync_current(self) -> None:
        """Refresh or create the one stored canonical message."""

        channel = self._channel
        if channel is None:
            channel = await self._fetch_channel()
            self._validate_channel(channel)
            self._channel = channel
            self._guild_id = int(channel.guild.id)

        rendered = render_persona_status(
            self._persona_service.current_snapshot,
            self._prompt_builder,
            active_nickname=(
                self._nickname_service.get_active_nickname()
                if self._nickname_service is not None
                else None
            ),
        )
        payload = {
            "content": rendered.content,
            "embed": rendered.embed,
            "allowed_mentions": _allowed_mentions(),
        }
        stored_id = await self._persona_service.get_status_message_id()
        existing = None
        if stored_id is not None:
            try:
                existing = await channel.fetch_message(stored_id)
            except discord.NotFound:
                existing = None
            except Exception as exc:
                self._logger.error(
                    "Persona status fetch failed channel_id=%s error_type=%s",
                    self._status_channel_id,
                    type(exc).__name__,
                )
                raise PersonaStatusError("fetch status message") from exc

        if existing is not None and self._is_bot_owned(existing):
            try:
                await existing.edit(**payload)
                return
            except Exception as exc:
                self._logger.error(
                    "Persona status edit failed channel_id=%s error_type=%s",
                    self._status_channel_id,
                    type(exc).__name__,
                )
                raise PersonaStatusError("edit status message") from exc

        try:
            created = await channel.send(**payload)
        except Exception as exc:
            self._logger.error(
                "Persona status create failed channel_id=%s error_type=%s",
                self._status_channel_id,
                type(exc).__name__,
            )
            raise PersonaStatusError("create status message") from exc
        created_id = getattr(created, "id", None)
        if not isinstance(created_id, int) or created_id <= 0:
            raise PersonaStatusError("create status message")
        try:
            await self._persona_service.set_status_message_id(created_id)
        except Exception as exc:
            self._logger.error(
                "Persona status pointer save failed channel_id=%s error_type=%s",
                self._status_channel_id,
                type(exc).__name__,
            )
            raise PersonaStatusError("save status message") from exc

    async def _fetch_channel(self) -> Any:
        try:
            return await self._client.fetch_channel(self._status_channel_id)
        except Exception as exc:
            self._logger.error(
                "Persona status channel fetch failed channel_id=%s error_type=%s",
                self._status_channel_id,
                type(exc).__name__,
            )
            raise PersonaStatusError("fetch status channel") from exc

    def _validate_channel(self, channel: Any) -> None:
        if getattr(channel, "id", None) != self._status_channel_id:
            raise PersonaStatusError("status channel identity")
        guild = getattr(channel, "guild", None)
        if guild is None or getattr(guild, "id", None) is None:
            raise PersonaStatusError("status channel guild")
        if not _is_supported_status_channel(channel):
            raise PersonaStatusError("status channel type")

    def _is_bot_owned(self, message: Any) -> bool:
        bot_user = self._client.user
        author = getattr(message, "author", None)
        return (
            bot_user is not None
            and author is not None
            and getattr(author, "id", None) == getattr(bot_user, "id", None)
            and bool(getattr(author, "bot", True))
        )


__all__ = [
    "PersonaStatusError",
    "PersonaStatusManager",
    "RenderedPersonaStatus",
    "STATUS_EMBED_DESCRIPTION_LIMIT",
    "STATUS_INSTRUCTIONS",
    "excerpt_persona_text",
    "render_persona_status",
]

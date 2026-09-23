"""Guild-scoped BotChat context-menu and session-control commands."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import discord
from discord import app_commands

from .attachment_metadata import classify_image_attachment
from .bot_dialogue import BotDialogueAlreadyActiveError, BotDialogueService
from .persona_commands import _permission_allowed

LOGGER = logging.getLogger(__name__)
BOT_DIALOGUE_CONTEXT_MENU_NAME = "讓六耳回覆此 Bot"


def _is_thread_channel(channel: object) -> bool:
    if isinstance(channel, discord.Thread) or bool(getattr(channel, "is_thread", False)):
        return True
    channel_type = getattr(channel, "type", None)
    return channel_type in {
        getattr(discord.ChannelType, "public_thread", None),
        getattr(discord.ChannelType, "private_thread", None),
        getattr(discord.ChannelType, "news_thread", None),
    }


def _display_name(author: object) -> str:
    value = getattr(author, "display_name", None)
    if isinstance(value, str):
        return value
    value = getattr(author, "name", None)
    return value if isinstance(value, str) else "Discord Bot"


def _channel_id(interaction: discord.Interaction) -> int | None:
    value = getattr(interaction, "channel_id", None)
    if isinstance(value, int):
        return value
    channel = getattr(interaction, "channel", None)
    value = getattr(channel, "id", None)
    return value if isinstance(value, int) else None


def _guild_id(value: object) -> int | None:
    guild_id = getattr(value, "id", None)
    return guild_id if isinstance(guild_id, int) else None


class BotDialogueCommandGroup(app_commands.Group):
    """The admin-only `/botchat` group without a start command."""

    def __init__(self, commands: BotDialogueCommands) -> None:
        self._commands = commands
        super().__init__(name="botchat", description="管理與查看 Bot 對話")

    @app_commands.command(name="stop", description="停止目前的 Bot 對話")
    @app_commands.guild_only()
    async def stop(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_stop(interaction)

    @app_commands.command(name="status", description="查看目前的 Bot 對話狀態")
    @app_commands.guild_only()
    async def status(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_status(interaction)


class BotDialogueCommands:
    """Keep BotChat interaction policy outside the Discord event handler."""

    def __init__(
        self,
        *,
        dialogue_service: BotDialogueService,
        chat_channel_id: int,
        status_channel_id: int | None,
        bot_user_id_provider: Callable[[], int | None],
        initial_message_handler: Callable[[discord.Message, str], Awaitable[None]],
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.dialogue_service = dialogue_service
        self.chat_channel_id = chat_channel_id
        self.status_channel_id = status_channel_id
        self._bot_user_id_provider = bot_user_id_provider
        self._initial_message_handler = initial_message_handler
        self._logger = logger
        self.group = BotDialogueCommandGroup(self)
        self.context_menu = app_commands.ContextMenu(
            name=BOT_DIALOGUE_CONTEXT_MENU_NAME,
            callback=self.handle_context_menu,
        )

    @property
    def commands(self) -> tuple[app_commands.Group, app_commands.ContextMenu]:
        return self.group, self.context_menu

    async def handle_context_menu(
        self,
        interaction: discord.Interaction,
        target: discord.Message,
    ) -> None:
        if interaction.guild is None:
            await self._respond(interaction, "此功能只能在伺服器中使用。")
            return
        if not _permission_allowed(interaction):
            await self._respond(interaction, "需要管理員或管理伺服器權限。")
            return
        if not self._target_is_eligible(interaction, target):
            await self._respond(
                interaction,
                "只能選取同一伺服器、設定聊天頻道中的其他 Bot 訊息。",
            )
            return
        try:
            session = self.dialogue_service.activate(
                target.author.id,
                _display_name(target.author),
            )
        except BotDialogueAlreadyActiveError:
            await self._respond(interaction, "目前已經有一段 Bot 對話進行中。")
            return
        except Exception as exc:
            self._logger.error(
                "BotChat activation failed error_type=%s",
                type(exc).__name__,
            )
            await self._respond(interaction, "Bot 對話目前無法開始。")
            return

        try:
            await self._respond(
                interaction,
                f"已開始與 {session.peer_display_name} 的 Bot 對話，六耳正在回覆這則訊息。",
            )
            await self._initial_message_handler(target, session.session_id)
        except asyncio.CancelledError:
            self.dialogue_service.close()
            raise
        except Exception as exc:
            self.dialogue_service.close()
            self._logger.error(
                "BotChat initial response failed error_type=%s",
                type(exc).__name__,
            )

    async def handle_stop(self, interaction: discord.Interaction) -> None:
        if not await self._authorize(interaction, allow_chat=True, allow_status=True):
            return
        closed = self.dialogue_service.close()
        message = "Bot 對話已停止。" if closed else "目前沒有進行中的 Bot 對話。"
        await self._respond(interaction, message)

    async def handle_status(self, interaction: discord.Interaction) -> None:
        if not await self._authorize(interaction, allow_chat=False, allow_status=True):
            return
        session = self.dialogue_service.get_session()
        if session is None:
            await self._respond(interaction, "Bot 對話：未啟用。")
            return
        idle_age = self.dialogue_service.idle_age_seconds()
        remaining = self.dialogue_service.remaining_timeout_seconds()
        await self._respond(
            interaction,
            "Bot 對話：進行中\n"
            f"對象：{session.peer_display_name}\n"
            f"六耳回覆：{session.liuer_reply_count} / "
            f"{self.dialogue_service.max_replies}\n"
            f"等待對方：{'是' if session.awaiting_peer else '否'}\n"
            f"閒置：{idle_age if idle_age is not None else 0} 秒\n"
            f"剩餘逾時：{remaining if remaining is not None else 0} 秒",
        )

    def _target_is_eligible(
        self,
        interaction: discord.Interaction,
        target: discord.Message,
    ) -> bool:
        guild = interaction.guild
        target_guild = getattr(target, "guild", None)
        author = getattr(target, "author", None)
        bot_user_id = self._bot_user_id_provider()
        if (
            guild is None
            or target_guild is None
            or _guild_id(target_guild) != _guild_id(guild)
        ):
            return False
        if getattr(target.channel, "id", None) != self.chat_channel_id:
            return False
        if _is_thread_channel(target.channel):
            return False
        if (
            author is None
            or not bool(getattr(author, "bot", False))
            or not isinstance(bot_user_id, int)
            or bot_user_id <= 0
        ):
            return False
        if not isinstance(getattr(author, "id", None), int) or author.id == bot_user_id:
            return False
        if getattr(target, "webhook_id", None) is not None:
            return False
        if isinstance(getattr(target, "content", None), str) and target.content.strip():
            return True
        try:
            return any(
                classify_image_attachment(attachment) is not None
                for attachment in target.attachments
            )
        except (AttributeError, TypeError, ValueError):
            return False

    async def _authorize(
        self,
        interaction: discord.Interaction,
        *,
        allow_chat: bool,
        allow_status: bool,
    ) -> bool:
        if interaction.guild is None:
            await self._respond(interaction, "此功能只能在伺服器中使用。")
            return False
        if not _permission_allowed(interaction):
            await self._respond(interaction, "需要管理員或管理伺服器權限。")
            return False
        channel_id = _channel_id(interaction)
        allowed = (allow_chat and channel_id == self.chat_channel_id) or (
            allow_status
            and self.status_channel_id is not None
            and channel_id == self.status_channel_id
        )
        if not allowed:
            await self._respond(interaction, "請在設定的聊天頻道或公告／狀態頻道使用此指令。")
            return False
        return True

    async def _respond(self, interaction: discord.Interaction, message: str) -> None:
        response = interaction.response
        done = response.is_done() if callable(getattr(response, "is_done", None)) else False
        if not done:
            await response.send_message(
                message,
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        else:
            await interaction.followup.send(
                message,
                ephemeral=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )


__all__ = [
    "BOT_DIALOGUE_CONTEXT_MENU_NAME",
    "BotDialogueCommandGroup",
    "BotDialogueCommands",
]

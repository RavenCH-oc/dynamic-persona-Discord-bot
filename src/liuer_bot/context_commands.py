"""Guild-scoped administrative conversation-context command boundary."""

from __future__ import annotations

import logging

import discord
from discord import app_commands

from .bot_dialogue import BotDialogueService
from .conversation_service import ConversationError, ConversationService
from .persona_commands import _permission_allowed

LOGGER = logging.getLogger(__name__)


class ContextCommandGroup(app_commands.Group):
    """The Phase 5B `/context` group with only the clear operation."""

    def __init__(self, commands: ContextCommands) -> None:
        self._commands = commands
        super().__init__(name="context", description="管理目前對話上下文")

    @app_commands.command(name="clear", description="清除目前對話上下文（管理員）")
    @app_commands.guild_only()
    async def clear(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_clear(interaction)


class ContextCommands:
    """Keep Discord interaction handling thin around ConversationService."""

    def __init__(
        self,
        *,
        conversation_service: ConversationService,
        chat_channel_id: int,
        logger: logging.Logger = LOGGER,
        bot_dialogue_service: BotDialogueService | None = None,
    ) -> None:
        self.conversation_service = conversation_service
        self.chat_channel_id = chat_channel_id
        self.bot_dialogue_service = bot_dialogue_service
        self._logger = logger
        self.group = ContextCommandGroup(self)

    async def handle_clear(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await self._respond(interaction, "Context commands are available in a server only.")
            return
        if not _permission_allowed(interaction):
            self._logger.warning(
                "Context command permission denied command=clear user_id=%s",
                getattr(interaction.user, "id", None),
            )
            await self._respond(
                interaction,
                "Administrator or Manage Server permission is required.",
            )
            return
        if _interaction_channel_id(interaction) != self.chat_channel_id:
            await self._respond(
                interaction,
                "Context clear is available in the configured chat channel only.",
            )
            return

        try:
            await self.conversation_service.clear_context()
        except ConversationError as exc:
            self._logger.error(
                "Context clear failed command=clear error_type=%s",
                type(exc).__name__,
            )
            await self._respond(interaction, "The conversation context could not be cleared.")
            return
        except Exception as exc:
            self._logger.error(
                "Context clear failed command=clear error_type=%s",
                type(exc).__name__,
            )
            await self._respond(interaction, "The conversation context could not be cleared.")
            return

        if self.bot_dialogue_service is not None:
            self.bot_dialogue_service.close()
        try:
            await self._respond(
                interaction,
                "六耳的對話上下文已由管理員清除。接下來將從新的對話開始。",
                ephemeral=False,
            )
        except Exception as exc:
            self._logger.error(
                "Context clear public response failed command=clear error_type=%s",
                type(exc).__name__,
            )

    async def _respond(
        self,
        interaction: discord.Interaction,
        message: str,
        *,
        ephemeral: bool = True,
    ) -> None:
        response = interaction.response
        done = response.is_done() if callable(getattr(response, "is_done", None)) else False
        if not done:
            await response.send_message(
                message,
                ephemeral=ephemeral,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        else:
            await interaction.followup.send(
                message,
                ephemeral=ephemeral,
                allowed_mentions=discord.AllowedMentions.none(),
            )


def _interaction_channel_id(interaction: discord.Interaction) -> int | None:
    channel_id = getattr(interaction, "channel_id", None)
    if isinstance(channel_id, int):
        return channel_id
    channel = getattr(interaction, "channel", None)
    value = getattr(channel, "id", None)
    return value if isinstance(value, int) else None


__all__ = ["ContextCommandGroup", "ContextCommands"]

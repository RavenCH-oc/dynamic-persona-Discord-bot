"""Guild-scoped ephemeral /status and /health command boundary."""

from __future__ import annotations

import logging

import discord
from discord import app_commands

from .persona_commands import _permission_allowed
from .runtime_health import RuntimeHealthService
from .runtime_status import RuntimeStatusService

LOGGER = logging.getLogger(__name__)


class RuntimeCommands:
    """Keep read-only operational commands separate from Discord probe logic."""

    def __init__(
        self,
        *,
        status_channel_id: int,
        status_service: RuntimeStatusService,
        health_service: RuntimeHealthService,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.status_channel_id = status_channel_id
        self.status_service = status_service
        self.health_service = health_service
        self._logger = logger

        async def status_callback(interaction: discord.Interaction) -> None:
            await self.handle_status(interaction)

        async def health_callback(interaction: discord.Interaction) -> None:
            await self.handle_health(interaction)

        self.status = app_commands.Command(
            name="status",
            description="查看六耳目前的執行狀態（管理員）",
            callback=status_callback,
        )
        self.health = app_commands.Command(
            name="health",
            description="檢查六耳、資料庫與 LM Studio 健康狀態（管理員）",
            callback=health_callback,
        )

    @property
    def commands(self) -> tuple[app_commands.Command[object, ..., object], ...]:
        return (self.status, self.health)

    async def handle_status(self, interaction: discord.Interaction) -> None:
        if not await self._authorize(interaction):
            return
        try:
            response = self.status_service.render()
        except Exception as exc:
            self._logger.error(
                "runtime status command failed error_type=%s",
                type(exc).__name__,
            )
            await self._respond(interaction, "Runtime status is temporarily unavailable.")
            return
        await self._respond(interaction, response)

    async def handle_health(self, interaction: discord.Interaction) -> None:
        if not await self._authorize(interaction):
            return
        try:
            response = await self.health_service.check_and_render()
        except Exception as exc:
            self._logger.error(
                "runtime health command failed error_type=%s",
                type(exc).__name__,
            )
            await self._respond(interaction, "Runtime health is temporarily unavailable.")
            return
        await self._respond(interaction, response)

    async def _authorize(self, interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            await self._respond(interaction, "This command is available in a server only.")
            return False
        if not _permission_allowed(interaction):
            self._logger.warning(
                "runtime command permission denied user_id=%s",
                getattr(interaction.user, "id", None),
            )
            await self._respond(
                interaction,
                "Administrator or Manage Server permission is required.",
            )
            return False
        if _interaction_channel_id(interaction) != self.status_channel_id:
            await self._respond(
                interaction,
                "請在六耳的公告／狀態頻道使用此指令。",
            )
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


def _interaction_channel_id(interaction: discord.Interaction) -> int | None:
    channel_id = getattr(interaction, "channel_id", None)
    if isinstance(channel_id, int):
        return channel_id
    channel = getattr(interaction, "channel", None)
    value = getattr(channel, "id", None)
    return value if isinstance(value, int) else None


__all__ = ["RuntimeCommands"]

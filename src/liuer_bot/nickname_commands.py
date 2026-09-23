"""Guild-scoped `/set name` command boundary for Active Nickname."""

from __future__ import annotations

import logging
from datetime import datetime

import discord
from discord import app_commands

from .nickname_models import (
    NicknameDailyLimitError,
    NicknameError,
    NicknameGlobalCooldownError,
    NicknameTooLongError,
    NicknameValidationError,
)
from .nickname_service import NicknameService
from .persona_status import PersonaStatusManager

LOGGER = logging.getLogger(__name__)


def _user_id(interaction: discord.Interaction) -> int:
    return int(interaction.user.id)


def _format_timestamp(value: datetime) -> str:
    return discord.utils.format_dt(value, style="R")


class NicknameSetCommandGroup(app_commands.Group):
    """The `/set` group containing the Phase 4A-A2 name command."""

    def __init__(self, commands: NicknameCommands) -> None:
        self._commands = commands
        super().__init__(name="set", description="設定六耳全域小名")

    @app_commands.command(name="name", description="修改六耳目前的全域小名")
    @app_commands.guild_only()
    @app_commands.describe(nickname="六耳的簡短全域小名")
    async def set_name_command(self, interaction: discord.Interaction, nickname: str) -> None:
        await self._commands.handle_set_name(interaction, nickname)


class NicknameCommands:
    """Keep slash-command transport thin; NicknameService owns all policy."""

    def __init__(
        self,
        *,
        nickname_service: NicknameService,
        status_manager: PersonaStatusManager,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.nickname_service = nickname_service
        self.status_manager = status_manager
        self._logger = logger
        self.group = NicknameSetCommandGroup(self)

    async def handle_set_name(
        self,
        interaction: discord.Interaction,
        nickname: str,
    ) -> None:
        if interaction.guild is None:
            await self._respond(interaction, "Nickname commands are available in a server only.")
            return
        try:
            result = await self.nickname_service.set_nickname(_user_id(interaction), nickname)
        except NicknameError as exc:
            await self._respond_error(interaction, exc)
            return
        except Exception as exc:
            self._logger.error(
                "Nickname command failed error_type=%s",
                type(exc).__name__,
            )
            await self._respond(
                interaction,
                "The nickname operation could not be completed.",
            )
            return

        warning = False
        try:
            await self.status_manager.sync_current()
        except Exception as exc:
            warning = True
            self._logger.error(
                "Nickname status sync failed error_type=%s",
                type(exc).__name__,
            )
        message = f"六耳目前的小名已更新為「{result.version.nickname_text}」。"
        if warning:
            message += " 公開 Persona 狀態訊息稍後才會更新。"
        await self._respond(interaction, message, ephemeral=warning)

    async def _respond_error(self, interaction: discord.Interaction, exc: NicknameError) -> None:
        if isinstance(exc, NicknameGlobalCooldownError):
            message = (
                "Nickname 更新仍在全域冷卻中；請稍後再試。"
                f"（約 {_format_timestamp(self._future_timestamp(exc.retry_after_seconds))}）"
            )
        elif isinstance(exc, NicknameDailyLimitError):
            message = (
                f"今日 nickname 更新次數已達上限（{exc.daily_limit} 次）；"
                f"請於 {_format_timestamp(exc.next_reset_utc)} 後再試。"
            )
        elif isinstance(exc, NicknameTooLongError):
            message = f"Nickname 最多 {exc.max_chars} 個字元。"
        elif isinstance(exc, NicknameValidationError):
            message = "Nickname 不符合格式要求。"
        else:
            message = "Nickname operation could not be completed."
        await self._respond(interaction, message)

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

    @staticmethod
    def _future_timestamp(seconds: int) -> datetime:
        from datetime import UTC, timedelta

        return datetime.now(UTC) + timedelta(seconds=seconds)


__all__ = ["NicknameCommands", "NicknameSetCommandGroup"]

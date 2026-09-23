"""Guild-scoped Discord Persona commands and modal interaction boundary."""

from __future__ import annotations

import logging
from datetime import datetime

import discord
from discord import app_commands

from .persona_models import (
    PersonaDailyLimitError,
    PersonaError,
    PersonaGlobalCooldownError,
    PersonaNoRollbackTargetError,
    PersonaTooLongError,
    PersonaValidationError,
)
from .persona_service import PersonaService
from .persona_status import PersonaStatusManager

LOGGER = logging.getLogger(__name__)
DISCORD_MODAL_MAX_LENGTH = 4_000


def _user_id(interaction: discord.Interaction) -> int:
    return int(interaction.user.id)


def _format_timestamp(value: datetime) -> str:
    return discord.utils.format_dt(value, style="R")


def _permission_allowed(interaction: discord.Interaction) -> bool:
    if interaction.guild is None:
        return False
    permissions = getattr(interaction.user, "guild_permissions", None)
    return bool(
        getattr(permissions, "administrator", False)
        or getattr(permissions, "manage_guild", False)
    )


class PersonaSetModal(discord.ui.Modal):
    """Multiline Persona submission modal with no title/name field."""

    def __init__(self, commands: PersonaCommands) -> None:
        super().__init__(title="Set Persona", timeout=300)
        self._commands = commands
        self.persona = discord.ui.TextInput(
            label="Persona",
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=min(commands.max_persona_chars, DISCORD_MODAL_MAX_LENGTH),
            placeholder="Describe the style and personality for the Bot.",
        )
        self.add_item(self.persona)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_set_submit(interaction, self.persona.value)


class PersonaCommandGroup(app_commands.Group):
    """The exact five-command `/persona` group, with no public `show` command."""

    def __init__(self, commands: PersonaCommands) -> None:
        self._commands = commands
        super().__init__(name="persona", description="管理與查看全域 Persona")

    @app_commands.command(name="set", description="設定新的全域 Persona")
    @app_commands.guild_only()
    async def set(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_set(interaction)

    @app_commands.command(name="status", description="查看目前 Persona 狀態與使用限制")
    @app_commands.guild_only()
    async def status(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_status(interaction)

    @app_commands.command(name="reset", description="恢復預設 Persona（管理員）")
    @app_commands.guild_only()
    async def reset(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_reset(interaction)

    @app_commands.command(name="rollback", description="回復上一版 Persona（管理員）")
    @app_commands.guild_only()
    async def rollback(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_rollback(interaction)

    @app_commands.command(name="history", description="查看 Persona 版本紀錄（管理員）")
    @app_commands.guild_only()
    async def history(self, interaction: discord.Interaction) -> None:
        await self._commands.handle_history(interaction)


class PersonaCommands:
    """Keep Interaction handling thin; PersonaService remains policy authority."""

    def __init__(
        self,
        *,
        persona_service: PersonaService,
        status_manager: PersonaStatusManager,
        max_persona_chars: int,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.persona_service = persona_service
        self.status_manager = status_manager
        self.max_persona_chars = max_persona_chars
        self._logger = logger
        self.group = PersonaCommandGroup(self)

    async def handle_set(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await self._respond(interaction, "Persona commands are available in a server only.")
            return
        try:
            policy = await self.persona_service.get_policy_status(_user_id(interaction))
        except PersonaError as exc:
            await self._respond_error(interaction, exc)
            return
        except Exception as exc:
            await self._respond_internal(interaction, exc)
            return
        if policy.global_cooldown_remaining_seconds > 0:
            await self._respond(
                interaction,
                "Persona publishing is on global cooldown; try again "
                f"{_format_timestamp(self._future_timestamp(policy.global_cooldown_remaining_seconds))}.",
            )
            return
        if policy.daily_remaining <= 0:
            await self._respond(
                interaction,
                "Your daily Persona publish limit is reached; try again "
                f"{_format_timestamp(policy.next_daily_reset_utc)}.",
            )
            return
        await interaction.response.send_modal(PersonaSetModal(self))

    async def handle_set_submit(self, interaction: discord.Interaction, persona_text: str) -> None:
        try:
            result = await self.persona_service.publish_persona(_user_id(interaction), persona_text)
        except PersonaError as exc:
            await self._respond_error(interaction, exc)
            return
        except Exception as exc:
            await self._respond_internal(interaction, exc)
            return
        warning = await self._sync_after_mutation()
        message = (
            f"Persona published as community version v{result.version.version_id}. "
            f"Global cooldown: {result.policy_status.global_cooldown_remaining_seconds}s."
        )
        if warning:
            message += " The public status message could not be refreshed yet."
        await self._respond(interaction, message)

    async def handle_status(self, interaction: discord.Interaction) -> None:
        if interaction.guild is None:
            await self._respond(interaction, "Persona commands are available in a server only.")
            return
        try:
            policy = await self.persona_service.get_policy_status(_user_id(interaction))
            snapshot = self.persona_service.current_snapshot
        except PersonaError as exc:
            await self._respond_error(interaction, exc)
            return
        except Exception as exc:
            await self._respond_internal(interaction, exc)
            return
        active_label = (
            f"Community v{snapshot.active_version_id}"
            if snapshot.active_version_id is not None
            else "Default"
        )
        cooldown = (
            f"{policy.global_cooldown_remaining_seconds}s remaining"
            if policy.global_cooldown_remaining_seconds
            else "available"
        )
        response = (
            f"Active Persona: {active_label}\n"
            f"Global cooldown: {cooldown}\n"
            f"Daily publishes: {policy.daily_used}/{policy.daily_limit} "
            f"({policy.daily_remaining} remaining)\n"
            f"Daily reset: {_format_timestamp(policy.next_daily_reset_utc)}\n"
            f"Public status: <#{self.status_manager.channel_id}>"
        )
        await self._respond(interaction, response)

    async def handle_reset(self, interaction: discord.Interaction) -> None:
        if not await self._require_admin(interaction, "reset"):
            return
        try:
            snapshot = await self.persona_service.reset_to_default(_user_id(interaction))
        except PersonaError as exc:
            await self._respond_error(interaction, exc)
            return
        except Exception as exc:
            await self._respond_internal(interaction, exc)
            return
        warning = await self._sync_after_mutation()
        message = "Active Persona reset to the packaged Default Persona."
        if snapshot.active_version_id is not None:
            message = "Active Persona reset operation completed."
        if warning:
            message += " The public status message could not be refreshed yet."
        await self._respond(interaction, message)

    async def handle_rollback(self, interaction: discord.Interaction) -> None:
        if not await self._require_admin(interaction, "rollback"):
            return
        try:
            snapshot = await self.persona_service.rollback(_user_id(interaction))
        except PersonaError as exc:
            await self._respond_error(interaction, exc)
            return
        except Exception as exc:
            await self._respond_internal(interaction, exc)
            return
        warning = await self._sync_after_mutation()
        message = f"Active Persona rolled back to community version v{snapshot.active_version_id}."
        if warning:
            message += " The public status message could not be refreshed yet."
        await self._respond(interaction, message)

    async def handle_history(self, interaction: discord.Interaction) -> None:
        if not await self._require_admin(interaction, "history"):
            return
        try:
            history = await self.persona_service.list_history(limit=10)
            active_id = self.persona_service.current_snapshot.active_version_id
        except PersonaError as exc:
            await self._respond_error(interaction, exc)
            return
        except Exception as exc:
            await self._respond_internal(interaction, exc)
            return
        if not history:
            await self._respond(interaction, "Persona history is empty.")
            return
        lines = ["Persona history (metadata only):"]
        for version in history:
            marker = " [active]" if version.version_id == active_id else ""
            lines.append(
                f"v{version.version_id} — author {version.author_discord_id} — "
                f"{_format_timestamp(version.created_at_utc)}{marker}"
            )
        await self._respond(interaction, "\n".join(lines))

    async def _require_admin(self, interaction: discord.Interaction, command: str) -> bool:
        if not _permission_allowed(interaction):
            self._logger.warning(
                "Persona command permission denied command=%s user_id=%s",
                command,
                getattr(interaction.user, "id", None),
            )
            await self._respond(
                interaction,
                "Administrator or Manage Server permission is required.",
            )
            return False
        return True

    async def _sync_after_mutation(self) -> bool:
        try:
            await self.status_manager.sync_current()
        except Exception as exc:
            self._logger.error("Persona status sync failed error_type=%s", type(exc).__name__)
            return True
        return False

    async def _respond_error(self, interaction: discord.Interaction, exc: PersonaError) -> None:
        if isinstance(exc, PersonaGlobalCooldownError):
            message = "Persona publishing is on global cooldown."
        elif isinstance(exc, PersonaDailyLimitError):
            message = (
                "Your daily Persona publish limit is reached; try again "
                f"{_format_timestamp(exc.next_reset_utc)}."
            )
        elif isinstance(exc, PersonaTooLongError):
            message = f"Persona must be at most {exc.max_chars} characters."
        elif isinstance(exc, PersonaValidationError):
            message = "Persona text must not be blank."
        elif isinstance(exc, PersonaNoRollbackTargetError):
            message = "No older Persona version is available for rollback."
        else:
            message = "The Persona operation could not be completed."
        await self._respond(interaction, message)

    async def _respond_internal(self, interaction: discord.Interaction, exc: Exception) -> None:
        self._logger.error(
            "Persona command failed error_type=%s",
            type(exc).__name__,
        )
        await self._respond(interaction, "The Persona operation could not be completed.")

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

    @staticmethod
    def _future_timestamp(seconds: int) -> datetime:
        from datetime import UTC, timedelta

        return datetime.now(UTC) + timedelta(seconds=seconds)


__all__ = [
    "DISCORD_MODAL_MAX_LENGTH",
    "PersonaCommandGroup",
    "PersonaCommands",
    "PersonaSetModal",
]

"""Best-effort Discord guild nickname presentation for Active Nickname."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import discord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NicknameSyncResult:
    """Metadata-only outcome; no nickname or Discord response body is retained."""

    succeeded: bool
    changed: bool


class NicknameSynchronizer(Protocol):
    async def synchronize(self, desired_nickname: str | None) -> NicknameSyncResult: ...


class DiscordNicknameSynchronizer:
    """Synchronize only the configured chat guild's own Bot member nickname."""

    def __init__(
        self,
        *,
        guild_provider: Callable[[], discord.Guild | None],
        bot_user_id_provider: Callable[[], int | None],
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._guild_provider = guild_provider
        self._bot_user_id_provider = bot_user_id_provider
        self._logger = logger

    async def synchronize(self, desired_nickname: str | None) -> NicknameSyncResult:
        """Edit only when needed; Discord failures never change domain state."""

        guild_id: int | None = None
        try:
            guild = self._guild_provider()
            if guild is None:
                raise RuntimeError("configured guild is unavailable")
            guild_id = guild.id
            member = guild.me
            if member is None:
                bot_user_id = self._bot_user_id_provider()
                if bot_user_id is None:
                    raise RuntimeError("current Bot user is unavailable")
                member = await guild.fetch_member(bot_user_id)
            if member.nick == desired_nickname:
                self._logger.debug(
                    "Discord nickname already synchronized guild_id=%s changed=False",
                    guild_id,
                )
                return NicknameSyncResult(succeeded=True, changed=False)
            await member.edit(nick=desired_nickname)
            self._logger.debug(
                "Discord nickname synchronized guild_id=%s changed=True",
                guild_id,
            )
            return NicknameSyncResult(succeeded=True, changed=True)
        except Exception as exc:
            self._logger.warning(
                "Discord nickname synchronization failed guild_id=%s error_type=%s",
                guild_id,
                type(exc).__name__,
            )
            return NicknameSyncResult(succeeded=False, changed=False)


__all__ = [
    "DiscordNicknameSynchronizer",
    "NicknameSynchronizer",
    "NicknameSyncResult",
]

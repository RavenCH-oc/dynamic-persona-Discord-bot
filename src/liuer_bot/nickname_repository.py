"""Synchronous SQLite repository for the global nickname domain."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from pathlib import Path

from .database import DatabaseError, connect_database, initialize_database
from .nickname_models import (
    NicknameDailyLimitError,
    NicknameGlobalCooldownError,
    NicknameStateSnapshot,
    NicknameVersion,
)
from .persona_time import format_utc, parse_utc


class NicknameDatabaseError(DatabaseError):
    """A nickname repository operation failed without exposing values."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"nickname database operation failed: {operation}")


def _safe_parse_utc(value: object, operation: str) -> datetime:
    try:
        return parse_utc(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise NicknameDatabaseError(operation) from exc


class NicknameRepository:
    """Short-connection repository; callers own the async boundary."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)

    def initialize(self) -> None:
        initialize_database(self._database_path)

    def load_state(self) -> NicknameStateSnapshot:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            row = connection.execute(
                "SELECT state.revision, state.active_version_id, state.updated_at_utc, "
                "version.nickname_text FROM nickname_state AS state "
                "LEFT JOIN nickname_versions AS version "
                "ON version.id = state.active_version_id "
                "WHERE state.singleton_id = 1"
            ).fetchone()
            if row is None:
                raise NicknameDatabaseError("load state")
            active_nickname = None if row[1] is None else row[3]
            if row[1] is not None and (
                not isinstance(active_nickname, str) or not active_nickname
            ):
                raise NicknameDatabaseError("load state")
            return NicknameStateSnapshot(
                revision=int(row[0]),
                active_version_id=None if row[1] is None else int(row[1]),
                active_nickname=active_nickname,
                updated_at_utc=_safe_parse_utc(row[2], "load state"),
            )
        except NicknameDatabaseError:
            raise
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise NicknameDatabaseError("load state") from exc
        finally:
            if connection is not None:
                connection.close()

    def policy_metrics(
        self,
        author_discord_id: str,
        quota_start_utc: datetime,
        quota_end_utc: datetime,
    ) -> tuple[datetime | None, int]:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            latest_row = connection.execute(
                "SELECT created_at_utc FROM nickname_versions "
                "ORDER BY created_at_utc DESC, id DESC LIMIT 1"
            ).fetchone()
            daily_row = connection.execute(
                "SELECT COUNT(*) FROM nickname_versions "
                "WHERE author_discord_id = ? AND created_at_utc >= ? AND created_at_utc < ?",
                (author_discord_id, format_utc(quota_start_utc), format_utc(quota_end_utc)),
            ).fetchone()
            latest = None if latest_row is None else _safe_parse_utc(latest_row[0], "policy")
            return latest, int(daily_row[0]) if daily_row is not None else 0
        except NicknameDatabaseError:
            raise
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise NicknameDatabaseError("policy metrics") from exc
        finally:
            if connection is not None:
                connection.close()

    def set_nickname(
        self,
        *,
        author_discord_id: str,
        nickname_text: str,
        now_utc: datetime,
        cooldown_seconds: int,
        quota_start_utc: datetime,
        quota_end_utc: datetime,
        daily_limit: int,
    ) -> tuple[NicknameVersion, NicknameStateSnapshot]:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            connection.execute("BEGIN IMMEDIATE")
            latest_row = connection.execute(
                "SELECT created_at_utc FROM nickname_versions "
                "ORDER BY created_at_utc DESC, id DESC LIMIT 1"
            ).fetchone()
            if latest_row is not None:
                latest = _safe_parse_utc(latest_row[0], "set nickname")
                remaining = cooldown_seconds - (now_utc - latest).total_seconds()
                if remaining > 0:
                    raise NicknameGlobalCooldownError(math.ceil(remaining))

            daily_row = connection.execute(
                "SELECT COUNT(*) FROM nickname_versions "
                "WHERE author_discord_id = ? AND created_at_utc >= ? AND created_at_utc < ?",
                (
                    author_discord_id,
                    format_utc(quota_start_utc),
                    format_utc(quota_end_utc),
                ),
            ).fetchone()
            if daily_row is not None and int(daily_row[0]) >= daily_limit:
                raise NicknameDailyLimitError(daily_limit, quota_end_utc)

            cursor = connection.execute(
                "INSERT INTO nickname_versions "
                "(author_discord_id, nickname_text, created_at_utc) VALUES (?, ?, ?)",
                (author_discord_id, nickname_text, format_utc(now_utc)),
            )
            version_id = int(cursor.lastrowid)
            updated = connection.execute(
                "UPDATE nickname_state SET active_version_id = ?, "
                "revision = revision + 1, updated_at_utc = ? WHERE singleton_id = 1",
                (version_id, format_utc(now_utc)),
            ).rowcount
            if updated != 1:
                raise NicknameDatabaseError("set nickname")
            connection.commit()
            state = self.load_state()
            return (
                NicknameVersion(
                    version_id=version_id,
                    author_discord_id=author_discord_id,
                    nickname_text=nickname_text,
                    created_at_utc=now_utc,
                ),
                state,
            )
        except (NicknameDatabaseError, NicknameGlobalCooldownError, NicknameDailyLimitError):
            if connection is not None:
                connection.rollback()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise NicknameDatabaseError("set nickname") from exc
        finally:
            if connection is not None:
                connection.close()


__all__ = ["NicknameDatabaseError", "NicknameRepository"]

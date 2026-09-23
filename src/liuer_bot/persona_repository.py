"""Synchronous SQLite repository for the Persona domain."""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from pathlib import Path

from .database import DatabaseError, connect_database, initialize_database
from .persona_models import (
    PersonaDailyLimitError,
    PersonaGlobalCooldownError,
    PersonaNoRollbackTargetError,
    PersonaStateSnapshot,
    PersonaVersion,
)
from .persona_time import format_utc, parse_utc


class PersonaDatabaseError(DatabaseError):
    """A Persona repository operation failed without exposing SQL or values."""

    def __init__(self, operation: str) -> None:
        super().__init__(f"persona database operation failed: {operation}")


def _safe_parse_utc(value: str, operation: str) -> datetime:
    try:
        return parse_utc(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PersonaDatabaseError(operation) from exc


class PersonaRepository:
    """Short-connection repository; callers own the async boundary."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)

    def initialize(self) -> None:
        initialize_database(self._database_path)

    def load_state(self) -> PersonaStateSnapshot:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            return self._load_state(connection)
        except PersonaDatabaseError:
            raise
        except sqlite3.Error as exc:
            raise PersonaDatabaseError("load state") from exc
        finally:
            if connection is not None:
                connection.close()

    def load_status_message_id(self) -> int | None:
        """Load only the Discord presentation pointer, never Persona content."""

        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            row = connection.execute(
                "SELECT status_message_id FROM persona_discord_state "
                "WHERE singleton_id = 1"
            ).fetchone()
            if row is None or row[0] is None:
                return None
            value = str(row[0]).strip()
            if not value.isdigit() or int(value) <= 0:
                raise PersonaDatabaseError("load status message")
            return int(value)
        except PersonaDatabaseError:
            raise
        except sqlite3.Error as exc:
            raise PersonaDatabaseError("load status message") from exc
        finally:
            if connection is not None:
                connection.close()

    def save_status_message_id(self, status_message_id: int | None) -> None:
        """Persist the canonical status message pointer in its own singleton row."""

        if status_message_id is not None:
            value = str(status_message_id).strip()
            if not value.isdigit() or int(value) <= 0:
                raise PersonaDatabaseError("save status message")
            stored_value: str | None = str(int(value))
        else:
            stored_value = None

        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE persona_discord_state SET status_message_id = ? "
                "WHERE singleton_id = 1",
                (stored_value,),
            ).rowcount
            if updated != 1:
                raise PersonaDatabaseError("save status message")
            connection.commit()
        except PersonaDatabaseError:
            if connection is not None:
                connection.rollback()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise PersonaDatabaseError("save status message") from exc
        finally:
            if connection is not None:
                connection.close()

    def publish(
        self,
        author_discord_id: str,
        persona_text: str,
        now_utc: datetime,
        cooldown_seconds: int,
        quota_start_utc: datetime,
        quota_end_utc: datetime,
        daily_limit: int,
    ) -> tuple[PersonaVersion, PersonaStateSnapshot]:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            connection.execute("BEGIN IMMEDIATE")
            latest_row = connection.execute(
                "SELECT created_at_utc FROM persona_versions "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if latest_row is not None:
                latest = _safe_parse_utc(latest_row[0], "publish cooldown")
                elapsed = (now_utc - latest).total_seconds()
                remaining = cooldown_seconds - elapsed
                if remaining > 0:
                    connection.rollback()
                    raise PersonaGlobalCooldownError(math.ceil(remaining))

            quota_start = format_utc(quota_start_utc)
            quota_end = format_utc(quota_end_utc)
            used_row = connection.execute(
                "SELECT COUNT(*) FROM persona_versions "
                "WHERE author_discord_id = ? AND created_at_utc >= ? AND created_at_utc < ?",
                (author_discord_id, quota_start, quota_end),
            ).fetchone()
            daily_used = int(used_row[0]) if used_row else 0
            if daily_used >= daily_limit:
                connection.rollback()
                raise PersonaDailyLimitError(daily_limit, quota_end_utc)

            cursor = connection.execute(
                "INSERT INTO persona_versions "
                "(author_discord_id, persona_text, created_at_utc) VALUES (?, ?, ?)",
                (author_discord_id, persona_text, format_utc(now_utc)),
            )
            version_id = int(cursor.lastrowid)
            state_row = connection.execute(
                "SELECT revision FROM persona_state WHERE singleton_id = 1"
            ).fetchone()
            if state_row is None:
                raise PersonaDatabaseError("publish state")
            revision = int(state_row[0]) + 1
            updated_at = format_utc(now_utc)
            connection.execute(
                "UPDATE persona_state SET active_version_id = ?, revision = ?, "
                "updated_at_utc = ? WHERE singleton_id = 1",
                (version_id, revision, updated_at),
            )
            connection.commit()
            version = PersonaVersion(
                version_id=version_id,
                author_discord_id=author_discord_id,
                persona_text=persona_text,
                created_at_utc=now_utc,
            )
            state = PersonaStateSnapshot(
                revision=revision,
                active_version_id=version_id,
                active_persona=persona_text,
                updated_at_utc=now_utc,
            )
            return version, state
        except (PersonaGlobalCooldownError, PersonaDailyLimitError, PersonaDatabaseError):
            if connection is not None:
                connection.rollback()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise PersonaDatabaseError("publish") from exc
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
                "SELECT created_at_utc FROM persona_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            used_row = connection.execute(
                "SELECT COUNT(*) FROM persona_versions "
                "WHERE author_discord_id = ? AND created_at_utc >= ? AND created_at_utc < ?",
                (author_discord_id, format_utc(quota_start_utc), format_utc(quota_end_utc)),
            ).fetchone()
            latest = (
                _safe_parse_utc(latest_row[0], "policy status")
                if latest_row is not None
                else None
            )
            return latest, int(used_row[0]) if used_row else 0
        except PersonaDatabaseError:
            raise
        except sqlite3.Error as exc:
            raise PersonaDatabaseError("policy status") from exc
        finally:
            if connection is not None:
                connection.close()

    def list_history(self, limit: int) -> tuple[PersonaVersion, ...]:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            rows = connection.execute(
                "SELECT id, author_discord_id, persona_text, created_at_utc "
                "FROM persona_versions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return tuple(
                PersonaVersion(
                    version_id=int(row[0]),
                    author_discord_id=str(row[1]),
                    persona_text=str(row[2]),
                    created_at_utc=_safe_parse_utc(row[3], "history"),
                )
                for row in rows
            )
        except PersonaDatabaseError:
            raise
        except sqlite3.Error as exc:
            raise PersonaDatabaseError("history") from exc
        finally:
            if connection is not None:
                connection.close()

    def reset(self, now_utc: datetime) -> PersonaStateSnapshot:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            connection.execute("BEGIN IMMEDIATE")
            current = self._load_state(connection)
            if current.active_version_id is None:
                connection.commit()
                return current
            revision = current.revision + 1
            connection.execute(
                "UPDATE persona_state SET active_version_id = NULL, revision = ?, "
                "updated_at_utc = ? WHERE singleton_id = 1",
                (revision, format_utc(now_utc)),
            )
            connection.commit()
            return PersonaStateSnapshot(revision, None, None, now_utc)
        except PersonaDatabaseError:
            if connection is not None:
                connection.rollback()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise PersonaDatabaseError("reset") from exc
        finally:
            if connection is not None:
                connection.close()

    def rollback(self, now_utc: datetime) -> PersonaStateSnapshot:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            connection.execute("BEGIN IMMEDIATE")
            current = self._load_state(connection)
            target_query = (
                "SELECT id, author_discord_id, persona_text, created_at_utc "
                "FROM persona_versions ORDER BY id DESC LIMIT 1"
                if current.active_version_id is None
                else "SELECT id, author_discord_id, persona_text, created_at_utc "
                "FROM persona_versions WHERE id < ? ORDER BY id DESC LIMIT 1"
            )
            target_row = connection.execute(
                target_query,
                () if current.active_version_id is None else (current.active_version_id,),
            ).fetchone()
            if target_row is None:
                connection.rollback()
                raise PersonaNoRollbackTargetError()
            revision = current.revision + 1
            connection.execute(
                "UPDATE persona_state SET active_version_id = ?, revision = ?, "
                "updated_at_utc = ? WHERE singleton_id = 1",
                (int(target_row[0]), revision, format_utc(now_utc)),
            )
            connection.commit()
            return PersonaStateSnapshot(
                revision=revision,
                active_version_id=int(target_row[0]),
                active_persona=str(target_row[2]),
                updated_at_utc=now_utc,
            )
        except (PersonaDatabaseError, PersonaNoRollbackTargetError):
            if connection is not None:
                connection.rollback()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise PersonaDatabaseError("rollback") from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _load_state(connection: sqlite3.Connection) -> PersonaStateSnapshot:
        row = connection.execute(
            "SELECT active_version_id, revision, updated_at_utc "
            "FROM persona_state WHERE singleton_id = 1"
        ).fetchone()
        if row is None:
            raise PersonaDatabaseError("load state")
        active_version_id = int(row[0]) if row[0] is not None else None
        active_persona: str | None = None
        if active_version_id is not None:
            version_row = connection.execute(
                "SELECT persona_text FROM persona_versions WHERE id = ?",
                (active_version_id,),
            ).fetchone()
            if version_row is None:
                raise PersonaDatabaseError("load active persona")
            active_persona = str(version_row[0])
        return PersonaStateSnapshot(
            revision=int(row[1]),
            active_version_id=active_version_id,
            active_persona=active_persona,
            updated_at_utc=_safe_parse_utc(row[2], "load state"),
        )

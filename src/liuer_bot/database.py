"""Small synchronous SQLite initialization and migration boundary."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 6
SQLITE_BUSY_TIMEOUT_MS = 5_000


class DatabaseError(RuntimeError):
    """Base class for privacy-safe local database failures."""


class DatabaseMigrationError(DatabaseError):
    """The local database could not be migrated to the supported schema."""

    def __init__(self) -> None:
        super().__init__("local database migration failed")


class UnsupportedDatabaseVersionError(DatabaseError):
    """The local database schema is newer than this application supports."""

    def __init__(self) -> None:
        super().__init__("local database schema version is unsupported")


def connect_database(path: Path) -> sqlite3.Connection:
    """Open one short-lived configured SQLite connection."""

    connection = sqlite3.connect(
        str(path),
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    return connection


def _create_v1_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE persona_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_discord_id TEXT NOT NULL,
            persona_text TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE persona_state (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            active_version_id INTEGER REFERENCES persona_versions(id),
            revision INTEGER NOT NULL CHECK (revision >= 0),
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX persona_versions_created_at_idx "
        "ON persona_versions(created_at_utc)"
    )
    connection.execute(
        "CREATE INDEX persona_versions_author_created_at_idx "
        "ON persona_versions(author_discord_id, created_at_utc)"
    )
    connection.execute(
        "INSERT INTO persona_state "
        "(singleton_id, active_version_id, revision, updated_at_utc) "
        "VALUES (1, NULL, 0, '1970-01-01T00:00:00.000000Z')"
    )


def _create_v2_schema(connection: sqlite3.Connection) -> None:
    """Create the Discord presentation singleton introduced by schema v2."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS persona_discord_state (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            status_message_id TEXT NULL
        )
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO persona_discord_state (singleton_id, status_message_id) "
        "VALUES (1, NULL)"
    )


def _create_v3_schema(connection: sqlite3.Connection) -> None:
    """Create completed-turn conversation storage introduced by schema v3."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_channel_id TEXT NOT NULL,
            user_message_id TEXT NOT NULL UNIQUE,
            user_author_id TEXT NOT NULL,
            user_display_name TEXT NOT NULL,
            user_content TEXT NOT NULL,
            user_image_count INTEGER NOT NULL CHECK (user_image_count >= 0),
            assistant_content TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            context_epoch INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS conversation_turns_channel_id_id_idx "
        "ON conversation_turns(chat_channel_id, id)"
    )


def _create_v4_schema(connection: sqlite3.Connection) -> None:
    """Create the global Active Nickname state introduced by schema v4."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS nickname_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_discord_id TEXT NOT NULL,
            nickname_text TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS nickname_state (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            active_version_id INTEGER REFERENCES nickname_versions(id),
            revision INTEGER NOT NULL CHECK (revision >= 0),
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS nickname_versions_created_at_idx "
        "ON nickname_versions(created_at_utc)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS nickname_versions_author_created_at_idx "
        "ON nickname_versions(author_discord_id, created_at_utc)"
    )
    connection.execute(
        "INSERT OR IGNORE INTO nickname_state "
        "(singleton_id, active_version_id, revision, updated_at_utc) "
        "VALUES (1, NULL, 0, '1970-01-01T00:00:00.000000Z')"
    )


def _create_v5_schema(connection: sqlite3.Connection) -> None:
    """Create the persistent conversation epoch boundary introduced in v5."""

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(conversation_turns)").fetchall()
    }
    if "context_epoch" not in columns:
        connection.execute(
            "ALTER TABLE conversation_turns ADD COLUMN "
            "context_epoch INTEGER NOT NULL DEFAULT 1"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_state (
            singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
            active_epoch INTEGER NOT NULL CHECK (active_epoch >= 1),
            revision INTEGER NOT NULL CHECK (revision >= 0),
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO conversation_state "
        "(singleton_id, active_epoch, revision, updated_at_utc) "
        "VALUES (1, 1, 0, '1970-01-01T00:00:00.000000Z')"
    )
    connection.execute("DROP INDEX IF EXISTS conversation_turns_channel_id_id_idx")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS conversation_turns_channel_epoch_id_idx "
        "ON conversation_turns(chat_channel_id, context_epoch, id)"
    )


def _create_v6_schema(connection: sqlite3.Connection) -> None:
    """Add explicit HUMAN/BOT speaker identity to completed turns."""

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(conversation_turns)").fetchall()
    }
    if "author_kind" not in columns:
        connection.execute(
            "ALTER TABLE conversation_turns ADD COLUMN "
            "author_kind TEXT NOT NULL DEFAULT 'HUMAN'"
        )


def initialize_database(path: Path) -> None:
    """Create the parent and migrate a database to the current schema."""

    connection: sqlite3.Connection | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = connect_database(path)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        version_row = connection.execute("PRAGMA user_version").fetchone()
        version = int(version_row[0]) if version_row else 0
        if version > SCHEMA_VERSION:
            raise UnsupportedDatabaseVersionError()
        if version == 0:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _create_v1_schema(connection)
                _create_v2_schema(connection)
                _create_v3_schema(connection)
                _create_v4_schema(connection)
                _create_v5_schema(connection)
                _create_v6_schema(connection)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        elif version == 1:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _create_v2_schema(connection)
                _create_v3_schema(connection)
                _create_v4_schema(connection)
                _create_v5_schema(connection)
                _create_v6_schema(connection)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        elif version == 2:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _create_v3_schema(connection)
                _create_v4_schema(connection)
                _create_v5_schema(connection)
                _create_v6_schema(connection)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        elif version == 3:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _create_v4_schema(connection)
                _create_v5_schema(connection)
                _create_v6_schema(connection)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        elif version == 4:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _create_v5_schema(connection)
                _create_v6_schema(connection)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        elif version == 5:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _create_v6_schema(connection)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
    except UnsupportedDatabaseVersionError:
        raise
    except (OSError, sqlite3.Error) as exc:
        raise DatabaseMigrationError() from exc
    except Exception as exc:
        raise DatabaseMigrationError() from exc
    finally:
        if connection is not None:
            connection.close()

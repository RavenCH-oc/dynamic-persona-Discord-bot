from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import liuer_bot.database as database
from liuer_bot.conversation_repository import ConversationRepository
from liuer_bot.database import (
    SCHEMA_VERSION,
    DatabaseMigrationError,
    UnsupportedDatabaseVersionError,
    connect_database,
    initialize_database,
)
from liuer_bot.persona_repository import PersonaRepository


def test_new_database_migrates_to_current_schema_with_expected_schema(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "persona.sqlite3"

    initialize_database(path)

    assert path.exists()
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"persona_versions", "persona_state"} <= table_names
        assert "persona_discord_state" in table_names
        assert "conversation_turns" in table_names
        assert {"nickname_versions", "nickname_state"} <= table_names
        index_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert "conversation_turns_channel_epoch_id_idx" in index_names
        assert "nickname_versions_created_at_idx" in index_names
        assert "nickname_versions_author_created_at_idx" in index_names
        state = connection.execute(
            "SELECT active_version_id, revision FROM persona_state WHERE singleton_id = 1"
        ).fetchone()
        assert state == (None, 0)
        assert connection.execute(
            "SELECT active_version_id, revision FROM nickname_state WHERE singleton_id = 1"
        ).fetchone() == (None, 0)
        assert connection.execute(
            "SELECT active_epoch, revision FROM conversation_state WHERE singleton_id = 1"
        ).fetchone() == (1, 0)
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
    finally:
        connection.close()

def test_v4_database_migrates_to_v6_preserving_rows_and_all_other_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v4.sqlite3"
    connection = sqlite3.connect(path)
    try:
        database._create_v1_schema(connection)
        database._create_v2_schema(connection)
        database._create_v3_schema(connection)
        database._create_v4_schema(connection)
        connection.execute(
            "INSERT INTO persona_versions "
            "(author_discord_id, persona_text, created_at_utc) "
            "VALUES ('123', 'persona body', '2026-01-01T00:00:00.000000Z')"
        )
        connection.execute(
            "UPDATE persona_state SET active_version_id = 1, revision = 1, "
            "updated_at_utc = '2026-01-01T00:00:00.000000Z' WHERE singleton_id = 1"
        )
        connection.execute(
            "UPDATE persona_discord_state SET status_message_id = '987' WHERE singleton_id = 1"
        )
        connection.execute(
            "INSERT INTO nickname_versions "
            "(author_discord_id, nickname_text, created_at_utc) "
            "VALUES ('456', '小六', '2026-01-01T00:00:00.000000Z')"
        )
        connection.execute(
            "UPDATE nickname_state SET active_version_id = 1, revision = 1, "
            "updated_at_utc = '2026-01-01T00:00:00.000000Z' WHERE singleton_id = 1"
        )
        connection.execute(
            "INSERT INTO conversation_turns "
            "(chat_channel_id, user_message_id, user_author_id, user_display_name, "
            "user_content, user_image_count, assistant_content, created_at_utc) "
            "VALUES ('123', '456', '789', 'Display', 'user body', 0, 'assistant body', "
            "'2026-01-01T00:00:00.000000Z')"
        )
        connection.execute("PRAGMA user_version = 4")
        connection.commit()
    finally:
        connection.close()

    initialize_database(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT context_epoch FROM conversation_turns WHERE id = 1"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT user_content, assistant_content FROM conversation_turns WHERE id = 1"
        ).fetchone() == ("user body", "assistant body")
        assert connection.execute(
            "SELECT active_version_id, revision FROM persona_state WHERE singleton_id = 1"
        ).fetchone() == (1, 1)
        assert connection.execute(
            "SELECT status_message_id FROM persona_discord_state WHERE singleton_id = 1"
        ).fetchone() == ("987",)
        assert connection.execute(
            "SELECT active_version_id, revision FROM nickname_state WHERE singleton_id = 1"
        ).fetchone() == (1, 1)
        assert connection.execute(
            "SELECT active_epoch, revision FROM conversation_state WHERE singleton_id = 1"
        ).fetchone() == (1, 0)
        index_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert "conversation_turns_channel_epoch_id_idx" in index_names
        assert "conversation_turns_channel_id_id_idx" not in index_names
    finally:
        connection.close()

    restored = ConversationRepository(path).load_recent_turns(123, 12)
    assert [turn.user_message_id for turn in restored] == [456]


def test_v1_database_migrates_to_v6_without_losing_persona_state(tmp_path: Path) -> None:
    path = tmp_path / "v1.sqlite3"
    connection = sqlite3.connect(path)
    try:
        database._create_v1_schema(connection)
        connection.execute(
            "INSERT INTO persona_versions "
            "(author_discord_id, persona_text, created_at_utc) "
            "VALUES ('123', 'private persona body', '2026-01-01T00:00:00.000000Z')"
        )
        connection.execute(
            "UPDATE persona_state SET active_version_id = 1, revision = 1, "
            "updated_at_utc = '2026-01-01T00:00:00.000000Z' WHERE singleton_id = 1"
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    initialize_database(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT author_discord_id, persona_text FROM persona_versions"
        ).fetchone() == ("123", "private persona body")
        assert connection.execute(
            "SELECT active_version_id, revision FROM persona_state"
        ).fetchone() == (1, 1)
        assert connection.execute(
            "SELECT status_message_id FROM persona_discord_state WHERE singleton_id = 1"
        ).fetchone() == (None,)
        assert connection.execute(
            "SELECT active_version_id, revision FROM nickname_state WHERE singleton_id = 1"
        ).fetchone() == (None, 0)
    finally:
        connection.close()


def test_v2_database_migrates_to_v6_preserving_persona_and_status_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v2.sqlite3"
    connection = sqlite3.connect(path)
    try:
        database._create_v1_schema(connection)
        database._create_v2_schema(connection)
        connection.execute(
            "INSERT INTO persona_versions "
            "(author_discord_id, persona_text, created_at_utc) "
            "VALUES ('123', 'private persona body', '2026-01-01T00:00:00.000000Z')"
        )
        connection.execute(
            "UPDATE persona_state SET active_version_id = 1, revision = 1, "
            "updated_at_utc = '2026-01-01T00:00:00.000000Z' WHERE singleton_id = 1"
        )
        connection.execute(
            "UPDATE persona_discord_state SET status_message_id = '987' WHERE singleton_id = 1"
        )
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    finally:
        connection.close()

    initialize_database(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT active_version_id, revision FROM persona_state"
        ).fetchone() == (1, 1)
        assert connection.execute(
            "SELECT status_message_id FROM persona_discord_state"
        ).fetchone() == ("987",)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'conversation_turns'"
        ).fetchone() == ("conversation_turns",)
        assert connection.execute(
            "SELECT active_version_id, revision FROM nickname_state"
        ).fetchone() == (None, 0)
    finally:
        connection.close()


def test_v3_database_migrates_to_v6_without_touching_conversation_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v3.sqlite3"
    connection = sqlite3.connect(path)
    try:
        database._create_v1_schema(connection)
        database._create_v2_schema(connection)
        database._create_v3_schema(connection)
        connection.execute(
            "INSERT INTO conversation_turns "
            "(chat_channel_id, user_message_id, user_author_id, user_display_name, "
            "user_content, user_image_count, assistant_content, created_at_utc) "
            "VALUES ('123', '456', '789', 'display', 'safe content', 0, 'answer', "
            "'2026-01-01T00:00:00.000000Z')"
        )
        connection.execute("PRAGMA user_version = 3")
        connection.commit()
    finally:
        connection.close()

    initialize_database(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
        assert connection.execute(
            "SELECT user_content, assistant_content FROM conversation_turns"
        ).fetchone() == ("safe content", "answer")
        assert connection.execute(
            "SELECT active_version_id, revision FROM nickname_state"
        ).fetchone() == (None, 0)
    finally:
        connection.close()


def test_v5_database_migrates_to_v6_with_existing_turns_as_human(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v5.sqlite3"
    connection = sqlite3.connect(path)
    try:
        database._create_v1_schema(connection)
        database._create_v2_schema(connection)
        database._create_v3_schema(connection)
        database._create_v4_schema(connection)
        database._create_v5_schema(connection)
        connection.execute(
            "INSERT INTO conversation_turns "
            "(chat_channel_id, user_message_id, user_author_id, user_display_name, "
            "user_content, user_image_count, assistant_content, created_at_utc, context_epoch) "
            "VALUES ('123', '456', '789', 'Display', 'human body', 0, 'answer', "
            "'2026-01-01T00:00:00.000000Z', 1)"
        )
        connection.execute("PRAGMA user_version = 5")
        connection.commit()
    finally:
        connection.close()

    initialize_database(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (6,)
        assert connection.execute(
            "SELECT user_content, assistant_content, context_epoch, author_kind "
            "FROM conversation_turns"
        ).fetchone() == ("human body", "answer", 1, "HUMAN")
    finally:
        connection.close()


def test_status_message_pointer_round_trip_is_separate_from_persona_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "presentation.sqlite3"
    repository = PersonaRepository(path)
    repository.initialize()

    assert repository.load_status_message_id() is None
    repository.save_status_message_id(987654321)
    assert repository.load_status_message_id() == 987654321
    repository.save_status_message_id(None)
    assert repository.load_status_message_id() is None

    configured = connect_database(path)
    try:
        assert configured.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert configured.execute("PRAGMA busy_timeout").fetchone() == (5000,)
    finally:
        configured.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "persona.sqlite3"

    initialize_database(path)
    first_connection = sqlite3.connect(path)
    try:
        first_state = first_connection.execute(
            "SELECT revision, updated_at_utc FROM persona_state"
        ).fetchone()
    finally:
        first_connection.close()
    initialize_database(path)
    second_connection = sqlite3.connect(path)
    try:
        second_state = second_connection.execute(
            "SELECT revision, updated_at_utc FROM persona_state"
        ).fetchone()
        assert second_state == first_state
        assert second_connection.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
    finally:
        second_connection.close()


def test_future_schema_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA user_version = 99")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(UnsupportedDatabaseVersionError):
        initialize_database(path)


def test_migration_failure_rolls_back_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed.sqlite3"

    def fail_schema(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE partial_table (id INTEGER)")
        raise RuntimeError("private migration test failure")

    monkeypatch.setattr(database, "_create_v1_schema", fail_schema)

    with pytest.raises(DatabaseMigrationError) as raised:
        initialize_database(path)

    assert "private migration test failure" not in str(raised.value)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'partial_table'"
        ).fetchone() is None
    finally:
        connection.close()

"""Synchronous SQLite repository for completed conversation turns."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .conversation_models import (
    AuthorKind,
    ConversationClearResult,
    ConversationState,
    ConversationTurn,
    sanitize_display_name,
)
from .database import DatabaseError, connect_database, initialize_database
from .persona_time import format_utc, parse_utc


class ConversationDatabaseError(DatabaseError):
    """A conversation database operation failed without exposing content."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"conversation database operation failed: {operation}")


def _safe_parse_utc(value: object, operation: str) -> datetime:
    try:
        return parse_utc(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConversationDatabaseError(operation) from exc


def _safe_author_kind(value: object) -> AuthorKind:
    try:
        return AuthorKind(str(value))
    except (TypeError, ValueError) as exc:
        raise ConversationDatabaseError("load recent turns") from exc


class ConversationRepository:
    """Short-lived SQLite connections for the conversation service boundary."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)

    def initialize(self) -> None:
        initialize_database(self._database_path)

    def load_conversation_state(self) -> ConversationState:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            row = connection.execute(
                "SELECT active_epoch, revision, updated_at_utc "
                "FROM conversation_state WHERE singleton_id = 1"
            ).fetchone()
            if row is None:
                raise ConversationDatabaseError("load conversation state")
            return ConversationState(
                active_epoch=int(row[0]),
                revision=int(row[1]),
                updated_at_utc=_safe_parse_utc(row[2], "load conversation state"),
            )
        except ConversationDatabaseError:
            raise
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise ConversationDatabaseError("load conversation state") from exc
        finally:
            if connection is not None:
                connection.close()

    def load_recent_turns(
        self,
        chat_channel_id: int,
        limit: int,
        *,
        context_epoch: int = 1,
    ) -> tuple[ConversationTurn, ...]:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            rows = connection.execute(
                "SELECT id, chat_channel_id, user_message_id, user_author_id, "
                "user_display_name, user_content, user_image_count, assistant_content, "
                "created_at_utc, context_epoch, author_kind FROM conversation_turns "
                "WHERE chat_channel_id = ? AND context_epoch = ? "
                "ORDER BY id DESC LIMIT ?",
                (str(chat_channel_id), context_epoch, limit),
            ).fetchall()
            return tuple(self._row_to_turn(row) for row in reversed(rows))
        except ConversationDatabaseError:
            raise
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise ConversationDatabaseError("load recent turns") from exc
        finally:
            if connection is not None:
                connection.close()

    def insert_turn(
        self,
        *,
        chat_channel_id: int,
        user_message_id: int,
        user_author_id: int,
        user_display_name: str,
        user_content: str,
        user_image_count: int,
        assistant_content: str,
        created_at_utc: datetime,
        context_epoch: int = 1,
        author_kind: AuthorKind = AuthorKind.HUMAN,
    ) -> ConversationTurn:
        try:
            normalized_author_kind = AuthorKind(author_kind)
        except (TypeError, ValueError) as exc:
            raise ConversationDatabaseError("insert turn") from exc
        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "INSERT INTO conversation_turns "
                "(chat_channel_id, user_message_id, user_author_id, user_display_name, "
                "user_content, user_image_count, assistant_content, created_at_utc, "
                "context_epoch, author_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(chat_channel_id),
                    str(user_message_id),
                    str(user_author_id),
                    sanitize_display_name(user_display_name),
                    user_content,
                    user_image_count,
                    assistant_content,
                    format_utc(created_at_utc),
                    context_epoch,
                    normalized_author_kind.value,
                ),
            )
            connection.commit()
            return ConversationTurn(
                turn_id=int(cursor.lastrowid),
                chat_channel_id=chat_channel_id,
                user_message_id=user_message_id,
                user_author_id=user_author_id,
                user_display_name=sanitize_display_name(user_display_name),
                user_content=user_content,
                user_image_count=user_image_count,
                assistant_content=assistant_content,
                created_at_utc=created_at_utc,
                context_epoch=context_epoch,
                author_kind=normalized_author_kind,
            )
        except ConversationDatabaseError:
            if connection is not None:
                connection.rollback()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise ConversationDatabaseError("insert turn") from exc
        finally:
            if connection is not None:
                connection.close()

    def advance_context_epoch(self, updated_at_utc: datetime) -> ConversationClearResult:
        """Atomically advance the active epoch without deleting conversation rows."""

        connection: sqlite3.Connection | None = None
        try:
            connection = connect_database(self._database_path)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT active_epoch, revision FROM conversation_state "
                "WHERE singleton_id = 1"
            ).fetchone()
            if row is None:
                raise ConversationDatabaseError("advance conversation epoch")
            previous_epoch = int(row[0])
            active_epoch = previous_epoch + 1
            revision = int(row[1]) + 1
            connection.execute(
                "UPDATE conversation_state SET active_epoch = ?, revision = ?, "
                "updated_at_utc = ? WHERE singleton_id = 1",
                (active_epoch, revision, format_utc(updated_at_utc)),
            )
            connection.commit()
            return ConversationClearResult(
                previous_epoch=previous_epoch,
                active_epoch=active_epoch,
                revision=revision,
                updated_at_utc=updated_at_utc,
            )
        except ConversationDatabaseError:
            if connection is not None:
                connection.rollback()
            raise
        except (sqlite3.Error, ValueError, TypeError) as exc:
            if connection is not None:
                connection.rollback()
            raise ConversationDatabaseError("advance conversation epoch") from exc
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _row_to_turn(row: tuple[object, ...]) -> ConversationTurn:
        try:
            return ConversationTurn(
                turn_id=int(row[0]),
                chat_channel_id=int(str(row[1])),
                user_message_id=int(str(row[2])),
                user_author_id=int(str(row[3])),
                user_display_name=sanitize_display_name(row[4]),
                user_content=str(row[5]),
                user_image_count=int(row[6]),
                assistant_content=str(row[7]),
                created_at_utc=_safe_parse_utc(row[8], "load recent turns"),
                context_epoch=int(row[9]),
                author_kind=_safe_author_kind(row[10]),
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise ConversationDatabaseError("load recent turns") from exc


__all__ = ["ConversationDatabaseError", "ConversationRepository"]

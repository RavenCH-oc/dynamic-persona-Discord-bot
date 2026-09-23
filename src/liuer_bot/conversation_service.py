"""Conversation persistence service and responder-side recording boundary."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .config import Config
from .conversation_context import ConversationContextProvider, ConversationGenerationContext
from .conversation_models import (
    ConversationClearResult,
    ConversationRuntimeStatus,
    ConversationTurn,
    sanitize_display_name,
)
from .conversation_repository import ConversationDatabaseError, ConversationRepository
from .responder import ChatRequest, Responder

LOGGER = logging.getLogger(__name__)


class ConversationError(RuntimeError):
    """Base class for privacy-safe conversation runtime failures."""


class ConversationNotInitializedError(ConversationError):
    """The conversation service was used before startup initialization."""

    def __init__(self) -> None:
        super().__init__("conversation service is not initialized")


class ConversationPersistenceError(ConversationError):
    """A completed turn could not be committed."""

    def __init__(self) -> None:
        super().__init__("conversation turn persistence failed")


class ConversationService(ConversationContextProvider):
    """Keep one bounded active-channel conversation snapshot in memory."""

    def __init__(
        self,
        database_path: Path,
        *,
        chat_channel_id: int,
        recent_turn_limit: int = 12,
        repository: ConversationRepository | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if chat_channel_id <= 0:
            raise ValueError("chat channel ID must be positive")
        if not 1 <= recent_turn_limit <= 50:
            raise ValueError("recent turn limit must be between 1 and 50")
        self._chat_channel_id = chat_channel_id
        self._recent_turn_limit = recent_turn_limit
        self._repository = repository or ConversationRepository(Path(database_path))
        self._clock = clock or (lambda: datetime.now(UTC))
        self._recent_turns: tuple[ConversationTurn, ...] = ()
        self._active_epoch = 1
        self._revision = 0
        self._state_lock = threading.RLock()
        self._initialized = False

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        repository: ConversationRepository | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> ConversationService:
        return cls(
            config.database_path,
            chat_channel_id=config.chat_channel_id,
            recent_turn_limit=config.context_recent_turns,
            repository=repository,
            clock=clock,
        )

    async def initialize(self) -> tuple[ConversationTurn, ...]:
        """Migrate storage and load only the active configured channel."""

        await asyncio.to_thread(self._repository.initialize)
        state = await asyncio.to_thread(self._repository.load_conversation_state)
        turns = await asyncio.to_thread(
            self._repository.load_recent_turns,
            self._chat_channel_id,
            self._recent_turn_limit,
            context_epoch=state.active_epoch,
        )
        with self._state_lock:
            self._active_epoch = state.active_epoch
            self._revision = state.revision
            self._recent_turns = tuple(turns)
            self._initialized = True
            return self._recent_turns

    @property
    def active_epoch(self) -> int:
        """Return the active epoch from process-local state."""

        self._require_initialized()
        with self._state_lock:
            return self._active_epoch

    def get_generation_context(self) -> ConversationGenerationContext:
        """Atomically capture the epoch and recent-turn snapshot."""

        self._require_initialized()
        with self._state_lock:
            return ConversationGenerationContext(
                context_epoch=self._active_epoch,
                recent_turns=self._recent_turns,
            )

    def get_recent_turns(self) -> tuple[ConversationTurn, ...]:
        """Return the in-memory snapshot without querying SQLite."""

        return self.get_generation_context().recent_turns

    def get_runtime_status(self) -> ConversationRuntimeStatus:
        """Return an atomic count-only snapshot without querying SQLite."""

        with self._state_lock:
            return ConversationRuntimeStatus(
                initialized=self._initialized,
                active_epoch=self._active_epoch if self._initialized else None,
                recent_turn_count=len(self._recent_turns) if self._initialized else 0,
            )

    async def clear_context(self) -> ConversationClearResult:
        """Commit a new epoch, then clear the active in-memory snapshot."""

        self._require_initialized()
        result = await asyncio.to_thread(
            self._repository.advance_context_epoch,
            self._clock(),
        )
        with self._state_lock:
            self._active_epoch = result.active_epoch
            self._revision = result.revision
            self._recent_turns = ()
        return result

    async def persist_successful_turn(
        self,
        request: ChatRequest,
        assistant_content: str,
        *,
        context_epoch: int | None = None,
    ) -> ConversationTurn:
        """Commit one completed generation, then update the memory snapshot."""

        self._require_initialized()
        if not isinstance(assistant_content, str) or not assistant_content.strip():
            raise ConversationPersistenceError()
        if not isinstance(request.content, str):
            raise ConversationPersistenceError()
        captured_epoch = context_epoch
        if captured_epoch is None and request.generation_context is not None:
            captured_epoch = request.generation_context.context_epoch
        if captured_epoch is None:
            captured_epoch = self.get_generation_context().context_epoch
        if not isinstance(captured_epoch, int) or captured_epoch < 1:
            raise ConversationPersistenceError()
        try:
            turn = await asyncio.to_thread(
                self._repository.insert_turn,
                chat_channel_id=self._chat_channel_id,
                user_message_id=request.message_id,
                user_author_id=request.author_id,
                user_display_name=sanitize_display_name(request.author_display_name),
                user_content=request.content,
                user_image_count=len(request.prepared_images),
                assistant_content=assistant_content,
                created_at_utc=self._clock(),
                context_epoch=captured_epoch,
                author_kind=request.author_kind,
            )
        except ConversationDatabaseError as exc:
            raise ConversationPersistenceError() from exc
        with self._state_lock:
            if turn.context_epoch == self._active_epoch:
                self._recent_turns = (*self._recent_turns, turn)[-self._recent_turn_limit :]
        return turn

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise ConversationNotInitializedError()


class ConversationRecordingResponder:
    """Record successful model output before the queue advances."""

    def __init__(
        self,
        responder: Responder,
        conversation_service: ConversationService,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._responder = responder
        self._conversation_service = conversation_service
        self._logger = logger

    async def generate(self, request: ChatRequest) -> str:
        generation_context = self._conversation_service.get_generation_context()
        generation_request = replace(request, generation_context=generation_context)
        generated = await self._responder.generate(generation_request)
        try:
            await self._conversation_service.persist_successful_turn(
                generation_request,
                generated,
                context_epoch=generation_context.context_epoch,
            )
        except Exception as exc:
            self._logger.error(
                "conversation persistence failed message_id=%s operation=insert_turn "
                "error_type=%s",
                request.message_id,
                type(exc).__name__,
            )
            raise
        return generated


__all__ = [
    "ConversationError",
    "ConversationNotInitializedError",
    "ConversationPersistenceError",
    "ConversationRecordingResponder",
    "ConversationService",
]

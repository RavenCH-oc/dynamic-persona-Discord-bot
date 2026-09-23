from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from liuer_bot.conversation_context import ConversationGenerationContext
from liuer_bot.conversation_service import ConversationRecordingResponder
from liuer_bot.discord_runtime import handle_message
from liuer_bot.generation_queue import SerializedGenerationQueue
from liuer_bot.responder import ChatRequest
from liuer_bot.typing_indicator import TypingIndicatorManager


class FakeTypingContext:
    def __init__(
        self,
        *,
        trace: list[str] | None = None,
        fail_enter: Exception | None = None,
        entered: asyncio.Event | None = None,
    ) -> None:
        self.trace = trace if trace is not None else []
        self.fail_enter = fail_enter
        self.entered = entered
        self.enter_count = 0
        self.exit_count = 0
        self.active = False

    async def __aenter__(self) -> None:
        self.enter_count += 1
        if self.fail_enter is not None:
            raise self.fail_enter
        self.active = True
        self.trace.append("typing-start")
        if self.entered is not None:
            self.entered.set()

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exit_count += 1
        self.active = False
        self.trace.append("typing-stop")


@dataclass
class FakeChannel:
    id: int = 123
    typing_context: FakeTypingContext = field(default_factory=FakeTypingContext)
    fetched_message: object | None = None
    fetch_started: asyncio.Event | None = None
    fetch_release: asyncio.Event | None = None
    sent_messages: list[dict[str, object]] = field(default_factory=list)

    def typing(self) -> FakeTypingContext:
        return self.typing_context

    async def fetch_message(self, message_id: int) -> object | None:
        if self.fetch_started is not None:
            self.fetch_started.set()
        if self.fetch_release is not None:
            await self.fetch_release.wait()
        return self.fetched_message

    async def send(self, content: str, **kwargs: object) -> None:
        self.sent_messages.append({"content": content, **kwargs})


@dataclass
class FakeMessage:
    id: int = 1
    content: str = "六耳 你好"
    channel: FakeChannel = field(default_factory=FakeChannel)
    author: object = field(
        default_factory=lambda: SimpleNamespace(id=10, bot=False, display_name="Human")
    )
    guild: object | None = field(default_factory=lambda: SimpleNamespace(id=20))
    mentions: list[object] = field(default_factory=list)
    attachments: list[object] = field(default_factory=list)
    reference: object | None = None
    webhook_id: int | None = None
    replies: list[dict[str, object]] = field(default_factory=list)

    async def reply(self, content: str, **kwargs: object) -> None:
        self.replies.append({"content": content, **kwargs})


class RecordingResponder:
    def __init__(self, result: str = "response") -> None:
        self.result = result
        self.calls: list[ChatRequest] = []
        self.started = asyncio.Event()

    async def generate(self, request: ChatRequest) -> str:
        self.calls.append(request)
        self.started.set()
        return self.result


class BlockingResponder(RecordingResponder):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def generate(self, request: ChatRequest) -> str:
        self.calls.append(request)
        self.started.set()
        await self.release.wait()
        return f"response-{request.message_id}"


class FailingResponder:
    async def generate(self, request: ChatRequest) -> str:
        raise RuntimeError("private-generation-content")


class FailingPersistence:
    def get_generation_context(self) -> ConversationGenerationContext:
        return ConversationGenerationContext(context_epoch=1, recent_turns=())

    async def persist_successful_turn(
        self,
        request: ChatRequest,
        assistant_content: str,
        *,
        context_epoch: int | None = None,
    ) -> None:
        raise RuntimeError("private-persistence-content")


class QueueProxy:
    def __init__(self, queue: SerializedGenerationQueue) -> None:
        self.queue = queue
        self.second_submitted = asyncio.Event()

    async def submit(self, request: ChatRequest) -> str:
        if request.message_id == 2:
            self.second_submitted.set()
        return await self.queue.submit(request)


async def _run_message(
    message: FakeMessage,
    responder: object,
    manager: TypingIndicatorManager,
    *,
    bot_dialogue_service: object | None = None,
    initial_bot_session_id: str | None = None,
    queue: object | None = None,
    logger: logging.Logger | None = None,
) -> None:
    generation_queue = queue or SerializedGenerationQueue(responder)  # type: ignore[arg-type]
    owns_queue = queue is None
    if owns_queue:
        await generation_queue.start()  # type: ignore[union-attr]
    try:
        await handle_message(
            message,
            chat_channel_id=123,
            generation_queue=generation_queue,  # type: ignore[arg-type]
            logger=logger or logging.getLogger("test.typing.runtime"),
            typing_manager=manager,
            bot_dialogue_service=bot_dialogue_service,  # type: ignore[arg-type]
            initial_bot_session_id=initial_bot_session_id,
        )
    finally:
        if owns_queue:
            await generation_queue.close()  # type: ignore[union-attr]


def test_typing_manager_reference_counts_one_public_context() -> None:
    async def run() -> None:
        context = FakeTypingContext()
        channel = FakeChannel(typing_context=context)
        manager = TypingIndicatorManager()

        first = await manager.acquire(channel)
        second = await manager.acquire(channel)
        assert context.enter_count == 1
        assert manager.get_active_count(123) == 2
        assert manager.is_typing(123) is True

        await first.release()
        assert manager.get_active_count(123) == 1
        assert context.exit_count == 0

        await second.release()
        await second.release()
        assert manager.get_active_count(123) == 0
        assert context.exit_count == 1

    asyncio.run(run())


def test_typing_manager_keeps_channels_independent_and_closes_cleanly() -> None:
    async def run() -> None:
        first_context = FakeTypingContext()
        second_context = FakeTypingContext()
        manager = TypingIndicatorManager()
        first = await manager.acquire(FakeChannel(123, first_context))
        second = await manager.acquire(FakeChannel(456, second_context))
        assert manager.get_active_count(123) == 1
        assert manager.get_active_count(456) == 1

        await manager.close()
        assert first_context.exit_count == 1
        assert second_context.exit_count == 1
        assert manager.get_active_count(123) == 0
        await first.release()
        await second.release()
        after_close = await manager.acquire(FakeChannel(123, FakeTypingContext()))
        await after_close.release()

    asyncio.run(run())


def test_accepted_human_typing_covers_generation_and_stops_before_delivery() -> None:
    async def run() -> list[str]:
        trace: list[str] = []
        context = FakeTypingContext(trace=trace)
        channel = FakeChannel(typing_context=context)
        message = FakeMessage(channel=channel)
        responder = RecordingResponder()
        original_generate = responder.generate

        async def traced_generate(request: ChatRequest) -> str:
            trace.append("generation")
            return await original_generate(request)

        responder.generate = traced_generate  # type: ignore[method-assign]
        original_reply = message.reply

        async def traced_reply(content: str, **kwargs: object) -> None:
            trace.append("delivery")
            await original_reply(content, **kwargs)

        message.reply = traced_reply  # type: ignore[method-assign]
        manager = TypingIndicatorManager()
        await _run_message(message, responder, manager)
        return trace

    assert asyncio.run(run()) == ["typing-start", "generation", "typing-stop", "delivery"]


@pytest.mark.parametrize(
    "message",
    [
        FakeMessage(content="普通訊息"),
        FakeMessage(channel=FakeChannel(id=999)),
        FakeMessage(author=SimpleNamespace(id=200, bot=True, display_name="Other Bot")),
        FakeMessage(
            author=SimpleNamespace(id=200, bot=True, display_name="Webhook Bot"),
            webhook_id=77,
        ),
    ],
)
def test_ignored_messages_do_not_acquire_typing(message: FakeMessage) -> None:
    async def run() -> FakeTypingContext:
        context = message.channel.typing_context
        manager = TypingIndicatorManager()
        await _run_message(message, RecordingResponder(), manager)
        return context

    context = asyncio.run(run())
    assert context.enter_count == 0
    assert message.replies == []


def test_botchat_turn_gate_rejection_does_not_acquire_typing() -> None:
    async def run() -> FakeTypingContext:
        from liuer_bot.bot_dialogue import BotDialogueService

        service = BotDialogueService()
        session = service.activate(200, "Peer Bot")
        service.complete_turn(session.session_id, delivered=True)
        assert service.claim_peer_turn(200) == session.session_id
        message = FakeMessage(
            author=SimpleNamespace(id=200, bot=True, display_name="Peer Bot"),
        )
        manager = TypingIndicatorManager()
        await _run_message(
            message,
            RecordingResponder(),
            manager,
            bot_dialogue_service=service,
        )
        return message.channel.typing_context

    context = asyncio.run(run())
    assert context.enter_count == 0


def test_queued_request_holds_second_typing_lease_until_generation_finishes() -> None:
    async def run() -> tuple[int, list[int]]:
        manager = TypingIndicatorManager()
        context = FakeTypingContext()
        channel = FakeChannel(typing_context=context)
        responder = BlockingResponder()
        queue = SerializedGenerationQueue(responder)
        await queue.start()
        proxy = QueueProxy(queue)
        first = asyncio.create_task(
            _run_message(FakeMessage(id=1, channel=channel), responder, manager, queue=proxy)
        )
        await responder.started.wait()
        second = asyncio.create_task(
            _run_message(FakeMessage(id=2, channel=channel), responder, manager, queue=proxy)
        )
        await proxy.second_submitted.wait()
        active_while_queued = manager.get_active_count(123)
        assert responder.calls == [responder.calls[0]]
        responder.release.set()
        await asyncio.gather(first, second)
        await queue.close()
        return active_while_queued, [request.message_id for request in responder.calls]

    active_count, started_ids = asyncio.run(run())
    assert active_count == 2
    assert started_ids == [1, 2]


def test_typing_stays_active_during_reply_rest_fallback() -> None:
    async def run() -> FakeTypingContext:
        fetch_started = asyncio.Event()
        fetch_release = asyncio.Event()
        referenced = SimpleNamespace(
            id=70,
            author=SimpleNamespace(id=71, bot=False, display_name="Referenced"),
            content="referenced content",
            attachments=[],
        )
        channel = FakeChannel(
            fetch_started=fetch_started,
            fetch_release=fetch_release,
            fetched_message=referenced,
        )
        message = FakeMessage(
            channel=channel,
            reference=SimpleNamespace(message_id=70, resolved=None, cached_message=None),
        )
        manager = TypingIndicatorManager()
        task = asyncio.create_task(_run_message(message, RecordingResponder(), manager))
        await fetch_started.wait()
        assert manager.is_typing(123) is True
        fetch_release.set()
        await task
        return channel.typing_context

    context = asyncio.run(run())
    assert context.exit_count == 1


def test_typing_stays_active_during_image_preparation() -> None:
    async def run() -> FakeTypingContext:
        read_started = asyncio.Event()
        read_release = asyncio.Event()

        class Attachment:
            id = 1
            content_type = "image/png"
            filename = "image.png"
            size = 1

            async def read(self, *, use_cached: bool = False) -> bytes:
                read_started.set()
                await read_release.wait()
                return b"not an image"

        channel = FakeChannel()
        message = FakeMessage(channel=channel, attachments=[Attachment()])
        manager = TypingIndicatorManager()
        task = asyncio.create_task(_run_message(message, RecordingResponder(), manager))
        await read_started.wait()
        assert manager.is_typing(123) is True
        read_release.set()
        await task
        return channel.typing_context

    context = asyncio.run(run())
    assert context.exit_count == 1


def test_typing_api_failure_is_safe_and_does_not_block_response(caplog) -> None:
    async def run() -> FakeMessage:
        context = FakeTypingContext(fail_enter=RuntimeError("private-typing-content"))
        message = FakeMessage(channel=FakeChannel(typing_context=context))
        manager = TypingIndicatorManager(logger=logging.getLogger("test.typing.failure"))
        await _run_message(
            message,
            RecordingResponder(),
            manager,
            logger=logging.getLogger("test.typing.failure"),
        )
        return message

    logger = logging.getLogger("test.typing.failure")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        message = asyncio.run(run())
    assert len(message.replies) == 1
    assert "private-typing-content" not in caplog.text
    assert "error_type=RuntimeError" in caplog.text


def test_generation_failure_releases_typing_without_reply(caplog) -> None:
    async def run() -> FakeMessage:
        message = FakeMessage()
        manager = TypingIndicatorManager()
        await _run_message(
            message,
            FailingResponder(),
            manager,
            logger=logging.getLogger("test.typing.generation-failure"),
        )
        assert manager.get_active_count(123) == 0
        return message

    message = asyncio.run(run())
    assert message.replies == []
    assert "private-generation-content" not in caplog.text


def test_persistence_failure_releases_typing_without_success_reply(caplog) -> None:
    async def run() -> FakeMessage:
        message = FakeMessage()
        manager = TypingIndicatorManager()
        responder = ConversationRecordingResponder(
            RecordingResponder(),  # type: ignore[arg-type]
            FailingPersistence(),  # type: ignore[arg-type]
        )
        await _run_message(
            message,
            responder,
            manager,
            logger=logging.getLogger("test.typing.persistence-failure"),
        )
        assert manager.get_active_count(123) == 0
        return message

    message = asyncio.run(run())
    assert message.replies == []
    assert "private-persistence-content" not in caplog.text


def test_cancellation_releases_typing_and_preserves_cancelled_error() -> None:
    async def run() -> tuple[FakeTypingContext, BlockingResponder]:
        message = FakeMessage()
        manager = TypingIndicatorManager()
        responder = BlockingResponder()
        queue = SerializedGenerationQueue(responder)
        await queue.start()
        task = asyncio.create_task(
            _run_message(message, responder, manager, queue=queue)
        )
        await responder.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert manager.get_active_count(123) == 0
        assert message.channel.typing_context.exit_count == 1
        responder.release.set()
        await queue.close()
        return message.channel.typing_context, responder

    context, _ = asyncio.run(run())
    assert context.active is False


def test_active_botchat_request_uses_typing_and_keeps_existing_semantics() -> None:
    async def run() -> tuple[FakeTypingContext, list[ChatRequest]]:
        from liuer_bot.bot_dialogue import BotDialogueService

        service = BotDialogueService()
        session = service.activate(200, "Peer Bot")
        responder = RecordingResponder()
        message = FakeMessage(
            author=SimpleNamespace(id=200, bot=True, display_name="Peer Bot"),
        )
        manager = TypingIndicatorManager()
        await _run_message(
            message,
            responder,
            manager,
            bot_dialogue_service=service,
            initial_bot_session_id=session.session_id,
        )
        return message.channel.typing_context, responder.calls

    context, calls = asyncio.run(run())
    assert context.enter_count == 1
    assert len(calls) == 1


def test_active_peer_bot_request_uses_typing_after_turn_claim() -> None:
    async def run() -> tuple[FakeTypingContext, list[ChatRequest]]:
        from liuer_bot.bot_dialogue import BotDialogueService

        service = BotDialogueService()
        session = service.activate(200, "Peer Bot")
        service.complete_turn(session.session_id, delivered=True)
        responder = RecordingResponder()
        message = FakeMessage(
            author=SimpleNamespace(id=200, bot=True, display_name="Peer Bot"),
        )
        manager = TypingIndicatorManager()
        await _run_message(
            message,
            responder,
            manager,
            bot_dialogue_service=service,
        )
        return message.channel.typing_context, responder.calls

    context, calls = asyncio.run(run())
    assert context.enter_count == 1
    assert len(calls) == 1

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import uuid4

import discord
import pytest
from PIL import Image

import liuer_bot.discord_runtime as discord_runtime
from liuer_bot.addressing import EMPTY_ADDRESSED_CONTENT
from liuer_bot.config import Config
from liuer_bot.conversation_service import ConversationService
from liuer_bot.discord_runtime import create_intents, handle_message
from liuer_bot.generation_queue import SerializedGenerationQueue
from liuer_bot.nickname_service import NicknameService
from liuer_bot.responder import ChatRequest


@dataclass
class FakeAuthor:
    id: int = 10
    bot: bool = False
    display_name: str = "Test User"


@dataclass
class FakeChannel:
    id: int = 123
    is_thread: bool = False
    fetched_message: object | None = None
    fetch_error: Exception | None = None
    fetch_calls: list[int] = field(default_factory=list)
    sent_messages: list[dict[str, object]] = field(default_factory=list)
    send_error: Exception | None = None

    async def fetch_message(self, message_id: int) -> object | None:
        self.fetch_calls.append(message_id)
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.fetched_message

    async def send(self, content: str, **kwargs: object) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent_messages.append({"content": content, **kwargs})


@dataclass
class FakeGuild:
    id: int = 20


@dataclass
class FakeMention:
    id: int


@dataclass
class FakeRole:
    name: str


def _png_bytes() -> bytes:
    image = Image.new("RGB", (1, 1), (20, 40, 60))
    try:
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    finally:
        image.close()


def _animated_gif_bytes() -> bytes:
    first = Image.new("RGB", (1, 1), (255, 0, 0))
    second = Image.new("RGB", (1, 1), (0, 0, 255))
    try:
        output = io.BytesIO()
        first.save(output, format="GIF", save_all=True, append_images=[second], loop=0)
        return output.getvalue()
    finally:
        first.close()
        second.close()


@dataclass
class FakeAttachment:
    id: int
    content_type: str | None
    filename: str
    size: int
    data: bytes = field(default_factory=_png_bytes)
    read_error: Exception | None = None
    read_delay_seconds: float = 0.0
    url: str = "https://example.invalid/private-discord-attachment"
    proxy_url: str = "https://example.invalid/private-discord-attachment-proxy"
    read_calls: list[bool] = field(default_factory=list)

    async def read(self, *, use_cached: bool = False) -> bytes:
        self.read_calls.append(use_cached)
        if self.read_delay_seconds:
            await asyncio.sleep(self.read_delay_seconds)
        if self.read_error is not None:
            raise self.read_error
        return self.data


@dataclass
class FakeMessage:
    author: FakeAuthor = field(default_factory=FakeAuthor)
    channel: FakeChannel = field(default_factory=FakeChannel)
    guild: FakeGuild | None = field(default_factory=FakeGuild)
    mentions: list[FakeMention] = field(default_factory=lambda: [FakeMention(999)])
    role_mentions: list[FakeRole] = field(default_factory=list)
    id: int = 1
    content: str = "六耳 private message content"
    attachments: list[object] = field(default_factory=list)
    reference: object | None = None
    replies: list[dict[str, object]] = field(default_factory=list)

    async def reply(self, content: str, **kwargs: object) -> None:
        self.replies.append({"content": content, **kwargs})


class RecordingResponder:
    def __init__(self, result: str = "六耳已收到訊息。") -> None:
        self.result = result
        self.calls: list[ChatRequest] = []

    async def generate(self, request: ChatRequest) -> str:
        self.calls.append(request)
        return self.result


class FailingResponder:
    async def generate(self, request: ChatRequest) -> str:
        raise RuntimeError(f"private message content: {request.content}")


class BlockingResponder:
    def __init__(self) -> None:
        self.started: list[int] = []
        self.active = 0
        self.max_active = 0
        self.first_release = asyncio.Event()
        self.first_started = asyncio.Event()

    async def generate(self, request: ChatRequest) -> str:
        self.started.append(request.message_id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if request.message_id == 1:
                self.first_started.set()
                await self.first_release.wait()
            return f"response-{request.message_id}"
        finally:
            self.active -= 1


def _run(
    message: FakeMessage,
    responder: object,
    logger: logging.Logger,
    *,
    config: Config | None = None,
    bot_user_id: int | None = None,
    active_nickname: str | None = None,
) -> None:
    async def run() -> None:
        queue = SerializedGenerationQueue(responder)  # type: ignore[arg-type]
        await queue.start()
        try:
            await handle_message(
                message,
                chat_channel_id=123,
                generation_queue=queue,
                logger=logger,
                config=config,
                bot_user_id=bot_user_id,
                active_nickname=active_nickname,
            )
        finally:
            await queue.close()

    asyncio.run(run())


def _image_config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "discord_token": "private-token",
        "chat_channel_id": 123,
        "lm_studio_model": "configured-model",
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


def test_concurrent_triggered_messages_queue_fifo_and_reply_once() -> None:
    async def run() -> None:
        responder = BlockingResponder()
        queue = SerializedGenerationQueue(responder)
        first_message = FakeMessage(id=1)
        second_message = FakeMessage(id=2)
        await queue.start()
        try:
            first = asyncio.create_task(
                handle_message(
                    first_message,
                    chat_channel_id=123,
                    generation_queue=queue,
                    logger=logging.getLogger("test.concurrent.first"),
                )
            )
            await asyncio.wait_for(responder.first_started.wait(), timeout=1)
            second = asyncio.create_task(
                handle_message(
                    second_message,
                    chat_channel_id=123,
                    generation_queue=queue,
                    logger=logging.getLogger("test.concurrent.second"),
                )
            )
            await asyncio.sleep(0)
            assert second_message.replies == []
            assert responder.started == [1]

            responder.first_release.set()
            await asyncio.gather(first, second)
        finally:
            await queue.close()

        assert responder.started == [1, 2]
        assert responder.max_active == 1
        assert [reply["content"] for reply in first_message.replies] == ["response-1"]
        assert [reply["content"] for reply in second_message.replies] == ["response-2"]

    asyncio.run(run())


def test_successful_message_calls_responder_and_replies_once() -> None:
    message = FakeMessage()
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.success"))

    assert len(responder.calls) == 1
    assert len(message.replies) == 1
    reply = message.replies[0]
    assert reply["content"] == "六耳已收到訊息。"
    assert reply["mention_author"] is False
    allowed_mentions = reply["allowed_mentions"]
    assert isinstance(allowed_mentions, discord.AllowedMentions)
    assert allowed_mentions.everyone is False
    assert allowed_mentions.users is False
    assert allowed_mentions.roles is False
    assert allowed_mentions.replied_user is False

    request = responder.calls[0]
    assert request.content == "private message content"
    assert "private message content" not in repr(request)


def test_long_response_is_delivered_in_order_with_mentions_suppressed() -> None:
    generated = "x" * 2001 + "y" * 2001
    message = FakeMessage()
    responder = RecordingResponder(result=generated)

    _run(message, responder, logging.getLogger("test.long-response"))

    assert len(responder.calls) == 1
    assert len(message.replies) == 1
    emitted = [message.replies[0], *message.channel.sent_messages]
    assert "".join(item["content"] for item in emitted) == generated
    assert all(len(item["content"]) <= 2000 for item in emitted)
    assert all(
        isinstance(item["allowed_mentions"], discord.AllowedMentions)
        and item["allowed_mentions"].everyone is False
        and item["allowed_mentions"].users is False
        and item["allowed_mentions"].roles is False
        and item["allowed_mentions"].replied_user is False
        for item in emitted
    )


def test_response_chunks_do_not_interleave_with_another_response() -> None:
    async def run() -> list[str]:
        trace: list[str] = []
        responder = RecordingResponder(result="z" * 2500)
        queue = SerializedGenerationQueue(responder)
        first = FakeMessage(id=1)
        second = FakeMessage(id=2)
        delivery_lock = asyncio.Lock()

        async def first_reply(content: str, **kwargs: object) -> None:
            trace.append("first-start")
            await asyncio.sleep(0)
            trace.append("first-end")
            first.replies.append({"content": content, **kwargs})

        async def second_reply(content: str, **kwargs: object) -> None:
            trace.append("second-start")
            await asyncio.sleep(0)
            trace.append("second-end")
            second.replies.append({"content": content, **kwargs})

        async def first_send(content: str, **kwargs: object) -> None:
            trace.append("first-send-start")
            await asyncio.sleep(0)
            trace.append("first-send-end")
            first.channel.sent_messages.append({"content": content, **kwargs})

        async def second_send(content: str, **kwargs: object) -> None:
            trace.append("second-send-start")
            await asyncio.sleep(0)
            trace.append("second-send-end")
            second.channel.sent_messages.append({"content": content, **kwargs})

        first.reply = first_reply  # type: ignore[method-assign]
        second.reply = second_reply  # type: ignore[method-assign]
        first.channel.send = first_send  # type: ignore[method-assign]
        second.channel.send = second_send  # type: ignore[method-assign]

        await queue.start()
        try:
            await asyncio.gather(
                handle_message(
                    first,
                    chat_channel_id=123,
                    generation_queue=queue,
                    logger=logging.getLogger("test.delivery.first"),
                    response_delivery_lock=delivery_lock,
                ),
                handle_message(
                    second,
                    chat_channel_id=123,
                    generation_queue=queue,
                    logger=logging.getLogger("test.delivery.second"),
                    response_delivery_lock=delivery_lock,
                ),
            )
        finally:
            await queue.close()
        return trace

    trace = asyncio.run(run())

    first_end = trace.index("first-send-end")
    second_start = trace.index("second-start")
    assert first_end < second_start


def test_delivery_failure_does_not_regenerate_or_log_response_body(caplog) -> None:
    generated = "phase5a-private-output-" + "q" * 2100
    message = FakeMessage(channel=FakeChannel(send_error=RuntimeError("send failed")))
    responder = RecordingResponder(result=generated)
    logger = logging.getLogger("test.delivery-failure")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        _run(message, responder, logger)

    assert len(responder.calls) == 1
    assert len(message.replies) == 1
    assert message.channel.sent_messages == []
    assert generated not in caplog.text
    assert "Discord response delivery failed" in caplog.text


def test_runtime_records_successful_turn_before_reply_without_mention(tmp_path) -> None:
    async def run() -> tuple[RecordingResponder, FakeMessage, ConversationService]:
        service = ConversationService(tmp_path / "conversation.sqlite3", chat_channel_id=123)
        await service.initialize()
        responder = RecordingResponder()
        client = discord_runtime.DiscordRuntimeClient(
            config=Config(discord_token="private-token", chat_channel_id=123),
            responder=responder,
            conversation_service=service,
        )
        message = FakeMessage(
            mentions=[],
            content="六耳 我今天換顯卡了",
            attachments=[FakeAttachment(1, "image/png", "image.png", 100)],
        )
        await client._generation_queue.start()
        try:
            await client.on_message(message)
        finally:
            await client._generation_queue.close()
        return responder, message, service

    responder, message, service = asyncio.run(run())

    assert len(responder.calls) == 1
    assert len(message.replies) == 1
    turns = service.get_recent_turns()
    assert len(turns) == 1
    assert turns[0].user_message_id == message.id
    assert turns[0].user_display_name == "Test User"
    assert responder.calls[0].content == "我今天換顯卡了"
    assert turns[0].user_content == "我今天換顯卡了"
    assert turns[0].user_image_count == 1


@pytest.mark.parametrize(
    ("content", "expected_content"),
    [
        ("六耳 你好", "你好"),
        ("六耳你好", "你好"),
        ("六耳，你在幹嘛？", "你在幹嘛？"),
        ("六耳: 幫我看看", "幫我看看"),
    ],
)
def test_leading_primary_name_forms_route_cleaned_content_once(
    content: str,
    expected_content: str,
) -> None:
    message = FakeMessage(content=content)
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.primary-address"))

    assert [request.content for request in responder.calls] == [expected_content]
    assert len(message.replies) == 1


def test_active_nickname_routes_and_persists_cleaned_content_once() -> None:
    message = FakeMessage(
        content="小六 我今天換顯卡了",
        attachments=[
            FakeAttachment(1, "image/png", "one.png", 100),
            FakeAttachment(2, "image/png", "two.png", 100),
        ],
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.nickname-address"), active_nickname="小六")

    assert [request.content for request in responder.calls] == ["我今天換顯卡了"]
    assert responder.calls[0].prepared_images and len(responder.calls[0].prepared_images) == 2
    assert len(message.replies) == 1


def test_client_on_message_reads_nickname_from_memory_snapshot(tmp_path) -> None:
    async def run() -> tuple[RecordingResponder, FakeMessage]:
        nickname_service = NicknameService(tmp_path / "runtime-nickname.sqlite3")
        await nickname_service.initialize()
        await nickname_service.set_nickname(1, "小六")
        responder = RecordingResponder()
        client = discord_runtime.DiscordRuntimeClient(
            config=Config(discord_token="private-token", chat_channel_id=123),
            responder=responder,
            nickname_service=nickname_service,
        )
        message = FakeMessage(content="小六 你好")
        await client._generation_queue.start()
        try:
            await client.on_message(message)
        finally:
            await client._generation_queue.close()
        return responder, message

    responder, message = asyncio.run(run())

    assert responder.calls[0].content == "你好"
    assert len(message.replies) == 1


@pytest.mark.parametrize(
    "content",
    ["你好", "我今天買4090", "我覺得六耳很好", "你問六耳看看"],
)
def test_ordinary_or_late_primary_name_does_not_trigger(content: str) -> None:
    message = FakeMessage(content=content)
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.not-addressed"))

    assert responder.calls == []
    assert message.replies == []


@pytest.mark.parametrize(
    "message",
    [
        FakeMessage(channel=FakeChannel(id=456)),
        FakeMessage(guild=None),
        FakeMessage(channel=FakeChannel(is_thread=True)),
    ],
)
def test_addressed_message_outside_chat_scope_is_ignored(message: FakeMessage) -> None:
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.outside-chat-scope"))

    assert responder.calls == []
    assert message.replies == []


def test_addressed_with_one_png_prepares_one_image_once() -> None:
    message = FakeMessage(
        attachments=[FakeAttachment(1, "image/png", "image.png", 100)],
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.one-image"))

    assert len(responder.calls) == 1
    assert len(message.replies) == 1
    assert len(responder.calls[0].prepared_images) == 1
    assert responder.calls[0].prepared_images[0].attachment_id == 1
    assert responder.calls[0].prepared_images[0].media_type == "image/png"
    assert responder.calls[0].prepared_images[0].data


def test_leading_bot_mention_remains_a_supported_addressing_form() -> None:
    message = FakeMessage(content="<@!999> 你好", mentions=[])
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.bot-mention"), bot_user_id=999)

    assert len(responder.calls) == 1
    assert responder.calls[0].content == "你好"
    assert len(message.replies) == 1


def test_late_bot_mention_does_not_bypass_leading_addressing() -> None:
    message = FakeMessage(content="你問 <@999> 看看", mentions=[])
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.late-bot-mention"), bot_user_id=999)

    assert responder.calls == []
    assert message.replies == []


def test_animated_gif_attachment_prepares_exactly_one_png() -> None:
    attachment = FakeAttachment(
        1,
        "image/gif",
        "image.gif",
        100,
        data=_animated_gif_bytes(),
    )
    message = FakeMessage(attachments=[attachment])
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.animated-gif"))

    assert len(responder.calls) == 1
    assert len(responder.calls[0].prepared_images) == 1
    assert responder.calls[0].prepared_images[0].media_type == "image/png"
    assert len(message.replies) == 1


def test_addressed_with_four_images_uses_one_request_and_reply() -> None:
    attachments = [
        FakeAttachment(1, "image/png", "one.png", 1),
        FakeAttachment(2, "image/jpeg", "two.jpg", 2),
        FakeAttachment(3, "image/webp", "three.webp", 3),
        FakeAttachment(4, "image/gif", "four.gif", 4),
    ]
    message = FakeMessage(attachments=attachments)
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.four-images"))

    assert len(responder.calls) == 1
    assert len(responder.calls[0].prepared_images) == 4
    assert [image.attachment_id for image in responder.calls[0].prepared_images] == [1, 2, 3, 4]
    assert [attachment.read_calls for attachment in attachments] == [[True], [True], [True], [True]]
    assert len(message.replies) == 1


def test_addressed_with_unsupported_attachment_still_replies_without_image_metadata() -> None:
    attachment = FakeAttachment(1, "application/pdf", "misleading.png", 100)
    message = FakeMessage(attachments=[attachment])
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.unsupported-attachment"))

    assert len(responder.calls) == 1
    assert responder.calls[0].prepared_images == ()
    assert len(message.replies) == 1
    assert attachment.read_calls == []


def test_unsupported_attachment_only_is_ignored_in_chat_channel() -> None:
    attachment = FakeAttachment(1, "application/pdf", "report.pdf", 100)
    message = FakeMessage(content="", mentions=[], attachments=[attachment])
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.unsupported-only"))

    assert responder.calls == []
    assert message.replies == []
    assert attachment.read_calls == []


def test_supported_image_and_pdf_prepares_only_the_image() -> None:
    png = FakeAttachment(1, "image/png", "image.png", 100)
    pdf = FakeAttachment(2, "application/pdf", "report.pdf", 100)
    message = FakeMessage(attachments=[png, pdf])
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.png-pdf"))

    assert len(responder.calls) == 1
    assert [image.attachment_id for image in responder.calls[0].prepared_images] == [1]
    assert png.read_calls == [True]
    assert pdf.read_calls == []
    assert len(message.replies) == 1


def test_too_many_selected_images_aborts_before_any_download(caplog) -> None:
    attachments = [
        FakeAttachment(index, "image/png", f"image-{index}.png", 1)
        for index in range(1, 6)
    ]
    message = FakeMessage(attachments=attachments)
    responder = RecordingResponder()
    logger = logging.getLogger("test.too-many-images")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        _run(message, responder, logger)

    assert [attachment.read_calls for attachment in attachments] == [[], [], [], [], []]
    assert responder.calls == []
    assert message.replies == []
    assert "error_type=TooManyImagesError" in caplog.text


def test_declared_per_image_limit_aborts_before_attachment_read(caplog) -> None:
    attachment = FakeAttachment(1, "image/png", "image.png", 11)
    message = FakeMessage(attachments=[attachment])
    responder = RecordingResponder()
    logger = logging.getLogger("test.declared-image-limit")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        _run(
            message,
            responder,
            logger,
            config=_image_config(max_image_bytes=10, max_total_image_bytes=100),
        )

    assert attachment.read_calls == []
    assert responder.calls == []
    assert message.replies == []
    assert "attachment_id=1" in caplog.text
    assert "error_type=ImageTooLargeError" in caplog.text


def test_declared_total_limit_aborts_before_any_attachment_read(caplog) -> None:
    attachments = [
        FakeAttachment(1, "image/png", "one.png", 6),
        FakeAttachment(2, "image/png", "two.png", 6),
    ]
    message = FakeMessage(attachments=attachments)
    responder = RecordingResponder()
    logger = logging.getLogger("test.declared-total-limit")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        _run(
            message,
            responder,
            logger,
            config=_image_config(max_image_bytes=100, max_total_image_bytes=10),
        )

    assert [attachment.read_calls for attachment in attachments] == [[], []]
    assert responder.calls == []
    assert message.replies == []
    assert "error_type=ImageTooLargeError" in caplog.text


def test_actual_downloaded_byte_limits_abort_without_generation() -> None:
    data = _png_bytes()
    attachment = FakeAttachment(1, "image/png", "image.png", 1, data=data)
    message = FakeMessage(attachments=[attachment])
    responder = RecordingResponder()

    _run(
        message,
        responder,
        logging.getLogger("test.actual-per-image-limit"),
        config=_image_config(max_image_bytes=len(data) - 1, max_total_image_bytes=10_000),
    )

    assert attachment.read_calls == [True]
    assert responder.calls == []
    assert message.replies == []


def test_actual_total_byte_limit_aborts_without_partial_generation() -> None:
    data = _png_bytes()
    attachments = [
        FakeAttachment(1, "image/png", "one.png", 1, data=data),
        FakeAttachment(2, "image/png", "two.png", 1, data=data),
    ]
    message = FakeMessage(attachments=attachments)
    responder = RecordingResponder()

    _run(
        message,
        responder,
        logging.getLogger("test.actual-total-limit"),
        config=_image_config(max_image_bytes=10_000, max_total_image_bytes=(len(data) * 2) - 1),
    )

    assert [attachment.read_calls for attachment in attachments] == [[True], [True]]
    assert responder.calls == []
    assert message.replies == []


def test_second_selected_image_failure_aborts_the_entire_message() -> None:
    attachments = [
        FakeAttachment(1, "image/png", "one.png", 1),
        FakeAttachment(2, "image/png", "two.png", 1, data=b"not an image"),
        FakeAttachment(3, "image/png", "three.png", 1),
    ]
    message = FakeMessage(attachments=attachments)
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.second-image-failure"))

    assert [attachment.read_calls for attachment in attachments] == [[True], [True], []]
    assert responder.calls == []
    assert message.replies == []


@pytest.mark.parametrize(
    "error",
    [
        discord.NotFound(SimpleNamespace(status=404, reason="private", headers={}), "private"),
        discord.Forbidden(SimpleNamespace(status=403, reason="private", headers={}), "private"),
        discord.HTTPException(SimpleNamespace(status=500, reason="private", headers={}), "private"),
    ],
)
def test_discord_attachment_download_errors_abort_without_generation(error: Exception) -> None:
    attachment = FakeAttachment(1, "image/png", "image.png", 1, read_error=error)
    message = FakeMessage(attachments=[attachment])
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.download-error"))

    assert attachment.read_calls == [True]
    assert responder.calls == []
    assert message.replies == []


def test_attachment_download_timeout_aborts_without_generation(caplog) -> None:
    attachment = FakeAttachment(
        1,
        "image/png",
        "image.png",
        1,
        read_delay_seconds=0.01,
    )
    message = FakeMessage(attachments=[attachment])
    responder = RecordingResponder()
    logger = logging.getLogger("test.download-timeout")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        _run(
            message,
            responder,
            logger,
            config=_image_config(image_download_timeout_seconds=0.0001),
        )

    assert attachment.read_calls == [True]
    assert responder.calls == []
    assert message.replies == []
    assert "error_type=ImageDownloadTimeoutError" in caplog.text


def test_image_preparation_runs_off_the_discord_event_loop(monkeypatch) -> None:
    original_to_thread = asyncio.to_thread
    calls: list[object] = []

    async def recording_to_thread(function, /, *args, **kwargs):
        calls.append(function)
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(discord_runtime.asyncio, "to_thread", recording_to_thread)
    message = FakeMessage(attachments=[FakeAttachment(1, "image/png", "image.png", 1)])
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.image-to-thread"))

    assert calls == [discord_runtime.prepare_image]
    assert len(responder.calls) == 1


def test_image_failure_logs_only_safe_metadata(caplog) -> None:
    attachment = FakeAttachment(
        1,
        "image/png",
        "phase2ba-private-filename-check.png",
        1,
        data=b"phase2ba-private-image-bytes-check",
    )
    message = FakeMessage(
        content="六耳 phase2ba-private-message-content-check",
        attachments=[attachment],
    )
    responder = RecordingResponder()
    logger = logging.getLogger("test.image-failure-privacy")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        _run(message, responder, logger)

    assert responder.calls == []
    assert message.replies == []
    assert "message_id=1" in caplog.text
    assert "attachment_id=1" in caplog.text
    assert "phase2ba-private-message-content-check" not in caplog.text
    assert "phase2ba-private-filename-check.png" not in caplog.text
    assert "https://example.invalid/private-discord-attachment" not in caplog.text
    assert "phase2ba-private-image-bytes-check" not in caplog.text


def test_supported_image_only_in_chat_channel_is_ignored() -> None:
    attachment = FakeAttachment(1, "image/png", "image.png", 100)
    message = FakeMessage(
        content="",
        mentions=[],
        attachments=[attachment],
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.pure-image"))

    assert responder.calls == []
    assert message.replies == []
    assert attachment.read_calls == []


def test_addressed_image_only_is_one_generation_with_minimal_text() -> None:
    message = FakeMessage(
        content="六耳",
        mentions=[],
        attachments=[FakeAttachment(1, "image/png", "image.png", 100)],
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.addressed-pure-image"))

    assert len(responder.calls) == 1
    assert responder.calls[0].content == EMPTY_ADDRESSED_CONTENT
    assert len(responder.calls[0].prepared_images) == 1
    assert len(message.replies) == 1


def test_address_only_without_image_uses_one_minimal_request() -> None:
    message = FakeMessage(content="六耳")
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.address-only"))

    assert [request.content for request in responder.calls] == [EMPTY_ADDRESSED_CONTENT]
    assert len(message.replies) == 1


def test_reply_relationship_never_bypasses_addressing() -> None:
    referenced_bot = SimpleNamespace(id=700, author=FakeAuthor(id=999, bot=True))
    message = FakeMessage(
        content="再說一次",
        reference=SimpleNamespace(resolved=referenced_bot),
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.reply-not-trigger"))

    assert responder.calls == []
    assert message.replies == []


def test_addressed_reply_to_human_enriches_context_without_changing_trigger() -> None:
    referenced_human = SimpleNamespace(
        id=701,
        author=FakeAuthor(id=777, display_name="Alice"),
        content="測試識別碼：X9Q7-AB12",
        attachments=[],
    )
    message = FakeMessage(
        content="六耳 再說一次",
        reference=SimpleNamespace(resolved=referenced_human),
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.addressed-human-reply"))

    assert [request.content for request in responder.calls] == ["再說一次"]
    assert responder.calls[0].reply_context is not None
    assert responder.calls[0].reply_context.content == "測試識別碼：X9Q7-AB12"
    assert len(message.replies) == 1


def test_addressed_reply_to_current_bot_marks_context_as_assistant() -> None:
    referenced_bot = SimpleNamespace(
        id=702,
        author=FakeAuthor(id=999, bot=True, display_name="Old Bot Name"),
        content="你剛才說的是4090。",
        attachments=[],
    )
    message = FakeMessage(
        content="六耳 再解釋一次",
        reference=SimpleNamespace(resolved=referenced_bot),
    )
    responder = RecordingResponder()

    _run(
        message,
        responder,
        logging.getLogger("test.reply-bot-context"),
        bot_user_id=999,
    )

    assert len(responder.calls) == 1
    assert responder.calls[0].reply_context is not None
    assert responder.calls[0].reply_context.is_current_bot is True


def test_addressed_deleted_reply_continues_one_generation_with_unavailable_context() -> None:
    channel = FakeChannel(
        fetch_error=discord.NotFound(
            SimpleNamespace(status=404, reason="private", headers={}), "private"
        )
    )
    message = FakeMessage(
        channel=channel,
        content="六耳 這則回覆還看得到嗎？",
        reference=SimpleNamespace(message_id=704),
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.reply-deleted"))

    assert channel.fetch_calls == [704]
    assert len(responder.calls) == 1
    assert responder.calls[0].reply_context is not None
    assert responder.calls[0].reply_context.is_available is False
    assert len(message.replies) == 1


def test_unaddressed_reply_does_not_resolve_or_fetch_reference() -> None:
    channel = FakeChannel()
    message = FakeMessage(
        channel=channel,
        content="真的假的",
        reference=SimpleNamespace(message_id=703),
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.reply-not-trigger-fetch"))

    assert channel.fetch_calls == []
    assert responder.calls == []
    assert message.replies == []


def test_reply_outside_chat_channel_does_not_fetch_reference() -> None:
    channel = FakeChannel(id=999)
    message = FakeMessage(
        channel=channel,
        content="六耳 這裡不該觸發",
        reference=SimpleNamespace(message_id=705),
    )
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.reply-outside-channel"))

    assert channel.fetch_calls == []
    assert responder.calls == []
    assert message.replies == []


def test_ignored_message_does_not_call_responder_or_reply() -> None:
    message = FakeMessage(mentions=[], content="你好")
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.ignored"))

    assert responder.calls == []
    assert message.replies == []


def test_bot_message_does_not_call_responder_or_reply() -> None:
    message = FakeMessage(author=FakeAuthor(bot=True))
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.bot"))

    assert responder.calls == []
    assert message.replies == []


def test_same_named_role_mention_does_not_trigger_bot(caplog) -> None:
    message = FakeMessage(
        mentions=[],
        role_mentions=[FakeRole(name="六耳")],
        content="",
    )
    responder = RecordingResponder()
    logger = logging.getLogger("test.role-mention")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        _run(message, responder, logger)

    assert "routing decision=IGNORE_NOT_ADDRESSED" in caplog.text
    assert "address_kind=none" in caplog.text
    assert responder.calls == []
    assert message.replies == []


def test_responder_failure_is_isolated_and_logs_no_content(caplog) -> None:
    message = FakeMessage(content="六耳 phase1a-private-routing-diagnostic-content")
    logger = logging.getLogger("test.failure")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        _run(message, FailingResponder(), logger)

    assert message.replies == []
    assert "phase1a-private-routing-diagnostic-content" not in caplog.text
    assert "message handling failed" in caplog.text


def test_empty_responder_result_is_not_sent(caplog) -> None:
    message = FakeMessage()
    logger = logging.getLogger("test.empty")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        _run(message, RecordingResponder(result="  \n"), logger)

    assert message.replies == []
    assert "empty responder result" in caplog.text
    assert "private message content" not in caplog.text


def test_phase1_fake_responder_output_is_deterministic() -> None:
    message = FakeMessage()
    responder = RecordingResponder()

    _run(message, responder, logging.getLogger("test.fake"))

    assert message.replies[0]["content"] == "六耳已收到訊息。"


def test_phase1a_intents_enable_only_required_message_intents() -> None:
    intents = create_intents()

    assert intents.guilds is True
    assert intents.guild_messages is True
    assert intents.message_content is True
    assert intents.members is False
    assert intents.presences is False
    assert intents.voice_states is False
    assert intents.reactions is False
    assert intents.typing is False


def test_on_message_enters_routing_and_logs_safe_metadata(monkeypatch, caplog) -> None:
    calls: list[tuple[FakeMessage, dict[str, object]]] = []

    async def fake_handle_message(message: FakeMessage, **kwargs: object) -> None:
        calls.append((message, kwargs))

    monkeypatch.setattr(discord_runtime, "handle_message", fake_handle_message)
    monkeypatch.setattr(
        discord_runtime.DiscordRuntimeClient,
        "user",
        SimpleNamespace(id=999),
        raising=False,
    )
    client = discord_runtime.DiscordRuntimeClient.__new__(discord_runtime.DiscordRuntimeClient)
    client._config = SimpleNamespace(chat_channel_id=123)
    client._responder = RecordingResponder()
    client._generation_queue = object()
    client._logger = logging.getLogger("test.event-entry")
    message = FakeMessage(
        content="六耳 phase1a-private-routing-diagnostic-content",
        attachments=[object()],
    )

    with caplog.at_level(logging.DEBUG, logger=client._logger.name):
        asyncio.run(client.on_message(message))

    assert len(calls) == 1
    assert calls[0][1]["chat_channel_id"] == 123
    assert calls[0][1]["generation_queue"] is client._generation_queue
    assert calls[0][1]["bot_user_id"] == 999
    assert "Discord message received" in caplog.text
    assert "message_id=1" in caplog.text
    assert "guild_id=20" in caplog.text
    assert "channel_id=123" in caplog.text
    assert "author_id=10" in caplog.text
    assert "author_bot=False" in caplog.text
    assert "mentions=1" in caplog.text
    assert "attachments=1" in caplog.text
    assert "phase1a-private-routing-diagnostic-content" not in caplog.text
    token = f"phase1a-{uuid4().hex}"
    assert token not in caplog.text


def test_setup_hook_starts_one_worker_for_repeated_ready_lifecycle() -> None:
    async def run() -> None:
        client = discord_runtime.DiscordRuntimeClient(
            config=Config(discord_token="private-token", chat_channel_id=123),
            responder=RecordingResponder(),
        )
        client._connection.application_id = 1

        async def fetch_channel(channel_id: int) -> object:
            return SimpleNamespace(
                id=channel_id,
                type=discord.ChannelType.text,
                guild=SimpleNamespace(id=20),
            )

        client.fetch_channel = fetch_channel  # type: ignore[method-assign]
        async def sync(*, guild: object) -> None:
            return None

        client.tree.sync = sync  # type: ignore[method-assign]
        queue = client._generation_queue

        await client.setup_hook()
        first_worker = queue.worker_task
        await client.setup_hook()

        assert first_worker is not None
        assert queue.worker_task is first_worker
        await queue.close()

    asyncio.run(run())


def test_client_close_shuts_down_queue_before_discord_close(monkeypatch) -> None:
    events: list[str] = []

    async def fake_discord_close(_client: discord.Client) -> None:
        events.append("discord-close")

    monkeypatch.setattr(discord.Client, "close", fake_discord_close)
    client = discord_runtime.DiscordRuntimeClient(
        config=Config(discord_token="private-token", chat_channel_id=123),
        responder=RecordingResponder(),
    )

    original_queue_close = client._generation_queue.close

    async def recording_queue_close() -> None:
        events.append("queue-close")
        await original_queue_close()

    client._generation_queue.close = recording_queue_close  # type: ignore[method-assign]
    asyncio.run(client.close())

    assert events == ["queue-close", "discord-close"]


def test_run_discord_uses_lm_studio_responder_and_preflights_by_default(monkeypatch) -> None:
    events: list[str] = []

    class FakeLmStudioResponder:
        def __init__(self, config: Config) -> None:
            self.config = config
            events.append("responder-created")

        async def preflight(self) -> None:
            events.append("preflight")

    class FakeClient:
        def run(self, token: str, **kwargs: object) -> None:
            events.append("discord-run")

    def fake_client_factory(config: Config, responder: object) -> FakeClient:
        assert isinstance(responder, FakeLmStudioResponder)
        return FakeClient()

    monkeypatch.setattr(discord_runtime, "create_llm_provider", FakeLmStudioResponder)
    discord_runtime.run_discord(
        Config(
            discord_token="private-token",
            chat_channel_id=123,
            lm_studio_model="configured-model",
        ),
        client_factory=fake_client_factory,  # type: ignore[arg-type]
    )

    assert events == ["responder-created", "preflight", "discord-run"]


def test_attachment_diagnostics_do_not_log_filename_or_url(caplog) -> None:
    message = FakeMessage(
        content="六耳 phase1a-private-routing-diagnostic-content",
        attachments=[FakeAttachment(1, "image/png", "private-file.png", 100)],
    )
    logger = logging.getLogger("test.attachment-diagnostics")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        _run(message, RecordingResponder(), logger)

    assert "attachment_count=1" in caplog.text
    assert "supported_image_count=1" in caplog.text
    assert "private-file.png" not in caplog.text
    assert "https://" not in caplog.text
    assert "phase1a-private-routing-diagnostic-content" not in caplog.text


@pytest.mark.parametrize(
    ("message", "expected_decision", "expected_addressed", "expected_kind"),
    [
        (
            FakeMessage(
                channel=FakeChannel(id=456),
                author=FakeAuthor(display_name="phase4aa-private-display-name"),
                content="六耳 phase4aa-private-routing-content",
            ),
            "IGNORE_CHANNEL",
            True,
            "primary_name",
        ),
        (
            FakeMessage(
                author=FakeAuthor(display_name="phase4aa-private-display-name"),
                content="phase4aa-private-routing-content",
            ),
            "IGNORE_NOT_ADDRESSED",
            False,
            "none",
        ),
        (
            FakeMessage(
                author=FakeAuthor(bot=True, display_name="phase4aa-private-display-name"),
                content="六耳 phase4aa-private-routing-content",
            ),
            "IGNORE_AUTHOR_BOT",
            True,
            "primary_name",
        ),
        (
            FakeMessage(
                author=FakeAuthor(display_name="phase4aa-private-display-name"),
                content="六耳 phase4aa-private-routing-content",
            ),
            "RESPOND",
            True,
            "primary_name",
        ),
    ],
)
def test_routing_decision_diagnostics_are_safe(
    message: FakeMessage,
    expected_decision: str,
    expected_addressed: bool,
    expected_kind: str,
    caplog,
) -> None:
    logger = logging.getLogger(f"test.routing.{expected_decision}")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        _run(message, RecordingResponder(), logger)

    assert f"routing decision={expected_decision}" in caplog.text
    assert f"addressed={expected_addressed}" in caplog.text
    assert f"address_kind={expected_kind}" in caplog.text
    assert "phase4aa-private-routing-content" not in caplog.text
    assert "phase4aa-private-display-name" not in caplog.text

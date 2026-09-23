"""Discord transport, single-channel chat, and Persona command lifecycle."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

import discord
from discord import app_commands

from .addressing import parse_addressed_message
from .attachment_metadata import classify_image_attachment
from .bot_dialogue import BotDialogueService
from .bot_dialogue_commands import BotDialogueCommands
from .config import (
    DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_MAX_IMAGES_PER_MESSAGE,
    DEFAULT_MAX_TOTAL_IMAGE_BYTES,
    Config,
)
from .context_commands import ContextCommands
from .conversation_models import AuthorKind
from .conversation_service import ConversationRecordingResponder, ConversationService
from .discord_output import split_discord_response
from .discord_routing import RoutingDecision, RoutingInput, route_input
from .generation_queue import GenerationQueueClosedError, SerializedGenerationQueue
from .image_preparation import (
    ImageDownloadError,
    ImageDownloadTimeoutError,
    ImagePreparationError,
    ImageTooLargeError,
    PreparedImage,
    TooManyImagesError,
    prepare_image,
)
from .llm_provider import create_llm_provider
from .nickname_commands import NicknameCommands
from .nickname_service import NicknameService
from .persona_commands import PersonaCommands
from .persona_service import PersonaService
from .persona_status import PersonaStatusManager
from .prompting import PromptBuilder
from .reply_context import resolve_reply_context
from .responder import ChatRequest, ImageAttachment, Responder
from .runtime_commands import RuntimeCommands
from .runtime_health import RuntimeHealthService
from .runtime_status import RuntimeStatusService
from .typing_indicator import TypingIndicatorManager

LOGGER = logging.getLogger(__name__)


def create_intents() -> discord.Intents:
    """Create only the guild text and message-content intents this Bot needs."""

    intents = discord.Intents.none()
    intents.guilds = True
    intents.guild_messages = True
    intents.message_content = True
    return intents


class DiscordRuntimeClient(discord.Client):
    """Transport adapter connecting discord.py events to the responder boundary."""

    def __init__(
        self,
        *,
        config: Config,
        responder: Responder,
        persona_service: PersonaService | None = None,
        nickname_service: NicknameService | None = None,
        prompt_builder: PromptBuilder | None = None,
        conversation_service: ConversationService | None = None,
    ) -> None:
        super().__init__(intents=create_intents(), allowed_mentions=discord.AllowedMentions.none())
        self.tree = app_commands.CommandTree(self)
        self._config = config
        self._nickname_service = nickname_service
        self._conversation_service = conversation_service
        self._bot_dialogue_service = BotDialogueService()
        self._responder = (
            ConversationRecordingResponder(responder, conversation_service)
            if conversation_service is not None
            else responder
        )
        self._generation_queue = SerializedGenerationQueue(self._responder)
        self._response_delivery_lock = asyncio.Lock()
        self._logger = LOGGER
        self._typing_manager = TypingIndicatorManager(logger=self._logger)
        self._runtime_status_service = RuntimeStatusService(
            config,
            generation_queue=self._generation_queue,
            conversation_service=conversation_service,
            persona_service=persona_service,
            nickname_service=nickname_service,
            runtime_ready=self.is_ready,
            discord_latency=lambda: self.latency,
        )
        self._runtime_health_service = RuntimeHealthService(
            config,
            status_service=self._runtime_status_service,
        )
        self._chat_channel: object | None = None
        self._persona_setup_complete = False
        self._persona_status_manager: PersonaStatusManager | None = None
        self._persona_commands: PersonaCommands | None = None
        self._nickname_commands: NicknameCommands | None = None
        self._context_commands: ContextCommands | None = (
            ContextCommands(
                conversation_service=conversation_service,
                chat_channel_id=config.chat_channel_id,
                logger=self._logger,
                bot_dialogue_service=self._bot_dialogue_service,
            )
            if conversation_service is not None
            else None
        )
        self._runtime_commands: RuntimeCommands | None = (
            RuntimeCommands(
                status_channel_id=config.persona_status_channel_id,
                status_service=self._runtime_status_service,
                health_service=self._runtime_health_service,
                logger=self._logger,
            )
            if config.persona_status_channel_id is not None
            else None
        )
        self._bot_dialogue_commands = BotDialogueCommands(
            dialogue_service=self._bot_dialogue_service,
            chat_channel_id=config.chat_channel_id,
            status_channel_id=config.persona_status_channel_id,
            bot_user_id_provider=lambda: self.user.id if self.user is not None else None,
            initial_message_handler=self._handle_initial_bot_message,
            logger=self._logger,
        )
        if persona_service is not None and config.persona_status_channel_id is not None:
            selected_prompt_builder = prompt_builder or PromptBuilder()
            self._persona_status_manager = PersonaStatusManager(
                client=self,
                status_channel_id=config.persona_status_channel_id,
                persona_service=persona_service,
                prompt_builder=selected_prompt_builder,
                nickname_service=nickname_service,
                logger=self._logger,
            )
            self._persona_commands = PersonaCommands(
                persona_service=persona_service,
                status_manager=self._persona_status_manager,
                max_persona_chars=config.persona_max_chars,
                logger=self._logger,
            )
            if nickname_service is not None:
                self._nickname_commands = NicknameCommands(
                    nickname_service=nickname_service,
                    status_manager=self._persona_status_manager,
                    logger=self._logger,
                )

    async def setup_hook(self) -> None:
        await self._generation_queue.start()
        status_manager = getattr(self, "_persona_status_manager", None)
        try:
            if getattr(self, "_persona_setup_complete", False):
                return
            chat_guild_id = await self._initialize_chat_channel()
            status_manager = getattr(self, "_persona_status_manager", None)
            guild = discord.Object(id=chat_guild_id)
            if self._persona_commands is not None:
                if status_manager is None:
                    raise RuntimeError("Persona status manager is unavailable")
                status_guild_id = await status_manager.initialize_channel()
                if status_guild_id != chat_guild_id:
                    raise RuntimeError("configured chat and Persona channels must share a guild")
                self.tree.add_command(self._persona_commands.group, guild=guild)
                if self._nickname_commands is not None:
                    self.tree.add_command(self._nickname_commands.group, guild=guild)
            if self._context_commands is not None:
                self.tree.add_command(self._context_commands.group, guild=guild)
            self.tree.add_command(self._bot_dialogue_commands.group, guild=guild)
            self.tree.add_command(self._bot_dialogue_commands.context_menu, guild=guild)
            if self._runtime_commands is not None:
                for command in self._runtime_commands.commands:
                    self.tree.add_command(command, guild=guild)
            if (
                self._persona_commands is not None
                or self._context_commands is not None
                or self._runtime_commands is not None
                or self._bot_dialogue_commands is not None
            ):
                await self.tree.sync(guild=guild)
            if status_manager is not None:
                await status_manager.sync_current()
            self._persona_setup_complete = True
        except Exception:
            await self._generation_queue.close()
            self._logger.error(
                "Discord startup validation failed channel_id=%s",
                status_manager.channel_id if status_manager is not None else None,
            )
            raise

    async def on_ready(self) -> None:
        if self.user is not None:
            self._logger.info(
                "Discord ready bot_user_id=%s configured_chat_channel_count=1",
                self.user.id,
            )

    async def on_message(self, message: discord.Message) -> None:
        metadata = _message_metadata(message)
        self._logger.debug(
            "Discord message received message_id=%s guild_id=%s channel_id=%s "
            "author_id=%s author_bot=%s mentions=%s attachments=%s",
            metadata["message_id"],
            metadata["guild_id"],
            metadata["channel_id"],
            metadata["author_id"],
            metadata["author_bot"],
            metadata["mention_count"],
            metadata["attachment_count"],
        )
        nickname_service = getattr(self, "_nickname_service", None)
        await handle_message(
            message,
            chat_channel_id=self._config.chat_channel_id,
            generation_queue=self._generation_queue,
            response_delivery_lock=getattr(self, "_response_delivery_lock", None),
            logger=self._logger,
            config=self._config,
            bot_user_id=self.user.id if self.user is not None else None,
            active_nickname=(
                nickname_service.get_active_nickname()
                if nickname_service is not None
                else None
            ),
            bot_dialogue_service=getattr(self, "_bot_dialogue_service", None),
            typing_manager=getattr(self, "_typing_manager", None),
        )

    async def _handle_initial_bot_message(
        self,
        message: discord.Message,
        session_id: str,
    ) -> None:
        nickname_service = getattr(self, "_nickname_service", None)
        await handle_message(
            message,
            chat_channel_id=self._config.chat_channel_id,
            generation_queue=self._generation_queue,
            response_delivery_lock=self._response_delivery_lock,
            logger=self._logger,
            config=self._config,
            bot_user_id=self.user.id if self.user is not None else None,
            active_nickname=(
                nickname_service.get_active_nickname()
                if nickname_service is not None
                else None
            ),
            bot_dialogue_service=self._bot_dialogue_service,
            initial_bot_session_id=session_id,
            typing_manager=self._typing_manager,
        )

    async def _initialize_chat_channel(self) -> int:
        if self._chat_channel is None:
            try:
                channel = await self.fetch_channel(self._config.chat_channel_id)
            except Exception as exc:
                self._logger.error(
                    "chat channel fetch failed channel_id=%s error_type=%s",
                    self._config.chat_channel_id,
                    type(exc).__name__,
                )
                raise RuntimeError("configured chat channel could not be fetched") from exc
            _validate_chat_channel(channel, self._config.chat_channel_id)
            self._chat_channel = channel
        guild = getattr(self._chat_channel, "guild", None)
        guild_id = getattr(guild, "id", None)
        if not isinstance(guild_id, int) or guild_id <= 0:
            raise RuntimeError("configured chat channel has no guild")
        return guild_id

    async def close(self) -> None:
        self._bot_dialogue_service.close()
        await self._typing_manager.close()
        await self._generation_queue.close()
        await super().close()


def _message_metadata(message: discord.Message) -> dict[str, int | bool | None]:
    guild = message.guild
    return {
        "message_id": message.id,
        "guild_id": guild.id if guild is not None else None,
        "channel_id": message.channel.id,
        "author_id": message.author.id,
        "author_bot": message.author.bot,
        "mention_count": len(message.mentions),
        "attachment_count": len(message.attachments),
    }


def _message_display_name(message: discord.Message) -> str:
    author = message.author
    display_name = getattr(author, "display_name", None)
    if isinstance(display_name, str):
        return display_name
    name = getattr(author, "name", None)
    return name if isinstance(name, str) else ""


def _is_thread_channel(channel: object) -> bool:
    if isinstance(channel, discord.Thread) or bool(getattr(channel, "is_thread", False)):
        return True
    channel_type = getattr(channel, "type", None)
    thread_types = {
        getattr(discord.ChannelType, "public_thread", None),
        getattr(discord.ChannelType, "private_thread", None),
        getattr(discord.ChannelType, "news_thread", None),
    }
    return channel_type in thread_types


def _is_supported_chat_channel(channel: object) -> bool:
    if _is_thread_channel(channel):
        return False
    channel_type = getattr(channel, "type", None)
    supported_types = {
        getattr(discord.ChannelType, "text", None),
        getattr(discord.ChannelType, "news", None),
    }
    if channel_type is not None:
        return channel_type in supported_types
    channel_classes = tuple(
        channel_class
        for channel_class in (
            getattr(discord, "TextChannel", None),
            getattr(discord, "NewsChannel", None),
        )
        if isinstance(channel_class, type)
    )
    return isinstance(channel, channel_classes)


def _validate_chat_channel(channel: object, channel_id: int) -> None:
    if getattr(channel, "id", None) != channel_id:
        raise RuntimeError("configured chat channel identity is invalid")
    guild = getattr(channel, "guild", None)
    if guild is None or not isinstance(getattr(guild, "id", None), int):
        raise RuntimeError("configured chat channel must belong to a guild")
    if not _is_supported_chat_channel(channel):
        raise RuntimeError("configured chat channel type is unsupported")


async def handle_message(
    message: discord.Message,
    *,
    chat_channel_id: int,
    generation_queue: SerializedGenerationQueue,
    logger: logging.Logger,
    response_delivery_lock: asyncio.Lock | None = None,
    config: Config | None = None,
    bot_user_id: int | None = None,
    active_nickname: str | None = None,
    bot_dialogue_service: BotDialogueService | None = None,
    initial_bot_session_id: str | None = None,
    typing_manager: TypingIndicatorManager | None = None,
) -> None:
    """Route, generate, and reply for one Discord message event."""

    image_candidates = _image_candidates(tuple(message.attachments))
    guild = message.guild
    is_webhook = getattr(message, "webhook_id", None) is not None
    is_bot_author = bool(getattr(message.author, "bot", False))
    is_self_message = bot_user_id is not None and message.author.id == bot_user_id
    is_bot_turn = initial_bot_session_id is not None
    bot_session_id = initial_bot_session_id
    addressing = parse_addressed_message(
        message.content,
        bot_user_id=bot_user_id,
        active_nickname=active_nickname,
    )
    bot_content = (
        addressing.conversational_content
        if addressing.is_addressed
        else message.content
    )
    is_thread = _is_thread_channel(message.channel)
    if not is_bot_turn and bot_dialogue_service is not None and is_bot_author:
        if (
            not is_webhook
            and not is_self_message
            and guild is not None
            and not is_thread
            and message.channel.id == chat_channel_id
            and bot_dialogue_service.is_active_peer(message.author.id)
        ):
            if not bot_content.strip() and not image_candidates:
                return
            bot_session_id = bot_dialogue_service.claim_peer_turn(message.author.id)
            is_bot_turn = bot_session_id is not None
    if is_bot_turn:
        if (
            guild is None
            or not is_bot_author
            or is_webhook
            or is_self_message
            or is_thread
            or message.channel.id != chat_channel_id
        ):
            if bot_session_id is not None and bot_dialogue_service is not None:
                bot_dialogue_service.complete_turn(bot_session_id, delivered=False)
            return
        decision_name = "RESPOND_BOT_CHAT"
        content = bot_content
        has_text_content = bool(content.strip())
    else:
        routing_input = RoutingInput(
            guild_id=guild.id if guild is not None else None,
            channel_id=message.channel.id,
            author_id=message.author.id,
            author_is_bot=message.author.bot,
            has_text_content=bool(addressing.conversational_content.strip()),
            has_supported_image=bool(image_candidates),
            is_thread=is_thread,
            is_addressed=addressing.is_addressed,
        )
        decision = route_input(
            routing_input,
            chat_channel_id=chat_channel_id,
        )
        decision_name = decision.name
        content = addressing.conversational_content
        has_text_content = routing_input.has_text_content
    metadata = _message_metadata(message)
    logger.debug(
        "routing decision=%s message_id=%s guild_id=%s channel_id=%s author_id=%s "
        "attachment_count=%s supported_image_count=%s text_present=%s is_thread=%s "
        "addressed=%s address_kind=%s",
        decision_name,
        metadata["message_id"],
        metadata["guild_id"],
        metadata["channel_id"],
        metadata["author_id"],
        len(message.attachments),
        len(image_candidates),
        has_text_content,
        is_thread,
        addressing.is_addressed if not is_bot_turn else bool(content.strip()),
        addressing.address_kind.value,
    )
    if not is_bot_turn and decision is not RoutingDecision.RESPOND:
        return

    guild = message.guild
    if guild is None:
        return
    typing_scope = (
        typing_manager.indicate(message.channel)
        if typing_manager is not None
        else _no_typing()
    )
    request: ChatRequest | None = None
    chunks: tuple[str, ...] | None = None
    try:
        async with typing_scope:
            try:
                prepared_images = await _prepare_images(
                    image_candidates,
                    message_id=message.id,
                    config=config,
                    logger=logger,
                )
            except ImagePreparationError:
                if bot_session_id is not None and bot_dialogue_service is not None:
                    bot_dialogue_service.complete_turn(bot_session_id, delivered=False)
                return
            reply_resolution = await resolve_reply_context(
                message,
                bot_user_id=bot_user_id,
                logger=logger,
            )
            request = ChatRequest(
                message_id=message.id,
                guild_id=guild.id,
                channel_id=message.channel.id,
                author_id=message.author.id,
                content=content,
                prepared_images=prepared_images,
                author_display_name=_message_display_name(message),
                reply_context=reply_resolution.context,
                author_kind=AuthorKind.BOT if is_bot_turn else AuthorKind.HUMAN,
            )
            generated = await generation_queue.submit(request)
            if not generated or not generated.strip():
                logger.warning(
                    "empty responder result message_id=%s guild_id=%s channel_id=%s",
                    request.message_id,
                    request.guild_id,
                    request.channel_id,
                )
                if bot_session_id is not None and bot_dialogue_service is not None:
                    bot_dialogue_service.complete_turn(bot_session_id, delivered=False)
                return
            chunks = split_discord_response(generated)
            if not chunks:
                logger.warning(
                    "empty responder result message_id=%s guild_id=%s channel_id=%s",
                    request.message_id,
                    request.guild_id,
                    request.channel_id,
                )
                if bot_session_id is not None and bot_dialogue_service is not None:
                    bot_dialogue_service.complete_turn(bot_session_id, delivered=False)
                return
    except GenerationQueueClosedError:
        logger.debug(
            "message generation skipped because queue is closed message_id=%s",
            message.id,
        )
        if bot_session_id is not None and bot_dialogue_service is not None:
            bot_dialogue_service.complete_turn(bot_session_id, delivered=False)
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(
            "message handling failed message_id=%s guild_id=%s channel_id=%s error_type=%s",
            message.id,
            guild.id,
            message.channel.id,
            type(exc).__name__,
        )
        if bot_session_id is not None and bot_dialogue_service is not None:
            bot_dialogue_service.complete_turn(bot_session_id, delivered=False)
        return

    if request is None or chunks is None:
        return
    try:
        if response_delivery_lock is None:
            delivered = await _send_response_chunks(message, chunks, logger=logger)
        else:
            async with response_delivery_lock:
                delivered = await _send_response_chunks(message, chunks, logger=logger)
        if bot_session_id is not None and bot_dialogue_service is not None:
            bot_dialogue_service.complete_turn(bot_session_id, delivered=delivered)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(
            "message handling failed message_id=%s guild_id=%s channel_id=%s error_type=%s",
            request.message_id,
            request.guild_id,
            request.channel_id,
            type(exc).__name__,
        )
        if bot_session_id is not None and bot_dialogue_service is not None:
            bot_dialogue_service.complete_turn(bot_session_id, delivered=False)


@asynccontextmanager
async def _no_typing():
    """Keep direct unit callers backward-compatible when no manager is supplied."""

    yield


async def _send_response_chunks(
    message: discord.Message,
    chunks: tuple[str, ...],
    *,
    logger: logging.Logger,
) -> bool:
    """Send one generated response sequentially without exposing its body."""

    for index, chunk in enumerate(chunks):
        try:
            if index == 0:
                await message.reply(
                    chunk,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await message.channel.send(
                    chunk,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Discord response delivery failed message_id=%s chunk_index=%s "
                "chunk_count=%s chunk_chars=%s error_type=%s",
                message.id,
                index,
                len(chunks),
                len(chunk),
                type(exc).__name__,
            )
            return False
    return True


def _image_candidates(
    attachments: tuple[discord.Attachment, ...],
) -> tuple[tuple[discord.Attachment, ImageAttachment], ...]:
    """Keep Discord objects only until their validated bytes are prepared."""

    candidates: list[tuple[discord.Attachment, ImageAttachment]] = []
    for attachment in attachments:
        metadata = classify_image_attachment(attachment)
        if metadata is not None:
            candidates.append((attachment, metadata))
    return tuple(candidates)


async def _prepare_images(
    candidates: tuple[tuple[discord.Attachment, ImageAttachment], ...],
    *,
    message_id: int,
    config: Config | None,
    logger: logging.Logger,
) -> tuple[PreparedImage, ...]:
    """Acquire selected Discord images in order before one queue submission."""

    (
        download_timeout_seconds,
        max_images_per_message,
        max_image_bytes,
        max_total_image_bytes,
        max_image_pixels,
    ) = _image_limits(config)
    try:
        if len(candidates) > max_images_per_message:
            raise TooManyImagesError()
        if sum(metadata.size_bytes for _, metadata in candidates) > max_total_image_bytes:
            raise ImageTooLargeError()
    except ImagePreparationError as exc:
        _log_image_failure(logger, message_id=message_id, attachment_id=None, exc=exc)
        raise

    prepared_images: list[PreparedImage] = []
    total_downloaded_bytes = 0
    for attachment, metadata in candidates:
        try:
            if metadata.size_bytes > max_image_bytes:
                raise ImageTooLargeError()
            data = await _read_attachment_bytes(attachment, download_timeout_seconds)
            if len(data) > max_image_bytes:
                raise ImageTooLargeError()
            total_downloaded_bytes += len(data)
            if total_downloaded_bytes > max_total_image_bytes:
                raise ImageTooLargeError()
            prepared_images.append(
                await asyncio.to_thread(
                    prepare_image,
                    metadata.attachment_id,
                    metadata.media_type,
                    data,
                    max_image_bytes=max_image_bytes,
                    max_image_pixels=max_image_pixels,
                )
            )
        except asyncio.CancelledError:
            raise
        except ImagePreparationError as exc:
            _log_image_failure(
                logger,
                message_id=message_id,
                attachment_id=metadata.attachment_id,
                exc=exc,
            )
            raise
    return tuple(prepared_images)


def _image_limits(config: Config | None) -> tuple[float, int, int, int, int]:
    if config is None:
        return (
            DEFAULT_IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
            DEFAULT_MAX_IMAGES_PER_MESSAGE,
            DEFAULT_MAX_IMAGE_BYTES,
            DEFAULT_MAX_TOTAL_IMAGE_BYTES,
            DEFAULT_MAX_IMAGE_PIXELS,
        )
    return (
        config.image_download_timeout_seconds,
        config.max_images_per_message,
        config.max_image_bytes,
        config.max_total_image_bytes,
        config.max_image_pixels,
    )


async def _read_attachment_bytes(
    attachment: discord.Attachment,
    timeout_seconds: float,
) -> bytes:
    try:
        async with asyncio.timeout(timeout_seconds):
            return await attachment.read(use_cached=True)
    except TimeoutError as exc:
        raise ImageDownloadTimeoutError() from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise ImageDownloadError() from exc


def _log_image_failure(
    logger: logging.Logger,
    *,
    message_id: int,
    attachment_id: int | None,
    exc: ImagePreparationError,
) -> None:
    logger.warning(
        "image preparation failed message_id=%s attachment_id=%s error_type=%s",
        message_id,
        attachment_id,
        type(exc).__name__,
    )


ClientFactory = Callable[..., DiscordRuntimeClient]


def create_client(
    config: Config,
    responder: Responder,
    *,
    persona_service: PersonaService | None = None,
    nickname_service: NicknameService | None = None,
    prompt_builder: PromptBuilder | None = None,
    conversation_service: ConversationService | None = None,
) -> DiscordRuntimeClient:
    """Construct the real Discord transport for the runtime boundary."""

    return DiscordRuntimeClient(
        config=config,
        responder=responder,
        persona_service=persona_service,
        nickname_service=nickname_service,
        prompt_builder=prompt_builder,
        conversation_service=conversation_service,
    )


def run_discord(
    config: Config,
    *,
    responder: Responder | None = None,
    persona_service: PersonaService | None = None,
    nickname_service: NicknameService | None = None,
    prompt_builder: PromptBuilder | None = None,
    conversation_service: ConversationService | None = None,
    client_factory: ClientFactory = create_client,
) -> None:
    """Start the long-running Discord client with safe library logging options."""

    selected_responder = responder
    if selected_responder is None:
        if persona_service is None and prompt_builder is None:
            selected_responder = create_llm_provider(config)
        else:
            selected_responder = create_llm_provider(
                config,
                prompt_builder=prompt_builder,
                active_persona_provider=persona_service,
                active_nickname_provider=nickname_service,
                conversation_context_provider=conversation_service,
            )
        asyncio.run(selected_responder.preflight())
    LOGGER.info("starting Discord runtime")
    if (
        persona_service is None
        and nickname_service is None
        and prompt_builder is None
        and conversation_service is None
    ):
        client = client_factory(
            config,
            selected_responder,
        )
    else:
        client = client_factory(
            config,
            selected_responder,
            persona_service=persona_service,
            nickname_service=nickname_service,
            prompt_builder=prompt_builder,
            conversation_service=conversation_service,
        )
    try:
        client.run(config.discord_token or "", log_handler=None, root_logger=False)
    except Exception as exc:
        LOGGER.error("Discord runtime failed error_type=%s", type(exc).__name__)
        raise RuntimeError("Discord runtime failed") from exc

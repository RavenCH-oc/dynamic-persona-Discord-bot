"""Small, testable command line interface for liuer-bot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TextIO

from .config import ConfigError, load_config
from .conversation_service import ConversationError, ConversationService
from .database import DatabaseError
from .discord_runtime import run_discord
from .doctor import run_doctor
from .llm_provider import LlmProviderError, create_llm_provider
from .logging_config import configure_logging
from .nickname_service import NicknameError, NicknameService
from .persona_service import PersonaError, PersonaService
from .prompting import PromptBuilder, PromptResourceError


def build_parser() -> argparse.ArgumentParser:
    """Build the current liuer-bot argument parser."""

    parser = argparse.ArgumentParser(
        prog="liuer-bot",
        description="Commands for the liuer-bot runtime.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="check Python and local configuration")
    subparsers.add_parser("run", help="start the Discord text runtime")
    return parser


def _run_discord_command(
    *,
    environ: Mapping[str, str] | None,
    dotenv_path: str | Path | None,
) -> int:
    try:
        config = load_config(environ, dotenv_path=dotenv_path)
    except ConfigError as exc:
        sys.stdout.write("Configuration: invalid\n")
        sys.stderr.write(f"liuer-bot: {exc}\n")
        return 2

    configure_logging(config.log_level)
    try:
        prompt_builder = PromptBuilder()
        persona_service = PersonaService.from_config(config)
        asyncio.run(persona_service.initialize())
        nickname_service = NicknameService.from_config(config)
        asyncio.run(nickname_service.initialize())
        conversation_service = ConversationService.from_config(config)
        asyncio.run(conversation_service.initialize())
        responder = create_llm_provider(
            config,
            prompt_builder=prompt_builder,
            active_persona_provider=persona_service,
            active_nickname_provider=nickname_service,
            conversation_context_provider=conversation_service,
        )
    except PromptResourceError as exc:
        logging.getLogger(__name__).error(
            "Prompt initialization failed error_type=%s",
            type(exc).__name__,
        )
        return 1
    except (DatabaseError, NicknameError, PersonaError, ConversationError) as exc:
        logging.getLogger(__name__).error(
            "Local Persona startup failed error_type=%s",
            type(exc).__name__,
        )
        return 1
    try:
        asyncio.run(responder.preflight())
    except LlmProviderError as exc:
        logging.getLogger(__name__).error(
            "LLM provider preflight failed provider=%s error_type=%s",
            config.llm_provider.value,
            type(exc).__name__,
        )
        return 1
    try:
        run_discord(
            config,
            responder=responder,
            persona_service=persona_service,
            nickname_service=nickname_service,
            prompt_builder=prompt_builder,
            conversation_service=conversation_service,
        )
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Discord runtime failed error_type=%s",
            type(exc).__name__,
        )
        return 1
    return 0


def _run(
    argv: Sequence[str] | None,
    *,
    environ: Mapping[str, str] | None,
    dotenv_path: str | Path | None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "doctor":
        return run_doctor(
            environ,
            dotenv_path=dotenv_path,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    if arguments.command == "run":
        return _run_discord_command(environ=environ, dotenv_path=dotenv_path)
    parser.print_help()
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return its process exit code.

    The optional streams and environment mapping make command behavior easy
    to exercise without mutating the process environment in tests.
    """

    with ExitStack() as stack:
        if stdout is not None:
            stack.enter_context(redirect_stdout(stdout))
        if stderr is not None:
            stack.enter_context(redirect_stderr(stderr))
        try:
            return _run(argv, environ=environ, dotenv_path=dotenv_path)
        except Exception as exc:
            # This is the process boundary: unexpected failures become the
            # documented internal-error status without exposing configuration.
            sys.stderr.write(f"liuer-bot: internal failure: {type(exc).__name__}\n")
            return 1

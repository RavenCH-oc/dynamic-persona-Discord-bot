"""Offline Python and active-provider configuration diagnostics."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

from .config import ConfigError, load_config
from .llm_provider import LlmProviderKind
from .logging_config import configure_logging


def _supported_python() -> bool:
    return sys.version_info[:2] == (3, 13)


def run_doctor(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | Path | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run diagnostics without contacting any external service.

    Returns ``0`` for a valid local setup and ``2`` for an unsupported Python
    version or invalid configuration. Unexpected failures are allowed to
    reach the CLI boundary, where they are reported as exit code ``1``.
    """

    output = stdout or sys.stdout
    error = stderr or sys.stderr
    python_ok = _supported_python()

    try:
        config = load_config(environ, dotenv_path=dotenv_path)
    except ConfigError as exc:
        output.write(
            "Python: "
            + ("valid (3.13.x)" if python_ok else "invalid (requires Python 3.13.x)")
            + "\n"
        )
        output.write("Configuration: invalid\n")
        error.write(f"doctor: {exc}\n")
        return 2

    configure_logging(config.log_level)
    output.write(
        "Python: "
        + ("valid (3.13.x)" if python_ok else "invalid (requires Python 3.13.x)")
        + "\n"
    )
    output.write("Configuration: valid\n")
    output.write(f"Discord token: {'configured' if config.discord_token else 'missing'}\n")
    output.write("Chat channel: valid\n")
    output.write("Persona status channel: valid\n")
    output.write("Context recent turns: valid\n")
    output.write(f"LLM provider: {config.llm_provider.value}\n")
    if config.llm_provider is LlmProviderKind.VENICE:
        output.write("Venice base URL: valid\n")
        output.write("Venice text model: configured\n")
        output.write("Venice vision model: configured\n")
        output.write("Venice API key: configured\n")
    else:
        output.write("LM Studio base URL: valid\n")
        output.write("LM Studio model: configured\n")
        output.write(
            "LM Studio API token: "
            f"{'configured' if config.lm_studio_api_token else 'not configured'}\n"
        )
    output.write("Provider request timeout: valid\n")
    output.write("Provider temperature: valid\n")
    output.write("Deep response max tokens: valid\n")
    output.write("Response chat-short max tokens: valid\n")
    output.write("Response normal max tokens: valid\n")
    output.write("Context history max chars chat-short: valid\n")
    output.write("Context history max chars normal: valid\n")
    output.write("Context history max chars deep: valid\n")
    output.write("Image download timeout: valid\n")
    output.write("Image limits: valid\n")
    output.write("Model image limits: valid\n")
    output.write("Database path: valid\n")
    output.write("Persona max chars: valid\n")
    output.write("Persona global cooldown: valid\n")
    output.write("Persona daily limit: valid\n")
    output.write("Persona daily reset offset: valid\n")
    output.write("Nickname max chars: valid\n")
    output.write("Log level: valid\n")
    return 0 if python_ok else 2

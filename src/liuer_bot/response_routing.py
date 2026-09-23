"""Pure deterministic response-mode routing for one current ChatRequest."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum, auto

from .config import DEFAULT_LM_STUDIO_MAX_TOKENS, Config
from .responder import ChatRequest

DEFAULT_RESPONSE_CHAT_SHORT_MAX_TOKENS = 1_024
DEFAULT_RESPONSE_NORMAL_MAX_TOKENS = 1_536
DEEP_INPUT_CHAR_THRESHOLD = 600
CASUAL_MAX_CHARS = 80

_DISCORD_USER_MENTION = re.compile(r"<@!?[0-9]+>")
_ENGLISH_BRIEF = re.compile(
    r"(?<![a-z0-9])(?:briefly|concise|short answer|one sentence|just answer|"
    r"no explanation)(?![a-z0-9])",
    re.IGNORECASE,
)
_ENGLISH_DEEP = re.compile(
    r"(?<![a-z0-9])(?:in detail|deep dive|step by step|comprehensive|thoroughly|"
    r"analyze in depth|detailed comparison)(?![a-z0-9])",
    re.IGNORECASE,
)

# Keep these small and high-confidence. They are intentionally not an NLP or
# translation dictionary; unknown language remains NORMAL by default.
_BRIEF_MARKERS = frozenset(
    {
        "簡短",
        "簡潔",
        "簡短回答",
        "一句話",
        "只要答案",
        "不要解釋",
        "簡要",
        "簡單回答",
    }
)
_DEEP_MARKERS = frozenset(
    {
        "詳細",
        "深入",
        "逐步",
        "一步一步",
        "完整分析",
        "全面",
        "徹底",
        "深入分析",
        "詳細比較",
        "詳述",
        "深度分析",
        "分步說明",
    }
)
_CASUAL_EXACT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "yo",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "thx",
        "lol",
        "haha",
        "how are you",
        "good morning",
        "good night",
        "bye",
        "嗨",
        "你好",
        "早安",
        "晚安",
        "收到",
        "了解",
        "好的",
        "謝謝",
        "哈哈",
        "掰掰",
        "你在幹嘛",
    }
)
_TECHNICAL_MARKERS = frozenset(
    {
        "api",
        "async",
        "code",
        "cpu",
        "database",
        "discord",
        "error",
        "gpu",
        "hardware",
        "http",
        "lm studio",
        "python",
        "ram",
        "sqlite",
        "stack trace",
        "traceback",
        "vram",
    }
)
_QUESTION_MARKERS = frozenset(
    {
        "can",
        "how",
        "what",
        "which",
        "why",
        "where",
        "when",
        "怎麼",
        "如何",
        "什麼",
        "為什麼",
        "哪個",
    }
)


class ResponseMode(Enum):
    """The only response detail modes supported in Phase 3C-A."""

    CHAT_SHORT = auto()
    NORMAL = auto()
    DEEP = auto()


class ResponseRouteReason(Enum):
    """Typed, privacy-safe explanation for a response-mode decision."""

    EXPLICIT_BRIEF = auto()
    EXPLICIT_DEEP = auto()
    CONFLICTING_EXPLICIT_REQUEST = auto()
    CASUAL_SMALL_TALK = auto()
    MULTIMODAL_MINIMUM = auto()
    LONG_INPUT = auto()
    CODE_OR_STRUCTURED_INPUT = auto()
    DEFAULT_NORMAL = auto()


# Short alias for callers that prefer the simpler public name.
ResponseReason = ResponseRouteReason


@dataclass(frozen=True, slots=True, repr=False)
class ResponseDecision:
    """Immutable routing result containing no user-provided text."""

    mode: ResponseMode
    max_tokens: int
    reason: ResponseRouteReason

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ResponseMode):
            raise TypeError("mode must be a ResponseMode")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not isinstance(self.reason, ResponseRouteReason):
            raise TypeError("reason must be a ResponseRouteReason")

    def __repr__(self) -> str:
        return (
            "ResponseDecision("
            f"mode={self.mode.name!r}, "
            f"max_tokens={self.max_tokens!r}, "
            f"reason={self.reason.name!r})"
        )


RESPONSE_MODE_INSTRUCTIONS: dict[ResponseMode, str] = {
    ResponseMode.CHAT_SHORT: (
        "Answer briefly and concisely, preferably in one or two sentences. "
        "Avoid unnecessary expansion."
    ),
    ResponseMode.NORMAL: (
        "Answer the user's actual question directly. For a simple single-topic "
        "knowledge question, give the core definition or explanation first, "
        "usually in one to three short paragraphs or a small number of concise "
        "bullets. Include only details materially useful to answering the "
        "question; do not automatically add buying advice, unrelated application "
        "scenarios, extended background, extra examples, or an offer to continue "
        "helping. Expand further only when the question requires it, while staying "
        "factually correct and consistent with the active Persona."
    ),
    ResponseMode.DEEP: (
        "Provide a complete, detailed analysis with useful reasoning, tradeoffs, "
        "and step-by-step detail where appropriate."
    ),
}


def response_instruction(mode: ResponseMode) -> str:
    """Return the fixed runtime prompt instruction for one response mode."""

    return RESPONSE_MODE_INSTRUCTIONS[mode]


def normalize_routing_text(content: str) -> str:
    """Normalize only the router's private inspection copy of message text."""

    normalized = unicodedata.normalize("NFKC", content)
    normalized = _DISCORD_USER_MENTION.sub(" ", normalized)
    return " ".join(normalized.casefold().strip().split())


class ResponseRouter:
    """Route one request synchronously without external state or inference."""

    def __init__(
        self,
        chat_short_max_tokens: int = DEFAULT_RESPONSE_CHAT_SHORT_MAX_TOKENS,
        normal_max_tokens: int = DEFAULT_RESPONSE_NORMAL_MAX_TOKENS,
        deep_max_tokens: int = DEFAULT_LM_STUDIO_MAX_TOKENS,
    ) -> None:
        _validate_token_budgets(chat_short_max_tokens, normal_max_tokens, deep_max_tokens)
        self._chat_short_max_tokens = chat_short_max_tokens
        self._normal_max_tokens = normal_max_tokens
        self._deep_max_tokens = deep_max_tokens

    @classmethod
    def from_config(cls, config: Config) -> ResponseRouter:
        return cls(
            chat_short_max_tokens=config.response_chat_short_max_tokens,
            normal_max_tokens=config.response_normal_max_tokens,
            deep_max_tokens=config.lm_studio_max_tokens,
        )

    def route(self, request: ChatRequest) -> ResponseDecision:
        """Return a deterministic decision from current text and prepared images."""

        text = normalize_routing_text(request.content)
        explicit_brief = _has_explicit_brief(text)
        explicit_deep = _has_explicit_deep(text)

        if explicit_brief and explicit_deep:
            return self._decision(
                ResponseMode.NORMAL,
                ResponseRouteReason.CONFLICTING_EXPLICIT_REQUEST,
            )
        if explicit_deep:
            return self._decision(ResponseMode.DEEP, ResponseRouteReason.EXPLICIT_DEEP)
        if request.prepared_images:
            return self._decision(
                ResponseMode.NORMAL,
                ResponseRouteReason.MULTIMODAL_MINIMUM,
            )
        if explicit_brief:
            return self._decision(ResponseMode.CHAT_SHORT, ResponseRouteReason.EXPLICIT_BRIEF)
        if len(text) >= DEEP_INPUT_CHAR_THRESHOLD:
            return self._decision(ResponseMode.DEEP, ResponseRouteReason.LONG_INPUT)
        if _is_code_or_structured(text):
            return self._decision(
                ResponseMode.NORMAL,
                ResponseRouteReason.CODE_OR_STRUCTURED_INPUT,
            )
        if _is_casual_small_talk(text):
            return self._decision(
                ResponseMode.CHAT_SHORT,
                ResponseRouteReason.CASUAL_SMALL_TALK,
            )
        return self._decision(ResponseMode.NORMAL, ResponseRouteReason.DEFAULT_NORMAL)

    def _decision(self, mode: ResponseMode, reason: ResponseRouteReason) -> ResponseDecision:
        max_tokens = {
            ResponseMode.CHAT_SHORT: self._chat_short_max_tokens,
            ResponseMode.NORMAL: self._normal_max_tokens,
            ResponseMode.DEEP: self._deep_max_tokens,
        }[mode]
        return ResponseDecision(mode=mode, max_tokens=max_tokens, reason=reason)


def _validate_token_budgets(chat_short: int, normal: int, deep: int) -> None:
    values = (chat_short, normal, deep)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError("response token budgets must be integers")
    if not 1 <= chat_short <= normal <= deep:
        raise ValueError("response token budgets must satisfy short <= normal <= deep")


def _has_explicit_brief(text: str) -> bool:
    return bool(_ENGLISH_BRIEF.search(text)) or any(marker in text for marker in _BRIEF_MARKERS)


def _has_explicit_deep(text: str) -> bool:
    return bool(_ENGLISH_DEEP.search(text)) or any(marker in text for marker in _DEEP_MARKERS)


def _is_code_or_structured(text: str) -> bool:
    if "```" in text:
        return True
    if "traceback (most recent call last)" in text or (
        "file \"" in text and ", line " in text
    ):
        return True
    if re.search(r"(?m)^\s*(?:def|class|import|from|return|select)\b", text):
        return True
    lines = text.splitlines()
    if len(lines) >= 3 and any(
        marker in text
        for marker in ("def ", "class ", "import ", "from ", "return ", "select ")
    ):
        return True
    if len(text) >= 80:
        code_markers = sum(text.count(marker) for marker in (" = ", "()", "{}", ";", "=>"))
        return code_markers >= 3
    return False


def _is_casual_small_talk(text: str) -> bool:
    if not text or len(text) > CASUAL_MAX_CHARS:
        return False
    if _is_short_technical_question(text):
        return False
    key = text.rstrip("!?.,:;。！？。 ")
    if key in _CASUAL_EXACT:
        return True
    words = set(key.split())
    return len(words) <= 4 and bool(words & {"hi", "hello", "hey", "thanks", "okay", "ok"})


def _is_short_technical_question(text: str) -> bool:
    """Keep short technical prompts in NORMAL rather than casual mode."""

    return any(marker in text for marker in _TECHNICAL_MARKERS)


__all__ = [
    "CASUAL_MAX_CHARS",
    "DEEP_INPUT_CHAR_THRESHOLD",
    "DEFAULT_RESPONSE_CHAT_SHORT_MAX_TOKENS",
    "DEFAULT_RESPONSE_NORMAL_MAX_TOKENS",
    "RESPONSE_MODE_INSTRUCTIONS",
    "ResponseDecision",
    "ResponseMode",
    "ResponseReason",
    "ResponseRouteReason",
    "ResponseRouter",
    "normalize_routing_text",
    "response_instruction",
]

from __future__ import annotations

from liuer_bot.addressing import parse_addressed_message
from liuer_bot.image_preparation import PreparedImage
from liuer_bot.responder import ChatRequest
from liuer_bot.response_routing import (
    ResponseMode,
    ResponseRouter,
    ResponseRouteReason,
    normalize_routing_text,
)


def _request(content: str, *, images: tuple[PreparedImage, ...] = ()) -> ChatRequest:
    return ChatRequest(1, 20, 123, 10, content, prepared_images=images)


def _image() -> PreparedImage:
    return PreparedImage(99, "image/png", b"prepared", 1, 1)


def test_explicit_brief_and_deep_markers_use_their_mode_budgets() -> None:
    router = ResponseRouter()

    brief = router.route(_request("Please answer briefly."))
    deep = router.route(_request("Give me a detailed comparison."))

    assert (brief.mode, brief.max_tokens, brief.reason) == (
        ResponseMode.CHAT_SHORT,
        1024,
        ResponseRouteReason.EXPLICIT_BRIEF,
    )
    assert (deep.mode, deep.max_tokens, deep.reason) == (
        ResponseMode.DEEP,
        4096,
        ResponseRouteReason.EXPLICIT_DEEP,
    )


def test_conflicting_explicit_requests_fail_toward_normal() -> None:
    decision = ResponseRouter().route(_request("Be concise, but explain in detail."))

    assert decision.mode is ResponseMode.NORMAL
    assert decision.max_tokens == 1536
    assert decision.reason is ResponseRouteReason.CONFLICTING_EXPLICIT_REQUEST


def test_casual_small_talk_is_short_but_technical_question_is_normal() -> None:
    router = ResponseRouter()

    assert router.route(_request("hello!")).reason is ResponseRouteReason.CASUAL_SMALL_TALK
    technical = router.route(_request("What is VRAM?"))
    assert technical.mode is ResponseMode.NORMAL
    assert technical.reason is ResponseRouteReason.DEFAULT_NORMAL


def test_live_qa_examples_route_to_expected_modes() -> None:
    router = ResponseRouter()

    casual = router.route(_request("你在幹嘛？"))
    normal = router.route(_request("VRAM是什麼？"))
    deep = router.route(_request("詳細比較VRAM與系統RAM的差異"))

    assert casual.mode is ResponseMode.CHAT_SHORT
    assert casual.reason is ResponseRouteReason.CASUAL_SMALL_TALK
    assert normal.mode is ResponseMode.NORMAL
    assert normal.reason is ResponseRouteReason.DEFAULT_NORMAL
    assert deep.mode is ResponseMode.DEEP
    assert deep.reason is ResponseRouteReason.EXPLICIT_DEEP


def test_multimodal_request_has_normal_floor_unless_explicitly_deep() -> None:
    router = ResponseRouter()

    brief_image = router.route(_request("Answer briefly.", images=(_image(),)))
    deep_image = router.route(_request("Analyze in depth.", images=(_image(),)))

    assert brief_image.reason is ResponseRouteReason.MULTIMODAL_MINIMUM
    assert brief_image.mode is ResponseMode.NORMAL
    assert deep_image.reason is ResponseRouteReason.EXPLICIT_DEEP
    assert deep_image.mode is ResponseMode.DEEP


def test_long_and_structured_inputs_do_not_use_chat_short() -> None:
    long_text = "Please examine this context. " + ("important detail " * 45)
    long_decision = ResponseRouter().route(_request(long_text))
    code_decision = ResponseRouter().route(_request("```python\nprint('hello')\n```"))

    assert long_decision.reason is ResponseRouteReason.LONG_INPUT
    assert long_decision.mode is ResponseMode.DEEP
    assert code_decision.reason is ResponseRouteReason.CODE_OR_STRUCTURED_INPUT
    assert code_decision.mode is ResponseMode.NORMAL


def test_routing_normalization_is_private_and_removes_only_user_mentions() -> None:
    content = "  BRIEFLY  <@!123456789>  answer  "
    request = _request(content)

    assert normalize_routing_text(content) == "briefly answer"
    decision = ResponseRouter().route(request)
    assert decision.mode is ResponseMode.CHAT_SHORT
    assert request.content == content


def test_router_is_deterministic_and_decision_repr_has_no_request_content() -> None:
    private_content = "phase3c-private-routing-content-check"
    router = ResponseRouter()
    decisions = [router.route(_request(private_content)) for _ in range(100)]

    assert len(set(decisions)) == 1
    assert private_content not in repr(decisions[0])


def test_address_prefix_is_removed_before_response_router() -> None:
    router = ResponseRouter()

    short = parse_addressed_message("六耳 你在幹嘛？")
    normal = parse_addressed_message("六耳 VRAM是什麼？")
    deep = parse_addressed_message("六耳 詳細比較VRAM與系統RAM的差異")

    assert short.conversational_content == "你在幹嘛？"
    assert normal.conversational_content == "VRAM是什麼？"
    assert deep.conversational_content == "詳細比較VRAM與系統RAM的差異"
    short_request = _request(short.conversational_content)
    normal_request = _request(normal.conversational_content)
    deep_request = _request(deep.conversational_content)
    assert short_request.content == "你在幹嘛？"
    assert normal_request.content == "VRAM是什麼？"
    assert deep_request.content == "詳細比較VRAM與系統RAM的差異"
    assert router.route(short_request).mode is ResponseMode.CHAT_SHORT
    assert router.route(normal_request).mode is ResponseMode.NORMAL
    assert router.route(deep_request).mode is ResponseMode.DEEP


def test_active_nickname_prefix_is_removed_before_all_response_modes() -> None:
    router = ResponseRouter()

    short = parse_addressed_message("小六 你在幹嘛？", active_nickname="小六")
    normal = parse_addressed_message("小六 VRAM是什麼？", active_nickname="小六")
    deep = parse_addressed_message(
        "小六 詳細比較VRAM與系統RAM的差異",
        active_nickname="小六",
    )

    assert short.conversational_content == "你在幹嘛？"
    assert normal.conversational_content == "VRAM是什麼？"
    assert deep.conversational_content == "詳細比較VRAM與系統RAM的差異"
    assert router.route(_request(short.conversational_content)).mode is ResponseMode.CHAT_SHORT
    assert router.route(_request(normal.conversational_content)).mode is ResponseMode.NORMAL
    assert router.route(_request(deep.conversational_content)).mode is ResponseMode.DEEP

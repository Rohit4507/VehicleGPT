"""Unit tests for Qwen3 and local SLM intent-classification router with fallback."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest

from vehiclegpt.nlu.evaluate import today_ns
from vehiclegpt.nlu.qwen_router import (
    DEFAULT_QWEN_WINDOWS_PATH,
    DEFAULT_SMOLLM2_WINDOWS_PATH,
    EXTENDED_INTENTS,
    SUPPORTED_INTENTS,
    QwenRouter,
    build_prompt,
    extract_json_payload,
)


@pytest.fixture
def now_ns():
    return today_ns("2026-03-12")


@pytest.fixture
def router():
    return QwenRouter(server_url="http://127.0.0.1:8080/v1", min_confidence=0.6)


def test_taxonomy_defined():
    """Verify strictly required intents exist in taxonomy."""
    for req in ("pricing", "maintenance", "comparison", "vehicle_diagnostics", "troubleshooting"):
        assert req in SUPPORTED_INTENTS
        assert req in EXTENDED_INTENTS


def test_extract_json_payload_valid():
    raw = '{"intent": "pricing", "confidence": 0.95, "reasoning": "User asks for car price."}'
    parsed = extract_json_payload(raw)
    assert parsed is not None
    assert parsed["intent"] == "pricing"
    assert parsed["confidence"] == 0.95


def test_extract_json_payload_with_codeblock():
    raw = (
        '```json\n{"intent": "maintenance", "confidence": 0.88, '
        '"reasoning": "Brake service"}\n```'
    )
    parsed = extract_json_payload(raw)
    assert parsed is not None
    assert parsed["intent"] == "maintenance"


def test_extract_json_payload_malformed():
    raw = "I think the intent is pricing because of cost."
    parsed = extract_json_payload(raw)
    assert parsed is None


def test_prompt_generation_qwen_and_smollm():
    q = "how much does brake fluid service cost?"
    qwen_p = build_prompt(q, model_type="qwen3")
    smol_p = build_prompt(q, model_type="smollm2")
    assert "<|im_start|>system" in qwen_p
    assert "<|im_start|>system" in smol_p
    assert q in qwen_p
    assert q in smol_p


def test_valid_routing_without_fallback(router, now_ns):
    mock_resp = json.dumps({
        "intent": "pricing",
        "confidence": 0.92,
        "reasoning": "Query asks about vehicle price."
    })
    res = router.route(
        "how much does a used Tesla Model 3 cost?", now_ns, raw_response_override=mock_resp
    )
    assert res.is_valid_json is True
    assert res.is_valid_taxonomy is True
    assert res.intent == "pricing"
    assert res.confidence == 0.92
    assert res.fallback_invoked is False
    assert res.final_intent_name == "pricing"


def test_malformed_json_triggers_deterministic_fallback(router, now_ns):
    bad_resp = "Here is my answer: it is maintenance."
    # The question mentions range and why -> deterministic routes to range_reduction
    res = router.route(
        "Why has my driving range reduced over the last two weeks?",
        now_ns,
        raw_response_override=bad_resp,
    )
    assert res.is_valid_json is False
    assert res.fallback_invoked is True
    assert res.fallback_intent is not None
    assert res.fallback_intent.name == "range_reduction"
    assert res.final_intent_name == "range_reduction"


def test_unsupported_intent_triggers_fallback(router, now_ns):
    hallucinated_resp = json.dumps({
        "intent": "alien_invasion_alert",
        "confidence": 0.99,
        "reasoning": "Unrecognized intent"
    })
    res = router.route(
        "Show all ADAS disengagement events from yesterday.",
        now_ns,
        raw_response_override=hallucinated_resp,
    )
    assert res.is_valid_json is True
    assert res.is_valid_taxonomy is False
    assert res.fallback_invoked is True
    assert res.fallback_intent is not None
    assert res.fallback_intent.name == "adas_events"
    assert res.final_intent_name == "adas_events"


def test_low_confidence_triggers_fallback(router, now_ns):
    low_conf_resp = json.dumps({
        "intent": "comparison",
        "confidence": 0.35,  # Below 0.6
        "reasoning": "Not sure"
    })
    res = router.route(
        "Compare energy consumption between city and highway driving.",
        now_ns,
        raw_response_override=low_conf_resp,
    )
    assert res.is_valid_json is True
    assert res.is_valid_taxonomy is True
    assert res.confidence == 0.35
    assert res.fallback_invoked is True
    assert res.fallback_intent is not None
    assert res.fallback_intent.name == "city_vs_highway"
    assert res.final_intent_name == "city_vs_highway"


def test_server_failure_triggers_fallback(router, now_ns):
    with patch.object(
        router, "_call_llama_server", side_effect=urllib.error.URLError("Connection refused")
    ):
        res = router.route("What happened before the battery overheating warning?", now_ns)
        assert res.fallback_invoked is True
        assert res.error is not None
        assert "Connection refused" in res.error
        assert res.fallback_intent is not None
        assert res.fallback_intent.name == "before_warning"
        assert res.final_intent_name == "before_warning"


def test_server_timeout_triggers_fallback(router, now_ns):
    with patch.object(router, "_call_llama_server", side_effect=TimeoutError("Request timed out")):
        res = router.route("When did fuel efficiency show the largest drop?", now_ns)
        assert res.fallback_invoked is True
        assert res.error is not None
        assert "timed out" in res.error
        assert res.fallback_intent is not None
        assert res.fallback_intent.name == "efficiency_drop"
        assert res.final_intent_name == "efficiency_drop"


def test_zero_hallucination_guard():
    """Verify that router result only produces routing decisions, never fake prices or evidence."""
    mock_resp = json.dumps({
        "intent": "pricing",
        "confidence": 0.9,
        "reasoning": "User wants quote"
    })
    router = QwenRouter()
    res = router.route("give me battery price", now_ns=0, raw_response_override=mock_resp)
    assert not hasattr(res, "invented_price")
    assert not hasattr(res, "evidence_data")
    assert res.intent in SUPPORTED_INTENTS or res.intent in EXTENDED_INTENTS


def test_smollm2_configuration():
    smol_router = QwenRouter(model_type="smollm2")
    assert smol_router.model_type == "smollm2"
    assert "smollm2" in smol_router.model_path.lower()
    assert smol_router.model_path == DEFAULT_SMOLLM2_WINDOWS_PATH

    qwen_router = QwenRouter(model_type="qwen3")
    assert qwen_router.model_type == "qwen3"
    assert "qwen3" in qwen_router.model_path.lower()
    assert qwen_router.model_path == DEFAULT_QWEN_WINDOWS_PATH

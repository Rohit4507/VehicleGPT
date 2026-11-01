"""Qwen3 and local SLM intent-classification router with deterministic fallback.

This module integrates local small language models (such as Qwen3 0.6B and SmolLM2 135M)
as an optional intent-classification router in VehicleGPT.

Design and Safety Principles:
1. Strict Intent Taxonomy: The model must classify only supported intents:
   - pricing
   - maintenance
   - comparison
   - vehicle_diagnostics
   - troubleshooting
   (and mapped existing analytical intents if appropriate, e.g. general_information
    or existing handlers: range_reduction, efficiency_drop, adas_events, before_warning,
    city_vs_highway).
2. Structured Output: The model must return valid JSON conforming to the schema:
   {
     "intent": "<supported_intent>",
     "confidence": <float between 0.0 and 1.0>,
     "reasoning": "<brief explanation>"
   }
3. Zero Hallucination of Vehicle Data: The SLM is strictly restricted to intent routing.
   It is never used to invent vehicle prices, specifications, diagnostic facts, or evidence.
   All facts and numbers remain strictly grounded in evidence packs produced by deterministic
   handlers.
4. Confidence & Abstention: If confidence is below ``min_confidence``, or if the model
   abstains, the router falls back to the deterministic router.
5. Robust Fallback: On malformed JSON, out-of-taxonomy intents, server timeouts, or connection
   errors, the system automatically and safely falls back to the existing deterministic
   NLU router (``vehiclegpt.nlu.intent.parse_question``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vehiclegpt.nlu.intent import Intent, parse_question

logger = logging.getLogger(__name__)

#: Supported intent taxonomy required by specification
SUPPORTED_INTENTS: tuple[str, ...] = (
    "pricing",
    "maintenance",
    "comparison",
    "vehicle_diagnostics",
    "troubleshooting",
)

#: Extended taxonomy including core VehicleGPT deterministic intents for compatibility
EXTENDED_INTENTS: tuple[str, ...] = (
    *SUPPORTED_INTENTS,
    "range_reduction",
    "efficiency_drop",
    "adas_events",
    "before_warning",
    "city_vs_highway",
    "unknown",
)

#: Default fallback confidence threshold
DEFAULT_MIN_CONFIDENCE = 0.6

#: Default server timeout in seconds
DEFAULT_TIMEOUT_S = 5.0

#: Default Windows and POSIX model paths
DEFAULT_QWEN_WINDOWS_PATH = r"C:\Users\rohit\models\qwen3-0.6b\Qwen3-0.6B-Q8_0.gguf"
DEFAULT_SMOLLM2_WINDOWS_PATH = r"C:\Users\rohit\VehicleGPT\models\smollm2-135m\SmolLM2-135M-Instruct-Q4_K_M.gguf"


@dataclass
class SLMRoutingResult:
    """Result of an SLM intent routing query."""

    intent: str = "unknown"
    confidence: float = 0.0
    reasoning: str = ""
    raw_response: str = ""
    is_valid_json: bool = False
    is_valid_taxonomy: bool = False
    fallback_invoked: bool = False
    fallback_intent: Intent | None = None
    latency_ms: float = 0.0
    model: str = "qwen3-0.6b"
    error: str | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def final_intent_name(self) -> str:
        """The final intent name, considering fallback if invoked."""
        if self.fallback_invoked and self.fallback_intent is not None:
            return self.fallback_intent.name
        return self.intent


ROUTER_SYSTEM_PROMPT = """You are an intent classification router for an on-device vehicle intelligence system.
Your job is ONLY to classify the user's vehicle-related question into one of the allowed intents.
DO NOT answer the user's question.
DO NOT generate any vehicle data, diagnostic facts, prices, or technical evidence.

Allowed intents:
- "pricing": Queries about purchase prices, lease costs, vehicle valuation, pricing estimates.
- "maintenance": Scheduled servicing, tyre changes, brake pad inspection, fluid replacements.
- "comparison": Comparing regimes (city vs highway), vehicle models, energy usage comparisons.
- "vehicle_diagnostics": Warning lights, battery overheating, sensor alerts, DTC codes, error logs.
- "troubleshooting": Range loss degradation, sudden efficiency drops, ADAS disengagements, failures.
- "unknown": Unclear queries, out-of-scope requests, or off-topic questions.

You must respond ONLY with a JSON object in the following format:
{
  "intent": "<pricing|maintenance|comparison|vehicle_diagnostics|troubleshooting|unknown>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<concise one-sentence rationale>"
}
No additional prose or explanation."""


def build_prompt(question: str, model_type: str = "qwen3") -> str:
    """Build prompt with chat formatting appropriate for the model."""
    clean_q = " ".join(question.strip().split())
    if model_type.lower().startswith("smollm"):
        return (
            f"<|im_start|>system\n{ROUTER_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nClassify this vehicle question: {clean_q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    # Default: Qwen / ChatML formatting
    return (
        f"<|im_start|>system\n{ROUTER_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nClassify this vehicle query: {clean_q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_json_payload(raw_text: str) -> dict[str, Any] | None:
    """Extract and parse structured JSON from model text generation."""
    text = raw_text.strip()
    if not text:
        return None

    # Handle optional markdown code blocks ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Search for first outermost {...}
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


class QwenRouter:
    """Optional SLM intent router with local llama.cpp server and deterministic fallback."""

    def __init__(
        self,
        server_url: str | None = None,
        model_path: str | Path | None = None,
        model_type: str = "qwen3",
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        taxonomy: tuple[str, ...] = EXTENDED_INTENTS,
    ):
        self.server_url = (
            server_url
            or os.environ.get("VEHICLEGPT_LLAMA_SERVER_URL")
            or "http://127.0.0.1:8080/v1"
        ).rstrip("/")
        self.model_type = model_type.lower()
        if self.model_type.startswith("smollm"):
            default_path = (
                os.environ.get("VEHICLEGPT_SMOLLM2_MODEL_PATH")
                or DEFAULT_SMOLLM2_WINDOWS_PATH
            )
        else:
            default_path = (
                os.environ.get("VEHICLEGPT_QWEN_MODEL_PATH")
                or DEFAULT_QWEN_WINDOWS_PATH
            )
        self.model_path = model_path or default_path
        self.min_confidence = min_confidence
        self.timeout_s = timeout_s
        self.taxonomy = taxonomy

    def _call_llama_server(self, prompt: str) -> str:
        """Call local llama.cpp OpenAI-compatible completions endpoint."""
        url = f"{self.server_url}/chat/completions"
        payload = {
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 128,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
            res_json = json.loads(body)
            choices = res_json.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
            return ""

    def route(
        self,
        question: str,
        now_ns: int,
        tz_minutes: int = 0,
        raw_response_override: str | None = None,
    ) -> SLMRoutingResult:
        """Route user question using SLM with deterministic fallback on failure/abstention."""
        start_t = time.perf_counter()
        result = SLMRoutingResult(model=self.model_type)

        raw_text = ""
        fetch_error = None

        if raw_response_override is not None:
            raw_text = raw_response_override
        else:
            try:
                prompt = f"Classify this vehicle query: {question}"
                raw_text = self._call_llama_server(prompt)
            except (urllib.error.URLError, TimeoutError, OSError, Exception) as exc:
                fetch_error = str(exc)
                result.error = fetch_error
                result.reasons.append(f"server call failed: {exc}")

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        result.latency_ms = round(elapsed_ms, 2)
        result.raw_response = raw_text

        # Validate structured JSON
        parsed = extract_json_payload(raw_text)
        if not parsed:
            result.is_valid_json = False
            result.reasons.append("malformed or missing JSON response")
            self._invoke_fallback(result, question, now_ns, tz_minutes, "invalid_json")
            return result

        result.is_valid_json = True
        predicted_intent = str(parsed.get("intent", "")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        reasoning = str(parsed.get("reasoning", "")).strip()

        result.confidence = round(confidence, 3)
        result.reasoning = reasoning

        # Validate taxonomy
        if predicted_intent not in self.taxonomy:
            result.is_valid_taxonomy = False
            result.intent = "unknown"
            result.reasons.append(f"intent '{predicted_intent}' not in supported taxonomy")
            self._invoke_fallback(result, question, now_ns, tz_minutes, "unsupported_intent")
            return result

        result.is_valid_taxonomy = True
        result.intent = predicted_intent

        # Validate confidence threshold
        if result.confidence < self.min_confidence or result.intent == "unknown":
            result.reasons.append(
                f"confidence {result.confidence:.2f} below threshold {self.min_confidence:.2f}"
            )
            self._invoke_fallback(result, question, now_ns, tz_minutes, "low_confidence_or_unknown")
            return result

        return result

    def _invoke_fallback(
        self,
        result: SLMRoutingResult,
        question: str,
        now_ns: int,
        tz_minutes: int,
        reason: str,
    ) -> None:
        """Invoke deterministic fallback without modifying original deterministic logic."""
        result.fallback_invoked = True
        result.reasons.append(f"fallback triggered: {reason}")
        deterministic_res = parse_question(question, now_ns, tz_minutes)
        result.fallback_intent = deterministic_res

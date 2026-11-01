"""Intent and slots, by rule, with abstention as the default.

The existing ``queries.classify`` requires *all* of an intent's keywords to
appear as substrings. That works for the five brief questions and collapses on
ordinary paraphrase: "why is my range down" has no "reduc", "how far can I go
on a charge" has neither, and "efficiency fell" has no "drop". Each of those
silently became ``unknown``.

This module scores instead of gating. Every intent has terms in two weights and
the highest score wins, so one strong word is enough and several weak ones can
accumulate. Two rules keep it honest:

* **Abstain below a floor.** If the best score is under ``MIN_SCORE`` the answer
  is ``unknown``, not the least-bad guess. A routing error is worse than an
  escalation, because escalation is visible and a wrong intent is not.
* **Time is a slot, not a word.** "yesterday" says nothing about *what* is being
  asked, so time expressions are removed before scoring. Otherwise "what
  happened yesterday" would match whatever intent mentions time.

The intents here are the ones ``analytics/queries.py`` can answer. Extending the
set means extending the handlers; the DSL is the contract between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vehiclegpt.nlu import timeparse
from vehiclegpt.nlu.guard import Finding, sanitise

#: Score below which no intent is claimed.
MIN_SCORE = 3.0

#: (strong terms, weak terms) per intent. Strong terms are specific enough to
#: nominate the intent alone; weak terms only tip a tie.
TERMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "range_reduction": (
        ("range", "how far", "distance per charge", "miles per charge",
         "km per charge", "goes less far", "fewer km"),
        ("reduc", "drop", "fell", "declin", "shorter", "why", "less"),
    ),
    "efficiency_drop": (
        ("efficien", "consumption", "wh per km", "wh/km", "kwh per 100",
         "energy use", "burning more", "using more energy"),
        ("drop", "fell", "worst", "largest", "spike", "increase", "when"),
    ),
    "adas_events": (
        ("adas", "disengag", "lane keep", "lane assist", "autopilot",
         "driver assist", "takeover", "hands on wheel", "acc off"),
        ("event", "events", "when", "show", "list", "all"),
    ),
    "before_warning": (
        ("before", "leading up to", "prior to", "precede",
         "warning", "overheat", "fault", "what caused"),
        # "what happened" is an opener, not a subject: on its own it says
        # nothing about *what* to look for, so it cannot nominate the intent.
        ("battery", "temp", "temperature", "alert", "issue", "problem",
         "what happened"),
    ),
    "city_vs_highway": (
        ("city", "highway", "motorway", "urban", "suburban", "traffic"),
        ("compare", "versus", "vs", "between", "consumption", "energy",
         "efficiency", "difference", "worse", "better"),
    ),
}

#: Surface forms mapped to one canonical term. A driver does not say
#: "consumption"; they say the car is thirstier, or that it burns more power per
#: distance. Patching each phrasing into the term lists would work once and fail
#: on the next wording, so the vocabulary is expressed as equivalence classes
#: instead: many ways of saying it collapse to one token that is then scored.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "consumption": ("thirstier", "thirsty", "burning more", "burns more",
                    "power per distance", "energy per distance",
                    "power per km", "energy per km", "juice", "drained",
                    "using more power", "drinks more"),
    "efficien": ("mpge", "economy", "miles per kwh", "km per kwh"),
    "range": ("last as long", "lasts as long", "how many km", "how many miles",
              "kilometres per charge", "kilometers per charge", "kms per charge",
              "distance on a charge", "driving distance", "shrinking",
              "hold a charge", "run out"),
    "reduc": ("less", "lower", "degrad", "worse", "not as good", "much less",
              "down on", "lost"),
    "disengag": ("handover", "hand over", "hand back", "give up", "gave up",
                 "dropped out", "dropout", "take over", "took over",
                 "steering control", "took control"),
    "warning": ("alarm", "alert", "amber", "red light", "error", "fault",
                "malfunction", "hot", "overheating"),
    "before": ("leading to", "led to", "leading up", "prior", "preceded",
               "run-up", "run up", "sequence of events", "what was going on"),
    "city": ("town", "stop-start", "stop and go", "traffic", "urban", "commute"),
    "highway": ("motorway", "freeway", "open road", "cruising", "cruise",
                "interstate", "high speed"),
    "compare": ("versus", "vs", "difference", "side by side", "against"),
}

#: Words that describe *when*, not *what*. Removed before scoring so that a
#: temporal phrase cannot decide the intent.
TIME_NOISE = re.compile(
    r"\b(today|yesterday|tonight|last|past|previous|recent|week|weeks|month|months|"
    r"day|days|hour|hours|minute|minutes|morning|afternoon|evening|night|"
    r"between|since|from|during|over|ago|this)\b")

#: Phrases that mark a question the deterministic layer must not answer.
OUT_OF_SCOPE = re.compile(
    r"\b(weather|stock|news|joke|recipe|who are you|what model|write me|"
    r"translate|summarise this article|play|order|book)\b")


@dataclass
class Intent:
    """A routing decision: which handler, with which slots, at what confidence."""

    name: str = "unknown"
    score: float = 0.0
    window: timeparse.Window | None = None
    slots: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.name == "unknown"

    @property
    def blocked(self) -> bool:
        return any(f.severity == "block" for f in self.findings)

    def to_dict(self) -> dict:
        return {"intent": self.name, "score": round(self.score, 2),
                "window": self.window.to_dict() if self.window else None,
                "slots": self.slots, "abstained": self.abstained,
                "blocked": self.blocked,
                "findings": [f.to_dict() for f in self.findings],
                "reasons": self.reasons}


def canonicalise(text: str) -> str:
    """Rewrite surface forms to their canonical term.

    Longest form first, so "last as long" is not eaten by "last" before the
    time-noise pass, and phrase matching does not restart inside a replacement.
    """
    pairs = sorted(((form, canonical) for canonical, forms in SYNONYMS.items()
                    for form in forms), key=lambda pair: -len(pair[0]))
    for form, canonical in pairs:
        if form in text:
            text = re.sub(rf"(?<![a-z]){re.escape(form)}(?![a-z])", canonical, text)
    return text


def _score(text: str, intent: str) -> tuple[float, list[str]]:
    strong, weak = TERMS[intent]
    score, hits = 0.0, []
    for term in strong:
        if term in text:
            score += 3.0
            hits.append(f"+3 {term!r}")
    for term in weak:
        if re.search(rf"\b{re.escape(term)}", text):
            score += 1.0
            hits.append(f"+1 {term!r}")
    return score, hits


def _distance_slot(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilomet)", text)
    return float(match.group(1)) if match else None


def _lookback_slot(text: str) -> float | None:
    match = re.search(r"(\d+)\s*(?:min|minute)", text)
    return float(match.group(1)) if match else None


def parse_question(question: str, now_ns: int, tz_minutes: int = 0) -> Intent:
    """Route one question. Never guesses: an unclear question abstains."""
    clean, findings = sanitise(question or "")
    intent = Intent(findings=findings)
    if not clean:
        intent.reasons.append("empty after sanitising")
        return intent

    lowered = " ".join(clean.lower().split())
    if OUT_OF_SCOPE.search(lowered):
        intent.reasons.append("out of scope for vehicle data")
        return intent

    intent.window = timeparse.parse(lowered, now_ns, tz_minutes)

    # Score on a copy with time expressions and their numbers removed, so that
    # "the last 2 weeks" cannot contribute the token "2" or the word "last".
    # Canonicalise *before* dropping time words: "not last as long" is a range
    # complaint, and stripping "last" first would leave "not as long" behind.
    scoring_text = canonicalise(lowered)
    scoring_text = TIME_NOISE.sub(" ", scoring_text)
    scoring_text = re.sub(r"\d+", " ", scoring_text)

    best_name, best_score, best_hits = "unknown", 0.0, []
    for name in TERMS:
        score, hits = _score(scoring_text, name)
        if score > best_score:
            best_name, best_score, best_hits = name, score, hits

    if best_score < MIN_SCORE:
        intent.reasons.append(
            f"best score {best_score:.1f} under floor {MIN_SCORE:.1f}")
        return intent

    # A question can carry two intents; the loser is reported rather than hidden.
    others = sorted(((s, n) for n in TERMS if n != best_name
                     for s, _ in [_score(scoring_text, n)]), reverse=True)
    if others and others[0][0] >= best_score - 1.0:
        intent.reasons.append(
            f"{others[0][1]!r} scored {others[0][0]:.1f}, close to "
            f"{best_name!r} at {best_score:.1f}")

    intent.name, intent.score = best_name, best_score
    intent.reasons.extend(best_hits)

    # Slots are read from the original text: scoring_text has had its digits
    # removed so that numbers cannot influence the score.
    distance = _distance_slot(lowered)
    if distance is not None and best_name == "efficiency_drop":
        intent.slots["min_km"] = distance
    lookback = _lookback_slot(lowered)
    if lookback is not None and best_name == "before_warning":
        intent.slots["lookback_min"] = lookback
    if best_name == "city_vs_highway":
        intent.slots["regimes"] = [r for r in ("city", "highway")
                                   if r in scoring_text]
    return intent

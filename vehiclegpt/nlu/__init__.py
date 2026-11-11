"""The language layer: what turns a sentence into a query, and a query into prose.

Nothing in this package generates a number. Text is assembled from facts that
were computed by ``vehiclegpt.analytics`` and registered in an evidence pack;
a verifier then re-reads the finished sentence and fails if any number in it
cannot be traced to a registered fact. That constraint is the reason the pack
exists, and it is checked on every answer rather than sampled.
"""

from vehiclegpt.nlu.qwen_router import (
    EXTENDED_INTENTS,
    SUPPORTED_INTENTS,
    QwenRouter,
    SLMRoutingResult,
)

__all__ = [
    "QwenRouter",
    "SLMRoutingResult",
    "SUPPORTED_INTENTS",
    "EXTENDED_INTENTS",
]

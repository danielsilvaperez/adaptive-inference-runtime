"""
AIR Routing Module

Implements dynamic small-to-large model routing based on query complexity,
confidence scores, and semantic analysis. The router decides when to escalate
from a lightweight model to a more capable one.

Key Components:
    - ConfidenceScorers: Token entropy, top-k disagreement, attention instability
    - LogprobSlopeTracker: Tracks log probability slopes for confidence estimation
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from air.routing.confidence import (
        AttentionInstabilityScorer,
        EntropyScorer,
        TopKDisagreementScorer,
    )
    from air.routing.logprob_slope import LogprobSlopeTracker

__all__ = [
    "EntropyScorer",
    "LogprobSlopeTracker",
    "TopKDisagreementScorer",
    "AttentionInstabilityScorer",
]


def __getattr__(name: str):
    """Lazy import mechanism for routing components."""
    if name in ("EntropyScorer", "TopKDisagreementScorer", "AttentionInstabilityScorer"):
        from air.routing import confidence
        return getattr(confidence, name)
    elif name == "LogprobSlopeTracker":
        from air.routing.logprob_slope import LogprobSlopeTracker
        return LogprobSlopeTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

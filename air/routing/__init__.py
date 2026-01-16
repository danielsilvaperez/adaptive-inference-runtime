"""
AIR Routing Module

Implements dynamic small-to-large model routing based on query complexity,
confidence scores, and semantic analysis. The router decides when to escalate
from a lightweight model to a more capable one.

Key Components:
    - AdaptiveRouter: Main routing engine combining multiple confidence scorers
    - EntropyScorer: Token entropy-based confidence metric
    - TopKDisagreementScorer: Top-k disagreement confidence metric
    - LogprobSlopeTracker: Tracks log probability slopes for confidence estimation
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from air.routing.confidence import EntropyScorer, TopKDisagreementScorer
    from air.routing.logprob_slope import LogprobSlopeTracker
    from air.routing.router import AdaptiveRouter

__all__ = [
    "AdaptiveRouter",
    "EntropyScorer",
    "TopKDisagreementScorer",
    "LogprobSlopeTracker",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for routing components."""
    if name == "AdaptiveRouter":
        from air.routing.router import AdaptiveRouter

        return AdaptiveRouter
    elif name in ("EntropyScorer", "TopKDisagreementScorer"):
        from air.routing import confidence

        return getattr(confidence, name)
    elif name == "LogprobSlopeTracker":
        from air.routing.logprob_slope import LogprobSlopeTracker

        return LogprobSlopeTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

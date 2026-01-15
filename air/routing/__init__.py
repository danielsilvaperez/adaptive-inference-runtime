"""
AIR Routing Module

Implements dynamic small-to-large model routing based on query complexity,
confidence scores, and semantic analysis. The router decides when to escalate
from a lightweight model to a more capable one.

Key Components:
    - Router: Base router interface and implementations
    - ConfidenceEstimator: Estimates model confidence for routing decisions
    - ComplexityAnalyzer: Analyzes query complexity
    - LogprobSlopeTracker: Tracks log probability slopes for confidence estimation
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from air.routing.router import Router, BaseRouter
    from air.routing.confidence import ConfidenceEstimator
    from air.routing.complexity import ComplexityAnalyzer
    from air.routing.logprob_slope import LogprobSlopeTracker

__all__ = [
    "Router",
    "BaseRouter",
    "ConfidenceEstimator",
    "ComplexityAnalyzer",
    "LogprobSlopeTracker",
]


def __getattr__(name: str):
    """Lazy import mechanism for routing components."""
    if name in ("Router", "BaseRouter"):
        from air.routing import router
        return getattr(router, name)
    elif name == "ConfidenceEstimator":
        from air.routing import confidence
        return confidence.ConfidenceEstimator
    elif name == "ComplexityAnalyzer":
        from air.routing import complexity
        return complexity.ComplexityAnalyzer
    elif name == "LogprobSlopeTracker":
        from air.routing.logprob_slope import LogprobSlopeTracker
        return LogprobSlopeTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

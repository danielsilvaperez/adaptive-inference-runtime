"""
Confidence scoring implementations for the Adaptive Inference Runtime.

This module provides confidence scorers that analyze model outputs to
estimate routing confidence.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from air.routing.confidence.token_entropy import EntropyScorer

__all__ = [
    "EntropyScorer",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for confidence scorers."""
    if name == "EntropyScorer":
        from air.routing.confidence.token_entropy import EntropyScorer

        return EntropyScorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

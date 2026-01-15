"""
Interface definitions for the Adaptive Inference Runtime.

This module exports all protocol/interface definitions used throughout AIR.
These interfaces define the contracts that implementations must follow:

- Router & ConfidenceScorer: Routing decisions and confidence scoring
- ModelAdapter: Unified model inference interface
- KVCompressor: KV cache compression strategies

All interfaces are defined as Protocols for structural subtyping (duck typing),
with optional ABC base classes for implementations preferring inheritance.

Example:
    >>> from air.interfaces import Router, ModelAdapter, KVCompressor
    >>> from air.interfaces import BaseRouter, BaseModelAdapter
    >>>
    >>> # Check if an object implements an interface
    >>> isinstance(my_router, Router)  # True if it has the right methods
    >>>
    >>> # Create an implementation using ABC
    >>> class MyRouter(BaseRouter):
    ...     def route(self, state): ...
    ...     def should_escalate(self, state): ...
"""

from air.interfaces.adapter import (
    BaseModelAdapter,
    ModelAdapter,
)
from air.interfaces.compressor import (
    BaseKVCompressor,
    CompressionResult,
    KVCompressor,
)
from air.interfaces.router import (
    BaseConfidenceScorer,
    BaseRouter,
    ConfidenceScorer,
    Router,
)

__all__ = [
    # Router interfaces
    "Router",
    "ConfidenceScorer",
    "BaseRouter",
    "BaseConfidenceScorer",
    # Adapter interfaces
    "ModelAdapter",
    "BaseModelAdapter",
    # Compressor interfaces
    "KVCompressor",
    "BaseKVCompressor",
    "CompressionResult",
]

"""
Adaptive Inference Runtime (AIR) - Core Package.

AIR is a drop-in inference runtime that makes large-model intelligence usable
everywhere by combining:
- Small to Large model routing
- Speculative decoding
- KV-cache compression
- Optional quantization awareness

Example:
    >>> from air import InferenceState, GenerationConfig
    >>> from air.interfaces import Router, ModelAdapter
    >>>
    >>> config = GenerationConfig(max_tokens=512, temperature=0.7)
    >>> state = InferenceState(model_id="llama-7b", generation_config=config)
"""

from typing import Any

__version__ = "0.1.0"
__author__ = "AIR Contributors"
__license__ = "MIT"


# Lazy imports to avoid circular dependencies and improve startup time
def __getattr__(name: str) -> Any:
    """Lazy import mechanism for package components."""
    if name in (
        "Token",
        "Logits",
        "KVCache",
        "ModelSelection",
        "RoutingThresholds",
        "GenerationConfig",
        "CompressionConfig",
    ):
        from air import types

        return getattr(types, name)
    if name == "InferenceState":
        from air import state

        return state.InferenceState
    if name in ("get_logger", "setup_logging"):
        from air.utils import logging as air_logging

        return getattr(air_logging, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Types
    "Token",
    "Logits",
    "KVCache",
    "ModelSelection",
    "RoutingThresholds",
    "GenerationConfig",
    "CompressionConfig",
    # State
    "InferenceState",
    # Logging
    "get_logger",
    "setup_logging",
]

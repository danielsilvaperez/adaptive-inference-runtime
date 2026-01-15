"""
AIR Model Adapters Module

Provides adapters for integrating various LLM backends with AIR.
Adapters normalize the interface for different model implementations,
allowing AIR to work with HuggingFace, vLLM, and other frameworks.

Key Components:
    - ModelAdapter: Base adapter interface
    - HuggingFaceAdapter: Adapter for HuggingFace Transformers
    - VLLMAdapter: Adapter for vLLM backend
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from air.adapters.base import ModelAdapter
    from air.adapters.huggingface import HuggingFaceAdapter
    from air.adapters.vllm import VLLMAdapter

__all__ = [
    "ModelAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
]


def __getattr__(name: str):
    """Lazy import mechanism for adapter components."""
    if name == "ModelAdapter":
        from air.adapters import base
        return base.ModelAdapter
    elif name == "HuggingFaceAdapter":
        from air.adapters import huggingface
        return huggingface.HuggingFaceAdapter
    elif name == "VLLMAdapter":
        from air.adapters import vllm
        return vllm.VLLMAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

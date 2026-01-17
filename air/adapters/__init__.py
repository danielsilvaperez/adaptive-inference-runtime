"""
AIR Model Adapters Module

Provides adapters for integrating various LLM backends with AIR.
Adapters normalize the interface for different model implementations,
allowing AIR to work with HuggingFace, vLLM, and other frameworks.

Key Components:
    - ModelAdapter: Base adapter interface
    - HuggingFaceAdapter: Adapter for HuggingFace Transformers
    - LlamaCppAdapter: Adapter for llama-cpp-python backend
    - VLLMAdapter: Adapter for vLLM backend
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from air.adapters.base import ModelAdapter
    from air.adapters.huggingface import HuggingFaceAdapter
    from air.adapters.vllm import VLLMAdapter
    from air.adapters.llama_cpp import LlamaCppAdapter

__all__ = [
    "ModelAdapter",
    "HuggingFaceAdapter",
    "VLLMAdapter",
    "LlamaCppAdapter",
]


def __getattr__(name: str) -> Any:
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
    elif name == "LlamaCppAdapter":
        from air.adapters import llama_cpp

        return llama_cpp.LlamaCppAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

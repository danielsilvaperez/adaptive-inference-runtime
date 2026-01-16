"""
HuggingFace Transformers adapter for AIR.

This module provides an adapter for using HuggingFace Transformers models
with the Adaptive Inference Runtime. It wraps the HF Transformers API to
provide the standard AIR ModelAdapter interface.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from air.adapters.base import ModelAdapter

if TYPE_CHECKING:
    from air.types import GenerationConfig, Logits, Token


class HuggingFaceAdapter(ModelAdapter):
    """
    Adapter for HuggingFace Transformers models.

    This adapter wraps HuggingFace's transformers library to provide
    a unified interface compatible with AIR's routing and speculation
    systems.

    Example:
        >>> adapter = HuggingFaceAdapter(
        ...     model_id="meta-llama/Llama-2-7b-hf",
        ...     device="cuda:0"
        ... )
        >>> config = GenerationConfig(max_tokens=100, temperature=0.7)
        >>> for token in adapter.generate("Hello, world!", config):
        ...     print(token.text, end="")
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        """
        Initialize the HuggingFace adapter.

        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf").
            device: Device to load the model on ("cpu", "cuda", "cuda:0", etc.).
            torch_dtype: PyTorch dtype for model weights ("auto", "float16", "bfloat16").
            load_in_8bit: Whether to load model in 8-bit quantization.
            load_in_4bit: Whether to load model in 4-bit quantization.
        """
        super().__init__(model_id)
        self._device = device
        self._torch_dtype = torch_dtype
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit

        # TODO: Lazy load model and tokenizer when first used
        self._model = None
        self._tokenizer = None

    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[Token]:
        """
        Generate tokens from the HuggingFace model.

        Args:
            prompt: Input text prompt.
            config: Generation configuration.

        Yields:
            Generated tokens with logprobs.
        """
        # TODO: Implement actual generation using HF transformers
        raise NotImplementedError(
            "HuggingFaceAdapter.generate() is not yet implemented. "
            "This is a stub for Phase 1 development."
        )

    def get_logits(self, prompt: str) -> Logits:
        """
        Get logits for next token prediction.

        Args:
            prompt: Input prompt.

        Returns:
            Logits tensor for next token.
        """
        # TODO: Implement logits extraction
        raise NotImplementedError(
            "HuggingFaceAdapter.get_logits() is not yet implemented. "
            "This is a stub for Phase 1 development."
        )

    def verify_tokens(self, prompt: str, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """
        Verify draft tokens for speculative decoding.

        Args:
            prompt: Original prompt.
            draft_tokens: Draft tokens to verify.

        Returns:
            Accepted tokens and count.
        """
        # TODO: Implement verification logic
        raise NotImplementedError(
            "HuggingFaceAdapter.verify_tokens() is not yet implemented. "
            "This is a stub for Phase 2 (speculative decoding) development."
        )


__all__ = ["HuggingFaceAdapter"]

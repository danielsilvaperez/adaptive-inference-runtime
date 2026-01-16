"""
Base adapter interface for AIR model backends.

This module defines the base ModelAdapter protocol that all backend adapters
must implement. It provides a unified interface for different model frameworks
(HuggingFace, vLLM, llama.cpp, etc.).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from air.interfaces.adapter import ModelAdapter as IModelAdapter

if TYPE_CHECKING:
    from air.types import GenerationConfig, Logits, Token


class ModelAdapter(IModelAdapter):
    """
    Base adapter for model backends.

    This is a concrete base class that provides common functionality
    for all model adapters. Specific adapters (HuggingFace, vLLM)
    should inherit from this class and override the required methods.
    """

    def __init__(self, model_id: str) -> None:
        """
        Initialize the adapter with a model identifier.

        Args:
            model_id: Identifier for the model (e.g., "llama-7b").
        """
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        return self._model_id

    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[Token]:
        """
        Generate tokens from the model.

        Args:
            prompt: Input text prompt.
            config: Generation configuration parameters.

        Yields:
            Generated tokens one at a time.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def get_logits(self, prompt: str) -> Logits:
        """
        Get logits for the next token given a prompt.

        Args:
            prompt: Input text prompt.

        Returns:
            Logits tensor for the next token prediction.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_logits()")

    def verify_tokens(self, prompt: str, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """
        Verify draft tokens against the model's predictions.

        Used in speculative decoding to check which draft tokens
        the model would have generated.

        Args:
            prompt: Input prompt that produced the draft tokens.
            draft_tokens: Draft tokens to verify.

        Returns:
            Tuple of (accepted_tokens, num_accepted) where accepted_tokens
            are the draft tokens that match model predictions and num_accepted
            is the count of accepted tokens.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement verify_tokens()")

    def __repr__(self) -> str:
        """Return string representation of the adapter."""
        return f"{self.__class__.__name__}(model_id={self._model_id!r})"


__all__ = ["ModelAdapter"]

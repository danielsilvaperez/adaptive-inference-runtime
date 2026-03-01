"""
Model adapter interface definitions for the Adaptive Inference Runtime.

This module defines the ModelAdapter protocol which provides a unified
interface for interacting with different inference backends (llama.cpp, vLLM).

The adapter abstracts away backend-specific details, allowing the runtime
to work with any compatible model implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from air.types import GenerationConfig, KVCache, Logits, Token


@runtime_checkable
class ModelAdapter(Protocol):
    """
    Protocol for model inference adapters.

    ModelAdapter provides a unified interface for model inference operations,
    abstracting the differences between backends like llama.cpp and vLLM.
    This allows the routing and speculation systems to work with any
    compatible model implementation.

    Key capabilities:
    - Token generation (streaming and batch)
    - Logits extraction for confidence scoring
    - Speculative verification
    - KV cache access

    Example:
        >>> class LlamaCppAdapter:
        ...     def __init__(self, model_path: str):
        ...         self._model = Llama(model_path)
        ...         self._model_id = "llama-7b"
        ...
        ...     @property
        ...     def model_id(self) -> str:
        ...         return self._model_id
        ...
        ...     def generate(self, prompt: str, config: GenerationConfig) -> Iterator[Token]:
        ...         for output in self._model.generate(prompt, **config.to_dict()):
        ...             yield Token(id=output.id, text=output.text, logprob=output.logprob)
    """

    @property
    def model_id(self) -> str:
        """
        Get the unique identifier for this model.

        The model ID is used for routing decisions and logging. It should
        be a human-readable string that identifies the model and its
        configuration (e.g., "llama-7b-q4", "llama-70b-fp16").

        Returns:
            A unique string identifier for the model.
        """
        ...

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size of the model.

        Returns:
            The number of tokens in the model's vocabulary.
        """
        ...

    @property
    def context_length(self) -> int:
        """
        Get the maximum context length supported by the model.

        Returns:
            The maximum number of tokens the model can process.
        """
        ...

    @property
    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded and ready for inference.

        Returns:
            True if the model is loaded, False otherwise.
        """
        ...

    def load_model(self, path: str) -> None:
        """
        Load a model from the specified path.

        This method initializes the model and prepares it for inference.
        It should handle all backend-specific loading logic.

        Args:
            path: Path to the model file or directory. The format depends
                on the backend (e.g., GGUF file for llama.cpp, HuggingFace
                model directory for vLLM).

        Raises:
            FileNotFoundError: If the model path doesn't exist.
            ValueError: If the model format is invalid or unsupported.
            RuntimeError: If loading fails for backend-specific reasons.
        """
        ...

    def unload_model(self) -> None:
        """
        Unload the current model and free resources.

        This method releases GPU/memory resources held by the model.
        After calling this, is_loaded should return False.
        """
        ...

    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[Token]:
        """
        Generate tokens from a prompt.

        This is the main generation method, producing a stream of tokens
        until a stopping condition is met (max_tokens, EOS, stop sequence).

        Args:
            prompt: The input prompt text.
            config: Generation configuration parameters.

        Yields:
            Token objects as they are generated.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If the prompt is empty or invalid.

        Example:
            >>> adapter = LlamaCppAdapter()
            >>> adapter.load_model("path/to/model.gguf")
            >>> config = GenerationConfig(max_tokens=100, temperature=0.7)
            >>> for token in adapter.generate("Hello, world!", config):
            ...     print(token.text, end="", flush=True)
        """
        ...

    def generate_batch(self, prompts: list[str], config: GenerationConfig) -> list[list[Token]]:
        """
        Generate tokens for multiple prompts in a batch.

        This method is more efficient than calling generate() multiple
        times when processing multiple prompts, as it can leverage
        batch parallelism in the backend.

        Args:
            prompts: List of input prompts.
            config: Generation configuration (applied to all prompts).

        Returns:
            List of token lists, one per input prompt.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If prompts list is empty.
        """
        ...

    def get_logits(self, tokens: list[int]) -> Logits:
        """
        Get the logits for a sequence of token IDs.

        This method performs a forward pass and returns the output logits,
        which are used for confidence scoring and routing decisions.

        Args:
            tokens: List of token IDs to process.

        Returns:
            Logits tensor of shape (seq_len, vocab_size) or
            (1, seq_len, vocab_size) depending on the backend.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If tokens list is empty.
        """
        ...

    def verify(self, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """
        Verify a sequence of draft tokens for speculative decoding.

        In speculative decoding, a small model generates draft tokens
        which are then verified by a larger model. This method performs
        the verification step, determining how many draft tokens can
        be accepted.

        Args:
            draft_tokens: List of tokens generated by the draft model.

        Returns:
            A tuple of (accepted_tokens, accepted_count) where:
            - accepted_tokens: The tokens that were accepted (may include
              a correction token at the end if the last draft was rejected)
            - accepted_count: Number of draft tokens that matched

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If draft_tokens is empty.

        Example:
            >>> # Draft model generates 5 tokens
            >>> draft_tokens = small_model.generate(prompt, config)[:5]
            >>> # Large model verifies
            >>> accepted, count = large_model.verify(draft_tokens)
            >>> # If count == 5, all drafts accepted
            >>> # If count < 5, draft[count] was rejected and replaced
        """
        ...

    def get_kv_cache(self) -> KVCache:
        """
        Get the current KV cache from the model.

        The KV cache stores key-value pairs from attention layers,
        enabling efficient autoregressive generation.

        Returns:
            The current KV cache object.

        Raises:
            RuntimeError: If the model is not loaded or doesn't support
                KV cache access.
        """
        ...

    def set_kv_cache(self, cache: KVCache) -> None:
        """
        Set the KV cache for the model.

        This method is used to restore a previous KV cache state,
        for example when transitioning between models during routing.

        Args:
            cache: The KV cache to set.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If the cache is incompatible with the model.
        """
        ...

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text into token IDs.

        Args:
            text: The text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            RuntimeError: If the model/tokenizer is not loaded.
        """
        ...

    def detokenize(self, tokens: list[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            The decoded text.

        Raises:
            RuntimeError: If the model/tokenizer is not loaded.
        """
        ...


class BaseModelAdapter(ABC):
    """
    Abstract base class for ModelAdapter implementations.

    Provides a concrete base with common functionality. Subclasses must
    implement the abstract methods for backend-specific operations.

    This class is provided as an alternative to the ModelAdapter protocol
    for implementations that prefer inheritance over duck typing.

    Attributes:
        _model_id: The model identifier.
        _is_loaded: Whether the model is currently loaded.

    Example:
        >>> class LlamaCppAdapter(BaseModelAdapter):
        ...     def __init__(self, model_id: str = "llama-cpp"):
        ...         super().__init__(model_id)
        ...         self._model = None
        ...
        ...     def load_model(self, path: str) -> None:
        ...         self._model = Llama(path)
        ...         self._is_loaded = True
    """

    def __init__(self, model_id: str) -> None:
        """
        Initialize the base adapter.

        Args:
            model_id: Unique identifier for this model.
        """
        self._model_id: str = model_id
        self._is_loaded: bool = False
        self._vocab_size: int = 0
        self._context_length: int = 0

    @property
    def model_id(self) -> str:
        """Get the model identifier."""
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size

    @property
    def context_length(self) -> int:
        """Get the maximum context length."""
        return self._context_length

    def _ensure_loaded(self) -> None:
        """
        Ensure the model is loaded.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError(f"Model '{self._model_id}' is not loaded. Call load_model() first.")

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a model from the specified path."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the current model."""
        ...

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[Token]:
        """Generate tokens from a prompt."""
        ...

    def generate_batch(self, prompts: list[str], config: GenerationConfig) -> list[list[Token]]:
        """
        Generate tokens for multiple prompts.

        Default implementation processes prompts sequentially.
        Subclasses should override for true batch processing.
        """
        return [list(self.generate(prompt, config)) for prompt in prompts]

    @abstractmethod
    def get_logits(self, tokens: list[int]) -> Logits:
        """Get logits for a token sequence."""
        ...

    @abstractmethod
    def verify(self, draft_tokens: list[Token]) -> tuple[list[Token], int]:
        """Verify draft tokens for speculative decoding."""
        ...

    @abstractmethod
    def get_kv_cache(self) -> KVCache:
        """Get the current KV cache."""
        ...

    @abstractmethod
    def set_kv_cache(self, cache: KVCache) -> None:
        """Set the KV cache."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Tokenize text into token IDs."""
        ...

    @abstractmethod
    def detokenize(self, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
        ...

    def __repr__(self) -> str:
        """Get string representation."""
        status = "loaded" if self._is_loaded else "not loaded"
        return f"{self.__class__.__name__}(model_id='{self._model_id}', status={status})"

"""
Core type definitions for the Adaptive Inference Runtime.

This module defines the foundational types and data structures used throughout AIR:
- Token: Represents a single generated token with metadata
- Logits: Type alias for model output logits
- KVCache: Protocol for key-value cache access
- ModelSelection: Result of routing decisions
- RoutingThresholds: Configurable thresholds for routing
- GenerationConfig: Parameters for text generation
- CompressionConfig: Settings for KV cache compression

All types use proper Python type hints and are compatible with mypy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import torch


# =============================================================================
# Token Types
# =============================================================================


class Token(NamedTuple):
    """
    Represents a single token in the generation sequence.

    A token contains its vocabulary ID, text representation, and the log
    probability assigned by the model. This is the fundamental unit of
    text generation in AIR.

    Attributes:
        id: The integer token ID from the model's vocabulary.
        text: The string representation of the token (decoded text).
        logprob: The log probability of this token being selected.
            Higher values (closer to 0) indicate higher confidence.
            Typical range is (-inf, 0].

    Example:
        >>> token = Token(id=1234, text="Hello", logprob=-0.5)
        >>> token.id
        1234
        >>> token.text
        'Hello'
        >>> token.logprob
        -0.5
    """

    id: int
    text: str
    logprob: float


# =============================================================================
# Tensor Type Aliases
# =============================================================================


# Type alias for logits tensor from model output
# Shape: (batch_size, sequence_length, vocab_size) or (batch_size, vocab_size)
# Note: We use string annotation to avoid importing torch at module level
Logits: TypeAlias = "torch.Tensor"


# =============================================================================
# KV Cache Protocol
# =============================================================================


@runtime_checkable
class KVCache(Protocol):
    """
    Protocol defining the interface for key-value cache access.

    The KV cache stores the key and value tensors from attention layers,
    enabling efficient autoregressive generation by avoiding recomputation
    of previous tokens' representations.

    Implementations must provide access to the cache size, layer count,
    and the actual key-value tensors. This protocol is designed to be
    backend-agnostic, supporting both llama.cpp and vLLM implementations.

    Example:
        >>> class MyKVCache:
        ...     @property
        ...     def size(self) -> int:
        ...         return self._current_tokens
        ...
        ...     @property
        ...     def num_layers(self) -> int:
        ...         return 32
        ...
        ...     def get_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        ...         return self._keys[layer], self._values[layer]
        ...
        ...     def set_kv(self, layer: int, keys: torch.Tensor, values: torch.Tensor):
        ...         self._keys[layer] = keys
        ...         self._values[layer] = values
    """

    @property
    def size(self) -> int:
        """
        Return the current number of tokens in the cache.

        Returns:
            The number of token positions currently stored in the cache.
        """
        ...

    @property
    def num_layers(self) -> int:
        """
        Return the number of transformer layers in the cache.

        Returns:
            The number of attention layers whose KV pairs are cached.
        """
        ...

    @property
    def max_size(self) -> int:
        """
        Return the maximum capacity of the cache in tokens.

        Returns:
            The maximum number of tokens the cache can hold before
            requiring eviction or extension.
        """
        ...

    def get_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the key and value tensors for a specific layer.

        Args:
            layer: The layer index (0-indexed).

        Returns:
            A tuple of (keys, values) tensors for the specified layer.
            Keys shape: (batch_size, num_heads, seq_len, head_dim)
            Values shape: (batch_size, num_heads, seq_len, head_dim)

        Raises:
            IndexError: If layer is out of bounds.
        """
        ...

    def set_kv(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Set the key and value tensors for a specific layer.

        Args:
            layer: The layer index (0-indexed).
            keys: The key tensor to store.
            values: The value tensor to store.

        Raises:
            IndexError: If layer is out of bounds.
            ValueError: If tensor shapes are incompatible.
        """
        ...

    def clear(self) -> None:
        """
        Clear all cached key-value pairs.

        This resets the cache to its initial empty state while
        maintaining the allocated memory buffers.
        """
        ...

    def clone(self) -> KVCache:
        """
        Create a deep copy of the cache.

        Returns:
            A new KVCache instance with copied tensors.
        """
        ...


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class ModelSelection:
    """
    Result of a routing decision indicating which model should be used.

    The router produces a ModelSelection to indicate which model should
    handle the current generation step. This includes a confidence score
    and human-readable reason for debugging and logging purposes.

    Attributes:
        model_id: Identifier of the selected model (e.g., "llama-7b", "llama-70b").
        confidence_score: Confidence in the selection, range [0.0, 1.0].
            Higher values indicate higher certainty in the routing decision.
        reason: Human-readable explanation of why this model was selected.
            Useful for debugging and understanding routing behavior.

    Example:
        >>> selection = ModelSelection(
        ...     model_id="llama-70b",
        ...     confidence_score=0.85,
        ...     reason="High entropy detected in token distribution"
        ... )
        >>> selection.model_id
        'llama-70b'
    """

    model_id: str
    confidence_score: float
    reason: str

    def __post_init__(self) -> None:
        """Validate the confidence score is within valid range."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"confidence_score must be in [0.0, 1.0], got {self.confidence_score}")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with model_id, confidence_score, and reason.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSelection:
        """
        Create a ModelSelection from a dictionary.

        Args:
            data: Dictionary containing model_id, confidence_score, and reason.

        Returns:
            A new ModelSelection instance.
        """
        return cls(
            model_id=data["model_id"],
            confidence_score=data["confidence_score"],
            reason=data["reason"],
        )


@dataclass
class RoutingThresholds:
    """
    Configurable thresholds for routing decisions.

    These thresholds control when the router decides to escalate from
    a small model to a large model based on various uncertainty signals.
    Lower thresholds result in more aggressive escalation.

    Attributes:
        entropy_threshold: Shannon entropy threshold for escalation.
            When token entropy exceeds this, escalation may be triggered.
            Typical range: [1.0, 5.0], default 2.5.
        logprob_slope_threshold: Threshold for confidence trajectory.
            Negative slopes below this threshold indicate declining confidence.
            Typical range: [-0.5, 0.0], default -0.2.
        top_k_disagreement_threshold: Threshold for top-k prediction consensus.
            Higher values mean more disagreement among top predictions.
            Range: [0.0, 1.0], default 0.5.
        attention_instability_threshold: Threshold for attention variance.
            Higher values indicate more unstable attention patterns.
            Typical range: [0.0, 1.0], default 0.3.
        min_confidence_for_small_model: Minimum confidence to stay on small model.
            If confidence drops below this, escalate to large model.
            Range: [0.0, 1.0], default 0.7.
        cooldown_tokens: Number of tokens to wait before re-evaluating routing.
            Prevents rapid model switching.
            Default: 5.

    Example:
        >>> thresholds = RoutingThresholds(
        ...     entropy_threshold=3.0,
        ...     min_confidence_for_small_model=0.8
        ... )
        >>> thresholds.entropy_threshold
        3.0
    """

    entropy_threshold: float = 2.5
    logprob_slope_threshold: float = -0.2
    top_k_disagreement_threshold: float = 0.5
    attention_instability_threshold: float = 0.3
    min_confidence_for_small_model: float = 0.7
    cooldown_tokens: int = 5

    def __post_init__(self) -> None:
        """Validate threshold values."""
        if self.entropy_threshold < 0:
            raise ValueError("entropy_threshold must be non-negative")
        if not -1.0 <= self.logprob_slope_threshold <= 0.0:
            raise ValueError("logprob_slope_threshold must be in [-1.0, 0.0]")
        if not 0.0 <= self.top_k_disagreement_threshold <= 1.0:
            raise ValueError("top_k_disagreement_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.attention_instability_threshold <= 1.0:
            raise ValueError("attention_instability_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.min_confidence_for_small_model <= 1.0:
            raise ValueError("min_confidence_for_small_model must be in [0.0, 1.0]")
        if self.cooldown_tokens < 0:
            raise ValueError("cooldown_tokens must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingThresholds:
        """Create from dictionary representation."""
        return cls(**data)

    @classmethod
    def conservative(cls) -> RoutingThresholds:
        """
        Create conservative thresholds (rarely escalate to large model).

        Use when minimizing large model usage is important.
        """
        return cls(
            entropy_threshold=4.0,
            logprob_slope_threshold=-0.4,
            top_k_disagreement_threshold=0.7,
            attention_instability_threshold=0.5,
            min_confidence_for_small_model=0.5,
            cooldown_tokens=10,
        )

    @classmethod
    def balanced(cls) -> RoutingThresholds:
        """
        Create balanced thresholds (default behavior).

        Provides a good balance between quality and efficiency.
        """
        return cls()  # Uses default values

    @classmethod
    def aggressive(cls) -> RoutingThresholds:
        """
        Create aggressive thresholds (escalate more often).

        Use when quality is more important than efficiency.
        """
        return cls(
            entropy_threshold=1.5,
            logprob_slope_threshold=-0.1,
            top_k_disagreement_threshold=0.3,
            attention_instability_threshold=0.2,
            min_confidence_for_small_model=0.85,
            cooldown_tokens=3,
        )


@dataclass
class GenerationConfig:
    """
    Configuration parameters for text generation.

    Controls the behavior of token sampling during generation, including
    temperature, top-k/top-p filtering, and length constraints.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
            Set to -1 for unlimited (until EOS or context length).
            Default: 512.
        temperature: Sampling temperature. Higher values increase randomness.
            Range: [0.0, 2.0], 0.0 means greedy/argmax sampling.
            Default: 1.0.
        top_k: Number of highest probability tokens to consider for sampling.
            Set to -1 to disable top-k filtering.
            Default: 50.
        top_p: Cumulative probability threshold for nucleus sampling.
            Range: [0.0, 1.0], 1.0 disables nucleus sampling.
            Default: 1.0.
        min_p: Minimum probability threshold relative to top token.
            Tokens with probability < min_p * max_prob are excluded.
            Range: [0.0, 1.0], 0.0 disables min-p filtering.
            Default: 0.0.
        repetition_penalty: Penalty for repeating tokens. Values > 1.0 discourage
            repetition, < 1.0 encourage it.
            Default: 1.0 (no penalty).
        presence_penalty: Additive penalty for tokens already in the sequence.
            Range: [-2.0, 2.0].
            Default: 0.0.
        frequency_penalty: Penalty based on token frequency in the sequence.
            Range: [-2.0, 2.0].
            Default: 0.0.
        stop_sequences: List of strings that will stop generation when produced.
            Default: empty list.
        seed: Random seed for reproducibility. None for non-deterministic.
            Default: None.

    Example:
        >>> config = GenerationConfig(
        ...     max_tokens=256,
        ...     temperature=0.7,
        ...     top_p=0.9
        ... )
        >>> config.temperature
        0.7
    """

    max_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate generation parameters."""
        if self.max_tokens < -1 or self.max_tokens == 0:
            raise ValueError("max_tokens must be -1 (unlimited) or positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError("top_k must be -1 (disabled) or positive")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0.0, 1.0]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0.0, 1.0]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2.0, 2.0]")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2.0, 2.0]")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationConfig:
        """Create from dictionary representation."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> GenerationConfig:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def greedy(cls, max_tokens: int = 512) -> GenerationConfig:
        """
        Create a greedy (deterministic) generation config.

        Uses argmax sampling with no randomness.
        """
        return cls(
            max_tokens=max_tokens,
            temperature=0.0,
            top_k=-1,
            top_p=1.0,
        )

    @classmethod
    def sampling(
        cls, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9
    ) -> GenerationConfig:
        """
        Create a standard sampling config with nucleus sampling.

        Good default for creative/conversational generation.
        """
        return cls(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=-1,
            top_p=top_p,
        )


@dataclass
class CompressionConfig:
    """
    Configuration for KV cache compression.

    Controls how the KV cache is compressed to reduce memory usage
    while maintaining generation quality. Supports multiple eviction
    strategies and optional quantization.

    Attributes:
        enabled: Whether compression is enabled.
            Default: True.
        target_ratio: Target compression ratio (compressed/original size).
            Range: (0.0, 1.0]. Lower values mean more compression.
            Default: 0.5 (50% of original size).
        eviction_policy: Policy for selecting tokens to evict.
            Options: "sliding_window", "heavy_hitter", "h2o", "lru".
            Default: "sliding_window".
        sliding_window_size: Size of sliding window for window-based eviction.
            Only used when eviction_policy is "sliding_window".
            Default: 512.
        heavy_hitter_ratio: Ratio of tokens to keep as heavy hitters.
            Only used when eviction_policy is "heavy_hitter" or "h2o".
            Range: [0.0, 1.0].
            Default: 0.1.
        quantize_kv: Whether to quantize KV cache to int8.
            Default: False.
        min_tokens_before_compression: Minimum tokens before starting compression.
            Compression only kicks in after this many tokens.
            Default: 256.
        per_layer_policy: Whether to apply different policies per layer.
            Default: False.
        protected_token_count: Number of recent tokens to never evict.
            Default: 32.

    Example:
        >>> config = CompressionConfig(
        ...     eviction_policy="heavy_hitter",
        ...     target_ratio=0.25,
        ...     heavy_hitter_ratio=0.2
        ... )
        >>> config.eviction_policy
        'heavy_hitter'
    """

    enabled: bool = True
    target_ratio: float = 0.5
    eviction_policy: str = "sliding_window"
    sliding_window_size: int = 512
    heavy_hitter_ratio: float = 0.1
    quantize_kv: bool = False
    min_tokens_before_compression: int = 256
    per_layer_policy: bool = False
    protected_token_count: int = 32

    _VALID_POLICIES = frozenset({"sliding_window", "heavy_hitter", "h2o", "lru"})

    def __post_init__(self) -> None:
        """Validate compression parameters."""
        if not 0.0 < self.target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0.0, 1.0]")
        if self.eviction_policy not in self._VALID_POLICIES:
            raise ValueError(
                f"eviction_policy must be one of {self._VALID_POLICIES}, "
                f"got '{self.eviction_policy}'"
            )
        if self.sliding_window_size < 1:
            raise ValueError("sliding_window_size must be positive")
        if not 0.0 <= self.heavy_hitter_ratio <= 1.0:
            raise ValueError("heavy_hitter_ratio must be in [0.0, 1.0]")
        if self.min_tokens_before_compression < 0:
            raise ValueError("min_tokens_before_compression must be non-negative")
        if self.protected_token_count < 0:
            raise ValueError("protected_token_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = asdict(self)
        # Remove private attributes
        result.pop("_VALID_POLICIES", None)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressionConfig:
        """Create from dictionary representation."""
        # Filter out any unknown keys
        valid_keys = {
            "enabled",
            "target_ratio",
            "eviction_policy",
            "sliding_window_size",
            "heavy_hitter_ratio",
            "quantize_kv",
            "min_tokens_before_compression",
            "per_layer_policy",
            "protected_token_count",
        }
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    @classmethod
    def disabled(cls) -> CompressionConfig:
        """Create a disabled compression config."""
        return cls(enabled=False)

    @classmethod
    def conservative(cls) -> CompressionConfig:
        """
        Create conservative compression settings.

        Minimal compression, high quality preservation.
        """
        return cls(
            enabled=True,
            target_ratio=0.75,
            eviction_policy="sliding_window",
            sliding_window_size=1024,
            protected_token_count=64,
        )

    @classmethod
    def balanced(cls) -> CompressionConfig:
        """
        Create balanced compression settings.

        Good tradeoff between memory and quality.
        """
        return cls()  # Uses default values

    @classmethod
    def aggressive(cls) -> CompressionConfig:
        """
        Create aggressive compression settings.

        Maximum memory savings, some quality impact possible.
        """
        return cls(
            enabled=True,
            target_ratio=0.25,
            eviction_policy="h2o",
            heavy_hitter_ratio=0.15,
            quantize_kv=True,
            min_tokens_before_compression=128,
            protected_token_count=16,
        )

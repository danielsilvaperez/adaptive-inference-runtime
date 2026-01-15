"""
KV cache compressor interface definitions for the Adaptive Inference Runtime.

This module defines the KVCompressor protocol which provides the interface
for compressing and managing KV caches to reduce memory usage while
maintaining generation quality.

Compression strategies include:
- Eviction (removing less important cached tokens)
- Quantization (reducing precision of cached values)
- Hybrid approaches
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Dict,
    Any,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from air.types import CompressionConfig, KVCache


@runtime_checkable
class KVCompressor(Protocol):
    """
    Protocol for KV cache compression.

    KVCompressor provides methods for compressing KV caches to reduce
    memory usage. This is essential for enabling long-context generation
    and running large models on memory-constrained devices.

    Key operations:
    - compress: Apply full compression to a cache
    - evict: Remove tokens to reach a target size
    - get_memory_usage: Measure current memory consumption

    Example:
        >>> class SlidingWindowCompressor:
        ...     def __init__(self, window_size: int = 512):
        ...         self.window_size = window_size
        ...
        ...     def compress(self, cache: KVCache) -> KVCache:
        ...         if cache.size <= self.window_size:
        ...             return cache
        ...         return self.evict(cache, self.window_size)
        ...
        ...     def evict(self, cache: KVCache, target_size: int) -> KVCache:
        ...         # Keep only the most recent target_size tokens
        ...         pass
        ...
        ...     def get_memory_usage(self, cache: KVCache) -> int:
        ...         return sum(t.numel() * t.element_size()
        ...                    for layer in range(cache.num_layers)
        ...                    for t in cache.get_kv(layer))
    """

    def compress(self, cache: "KVCache") -> "KVCache":
        """
        Compress the KV cache according to the configured policy.

        This method applies the full compression strategy, which may include
        eviction, quantization, or other techniques. The specific behavior
        depends on the implementation and configuration.

        Args:
            cache: The KV cache to compress.

        Returns:
            A new compressed KVCache, or the same cache if no compression
            was needed. The original cache should not be modified.

        Note:
            Implementations should ensure the returned cache is valid and
            usable for continued generation. Quality degradation should be
            minimized according to the configured compression level.
        """
        ...

    def evict(self, cache: "KVCache", target_size: int) -> "KVCache":
        """
        Evict tokens from the cache to reach a target size.

        This method removes cached tokens until the cache size is at or
        below the target. The eviction policy (which tokens to remove)
        depends on the implementation (e.g., oldest tokens, least
        attended tokens, etc.).

        Args:
            cache: The KV cache to evict from.
            target_size: The desired cache size in tokens. Must be positive
                and less than or equal to the current cache size.

        Returns:
            A new KVCache with size <= target_size. If the cache is
            already at or below the target size, it may be returned
            unchanged.

        Raises:
            ValueError: If target_size is invalid (negative or zero).

        Note:
            Some tokens may be protected from eviction (e.g., recent tokens,
            high-attention tokens). The actual size may be larger than
            target_size if protected tokens prevent reaching the target.
        """
        ...

    def get_memory_usage(self, cache: "KVCache") -> int:
        """
        Get the current memory usage of the cache in bytes.

        This method calculates the total memory consumed by the KV cache,
        including all layers and both key and value tensors.

        Args:
            cache: The KV cache to measure.

        Returns:
            Memory usage in bytes.

        Note:
            This should return the actual memory footprint, which may
            differ from the theoretical size due to memory alignment,
            allocator overhead, etc. For accurate measurements,
            implementations should query the underlying tensor storage.
        """
        ...


class BaseKVCompressor(ABC):
    """
    Abstract base class for KVCompressor implementations.

    Provides a concrete base with common functionality for KV cache
    compression. Subclasses must implement the abstract methods for
    the specific compression strategy.

    Attributes:
        config: The compression configuration.

    Example:
        >>> class H2OCompressor(BaseKVCompressor):
        ...     def __init__(self, config: CompressionConfig):
        ...         super().__init__(config)
        ...         self._attention_scores = {}
        ...
        ...     def compress(self, cache: KVCache) -> KVCache:
        ...         if not self._should_compress(cache):
        ...             return cache
        ...         target = int(cache.size * self.config.target_ratio)
        ...         return self.evict(cache, target)
    """

    def __init__(self, config: "CompressionConfig") -> None:
        """
        Initialize the compressor with configuration.

        Args:
            config: Compression configuration settings.
        """
        self._config: "CompressionConfig" = config

    @property
    def config(self) -> "CompressionConfig":
        """Get the compression configuration."""
        return self._config

    @config.setter
    def config(self, value: "CompressionConfig") -> None:
        """Set the compression configuration."""
        self._config = value

    @property
    def is_enabled(self) -> bool:
        """Check if compression is enabled."""
        return self._config.enabled

    def _should_compress(self, cache: "KVCache") -> bool:
        """
        Determine if compression should be applied.

        Args:
            cache: The cache to potentially compress.

        Returns:
            True if compression should be applied.
        """
        if not self.is_enabled:
            return False
        if cache.size < self._config.min_tokens_before_compression:
            return False
        target_size = int(cache.max_size * self._config.target_ratio)
        return cache.size > target_size

    def get_memory_usage(self, cache: "KVCache") -> int:
        """
        Get the memory usage of a cache.

        Default implementation estimates memory based on cache properties.
        Subclasses should override for accurate measurements.

        Args:
            cache: The cache to measure.

        Returns:
            Estimated memory usage in bytes.
        """
        # Estimate: assume float16 (2 bytes) for KV values
        # Shape: (batch, num_heads, seq_len, head_dim)
        # This is a rough estimate; subclasses should provide accurate values
        bytes_per_element = 2  # float16
        # Assuming typical dimensions
        num_heads = 32  # typical for 7B model
        head_dim = 128  # typical dimension

        kv_size_per_layer = (
            2  # key and value
            * cache.size  # sequence length
            * num_heads
            * head_dim
            * bytes_per_element
        )

        return kv_size_per_layer * cache.num_layers

    def get_compression_stats(self, original: "KVCache", compressed: "KVCache") -> Dict[str, Any]:
        """
        Get statistics about a compression operation.

        Args:
            original: The original cache before compression.
            compressed: The compressed cache.

        Returns:
            Dictionary with compression statistics.
        """
        original_size = self.get_memory_usage(original)
        compressed_size = self.get_memory_usage(compressed)

        return {
            "original_tokens": original.size,
            "compressed_tokens": compressed.size,
            "tokens_evicted": original.size - compressed.size,
            "original_memory_bytes": original_size,
            "compressed_memory_bytes": compressed_size,
            "memory_saved_bytes": original_size - compressed_size,
            "compression_ratio": compressed_size / original_size if original_size > 0 else 1.0,
            "eviction_policy": self._config.eviction_policy,
        }

    @abstractmethod
    def compress(self, cache: "KVCache") -> "KVCache":
        """
        Compress the KV cache.

        Args:
            cache: The cache to compress.

        Returns:
            The compressed cache.
        """
        ...

    @abstractmethod
    def evict(self, cache: "KVCache", target_size: int) -> "KVCache":
        """
        Evict tokens to reach target size.

        Args:
            cache: The cache to evict from.
            target_size: The target cache size.

        Returns:
            The evicted cache.
        """
        ...

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"{self.__class__.__name__}("
            f"policy='{self._config.eviction_policy}', "
            f"target_ratio={self._config.target_ratio}, "
            f"enabled={self.is_enabled})"
        )


class CompressionResult:
    """
    Result of a compression operation.

    Encapsulates the compressed cache along with metadata about
    the compression operation for logging and analysis.

    Attributes:
        cache: The compressed KV cache.
        original_size: Original cache size in tokens.
        compressed_size: Compressed cache size in tokens.
        memory_saved: Memory saved in bytes.
        tokens_evicted: Number of tokens evicted.
        policy_used: Name of the eviction policy used.
    """

    def __init__(
        self,
        cache: "KVCache",
        original_size: int,
        compressed_size: int,
        memory_saved: int,
        tokens_evicted: int,
        policy_used: str,
    ) -> None:
        """
        Initialize the compression result.

        Args:
            cache: The compressed cache.
            original_size: Original size in tokens.
            compressed_size: Compressed size in tokens.
            memory_saved: Bytes saved.
            tokens_evicted: Number of tokens removed.
            policy_used: Eviction policy name.
        """
        self.cache = cache
        self.original_size = original_size
        self.compressed_size = compressed_size
        self.memory_saved = memory_saved
        self.tokens_evicted = tokens_evicted
        self.policy_used = policy_used

    @property
    def compression_ratio(self) -> float:
        """Get the compression ratio (compressed/original)."""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size

    @property
    def eviction_ratio(self) -> float:
        """Get the ratio of tokens evicted."""
        if self.original_size == 0:
            return 0.0
        return self.tokens_evicted / self.original_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "memory_saved": self.memory_saved,
            "tokens_evicted": self.tokens_evicted,
            "eviction_ratio": self.eviction_ratio,
            "policy_used": self.policy_used,
        }

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"CompressionResult("
            f"ratio={self.compression_ratio:.2f}, "
            f"evicted={self.tokens_evicted}, "
            f"saved={self.memory_saved}B)"
        )

"""
Sliding Window KV Cache Compression.

Implements a simple but effective eviction policy that retains only the most
recent N tokens in the cache. This is useful for maintaining a fixed memory
footprint during long-sequence generation.

The sliding window policy:
- Keeps the last N tokens (configurable via sliding_window_size)
- Protects recent tokens from eviction (configurable via protected_token_count)
- Evicts older tokens when the cache exceeds the window size
- Minimal computational overhead (constant time eviction)

Example:
    >>> config = CompressionConfig(
    ...     eviction_policy="sliding_window",
    ...     sliding_window_size=512,
    ...     protected_token_count=32
    ... )
    >>> compressor = SlidingWindowCompressor(config)
    >>> compressed_cache = compressor.compress(cache)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from air.types import CompressionConfig, KVCache

from air.interfaces.compressor import BaseKVCompressor


class SlidingWindowCompressor(BaseKVCompressor):
    """
    Sliding window eviction policy for KV cache compression.

    This compressor maintains a fixed-size window of the most recent tokens,
    evicting older tokens as new ones are added. This ensures predictable
    memory usage while maintaining good performance for many generation tasks.

    The policy guarantees:
    - Recent tokens are never evicted (up to protected_token_count)
    - Memory usage is bounded by sliding_window_size
    - O(1) decision time per eviction operation

    Attributes:
        config: Compression configuration containing window size and settings.

    Example:
        >>> config = CompressionConfig(sliding_window_size=512)
        >>> compressor = SlidingWindowCompressor(config)
        >>> if cache.size > config.sliding_window_size:
        ...     compressed = compressor.compress(cache)
        ...     assert compressed.size <= config.sliding_window_size
    """

    def __init__(self, config: CompressionConfig) -> None:
        """
        Initialize the sliding window compressor.

        Args:
            config: Compression configuration. Must have eviction_policy
                set to "sliding_window" and a valid sliding_window_size.

        Raises:
            ValueError: If eviction_policy is not "sliding_window".
        """
        super().__init__(config)
        if config.eviction_policy != "sliding_window":
            raise ValueError(
                f"SlidingWindowCompressor requires eviction_policy='sliding_window', "
                f"got '{config.eviction_policy}'"
            )

    @property
    def window_size(self) -> int:
        """Get the configured sliding window size."""
        return self.config.sliding_window_size

    @property
    def protected_count(self) -> int:
        """Get the number of recent tokens protected from eviction."""
        return self.config.protected_token_count

    def compress(self, cache: KVCache) -> KVCache:
        """
        Compress the cache using sliding window eviction.

        Keeps only the most recent sliding_window_size tokens. If the cache
        is already within the window size, no compression is performed.

        Args:
            cache: The KV cache to compress.

        Returns:
            A compressed cache containing at most sliding_window_size tokens.
            If no compression is needed, returns the original cache.

        Note:
            This method respects compression enabled/disabled state and the
            minimum token threshold before starting compression.
        """
        # Check if compression is enabled
        if not self.is_enabled:
            return cache

        # Check if we've reached the minimum token threshold
        if cache.size < self.config.min_tokens_before_compression:
            return cache

        # Check if cache exceeds window size
        if cache.size <= self.window_size:
            return cache

        # Evict to window size
        return self.evict(cache, self.window_size)

    def evict(self, cache: KVCache, target_size: int) -> KVCache:
        """
        Evict tokens to reach the target size using sliding window policy.

        This method keeps the most recent target_size tokens and removes
        older tokens. Protected tokens (the most recent protected_token_count
        tokens) are guaranteed to be retained.

        Args:
            cache: The KV cache to evict from.
            target_size: The desired cache size in tokens. Must be positive.

        Returns:
            A new cache with size <= max(target_size, protected_token_count).
            If the cache is already at or below the target size, returns it
            unchanged.

        Raises:
            ValueError: If target_size is invalid (non-positive).

        Note:
            If target_size < protected_token_count, the resulting cache will
            have protected_token_count tokens to ensure recency protection.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")

        current_size = cache.size

        # If already at or below target, no eviction needed
        if current_size <= target_size:
            return cache

        # Calculate how many tokens to keep
        # Ensure we keep at least the protected count
        tokens_to_keep = max(target_size, self.protected_count)
        tokens_to_keep = min(tokens_to_keep, current_size)

        # Calculate the starting index for the sliding window
        # We keep the last tokens_to_keep tokens
        start_idx = current_size - tokens_to_keep

        # Create a new cache with only the windowed tokens
        compressed_cache = self._slice_cache(cache, start_idx, current_size)

        return compressed_cache

    def _slice_cache(self, cache: KVCache, start_idx: int, end_idx: int) -> KVCache:
        """
        Slice the cache to keep only tokens from start_idx to end_idx.

        This is a helper method that performs the actual tensor slicing
        operation across all layers in the cache.

        Args:
            cache: The original cache to slice.
            start_idx: Starting token index (inclusive).
            end_idx: Ending token index (exclusive).

        Returns:
            A new cache containing only the specified token range.

        Note:
            This method assumes that the cache stores KV tensors with shape
            (batch_size, num_heads, seq_len, head_dim) where seq_len is the
            dimension to slice along (dim=2).
        """
        # Note: torch.Tensor in type annotations is safe here because with
        # `from __future__ import annotations`, they are evaluated as strings
        # at runtime and won't cause import errors.

        # Create a new cache by slicing each layer's KV tensors
        # We use a simple dictionary-based implementation to wrap the sliced data

        class SlicedKVCache:
            """Temporary cache implementation for sliced data."""

            def __init__(self, original_cache: KVCache, start: int, end: int):
                self._original = original_cache
                self._start = start
                self._end = end
                self._cached_kvs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

            @property
            def size(self) -> int:
                return self._end - self._start

            @property
            def num_layers(self) -> int:
                return self._original.num_layers

            @property
            def max_size(self) -> int:
                return self._original.max_size

            def get_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
                if layer in self._cached_kvs:
                    return self._cached_kvs[layer]

                keys, values = self._original.get_kv(layer)
                # Slice along the sequence dimension (dim=2 for standard KV shape)
                # Shape: (batch_size, num_heads, seq_len, head_dim)
                sliced_keys = keys[:, :, self._start : self._end, :]
                sliced_values = values[:, :, self._start : self._end, :]

                self._cached_kvs[layer] = (sliced_keys, sliced_values)
                return sliced_keys, sliced_values

            def set_kv(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
                self._cached_kvs[layer] = (keys, values)

            def clear(self) -> None:
                self._cached_kvs.clear()

            def clone(self) -> KVCache:
                # Create a deep copy with cloned tensors
                cloned = SlicedKVCache(self._original, self._start, self._end)
                for layer in range(self.num_layers):
                    keys, values = self.get_kv(layer)
                    cloned.set_kv(layer, keys.clone(), values.clone())
                return cloned

        return SlicedKVCache(cache, start_idx, end_idx)

    def get_eviction_stats(self, cache: KVCache) -> dict:
        """
        Get statistics about what would be evicted from the cache.

        This method provides information about the current cache state
        and what would happen if compression were applied.

        Args:
            cache: The cache to analyze.

        Returns:
            Dictionary containing:
                - current_size: Current number of tokens in cache
                - window_size: Configured sliding window size
                - would_evict: Number of tokens that would be evicted
                - protected_count: Number of protected tokens
                - compression_needed: Whether compression would be applied

        Example:
            >>> stats = compressor.get_eviction_stats(cache)
            >>> if stats['compression_needed']:
            ...     print(f"Would evict {stats['would_evict']} tokens")
        """
        current_size = cache.size

        # Determine if compression would be applied
        compression_needed = (
            self.is_enabled
            and current_size >= self.config.min_tokens_before_compression
            and current_size > self.window_size
        )

        if compression_needed:
            target_size = min(self.window_size, current_size)
            tokens_to_keep = max(target_size, self.protected_count)
            tokens_to_keep = min(tokens_to_keep, current_size)
            would_evict = current_size - tokens_to_keep
        else:
            tokens_to_keep = current_size
            would_evict = 0

        return {
            "current_size": current_size,
            "window_size": self.window_size,
            "would_evict": would_evict,
            "protected_count": self.protected_count,
            "compression_needed": compression_needed,
            "tokens_after_compression": tokens_to_keep,
        }

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"SlidingWindowCompressor("
            f"window_size={self.window_size}, "
            f"protected_count={self.protected_count}, "
            f"enabled={self.is_enabled})"
        )

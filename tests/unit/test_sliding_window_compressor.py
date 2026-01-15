"""
Unit tests for the Sliding Window KV Cache Compressor.

Tests cover:
- Basic sliding window eviction behavior
- Protected token handling
- Edge cases (empty cache, small cache, exact window size)
- Configuration validation
- Long sequence handling
- Memory usage statistics
"""

import pytest
import torch

from air.compression.sliding_window import SlidingWindowCompressor
from air.types import CompressionConfig

# Test constants
COMPRESSION_RATIO_TOLERANCE = 0.1  # Tolerance for memory compression ratio tests


class MockKVCache:
    """
    Mock KV cache for testing.

    Simulates a simple KV cache with configurable size and layers.
    Uses small tensors to keep tests fast.
    """

    def __init__(
        self,
        num_tokens: int,
        num_layers: int = 4,
        num_heads: int = 8,
        head_dim: int = 64,
        max_size: int = 2048,
    ):
        """
        Initialize mock cache.

        Args:
            num_tokens: Number of tokens currently in cache.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads per layer.
            head_dim: Dimension of each attention head.
            max_size: Maximum cache capacity.
        """
        self._size = num_tokens
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._max_size = max_size

        # Create KV tensors for each layer
        # Shape: (batch=1, num_heads, seq_len, head_dim)
        self._kvs = {}
        for layer in range(num_layers):
            keys = torch.randn(1, num_heads, num_tokens, head_dim)
            values = torch.randn(1, num_heads, num_tokens, head_dim)
            self._kvs[layer] = (keys, values)

    @property
    def size(self) -> int:
        """Return the current number of tokens in the cache."""
        return self._size

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        return self._num_layers

    @property
    def max_size(self) -> int:
        """Return the maximum capacity of the cache."""
        return self._max_size

    def get_kv(self, layer: int):
        """Get key and value tensors for a specific layer."""
        if layer < 0 or layer >= self._num_layers:
            raise IndexError(f"Layer {layer} out of bounds [0, {self._num_layers})")
        return self._kvs[layer]

    def set_kv(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Set key and value tensors for a specific layer."""
        if layer < 0 or layer >= self._num_layers:
            raise IndexError(f"Layer {layer} out of bounds [0, {self._num_layers})")
        self._kvs[layer] = (keys, values)

    def clear(self) -> None:
        """Clear all cached key-value pairs."""
        self._size = 0
        self._kvs.clear()

    def clone(self):
        """Create a deep copy of the cache."""
        new_cache = MockKVCache(
            self._size, self._num_layers, self._num_heads, self._head_dim, self._max_size
        )
        for layer in range(self._num_layers):
            keys, values = self._kvs[layer]
            new_cache.set_kv(layer, keys.clone(), values.clone())
        return new_cache


class TestSlidingWindowCompressorInit:
    """Tests for SlidingWindowCompressor initialization."""

    def test_initialization_with_correct_policy(self):
        """Test that compressor initializes with sliding_window policy."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
        )
        compressor = SlidingWindowCompressor(config)

        assert compressor.window_size == 512
        assert compressor.is_enabled

    def test_initialization_with_wrong_policy_raises_error(self):
        """Test that wrong policy raises ValueError."""
        config = CompressionConfig(eviction_policy="heavy_hitter")

        with pytest.raises(ValueError, match="requires eviction_policy='sliding_window'"):
            SlidingWindowCompressor(config)

    def test_protected_count_property(self):
        """Test that protected_count property returns correct value."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            protected_token_count=64,
        )
        compressor = SlidingWindowCompressor(config)

        assert compressor.protected_count == 64


class TestSlidingWindowCompression:
    """Tests for the compress method."""

    def test_compress_cache_larger_than_window(self):
        """Test compression when cache exceeds window size."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
            min_tokens_before_compression=100,
        )
        compressor = SlidingWindowCompressor(config)

        # Create a cache with 1000 tokens
        cache = MockKVCache(num_tokens=1000)

        compressed = compressor.compress(cache)

        # Should keep only the last 512 tokens
        assert compressed.size == 512
        assert compressed.num_layers == cache.num_layers

    def test_compress_cache_smaller_than_window(self):
        """Test that compression doesn't happen when cache is small."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
            min_tokens_before_compression=100,
        )
        compressor = SlidingWindowCompressor(config)

        # Create a small cache
        cache = MockKVCache(num_tokens=300)

        compressed = compressor.compress(cache)

        # Should not compress (cache is within window and below target ratio)
        assert compressed.size == cache.size

    def test_compress_respects_min_tokens_threshold(self):
        """Test that compression doesn't happen below min_tokens threshold."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=200,
            min_tokens_before_compression=300,
        )
        compressor = SlidingWindowCompressor(config)

        # Create cache just below threshold
        cache = MockKVCache(num_tokens=250)

        compressed = compressor.compress(cache)

        # Should not compress due to min_tokens threshold
        assert compressed.size == cache.size

    def test_compress_when_disabled(self):
        """Test that compression doesn't happen when disabled."""
        config = CompressionConfig(
            enabled=False,
            eviction_policy="sliding_window",
            sliding_window_size=100,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=500)
        compressed = compressor.compress(cache)

        assert compressed.size == cache.size


class TestSlidingWindowEviction:
    """Tests for the evict method."""

    def test_evict_to_target_size(self):
        """Test eviction to a specific target size."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=1000)

        evicted = compressor.evict(cache, target_size=300)

        assert evicted.size == 300

    def test_evict_keeps_most_recent_tokens(self):
        """Test that eviction keeps the most recent tokens."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
        )
        compressor = SlidingWindowCompressor(config)

        # Create cache with identifiable token pattern
        cache = MockKVCache(num_tokens=100)

        # Store original last layer KV values
        original_keys, original_values = cache.get_kv(0)

        evicted = compressor.evict(cache, target_size=50)

        # Get the evicted cache KVs
        evicted_keys, evicted_values = evicted.get_kv(0)

        # Check that the evicted cache contains the last 50 tokens
        # from the original cache (tokens 50-99)
        assert evicted_keys.shape[2] == 50  # seq_len dimension

        # The last token in evicted should match the last token in original
        torch.testing.assert_close(
            evicted_keys[:, :, -1, :],
            original_keys[:, :, -1, :],
        )

    def test_evict_respects_protected_tokens(self):
        """Test that protected tokens are never evicted."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            protected_token_count=32,
            sliding_window_size=512,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=100)

        # Try to evict to a size smaller than protected count
        evicted = compressor.evict(cache, target_size=20)

        # Should keep at least protected_token_count
        assert evicted.size == 32

    def test_evict_invalid_target_size(self):
        """Test that invalid target size raises ValueError."""
        config = CompressionConfig(eviction_policy="sliding_window")
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=100)

        with pytest.raises(ValueError, match="target_size must be positive"):
            compressor.evict(cache, target_size=0)

        with pytest.raises(ValueError, match="target_size must be positive"):
            compressor.evict(cache, target_size=-10)

    def test_evict_cache_already_at_target(self):
        """Test eviction when cache is already at or below target."""
        config = CompressionConfig(eviction_policy="sliding_window")
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=100)

        # Target is equal to current size
        evicted = compressor.evict(cache, target_size=100)
        assert evicted.size == 100

        # Target is larger than current size
        evicted = compressor.evict(cache, target_size=200)
        assert evicted.size == 100


class TestSlidingWindowLongSequences:
    """Tests for handling long sequences."""

    def test_compress_very_long_sequence(self):
        """Test compression on very long sequences."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
            min_tokens_before_compression=100,
        )
        compressor = SlidingWindowCompressor(config)

        # Create a long sequence cache
        cache = MockKVCache(num_tokens=10000)

        compressed = compressor.compress(cache)

        # Should compress to window size
        assert compressed.size == 512

        # Verify all layers are compressed
        for layer in range(compressed.num_layers):
            keys, values = compressed.get_kv(layer)
            assert keys.shape[2] == 512  # seq_len dimension
            assert values.shape[2] == 512

    def test_memory_reduction_on_long_sequence(self):
        """Test that memory is actually reduced on long sequences."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
            min_tokens_before_compression=100,
        )
        compressor = SlidingWindowCompressor(config)

        # Create long sequence
        original_cache = MockKVCache(num_tokens=5000)

        # Get memory usage before compression
        original_memory = compressor.get_memory_usage(original_cache)

        # Compress
        compressed_cache = compressor.compress(original_cache)
        compressed_memory = compressor.get_memory_usage(compressed_cache)

        # Memory should be significantly reduced
        assert compressed_memory < original_memory

        # Compression ratio should be approximately 512/5000
        expected_ratio = 512 / 5000
        actual_ratio = compressed_memory / original_memory

        # Allow some tolerance due to memory usage estimation
        assert abs(actual_ratio - expected_ratio) < COMPRESSION_RATIO_TOLERANCE

    def test_iterative_compression(self):
        """Test applying compression multiple times (simulating generation)."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=100,
            min_tokens_before_compression=50,
        )
        compressor = SlidingWindowCompressor(config)

        # Start with initial cache
        cache = MockKVCache(num_tokens=200)

        # First compression
        cache = compressor.compress(cache)
        assert cache.size == 100

        # Simulate adding more tokens and compressing again
        # In real scenario, cache would grow, here we just verify idempotency
        cache = compressor.compress(cache)
        assert cache.size == 100


class TestSlidingWindowEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_cache(self):
        """Test handling of empty cache."""
        config = CompressionConfig(eviction_policy="sliding_window")
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=0)

        compressed = compressor.compress(cache)
        assert compressed.size == 0

    def test_single_token_cache(self):
        """Test handling of single token cache."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=1)

        compressed = compressor.compress(cache)
        assert compressed.size == 1

    def test_exact_window_size(self):
        """Test cache exactly at window size."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
            min_tokens_before_compression=0,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=512)

        compressed = compressor.compress(cache)
        assert compressed.size == 512

    def test_window_size_one(self):
        """Test with minimal window size of 1."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=1,
            protected_token_count=1,
            min_tokens_before_compression=0,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=100)

        compressed = compressor.compress(cache)
        assert compressed.size == 1


class TestSlidingWindowStats:
    """Tests for statistics and monitoring methods."""

    def test_get_eviction_stats(self):
        """Test eviction statistics calculation."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
            protected_token_count=32,
            min_tokens_before_compression=100,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=1000)

        stats = compressor.get_eviction_stats(cache)

        assert stats["current_size"] == 1000
        assert stats["window_size"] == 512
        assert stats["would_evict"] == 1000 - 512
        assert stats["protected_count"] == 32
        assert stats["compression_needed"] is True
        assert stats["tokens_after_compression"] == 512

    def test_get_eviction_stats_no_compression_needed(self):
        """Test stats when no compression is needed."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
            min_tokens_before_compression=100,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=50)

        stats = compressor.get_eviction_stats(cache)

        assert stats["compression_needed"] is False
        assert stats["would_evict"] == 0

    def test_get_compression_stats(self):
        """Test compression statistics from base class."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=200,
            min_tokens_before_compression=0,
        )
        compressor = SlidingWindowCompressor(config)

        original = MockKVCache(num_tokens=1000)
        compressed = compressor.compress(original)

        stats = compressor.get_compression_stats(original, compressed)

        assert stats["original_tokens"] == 1000
        assert stats["compressed_tokens"] == 200
        assert stats["tokens_evicted"] == 800
        assert stats["compression_ratio"] < 1.0
        assert stats["eviction_policy"] == "sliding_window"


class TestSlidingWindowStringRepresentation:
    """Tests for string representation and debugging."""

    def test_repr(self):
        """Test string representation of compressor."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=512,
            protected_token_count=32,
        )
        compressor = SlidingWindowCompressor(config)

        repr_str = repr(compressor)

        assert "SlidingWindowCompressor" in repr_str
        assert "window_size=512" in repr_str
        assert "protected_count=32" in repr_str
        assert "enabled=True" in repr_str

    def test_repr_disabled(self):
        """Test string representation when disabled."""
        config = CompressionConfig(
            enabled=False,
            eviction_policy="sliding_window",
        )
        compressor = SlidingWindowCompressor(config)

        repr_str = repr(compressor)

        assert "enabled=False" in repr_str


class TestSlidingWindowMultiLayer:
    """Tests for multi-layer cache handling."""

    def test_compress_preserves_all_layers(self):
        """Test that compression preserves all layers."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=100,
            min_tokens_before_compression=0,
        )
        compressor = SlidingWindowCompressor(config)

        # Create cache with multiple layers
        cache = MockKVCache(num_tokens=500, num_layers=32)

        compressed = compressor.compress(cache)

        assert compressed.num_layers == 32

        # Verify each layer is properly compressed
        for layer in range(32):
            keys, values = compressed.get_kv(layer)
            assert keys.shape[2] == 100
            assert values.shape[2] == 100

    def test_layer_independence(self):
        """Test that layers are handled independently."""
        config = CompressionConfig(
            eviction_policy="sliding_window",
            sliding_window_size=50,
            min_tokens_before_compression=0,
        )
        compressor = SlidingWindowCompressor(config)

        cache = MockKVCache(num_tokens=100, num_layers=4)

        # Store original values from different layers
        originals = [cache.get_kv(i) for i in range(4)]

        compressed = compressor.compress(cache)

        # Each compressed layer should contain the last 50 tokens
        for layer in range(4):
            orig_keys, orig_values = originals[layer]
            comp_keys, comp_values = compressed.get_kv(layer)

            # Check sequence length
            assert comp_keys.shape[2] == 50

            # Last token should match
            torch.testing.assert_close(
                comp_keys[:, :, -1, :],
                orig_keys[:, :, -1, :],
            )

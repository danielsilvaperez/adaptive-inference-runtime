"""
Unit tests for Heavy Hitter Retention Policy.
"""

import pytest

torch = pytest.importorskip("torch")

from air.compression.cache_impl import SimpleKVCache  # noqa: E402
from air.compression.heavy_hitter import HeavyHitterCompressor  # noqa: E402
from air.types import CompressionConfig  # noqa: E402


class TestHeavyHitterCompressor:
    """Tests for HeavyHitterCompressor class."""

    @pytest.fixture
    def config(self):
        """Create a default compression config for heavy hitter policy."""
        return CompressionConfig(
            enabled=True,
            eviction_policy="heavy_hitter",
            target_ratio=0.5,
            heavy_hitter_ratio=0.3,
            protected_token_count=4,
            min_tokens_before_compression=8,
        )

    @pytest.fixture
    def compressor(self, config):
        """Create a HeavyHitterCompressor instance."""
        return HeavyHitterCompressor(config)

    @pytest.fixture
    def sample_cache(self):
        """Create a sample KV cache for testing."""
        num_layers = 2
        max_size = 100
        cache = SimpleKVCache(num_layers=num_layers, max_size=max_size)

        # Create dummy tensors: shape [batch=1, heads=4, seq_len=16, dim=64]
        batch_size = 1
        num_heads = 4
        seq_len = 16
        head_dim = 64

        for layer in range(num_layers):
            keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
            values = torch.randn(batch_size, num_heads, seq_len, head_dim)
            cache.set_kv(layer, keys, values)

        return cache

    def test_initialization_valid(self, config):
        """Test that compressor initializes correctly with valid config."""
        compressor = HeavyHitterCompressor(config)
        assert compressor.config == config
        assert compressor.is_enabled is True

    def test_initialization_invalid_policy(self):
        """Test that initialization fails with invalid eviction policy."""
        config = CompressionConfig(eviction_policy="sliding_window")
        with pytest.raises(ValueError, match="eviction_policy"):
            HeavyHitterCompressor(config)

    def test_initialization_with_h2o_policy(self):
        """Test that h2o policy is also accepted."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = HeavyHitterCompressor(config)
        assert compressor.config.eviction_policy == "h2o"

    def test_update_attention_scores(self, compressor):
        """Test updating attention scores for tokens."""
        layer = 0
        positions = [0, 1, 2]
        scores = [0.5, 1.0, 0.2]

        compressor.update_attention_scores(layer, positions, scores)

        # Verify scores are tracked
        assert compressor.get_cumulative_attention(0) == 0.5
        assert compressor.get_cumulative_attention(1) == 1.0
        assert compressor.get_cumulative_attention(2) == 0.2

    def test_update_attention_scores_accumulates(self, compressor):
        """Test that attention scores accumulate over multiple updates."""
        layer = 0
        positions = [0, 1]
        scores = [0.5, 1.0]

        compressor.update_attention_scores(layer, positions, scores)
        compressor.update_attention_scores(layer, positions, scores)

        # Scores should be doubled
        assert compressor.get_cumulative_attention(0) == 1.0
        assert compressor.get_cumulative_attention(1) == 2.0

    def test_get_cumulative_attention_across_layers(self, compressor):
        """Test cumulative attention calculation across multiple layers."""
        # Add scores for same position across different layers
        compressor.update_attention_scores(0, [0], [0.5])
        compressor.update_attention_scores(1, [0], [0.3])
        compressor.update_attention_scores(2, [0], [0.2])

        # Should sum across all layers
        assert compressor.get_cumulative_attention(0) == 1.0

    def test_get_cumulative_attention_no_scores(self, compressor):
        """Test cumulative attention returns 0 for unseen positions."""
        assert compressor.get_cumulative_attention(999) == 0.0

    def test_reset_attention_scores(self, compressor):
        """Test resetting attention scores."""
        compressor.update_attention_scores(0, [0, 1], [0.5, 1.0])
        assert compressor.get_cumulative_attention(0) > 0

        compressor.reset_attention_scores()

        # All scores should be cleared
        assert compressor.get_cumulative_attention(0) == 0.0
        assert compressor.get_cumulative_attention(1) == 0.0

    def test_eviction_candidates_selection(self, compressor, sample_cache):
        """Test selection of eviction candidates based on attention."""
        # Set up attention scores (higher = more important)
        # Positions: 0=0.1, 1=0.9, 2=0.2, 3=0.8, 4=0.3, etc.
        for i in range(sample_cache.size):
            score = 0.1 if i % 2 == 0 else 0.9
            compressor.update_attention_scores(0, [i], [score])

        # Get candidates for eviction to target size of 10
        # Protected last 4 tokens, so can evict from positions 0-11
        candidates = compressor._get_eviction_candidates(sample_cache, target_size=10)

        # Should evict 6 tokens (16 - 10)
        assert len(candidates) == 6

        # Should evict tokens with low attention (even positions)
        for pos in candidates:
            # Evicted tokens should have lower attention than retained ones
            assert pos % 2 == 0  # Even positions have low attention

    def test_eviction_protects_recent_tokens(self, compressor, sample_cache):
        """Test that recent tokens are protected from eviction."""
        # Set high attention for early tokens, low for recent tokens
        for i in range(sample_cache.size - 4):
            compressor.update_attention_scores(0, [i], [10.0])

        # Recent tokens get very low scores
        for i in range(sample_cache.size - 4, sample_cache.size):
            compressor.update_attention_scores(0, [i], [0.001])

        # Get eviction candidates
        candidates = compressor._get_eviction_candidates(sample_cache, target_size=10)

        # Recent tokens (last 4) should NOT be in eviction candidates
        protected_start = sample_cache.size - 4
        for pos in candidates:
            assert pos < protected_start

    def test_compress_when_not_needed(self, compressor, sample_cache):
        """Test that compression is skipped when not needed."""
        # Cache is too small to compress (< min_tokens_before_compression)
        small_cache = SimpleKVCache(num_layers=2, max_size=100)
        keys = torch.randn(1, 4, 4, 64)  # Only 4 tokens
        values = torch.randn(1, 4, 4, 64)
        small_cache.set_kv(0, keys, values)
        small_cache.set_kv(1, keys, values)

        result = compressor.compress(small_cache)

        # Should return the same cache
        assert result is small_cache

    def test_compress_reduces_cache_size(self):
        """Test that compression actually reduces cache size."""
        # Create a config that will definitely compress
        config = CompressionConfig(
            enabled=True,
            eviction_policy="heavy_hitter",
            target_ratio=0.3,  # More aggressive compression
            heavy_hitter_ratio=0.2,
            protected_token_count=2,
            min_tokens_before_compression=8,
        )
        compressor = HeavyHitterCompressor(config)

        # Create cache with max_size closer to actual size to trigger compression
        cache = SimpleKVCache(num_layers=2, max_size=25)
        keys = torch.randn(1, 4, 20, 64)  # 20 tokens
        values = torch.randn(1, 4, 20, 64)
        cache.set_kv(0, keys, values)
        cache.set_kv(1, keys, values)

        # Set up some attention scores
        for i in range(cache.size):
            score = 1.0 if i < 5 else 0.1  # First 5 tokens are heavy hitters
            compressor.update_attention_scores(0, [i], [score])

        original_size = cache.size
        compressed = compressor.compress(cache)

        # Should be smaller
        assert compressed.size < original_size

    def test_evict_with_valid_target(self, compressor, sample_cache):
        """Test eviction with a valid target size."""
        # Set up attention scores
        for i in range(sample_cache.size):
            score = float(i)  # Increasing scores
            compressor.update_attention_scores(0, [i], [score])

        target_size = 10
        evicted = compressor.evict(sample_cache, target_size)

        # New cache should be at target size
        assert evicted.size == target_size

    def test_evict_with_invalid_target(self, compressor, sample_cache):
        """Test that eviction raises error with invalid target."""
        with pytest.raises(ValueError, match="target_size must be positive"):
            compressor.evict(sample_cache, target_size=0)

        with pytest.raises(ValueError, match="target_size must be positive"):
            compressor.evict(sample_cache, target_size=-1)

    def test_evict_no_op_when_target_larger(self, compressor, sample_cache):
        """Test that eviction is no-op when target >= current size."""
        result = compressor.evict(sample_cache, target_size=sample_cache.size + 10)
        assert result is sample_cache

    def test_update_scores_after_eviction(self, compressor):
        """Test that scores are remapped after eviction."""
        # Set up scores for positions 0-9
        for i in range(10):
            compressor.update_attention_scores(0, [i], [float(i)])

        # Evict positions 2, 5, 7
        evicted = [2, 5, 7]
        compressor._update_scores_after_eviction(evicted)

        # Position 0 should stay at 0
        assert compressor.get_cumulative_attention(0) == 0.0

        # Position 1 should stay at 1
        assert compressor.get_cumulative_attention(1) == 1.0

        # Position 2 was evicted, old position 3 should now be at 2
        assert compressor.get_cumulative_attention(2) == 3.0

        # Old position 4 should now be at 3
        assert compressor.get_cumulative_attention(3) == 4.0

        # Old position 6 should now be at 4
        assert compressor.get_cumulative_attention(4) == 6.0

    def test_get_compression_stats(self, compressor, sample_cache):
        """Test compression statistics generation."""
        # Set up attention scores
        for i in range(sample_cache.size):
            compressor.update_attention_scores(0, [i], [float(i)])

        compressed = compressor.evict(sample_cache, target_size=8)
        stats = compressor.get_compression_stats(sample_cache, compressed)

        # Check required fields
        assert "original_tokens" in stats
        assert "compressed_tokens" in stats
        assert "tokens_evicted" in stats
        assert "compression_ratio" in stats
        assert "heavy_hitter_ratio" in stats

        # Check values
        assert stats["original_tokens"] == sample_cache.size
        assert stats["compressed_tokens"] == compressed.size
        assert stats["tokens_evicted"] == sample_cache.size - compressed.size

    def test_heavy_hitter_stats_with_attention_scores(self, compressor, sample_cache):
        """Test that stats include attention metrics when available."""
        # Set up attention scores
        for i in range(sample_cache.size):
            compressor.update_attention_scores(0, [i], [float(i) * 0.1])

        compressed = compressor.evict(sample_cache, target_size=8)
        stats = compressor.get_compression_stats(sample_cache, compressed)

        # Should include attention statistics
        assert "avg_attention_score" in stats
        assert "max_attention_score" in stats
        assert "min_attention_score" in stats

    def test_memory_usage_calculation(self, compressor, sample_cache):
        """Test memory usage calculation."""
        memory = compressor.get_memory_usage(sample_cache)

        # Should return a positive value
        assert memory > 0

        # Memory should be proportional to cache size
        larger_cache = SimpleKVCache(num_layers=2, max_size=100)
        keys = torch.randn(1, 4, 32, 64)  # 32 tokens
        values = torch.randn(1, 4, 32, 64)
        larger_cache.set_kv(0, keys, values)
        larger_cache.set_kv(1, keys, values)

        larger_memory = compressor.get_memory_usage(larger_cache)
        assert larger_memory > memory

    def test_repr(self, compressor):
        """Test string representation."""
        repr_str = repr(compressor)
        assert "HeavyHitterCompressor" in repr_str
        assert "heavy_hitter" in repr_str or "h2o" in repr_str

    def test_integration_compress_and_stats(self):
        """Test full workflow: update scores, compress, get stats."""
        # Create a config that will definitely compress
        config = CompressionConfig(
            enabled=True,
            eviction_policy="heavy_hitter",
            target_ratio=0.4,
            heavy_hitter_ratio=0.2,
            protected_token_count=3,
            min_tokens_before_compression=8,
        )
        compressor = HeavyHitterCompressor(config)

        # Create cache with max_size closer to actual size to trigger compression
        cache = SimpleKVCache(num_layers=2, max_size=25)
        keys = torch.randn(1, 4, 20, 64)  # 20 tokens
        values = torch.randn(1, 4, 20, 64)
        cache.set_kv(0, keys, values)
        cache.set_kv(1, keys, values)

        # Simulate generation with attention tracking
        for step in range(5):
            # Each step, update attention for some tokens
            positions = list(range(min(step + 3, cache.size)))
            scores = [1.0 / (i + 1) for i in positions]  # Decreasing attention
            compressor.update_attention_scores(0, positions, scores)

        # Compress the cache
        original_size = cache.size
        compressed = compressor.compress(cache)

        # Verify compression occurred
        assert compressed.size < original_size

        # Get stats
        stats = compressor.get_compression_stats(cache, compressed)

        # Verify stats are reasonable
        assert 0 < stats["compression_ratio"] < 1
        assert stats["tokens_evicted"] > 0


class TestSimpleKVCache:
    """Tests for SimpleKVCache implementation."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = SimpleKVCache(num_layers=4, max_size=1024)
        assert cache.num_layers == 4
        assert cache.max_size == 1024
        assert cache.size == 0

    def test_set_and_get_kv(self):
        """Test setting and getting KV tensors."""
        cache = SimpleKVCache(num_layers=2, max_size=100)

        keys = torch.randn(1, 4, 10, 64)
        values = torch.randn(1, 4, 10, 64)

        cache.set_kv(0, keys, values)

        retrieved_keys, retrieved_values = cache.get_kv(0)

        assert torch.equal(retrieved_keys, keys)
        assert torch.equal(retrieved_values, values)
        assert cache.size == 10

    def test_clear(self):
        """Test clearing the cache."""
        cache = SimpleKVCache(num_layers=2, max_size=100)

        keys = torch.randn(1, 4, 10, 64)
        values = torch.randn(1, 4, 10, 64)
        cache.set_kv(0, keys, values)

        assert cache.size == 10

        cache.clear()

        assert cache.size == 0

    def test_clone(self):
        """Test cloning the cache."""
        cache = SimpleKVCache(num_layers=2, max_size=100)

        keys = torch.randn(1, 4, 10, 64)
        values = torch.randn(1, 4, 10, 64)
        cache.set_kv(0, keys, values)

        cloned = cache.clone()

        # Should have same data
        assert cloned.size == cache.size
        assert cloned.num_layers == cache.num_layers

        # But different tensor instances
        original_keys, _ = cache.get_kv(0)
        cloned_keys, _ = cloned.get_kv(0)
        assert torch.equal(original_keys, cloned_keys)
        assert original_keys is not cloned_keys

    def test_out_of_bounds_layer(self):
        """Test accessing out of bounds layer."""
        cache = SimpleKVCache(num_layers=2, max_size=100)

        with pytest.raises(IndexError):
            cache.get_kv(5)

        with pytest.raises(IndexError):
            cache.set_kv(5, torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64))

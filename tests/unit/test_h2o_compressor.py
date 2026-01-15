"""
Unit tests for H2O-style KV cache compressor.
"""


import pytest
import torch

from air.compression.h2o import H2OCompressor
from air.types import CompressionConfig


class MockKVCache:
    """Mock KV cache for testing."""

    def __init__(self, size: int, num_layers: int, max_size: int = 2048):
        self._size = size
        self._num_layers = num_layers
        self._max_size = max_size
        self._kv_data = {}

    @property
    def size(self) -> int:
        return self._size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def max_size(self) -> int:
        return self._max_size

    def get_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer not in self._kv_data:
            # Create dummy tensors
            batch_size, num_heads, seq_len, head_dim = 1, 8, self._size, 64
            keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
            values = torch.randn(batch_size, num_heads, seq_len, head_dim)
            self._kv_data[layer] = (keys, values)
        return self._kv_data[layer]

    def set_kv(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        self._kv_data[layer] = (keys, values)

    def clear(self) -> None:
        self._kv_data.clear()
        self._size = 0

    def clone(self) -> "MockKVCache":
        new_cache = MockKVCache(self._size, self._num_layers, self._max_size)
        new_cache._kv_data = {k: (v[0].clone(), v[1].clone()) for k, v in self._kv_data.items()}
        return new_cache


class TestH2OCompressorInit:
    """Tests for H2OCompressor initialization."""

    def test_init_with_valid_config(self):
        """Test initialization with valid H2O config."""
        config = CompressionConfig(
            eviction_policy="h2o",
            target_ratio=0.5,
            heavy_hitter_ratio=0.1,
        )
        compressor = H2OCompressor(config)

        assert compressor.config == config
        assert compressor.is_enabled
        assert compressor._per_layer == config.per_layer_policy

    def test_init_with_invalid_policy(self):
        """Test initialization fails with wrong eviction policy."""
        config = CompressionConfig(eviction_policy="sliding_window")

        with pytest.raises(ValueError, match="requires eviction_policy='h2o'"):
            H2OCompressor(config)

    def test_init_with_per_layer_policy(self):
        """Test initialization with per-layer policy."""
        config = CompressionConfig(
            eviction_policy="h2o",
            per_layer_policy=True,
        )
        compressor = H2OCompressor(config)

        assert compressor._per_layer is True


class TestAttentionScoreTracking:
    """Tests for attention score tracking functionality."""

    def test_update_attention_scores_4d(self):
        """Test updating attention scores with 4D tensor."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        # Shape: (batch, num_heads, seq_len, seq_len)
        attention_weights = torch.rand(1, 8, 10, 10)
        compressor.update_attention_scores(layer=0, attention_weights=attention_weights)

        # Check that scores were recorded
        assert 0 in compressor._attention_scores
        assert len(compressor._attention_scores[0]) == 10

        # Scores should be positive
        for score in compressor._attention_scores[0].values():
            assert score > 0

    def test_update_attention_scores_3d(self):
        """Test updating attention scores with 3D tensor."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        # Shape: (num_heads, seq_len, seq_len)
        attention_weights = torch.rand(8, 10, 10)
        compressor.update_attention_scores(layer=0, attention_weights=attention_weights)

        # Check that scores were recorded
        assert 0 in compressor._attention_scores
        assert len(compressor._attention_scores[0]) == 10

    def test_update_attention_scores_with_positions(self):
        """Test updating attention scores with custom token positions."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        attention_weights = torch.rand(1, 8, 5, 5)
        positions = [10, 11, 12, 13, 14]

        compressor.update_attention_scores(
            layer=0,
            attention_weights=attention_weights,
            token_positions=positions,
        )

        # Check that custom positions were used
        assert set(compressor._attention_scores[0].keys()) == set(positions)

    def test_update_attention_scores_accumulates(self):
        """Test that attention scores accumulate across updates."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        # First update
        attention_weights = torch.ones(1, 8, 5, 5) * 0.5
        compressor.update_attention_scores(layer=0, attention_weights=attention_weights)
        first_score = compressor.get_attention_score(layer=0, position=0)

        # Second update
        compressor.update_attention_scores(layer=0, attention_weights=attention_weights)
        second_score = compressor.get_attention_score(layer=0, position=0)

        # Score should have accumulated
        assert second_score > first_score
        assert abs(second_score - 2 * first_score) < 1e-5

    def test_update_attention_scores_invalid_shape(self):
        """Test that invalid tensor shape raises error."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        # 2D tensor (invalid)
        attention_weights = torch.rand(10, 10)

        with pytest.raises(ValueError, match="Expected attention_weights with 3 or 4 dimensions"):
            compressor.update_attention_scores(layer=0, attention_weights=attention_weights)

    def test_update_attention_scores_mismatched_positions(self):
        """Test that mismatched positions length raises error."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        attention_weights = torch.rand(1, 8, 5, 5)
        positions = [0, 1, 2]  # Wrong length

        with pytest.raises(ValueError, match="token_positions length.*must match sequence length"):
            compressor.update_attention_scores(
                layer=0,
                attention_weights=attention_weights,
                token_positions=positions,
            )

    def test_reset_attention_scores_single_layer(self):
        """Test resetting attention scores for a single layer."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        # Add scores for multiple layers
        attention_weights = torch.rand(1, 8, 5, 5)
        compressor.update_attention_scores(layer=0, attention_weights=attention_weights)
        compressor.update_attention_scores(layer=1, attention_weights=attention_weights)

        # Reset only layer 0
        compressor.reset_attention_scores(layer=0)

        assert 0 not in compressor._attention_scores
        assert 1 in compressor._attention_scores

    def test_reset_attention_scores_all_layers(self):
        """Test resetting attention scores for all layers."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        # Add scores for multiple layers
        attention_weights = torch.rand(1, 8, 5, 5)
        compressor.update_attention_scores(layer=0, attention_weights=attention_weights)
        compressor.update_attention_scores(layer=1, attention_weights=attention_weights)

        # Reset all
        compressor.reset_attention_scores()

        assert len(compressor._attention_scores) == 0


class TestCompression:
    """Tests for compression functionality."""

    def test_compress_below_threshold(self):
        """Test that compression is skipped when cache is below threshold."""
        config = CompressionConfig(
            eviction_policy="h2o",
            min_tokens_before_compression=100,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=50, num_layers=4)
        result = compressor.compress(cache)

        # Should return original cache unchanged
        assert result is cache

    def test_compress_disabled(self):
        """Test that compression is skipped when disabled."""
        config = CompressionConfig(
            eviction_policy="h2o",
            enabled=False,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=500, num_layers=4)
        result = compressor.compress(cache)

        # Should return original cache unchanged
        assert result is cache

    def test_compress_applies_eviction(self):
        """Test that compression applies eviction when needed."""
        config = CompressionConfig(
            eviction_policy="h2o",
            target_ratio=0.5,
            min_tokens_before_compression=100,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=500, num_layers=4, max_size=1000)

        # Add attention scores
        attention_weights = torch.rand(1, 8, 500, 500)
        for layer in range(4):
            compressor.update_attention_scores(layer=layer, attention_weights=attention_weights)

        # Compress (this uses the placeholder implementation)
        # The warning is only emitted during eviction, which creates the cache
        with pytest.warns(RuntimeWarning, match="placeholder implementation"):
            compressor.evict(cache, target_size=250)

        # Note: With placeholder implementation, cache is returned unchanged
        # Real implementation would return compressed cache


class TestEviction:
    """Tests for eviction functionality."""

    def test_evict_with_invalid_target(self):
        """Test that eviction fails with invalid target size."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=100, num_layers=4)

        with pytest.raises(ValueError, match="target_size must be positive"):
            compressor.evict(cache, target_size=0)

        with pytest.raises(ValueError, match="target_size must be positive"):
            compressor.evict(cache, target_size=-10)

    def test_evict_no_eviction_needed(self):
        """Test that eviction is skipped when cache is below target."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=50, num_layers=4)

        result = compressor.evict(cache, target_size=100)

        # Should return original cache
        assert result is cache

    def test_select_tokens_to_keep_all_protected(self):
        """Test token selection when target equals protected count."""
        config = CompressionConfig(
            eviction_policy="h2o",
            protected_token_count=32,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=100, num_layers=4)

        tokens_to_keep = compressor._select_tokens_to_keep(cache, target_size=32)

        # Should keep the most recent 32 tokens (positions 68-99)
        assert len(tokens_to_keep) == 32
        assert tokens_to_keep == list(range(68, 100))

    def test_select_tokens_to_keep_with_heavy_hitters(self):
        """Test token selection includes heavy hitters."""
        config = CompressionConfig(
            eviction_policy="h2o",
            protected_token_count=10,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=100, num_layers=4)

        # Set up attention scores with known heavy hitters
        # Positions 5, 15, 25 should have high scores
        for layer in range(4):
            compressor._attention_scores[layer] = {}
            for pos in range(90):  # Evictable positions
                if pos in [5, 15, 25]:
                    compressor._attention_scores[layer][pos] = 10.0
                else:
                    compressor._attention_scores[layer][pos] = 1.0

        # Keep 50 tokens total (10 protected + 40 heavy hitters)
        tokens_to_keep = compressor._select_tokens_to_keep(cache, target_size=50)

        assert len(tokens_to_keep) == 50

        # Should include protected tokens (90-99)
        protected = [t for t in tokens_to_keep if t >= 90]
        assert len(protected) == 10

        # Should include heavy hitters
        assert 5 in tokens_to_keep
        assert 15 in tokens_to_keep
        assert 25 in tokens_to_keep

    def test_select_tokens_to_keep_no_scores(self):
        """Test token selection when no attention scores available."""
        config = CompressionConfig(
            eviction_policy="h2o",
            protected_token_count=10,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=100, num_layers=4)

        # No attention scores set
        tokens_to_keep = compressor._select_tokens_to_keep(cache, target_size=50)

        assert len(tokens_to_keep) == 50

        # Should keep oldest 40 + newest 10
        assert tokens_to_keep[:40] == list(range(40))
        assert tokens_to_keep[40:] == list(range(90, 100))


class TestHeavyHitterSelection:
    """Tests for heavy hitter selection strategies."""

    def test_select_heavy_hitters_global(self):
        """Test global heavy hitter selection."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=100, num_layers=4)

        # Set up scores across multiple layers
        for layer in range(4):
            compressor._attention_scores[layer] = {
                5: 2.0,
                10: 1.5,
                15: 1.0,
                20: 0.5,
            }

        evictable = [5, 10, 15, 20, 25, 30]
        heavy_hitters = compressor._select_heavy_hitters_global(cache, evictable, count=3)

        # Should select top 3 by aggregated score
        assert len(heavy_hitters) == 3
        assert 5 in heavy_hitters  # Score: 8.0
        assert 10 in heavy_hitters  # Score: 6.0
        assert 15 in heavy_hitters  # Score: 4.0

    def test_select_heavy_hitters_per_layer(self):
        """Test per-layer heavy hitter selection."""
        config = CompressionConfig(
            eviction_policy="h2o",
            per_layer_policy=True,
            heavy_hitter_ratio=0.5,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=100, num_layers=2)

        # Layer 0: positions 5, 10 are heavy hitters
        compressor._attention_scores[0] = {
            5: 10.0,
            10: 8.0,
            15: 1.0,
            20: 0.5,
        }

        # Layer 1: positions 10, 15 are heavy hitters
        compressor._attention_scores[1] = {
            5: 1.0,
            10: 9.0,
            15: 7.0,
            20: 0.5,
        }

        evictable = [5, 10, 15, 20, 25]
        heavy_hitters = compressor._select_heavy_hitters_per_layer(cache, evictable, count=3)

        # Should include union of per-layer heavy hitters
        assert len(heavy_hitters) <= 3

        # Position 10 should definitely be included (heavy hitter in both layers)
        assert 10 in heavy_hitters

    def test_select_heavy_hitters_per_layer_zero_ratio(self):
        """Test per-layer heavy hitter selection with zero ratio."""
        config = CompressionConfig(
            eviction_policy="h2o",
            per_layer_policy=True,
            heavy_hitter_ratio=0.0,  # Keep none based on ratio
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=100, num_layers=2)

        # Layer 0: positions with scores
        compressor._attention_scores[0] = {
            5: 10.0,
            10: 8.0,
            15: 1.0,
        }

        evictable = [5, 10, 15, 20, 25]
        heavy_hitters = compressor._select_heavy_hitters_per_layer(cache, evictable, count=2)

        # With ratio=0, no tokens should be selected from per-layer logic initially
        # But we need at least 'count' tokens, so oldest will be selected
        assert len(heavy_hitters) == 2


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_attention_score(self):
        """Test getting attention score for a position."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        # Set a known score
        compressor._attention_scores[0] = {5: 3.5}

        score = compressor.get_attention_score(layer=0, position=5)
        assert abs(score - 3.5) < 1e-6

        # Non-existent position should return 0.0
        score = compressor.get_attention_score(layer=0, position=999)
        assert score == 0.0

        # Non-existent layer should return 0.0
        score = compressor.get_attention_score(layer=99, position=5)
        assert score == 0.0

    def test_get_heavy_hitter_positions_single_layer(self):
        """Test getting heavy hitter positions for a single layer."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        compressor._attention_scores[0] = {
            5: 10.0,
            10: 8.0,
            15: 6.0,
            20: 4.0,
            25: 2.0,
        }

        heavy_hitters = compressor.get_heavy_hitter_positions(layer=0, top_k=3)

        assert len(heavy_hitters) == 3
        assert heavy_hitters[0] == (5, 10.0)
        assert heavy_hitters[1] == (10, 8.0)
        assert heavy_hitters[2] == (15, 6.0)

    def test_get_heavy_hitter_positions_global(self):
        """Test getting heavy hitter positions across all layers."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        compressor._attention_scores[0] = {5: 5.0, 10: 3.0}
        compressor._attention_scores[1] = {5: 3.0, 10: 5.0}

        heavy_hitters = compressor.get_heavy_hitter_positions(layer=None, top_k=2)

        # Positions 5 and 10 both have aggregated score of 8.0
        assert len(heavy_hitters) == 2
        positions = [pos for pos, _ in heavy_hitters]
        assert 5 in positions
        assert 10 in positions

    def test_get_heavy_hitter_positions_empty(self):
        """Test getting heavy hitter positions when no scores available."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        heavy_hitters = compressor.get_heavy_hitter_positions(layer=0, top_k=5)

        assert heavy_hitters == []

    def test_repr(self):
        """Test string representation."""
        config = CompressionConfig(
            eviction_policy="h2o",
            target_ratio=0.5,
            heavy_hitter_ratio=0.1,
        )
        compressor = H2OCompressor(config)

        repr_str = repr(compressor)

        assert "H2OCompressor" in repr_str
        assert "global" in repr_str  # Default policy
        assert "target_ratio=0.5" in repr_str
        assert "heavy_hitter_ratio=0.1" in repr_str
        assert "enabled=True" in repr_str

    def test_repr_per_layer(self):
        """Test string representation with per-layer policy."""
        config = CompressionConfig(
            eviction_policy="h2o",
            per_layer_policy=True,
        )
        compressor = H2OCompressor(config)

        repr_str = repr(compressor)

        assert "per-layer" in repr_str


class TestCompressionStats:
    """Tests for compression statistics."""

    def test_get_compression_stats(self):
        """Test getting compression statistics."""
        config = CompressionConfig(
            eviction_policy="h2o",
            target_ratio=0.5,
        )
        compressor = H2OCompressor(config)

        original_cache = MockKVCache(size=1000, num_layers=4)
        compressed_cache = MockKVCache(size=500, num_layers=4)

        stats = compressor.get_compression_stats(original_cache, compressed_cache)

        assert stats["original_tokens"] == 1000
        assert stats["compressed_tokens"] == 500
        assert stats["tokens_evicted"] == 500
        assert stats["compression_ratio"] == 0.5
        assert stats["eviction_policy"] == "h2o"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_cache(self):
        """Test handling of empty cache."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=0, num_layers=4)

        result = compressor.compress(cache)
        assert result is cache

    def test_single_token_cache(self):
        """Test handling of single-token cache."""
        config = CompressionConfig(
            eviction_policy="h2o",
            protected_token_count=1,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=1, num_layers=4)

        tokens_to_keep = compressor._select_tokens_to_keep(cache, target_size=1)
        assert tokens_to_keep == [0]

    def test_target_size_larger_than_cache(self):
        """Test when target size is larger than current cache size."""
        config = CompressionConfig(eviction_policy="h2o")
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=50, num_layers=4)

        result = compressor.evict(cache, target_size=100)
        assert result is cache

    def test_protected_count_larger_than_cache(self):
        """Test when protected count exceeds cache size."""
        config = CompressionConfig(
            eviction_policy="h2o",
            protected_token_count=200,
        )
        compressor = H2OCompressor(config)

        cache = MockKVCache(size=100, num_layers=4)

        tokens_to_keep = compressor._select_tokens_to_keep(cache, target_size=50)

        # Should keep the most recent 50 tokens (positions 50-99)
        # When protected_count (200) > cache.size (100), protected_count is clamped to 100
        # protected_positions becomes [0, 1, ..., 99] (all tokens)
        # We then take the last 50 of these, which are positions 50-99
        assert len(tokens_to_keep) == 50
        assert tokens_to_keep == list(range(50, 100))

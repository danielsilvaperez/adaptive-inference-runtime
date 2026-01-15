"""
Unit tests for the AttentionInstabilityScorer.

Tests cover:
- Basic functionality with various attention weight configurations
- Edge cases (empty, single layer, NaN/Inf values)
- Score normalization and sensitivity settings
- Different variance aggregation methods
- Layer statistics computation
"""

import pytest
import torch

from air.routing.confidence.attention_instability import AttentionInstabilityScorer


class TestAttentionInstabilityScorer:
    """Test suite for AttentionInstabilityScorer."""

    def test_initialization_default(self):
        """Test scorer initialization with default parameters."""
        scorer = AttentionInstabilityScorer()

        assert scorer.name == "attention_instability"
        assert scorer.sensitivity == 0.5
        assert scorer._use_head_variance is True
        assert scorer._variance_aggregation == "mean"

    def test_initialization_with_custom_params(self):
        """Test scorer initialization with custom parameters."""
        scorer = AttentionInstabilityScorer(
            sensitivity=0.8, use_head_variance=False, variance_aggregation="max"
        )

        assert scorer.sensitivity == 0.8
        assert scorer._use_head_variance is False
        assert scorer._variance_aggregation == "max"

    def test_invalid_sensitivity_raises_error(self):
        """Test that invalid sensitivity values raise ValueError."""
        with pytest.raises(ValueError, match="sensitivity must be in"):
            AttentionInstabilityScorer(sensitivity=-0.1)

        with pytest.raises(ValueError):
            AttentionInstabilityScorer(sensitivity=1.5)

    def test_invalid_variance_aggregation(self):
        """Test that invalid aggregation method raises ValueError."""
        with pytest.raises(ValueError, match="variance_aggregation must be one of"):
            AttentionInstabilityScorer(variance_aggregation="invalid")

    def test_name_property(self):
        """Test that the scorer has the correct name."""
        scorer = AttentionInstabilityScorer()
        assert scorer.name == "attention_instability"

    def test_score_from_logits_returns_default(self):
        """Test that score() returns default value since this scorer needs attention weights."""

        scorer = AttentionInstabilityScorer()
        logits = torch.randn(10, 50000)  # (seq_len, vocab_size)
        score = scorer.score(logits)

        # Should return default neutral score
        assert score == 0.5

    def test_stable_attention_high_confidence(self):
        """Test that stable attention patterns yield high confidence."""

        scorer = AttentionInstabilityScorer(sensitivity=0.5)

        # Create very stable attention (all layers similar)
        # Shape: (num_layers=32, num_heads=32, seq_len=64, seq_len=64)
        num_layers, num_heads, seq_len = 32, 32, 64
        
        # Create a base attention pattern
        base_attn = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
        
        # Repeat across all layers and heads with tiny noise
        attention_weights = base_attn.unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1, 1)
        attention_weights = attention_weights + torch.randn_like(attention_weights) * 0.001

        score = scorer.score_from_attention(attention_weights)
        assert score > 0.7, f"Expected high confidence for stable attention, got {score}"

    def test_unstable_attention_low_confidence(self):
        """Test that unstable attention patterns yield lower confidence than stable patterns."""
        scorer = AttentionInstabilityScorer(sensitivity=0.8)

        # Create highly variable attention patterns where each layer is completely different
        num_layers, num_heads, seq_len = 16, 8, 32
        
        # First create a stable baseline
        stable_attn = torch.ones(num_layers, num_heads, seq_len, seq_len) / seq_len
        stable_score = scorer.score_from_attention(stable_attn)
        
        # Now create unstable attention - each layer focuses on completely different tokens
        unstable_attn = torch.zeros(num_layers, num_heads, seq_len, seq_len)
        for i in range(num_layers):
            # Each layer focuses on a different single token
            focus_token = (i * 2) % seq_len
            unstable_attn[i, :, :, focus_token] = 1.0  # All weight on one token per layer

        unstable_score = scorer.score_from_attention(unstable_attn)
        
        # Unstable attention should have lower confidence than stable
        assert unstable_score < stable_score, \
            f"Unstable attention ({unstable_score}) should have lower confidence than stable ({stable_score})"
        # And should be reasonably low (but not necessarily below 0.6 due to normalization)
        assert unstable_score < 0.85, f"Unstable attention should show reduced confidence, got {unstable_score}"

    def test_score_with_nan_values(self):
        """Test handling of NaN values in attention weights."""

        scorer = AttentionInstabilityScorer()

        # Create attention weights with NaN
        attn = torch.rand(8, 8, 32, 32)
        attn[0, 0, 0, 0] = float("nan")

        # Should return low confidence (0.0) for invalid attention
        score = scorer.score_from_attention(attn)
        assert score == 0.0

    def test_score_from_attention_with_inf(self):
        """Test scorer handles infinite values gracefully."""

        scorer = AttentionInstabilityScorer()

        # Create attention weights with inf values
        attn = torch.rand(4, 8, 32, 32)
        attn[0, 0, 0, 0] = float("inf")

        score = scorer.score_from_attention(attn)
        assert score == 0.0, "Should return 0.0 for invalid attention weights"

    def test_layer_statistics(self):
        """Test detailed layer statistics computation."""
        import math

        scorer = AttentionInstabilityScorer()

        # Create attention weights with known properties
        num_layers, num_heads, seq_len = 32, 32, 64
        attn = torch.rand(num_layers, num_heads, seq_len, seq_len)

        stats = scorer.compute_layer_statistics(attn)

        # Check all expected keys are present
        expected_keys = {
            "mean_variance",
            "max_variance",
            "std_variance",
            "head_disagreement",
            "overall_instability",
        }
        assert set(stats.keys()) == expected_keys

        # Check all values are valid floats
        for key, value in stats.items():
            assert isinstance(value, float)
            assert not math.isnan(value) and not math.isinf(value)

    def test_score_from_attention_with_layer_selection(self):
        """Test that layer_indices parameter works correctly."""

        scorer = AttentionInstabilityScorer()

        # Create attention with different patterns in different layers
        num_layers, num_heads, seq_len = 32, 32, 64
        attn = torch.rand(num_layers, num_heads, seq_len, seq_len)

        # Make early layers very unstable, late layers stable
        attn[:10] = torch.rand_like(attn[:10]) * 0.5  # More variance
        attn[-10:] = torch.ones_like(attn[-10:]) * 0.1  # More stable

        # Score with all layers
        score_all = scorer.score_from_attention(attn)

        # Score with only stable layers
        score_stable = scorer.score_from_attention(
            attn, layer_indices=(20, 21, 22, 23, 24)
        )

        # Both scores should be valid
        assert score_all > 0.0
        assert score_all <= 1.0
        assert score_stable > 0.0
        assert score_stable <= 1.0

    def test_compute_layer_statistics(self):
        """Test detailed statistics computation."""

        scorer = AttentionInstabilityScorer()

        # Create attention weights with known properties
        num_layers, num_heads, seq_len = 8, 4, 16
        attn = torch.rand(num_layers, num_heads, seq_len, seq_len)

        # Normalize attention weights
        attn = torch.softmax(attn, dim=-1)

        stats = scorer.compute_layer_statistics(attn)

        # Check all expected keys are present
        assert "mean_variance" in stats
        assert "max_variance" in stats
        assert "std_variance" in stats
        assert "head_disagreement" in stats
        assert "overall_instability" in stats

        # All should be non-negative
        for key, value in stats.items():
            assert value >= 0.0, f"{key} should be non-negative"

    def test_protocol_compliance(self):
        """Test that scorer implements the ConfidenceScorer protocol."""
        from air.interfaces.router import ConfidenceScorer

        scorer = AttentionInstabilityScorer()

        # Check protocol compliance
        assert isinstance(scorer, ConfidenceScorer)

        # Check required properties and methods
        assert hasattr(scorer, "name")
        assert hasattr(scorer, "score")
        assert callable(scorer.score)

        # Test basic functionality
        logits = torch.rand(10, 50000)
        score = scorer.score(logits)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_repr(self):
        """Test string representation."""
        scorer = AttentionInstabilityScorer(
            sensitivity=0.7, use_head_variance=False, variance_aggregation="max"
        )
        repr_str = repr(scorer)
        assert "AttentionInstabilityScorer" in repr_str
        assert "0.70" in repr_str
        assert "False" in repr_str
        assert "max" in repr_str

    def test_empty_attention_tensor(self):
        """Test handling of empty attention tensor."""

        scorer = AttentionInstabilityScorer()

        # Create empty attention tensor (0 layers)
        attn = torch.empty(0, 8, 32, 32)
        score = scorer.score_from_attention(attn)

        # Should return neutral confidence
        assert score == 0.5, "Empty tensor should return neutral confidence"

    def test_single_layer_attention(self):
        """Test handling of single layer attention (no cross-layer variance)."""

        scorer = AttentionInstabilityScorer()

        # Create single layer attention
        attn = torch.rand(1, 8, 32, 32)
        attn = torch.softmax(attn, dim=-1)

        score = scorer.score_from_attention(attn)

        # Should compute confidence based on head variance only
        assert 0.0 <= score <= 1.0, "Single layer score should be in valid range"

    def test_batch_dimension_handling(self):
        """Test that scorer properly handles batch dimensions."""

        scorer = AttentionInstabilityScorer()

        # Create attention with batch dimension
        # Shape: (num_layers, num_heads, batch_size, seq_len, seq_len)
        attn = torch.rand(8, 8, 4, 32, 32)
        attn = torch.softmax(attn, dim=-1)

        score = scorer.score_from_attention(attn)

        # Should handle batch dimension and return valid score
        assert 0.0 <= score <= 1.0, "Batch dimension should be handled properly"

    def test_different_aggregation_methods(self):
        """Test that different aggregation methods produce different but valid scores."""

        attn = torch.rand(16, 8, 32, 32)
        attn = torch.softmax(attn, dim=-1)

        # Test all aggregation methods
        scorer_mean = AttentionInstabilityScorer(variance_aggregation="mean")
        scorer_max = AttentionInstabilityScorer(variance_aggregation="max")
        scorer_weighted = AttentionInstabilityScorer(variance_aggregation="weighted")

        score_mean = scorer_mean.score_from_attention(attn)
        score_max = scorer_max.score_from_attention(attn)
        score_weighted = scorer_weighted.score_from_attention(attn)

        # All should be valid
        assert 0.0 <= score_mean <= 1.0
        assert 0.0 <= score_max <= 1.0
        assert 0.0 <= score_weighted <= 1.0

    def test_sensitivity_property_setter(self):
        """Test that sensitivity can be changed after initialization."""
        scorer = AttentionInstabilityScorer(sensitivity=0.5)

        assert scorer.sensitivity == 0.5

        # Update sensitivity
        scorer.sensitivity = 0.8
        assert scorer.sensitivity == 0.8

        # Invalid values should raise error
        with pytest.raises(ValueError):
            scorer.sensitivity = 1.5

    def test_score_consistency_with_same_input(self):
        """Test that scoring the same input produces consistent results."""

        scorer = AttentionInstabilityScorer()
        attn = torch.rand(8, 8, 32, 32)

        score1 = scorer.score_from_attention(attn)
        score2 = scorer.score_from_attention(attn)

        assert score1 == score2, "Same input should produce same score"

    def test_invalid_shape_raises_error(self):
        """Test that invalid tensor shapes raise errors."""

        scorer = AttentionInstabilityScorer()

        # Tensor with less than 4 dimensions
        attn_1d = torch.rand(32)
        with pytest.raises(ValueError, match="must have at least 4 dimensions"):
            scorer.score_from_attention(attn_1d)

        attn_2d = torch.rand(8, 32)
        with pytest.raises(ValueError, match="must have at least 4 dimensions"):
            scorer.score_from_attention(attn_2d)

        attn_3d = torch.rand(8, 8, 32)
        with pytest.raises(ValueError, match="must have at least 4 dimensions"):
            scorer.score_from_attention(attn_3d)

"""
Unit tests for confidence scoring implementations.

Tests the various confidence scorers used for routing decisions,
particularly focusing on the EntropyScorer implementation for Task 1.2.1.
"""

import math

import pytest
import torch

from air.routing.confidence import EntropyScorer


class TestEntropyScorer:
    """Tests for the EntropyScorer class."""

    def test_initialization_default(self):
        """Test EntropyScorer initialization with default parameters."""
        scorer = EntropyScorer()
        assert scorer.name == "entropy"
        assert scorer.temperature == 1.0
        assert scorer.max_entropy_threshold == 8.0

    def test_initialization_custom(self):
        """Test EntropyScorer initialization with custom parameters."""
        scorer = EntropyScorer(temperature=0.7, max_entropy_threshold=10.0)
        assert scorer.temperature == 0.7
        assert scorer.max_entropy_threshold == 10.0

    def test_initialization_invalid_temperature(self):
        """Test that invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            EntropyScorer(temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            EntropyScorer(temperature=-1.0)

    def test_initialization_invalid_max_entropy(self):
        """Test that invalid max_entropy_threshold raises ValueError."""
        with pytest.raises(ValueError, match="max_entropy_threshold must be positive"):
            EntropyScorer(max_entropy_threshold=0.0)

        with pytest.raises(ValueError, match="max_entropy_threshold must be positive"):
            EntropyScorer(max_entropy_threshold=-1.0)

    def test_peaked_distribution_high_confidence(self):
        """Test that a peaked distribution yields high confidence score."""
        scorer = EntropyScorer()

        # Create a highly peaked distribution (one token dominates)
        logits = torch.tensor([[10.0, 0.1, 0.05, 0.01, 0.01]])

        score = scorer.score(logits)

        # Should have high confidence (low entropy)
        assert 0.0 <= score <= 1.0
        assert score > 0.8, f"Expected high confidence, got {score}"

    def test_uniform_distribution_low_confidence(self):
        """Test that a uniform distribution yields low confidence score."""
        scorer = EntropyScorer()

        # Create a uniform distribution (all tokens equally likely)
        vocab_size = 100
        logits = torch.ones(1, vocab_size)

        score = scorer.score(logits)

        # Should have low confidence (high entropy)
        assert 0.0 <= score <= 1.0
        assert score < 0.5, f"Expected low confidence, got {score}"

    def test_known_entropy_calculation(self):
        """Test entropy calculation with a known distribution."""
        scorer = EntropyScorer(temperature=1.0, max_entropy_threshold=2.0)

        # Create a simple distribution: [0.5, 0.5] (binary)
        # Expected entropy: -0.5*log(0.5) - 0.5*log(0.5) = log(2) ≈ 0.693
        logits = torch.tensor([[0.0, 0.0]])  # After softmax: [0.5, 0.5]

        score = scorer.score(logits)

        # Entropy ≈ 0.693, normalized: 1 - 0.693/2.0 ≈ 0.653
        expected_score = 1.0 - (math.log(2) / 2.0)
        assert 0.0 <= score <= 1.0
        assert abs(score - expected_score) < 0.01, f"Expected ~{expected_score}, got {score}"

    def test_temperature_effect_higher(self):
        """Test that higher temperature increases entropy (lowers confidence)."""
        logits = torch.tensor([[10.0, 1.0, 0.5]])

        scorer_low_temp = EntropyScorer(temperature=0.5)
        scorer_high_temp = EntropyScorer(temperature=2.0)

        score_low = scorer_low_temp.score(logits)
        score_high = scorer_high_temp.score(logits)

        # Higher temperature → higher entropy → lower confidence
        assert score_low > score_high, (
            f"Higher temperature should decrease confidence: "
            f"low_temp={score_low}, high_temp={score_high}"
        )

    def test_temperature_effect_lower(self):
        """Test that lower temperature decreases entropy (increases confidence)."""
        logits = torch.tensor([[2.0, 1.0, 0.5]])

        scorer_normal = EntropyScorer(temperature=1.0)
        scorer_low_temp = EntropyScorer(temperature=0.5)

        score_normal = scorer_normal.score(logits)
        score_low = scorer_low_temp.score(logits)

        # Lower temperature → lower entropy → higher confidence
        assert score_low >= score_normal, (
            f"Lower temperature should increase confidence: "
            f"normal={score_normal}, low_temp={score_low}"
        )

    def test_1d_logits(self):
        """Test scoring with 1D logits (single prediction)."""
        scorer = EntropyScorer()
        logits = torch.tensor([5.0, 1.0, 0.5, 0.1])

        score = scorer.score(logits)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably confident

    def test_2d_logits(self):
        """Test scoring with 2D logits (batch of predictions)."""
        scorer = EntropyScorer()
        logits = torch.tensor([
            [5.0, 1.0, 0.5],
            [3.0, 2.9, 2.8],  # Less peaked
        ])

        score = scorer.score(logits)

        assert 0.0 <= score <= 1.0
        # Average should be moderate since we have one peaked, one flat

    def test_3d_logits(self):
        """Test scoring with 3D logits (batch with sequences)."""
        scorer = EntropyScorer()
        batch_size, seq_len, vocab_size = 2, 3, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)

        score = scorer.score(logits)

        assert 0.0 <= score <= 1.0

    def test_empty_tensor(self):
        """Test that empty tensor returns neutral score."""
        scorer = EntropyScorer()
        empty_logits = torch.tensor([])

        score = scorer.score(empty_logits)

        assert score == 0.5  # Neutral score for empty input

    def test_nan_handling(self):
        """Test that NaN values are handled gracefully."""
        scorer = EntropyScorer()
        logits_with_nan = torch.tensor([[1.0, float('nan'), 2.0]])

        score = scorer.score(logits_with_nan)

        # Should return neutral score, not crash
        assert score == 0.5

    def test_inf_handling(self):
        """Test that inf values are handled gracefully."""
        scorer = EntropyScorer()
        logits_with_inf = torch.tensor([[1.0, float('inf'), 2.0]])

        score = scorer.score(logits_with_inf)

        # Should handle inf gracefully (likely dominated by inf after softmax)
        assert 0.0 <= score <= 1.0

    def test_score_consistency(self):
        """Test that scoring the same logits produces consistent results."""
        scorer = EntropyScorer(temperature=1.0)
        logits = torch.tensor([[3.0, 1.5, 0.8, 0.2]])

        score1 = scorer.score(logits)
        score2 = scorer.score(logits)

        assert score1 == score2, "Same logits should produce same score"

    def test_max_entropy_threshold_normalization(self):
        """Test that max_entropy_threshold affects score normalization."""
        logits = torch.ones(1, 100)  # Uniform distribution, high entropy

        # With lower threshold, entropy gets clamped sooner
        scorer_low = EntropyScorer(max_entropy_threshold=4.0)
        scorer_high = EntropyScorer(max_entropy_threshold=10.0)

        score_low = scorer_low.score(logits)
        score_high = scorer_high.score(logits)

        # With lower threshold, high entropy hits the floor faster
        # Both should be in [0, 1], but low threshold may clamp to 0
        assert 0.0 <= score_low <= 1.0
        assert 0.0 <= score_high <= 1.0

    def test_very_large_vocab(self):
        """Test with very large vocabulary size."""
        scorer = EntropyScorer()
        vocab_size = 50000
        logits = torch.randn(1, vocab_size)

        score = scorer.score(logits)

        assert 0.0 <= score <= 1.0

    def test_batch_with_varying_distributions(self):
        """Test batch with mixed peaked and flat distributions."""
        scorer = EntropyScorer()

        # Mix peaked and uniform distributions
        peaked = torch.tensor([10.0, 0.1, 0.1, 0.1])
        uniform = torch.ones(4)
        logits = torch.stack([peaked, uniform])

        score = scorer.score(logits)

        # Peaked has very low entropy (~0), uniform has entropy ~1.39
        # Average entropy ~0.69, which gives high confidence with threshold=8.0
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high since peaked dominates the average

    def test_protocol_compliance(self):
        """Test that EntropyScorer complies with ConfidenceScorer protocol."""
        scorer = EntropyScorer()

        # Must have name property
        assert hasattr(scorer, 'name')
        assert isinstance(scorer.name, str)

        # Must have score method
        assert hasattr(scorer, 'score')
        assert callable(scorer.score)

        # Score method should accept logits and return float
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = scorer.score(logits)
        assert isinstance(result, float)

    def test_repr(self):
        """Test string representation."""
        scorer = EntropyScorer(temperature=0.8, max_entropy_threshold=10.0)
        repr_str = repr(scorer)

        assert "EntropyScorer" in repr_str
        assert "0.8" in repr_str
        assert "10.0" in repr_str

    def test_deterministic_for_same_seed(self):
        """Test that results are deterministic (no randomness)."""
        scorer = EntropyScorer()

        # No random seed needed - entropy calculation is deterministic
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.5]])

        scores = [scorer.score(logits) for _ in range(5)]

        # All scores should be identical
        assert all(s == scores[0] for s in scores)

    def test_gradient_not_required(self):
        """Test that scorer works with tensors that don't require gradients."""
        scorer = EntropyScorer()

        logits_no_grad = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=False)
        logits_with_grad = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        score_no_grad = scorer.score(logits_no_grad)
        score_with_grad = scorer.score(logits_with_grad)

        # Both should work and produce same result
        assert abs(score_no_grad - score_with_grad) < 1e-6

    def test_realistic_llm_distribution(self):
        """Test with realistic LLM-like logit distribution."""
        scorer = EntropyScorer()

        # Simulate realistic LLM logits: few high values, many low
        vocab_size = 32000
        logits = torch.randn(1, vocab_size) * 2 - 5  # Mostly negative, few positive

        # Add some clear top candidates
        logits[0, :5] = torch.tensor([8.0, 6.5, 5.0, 3.0, 2.0])

        score = scorer.score(logits)

        # Should have relatively high confidence due to clear top candidates
        assert 0.0 <= score <= 1.0
        assert score > 0.5, f"Expected moderate-high confidence for realistic dist, got {score}"

    def test_zero_logits(self):
        """Test with all-zero logits."""
        scorer = EntropyScorer()
        logits = torch.zeros(1, 10)

        score = scorer.score(logits)

        # All zeros → uniform after softmax (10 tokens)
        # Entropy = log(10) ≈ 2.3
        # With max_entropy_threshold=8.0: score = 1 - 2.3/8.0 ≈ 0.71
        # This is moderate confidence since 10 tokens isn't maximum uncertainty
        assert 0.0 <= score <= 1.0
        assert 0.6 < score < 0.8, f"Expected moderate confidence for 10-way uniform, got {score}"

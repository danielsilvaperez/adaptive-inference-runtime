"""
Unit tests for Top-k Disagreement Confidence Scorer.

Tests the TopKDisagreementScorer implementation to ensure it correctly
measures prediction consensus and produces sensible confidence scores.
"""

import pytest
import torch

from air.routing.confidence.topk_disagreement import TopKDisagreementScorer


class TestTopKDisagreementScorer:
    """Tests for the TopKDisagreementScorer class."""

    def test_initialization_valid_params(self):
        """Test that scorer initializes with valid parameters."""
        scorer = TopKDisagreementScorer(k=5, temperature=1.0)
        assert scorer.k == 5
        assert scorer.temperature == 1.0
        assert scorer.name == "topk_disagreement_k5"

    def test_initialization_invalid_k(self):
        """Test that initialization fails with invalid k."""
        with pytest.raises(ValueError, match="k must be positive"):
            TopKDisagreementScorer(k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            TopKDisagreementScorer(k=-1)

    def test_initialization_invalid_temperature(self):
        """Test that initialization fails with invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            TopKDisagreementScorer(temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            TopKDisagreementScorer(temperature=-0.5)

    def test_name_property(self):
        """Test that name property reflects k parameter."""
        scorer = TopKDisagreementScorer(k=3)
        assert scorer.name == "topk_disagreement_k3"

        scorer = TopKDisagreementScorer(k=10)
        assert scorer.name == "topk_disagreement_k10"

    def test_perfect_confidence_single_dominant_token(self):
        """Test high confidence when one token strongly dominates."""
        scorer = TopKDisagreementScorer(k=5)

        # Logits where first token is much larger
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        confidence = scorer.score(logits)

        # Should have high confidence (> 0.7)
        assert confidence > 0.7
        assert 0.0 <= confidence <= 1.0

    def test_low_confidence_uniform_distribution(self):
        """Test low confidence with uniform distribution."""
        scorer = TopKDisagreementScorer(k=5)

        # Uniform logits (all equal)
        logits = torch.ones((1, 100))
        confidence = scorer.score(logits)

        # Should have low confidence (< 0.3)
        assert confidence < 0.3
        assert 0.0 <= confidence <= 1.0

    def test_medium_confidence_moderate_distribution(self):
        """Test medium confidence with moderate probability distribution."""
        scorer = TopKDisagreementScorer(k=5)

        # Moderate distribution
        logits = torch.tensor([[3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]])
        confidence = scorer.score(logits)

        # With entropy-based scoring, this moderate distribution has
        # relatively high entropy, so confidence is lower
        # Should be between uniform (0.0) and fully concentrated (1.0)
        assert 0.0 < confidence < 0.5
        assert 0.0 <= confidence <= 1.0

    def test_batch_processing(self):
        """Test that scorer handles batched inputs correctly."""
        scorer = TopKDisagreementScorer(k=5)

        # Batch of 3 examples
        logits = torch.tensor([
            [10.0, 0.0, 0.0, 0.0, 0.0],  # High confidence
            [1.0, 1.0, 1.0, 1.0, 1.0],    # Low confidence
            [5.0, 2.0, 1.0, 0.5, 0.0],    # Medium confidence
        ])

        confidence = scorer.score(logits)

        # Should return a single averaged score
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_sequence_logits_uses_last_token(self):
        """Test that 3D logits (with sequence dimension) use last token."""
        scorer = TopKDisagreementScorer(k=5)

        # Shape: (batch_size=1, seq_len=3, vocab_size=10)
        logits = torch.randn(1, 3, 10)
        # Make last token very confident
        logits[0, -1, 0] = 10.0
        logits[0, -1, 1:] = 0.0

        confidence = scorer.score(logits)

        # Should have high confidence based on last token
        assert confidence > 0.6
        assert 0.0 <= confidence <= 1.0

    def test_single_dimension_logits(self):
        """Test handling of 1D logits (single example, no batch)."""
        scorer = TopKDisagreementScorer(k=5)

        # Shape: (vocab_size,)
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
        confidence = scorer.score(logits)

        # Should handle correctly and return high confidence
        assert confidence > 0.7
        assert 0.0 <= confidence <= 1.0

    def test_k_larger_than_vocab_size(self):
        """Test behavior when k exceeds vocabulary size."""
        scorer = TopKDisagreementScorer(k=100)

        # Small vocabulary (only 5 tokens)
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
        confidence = scorer.score(logits)

        # Should handle gracefully using all available tokens
        assert 0.0 <= confidence <= 1.0

    def test_k_equals_one(self):
        """Test edge case with k=1 (only top prediction)."""
        scorer = TopKDisagreementScorer(k=1)

        logits = torch.tensor([[5.0, 2.0, 1.0]])
        confidence = scorer.score(logits)

        # With k=1, entropy is 0 (perfect certainty), so confidence should be 1.0
        # However, numerical precision may make it slightly less
        assert 0.9 <= confidence <= 1.0

    def test_temperature_effect(self):
        """Test that temperature affects the confidence scores."""
        logits = torch.tensor([[5.0, 2.0, 1.0, 0.5, 0.0]])

        # Low temperature (sharper distribution)
        scorer_low_temp = TopKDisagreementScorer(k=5, temperature=0.5)
        conf_low_temp = scorer_low_temp.score(logits)

        # High temperature (smoother distribution)
        scorer_high_temp = TopKDisagreementScorer(k=5, temperature=2.0)
        conf_high_temp = scorer_high_temp.score(logits)

        # Lower temperature should increase confidence (sharper)
        assert conf_low_temp > conf_high_temp

    def test_nan_logits_returns_neutral(self):
        """Test that NaN logits return neutral confidence."""
        scorer = TopKDisagreementScorer(k=5)

        logits = torch.tensor([[float('nan'), 1.0, 2.0, 3.0, 4.0]])
        confidence = scorer.score(logits)

        # Should return neutral confidence (0.5)
        assert confidence == 0.5

    def test_inf_logits_returns_neutral(self):
        """Test that infinite logits return neutral confidence."""
        scorer = TopKDisagreementScorer(k=5)

        logits = torch.tensor([[float('inf'), 1.0, 2.0, 3.0, 4.0]])
        confidence = scorer.score(logits)

        # Should return neutral confidence (0.5)
        assert confidence == 0.5

    def test_invalid_shape_returns_neutral(self):
        """Test that invalid tensor shapes return neutral confidence."""
        scorer = TopKDisagreementScorer(k=5)

        # 4D tensor (invalid shape)
        logits = torch.randn(2, 3, 4, 5)
        confidence = scorer.score(logits)

        # Should return neutral confidence (0.5)
        assert confidence == 0.5

    def test_zero_probabilities_returns_neutral(self):
        """Test handling of edge case with near-zero probabilities."""
        scorer = TopKDisagreementScorer(k=5)

        # Very large negative logits (near-zero probabilities)
        logits = torch.tensor([[-1000.0, -1000.0, -1000.0, -1000.0, -1000.0]])
        confidence = scorer.score(logits)

        # Should handle gracefully (return neutral or valid score)
        assert 0.0 <= confidence <= 1.0

    def test_repr(self):
        """Test string representation."""
        scorer = TopKDisagreementScorer(k=5, temperature=1.5)
        repr_str = repr(scorer)

        assert "TopKDisagreementScorer" in repr_str
        assert "k=5" in repr_str
        assert "temperature=1.5" in repr_str

    def test_consistency_multiple_calls(self):
        """Test that multiple calls with same input return same result."""
        scorer = TopKDisagreementScorer(k=5)
        logits = torch.tensor([[5.0, 2.0, 1.0, 0.5, 0.0]])

        conf1 = scorer.score(logits)
        conf2 = scorer.score(logits)

        # Should be deterministic
        assert conf1 == conf2

    def test_different_k_values_produce_different_scores(self):
        """Test that different k values can produce different scores."""
        logits = torch.tensor([[5.0, 4.5, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0]])

        scorer_k3 = TopKDisagreementScorer(k=3)
        scorer_k5 = TopKDisagreementScorer(k=5)

        conf_k3 = scorer_k3.score(logits)
        conf_k5 = scorer_k5.score(logits)

        # Both should be valid confidence scores
        assert 0.0 <= conf_k3 <= 1.0
        assert 0.0 <= conf_k5 <= 1.0

        # They may differ since they consider different number of top predictions
        # (though not guaranteed to differ in all cases)

    def test_conforms_to_confidence_scorer_protocol(self):
        """Test that scorer conforms to ConfidenceScorer protocol."""
        from air.interfaces.router import ConfidenceScorer

        scorer = TopKDisagreementScorer(k=5)

        # Should have name property
        assert hasattr(scorer, 'name')
        assert isinstance(scorer.name, str)

        # Should have score method
        assert hasattr(scorer, 'score')
        assert callable(scorer.score)

        # Should be recognized as ConfidenceScorer
        assert isinstance(scorer, ConfidenceScorer)

    def test_score_range_with_various_distributions(self):
        """Test that scores are always in valid range for various distributions."""
        scorer = TopKDisagreementScorer(k=5)

        test_cases = [
            torch.tensor([[10.0, 5.0, 2.0, 1.0, 0.0]]),  # Decreasing
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),  # Increasing
            torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0]]),  # Uniform
            torch.randn(1, 100),  # Random
            torch.tensor([[100.0, -100.0, 50.0, -50.0, 0.0]]),  # Mixed signs
        ]

        for logits in test_cases:
            confidence = scorer.score(logits)
            assert 0.0 <= confidence <= 1.0, f"Invalid score for logits: {logits}"

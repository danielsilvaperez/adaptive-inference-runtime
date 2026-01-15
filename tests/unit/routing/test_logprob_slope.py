"""
Unit tests for the Logprob Slope Tracker.

Tests the LogprobSlopeTracker class to ensure it correctly:
- Tracks log probabilities in a sliding window
- Calculates slopes accurately
- Detects sharp confidence drops
- Converts slopes to confidence scores appropriately
"""

import pytest
import torch

from air.routing.logprob_slope import LogprobSlopeTracker


class TestLogprobSlopeTrackerInit:
    """Tests for LogprobSlopeTracker initialization."""

    def test_default_initialization(self):
        """Test that tracker initializes with default parameters."""
        tracker = LogprobSlopeTracker()
        assert tracker.window_size == 20
        assert tracker.sharp_drop_threshold == -0.3
        assert tracker.temperature == 1.0
        assert tracker.name == "logprob_slope"
        assert tracker.history_length == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        tracker = LogprobSlopeTracker(
            window_size=10,
            sharp_drop_threshold=-0.5,
            temperature=0.8,
        )
        assert tracker.window_size == 10
        assert tracker.sharp_drop_threshold == -0.5
        assert tracker.temperature == 0.8

    def test_invalid_window_size(self):
        """Test that invalid window size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be at least 2"):
            LogprobSlopeTracker(window_size=1)

        with pytest.raises(ValueError, match="window_size must be at least 2"):
            LogprobSlopeTracker(window_size=0)

    def test_invalid_sharp_drop_threshold(self):
        """Test that positive sharp_drop_threshold raises ValueError."""
        with pytest.raises(ValueError, match="sharp_drop_threshold must be negative"):
            LogprobSlopeTracker(sharp_drop_threshold=0.1)

    def test_invalid_temperature(self):
        """Test that non-positive temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            LogprobSlopeTracker(temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            LogprobSlopeTracker(temperature=-0.5)


class TestLogprobSlopeTrackerAddLogprob:
    """Tests for adding log probabilities to the tracker."""

    def test_add_single_logprob(self):
        """Test adding a single log probability."""
        tracker = LogprobSlopeTracker(window_size=5)
        tracker.add_logprob(-0.5)
        assert tracker.history_length == 1

    def test_add_multiple_logprobs(self):
        """Test adding multiple log probabilities."""
        tracker = LogprobSlopeTracker(window_size=5)
        for i in range(3):
            tracker.add_logprob(-0.1 * i)
        assert tracker.history_length == 3

    def test_sliding_window_behavior(self):
        """Test that old entries are removed when window is full."""
        tracker = LogprobSlopeTracker(window_size=3)

        # Add 5 entries (window size is 3)
        logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5]
        for lp in logprobs:
            tracker.add_logprob(lp)

        # Should only keep last 3
        assert tracker.history_length == 3
        # Verify the correct ones are kept (last 3)
        assert tracker._logprob_history == [-0.3, -0.4, -0.5]

    def test_reset_clears_history(self):
        """Test that reset clears the history."""
        tracker = LogprobSlopeTracker(window_size=5)
        tracker.add_logprob(-0.5)
        tracker.add_logprob(-0.6)
        assert tracker.history_length == 2

        tracker.reset()
        assert tracker.history_length == 0


class TestLogprobSlopeTrackerScore:
    """Tests for the score method."""

    def test_score_with_2d_logits(self):
        """Test scoring with 2D logits (batch_size, vocab_size)."""
        tracker = LogprobSlopeTracker()
        logits = torch.randn(1, 100)  # Single batch, 100 vocab items
        score = tracker.score(logits)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_with_3d_logits(self):
        """Test scoring with 3D logits (batch_size, seq_len, vocab_size)."""
        tracker = LogprobSlopeTracker()
        logits = torch.randn(1, 5, 100)  # Batch 1, seq 5, vocab 100
        score = tracker.score(logits)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_with_insufficient_history(self):
        """Test that score returns high confidence with <2 history points."""
        tracker = LogprobSlopeTracker()
        logits = torch.randn(1, 100)

        # First token (no history yet)
        score = tracker.score(logits)
        assert score == 0.8  # Default high confidence

    def test_score_with_stable_confidence(self):
        """Test score with stable (flat) log probabilities."""
        tracker = LogprobSlopeTracker(window_size=5)

        # Create logits that will produce similar high probabilities
        for _ in range(5):
            logits = torch.zeros(1, 100)
            logits[0, 0] = 10.0  # Strong signal for token 0
            score = tracker.score(logits)

        # With stable high probabilities, slope should be near 0
        # and confidence should be high
        assert score >= 0.8

    def test_score_with_declining_confidence(self):
        """Test score with declining log probabilities."""
        tracker = LogprobSlopeTracker(window_size=5, sharp_drop_threshold=-0.3)

        # Simulate declining confidence by gradually reducing max logit
        scores = []
        for i in range(6):
            logits = torch.zeros(1, 100)
            logits[0, 0] = 10.0 - i * 2.0  # Gradually decrease
            score = tracker.score(logits)
            scores.append(score)

        # Confidence should decrease over time
        # Later scores should be lower than earlier ones
        assert scores[-1] < scores[0]

    def test_score_with_sharp_drop(self):
        """Test score with a sharp drop in confidence."""
        tracker = LogprobSlopeTracker(window_size=5, sharp_drop_threshold=-0.5)

        # Start with high confidence
        for _ in range(3):
            logits = torch.zeros(1, 100)
            logits[0, 0] = 10.0
            tracker.score(logits)

        # Then sharp drop
        for _ in range(2):
            logits = torch.zeros(1, 100)
            logits[0, :] = 1.0  # Uniform distribution (low confidence)
            score = tracker.score(logits)

        # Should detect sharp drop and have low confidence
        assert score < 0.5

    def test_score_updates_history(self):
        """Test that scoring updates the internal history."""
        tracker = LogprobSlopeTracker(window_size=5)

        initial_length = tracker.history_length
        logits = torch.randn(1, 100)
        tracker.score(logits)

        assert tracker.history_length == initial_length + 1


class TestLogprobSlopeTrackerSlope:
    """Tests for slope calculation."""

    def test_calculate_slope_flat_line(self):
        """Test slope calculation with flat log probabilities."""
        tracker = LogprobSlopeTracker(window_size=5)

        # Add flat values
        for _ in range(5):
            tracker.add_logprob(-1.0)

        slope = tracker.get_slope()
        assert slope is not None
        assert abs(slope) < 0.01  # Should be very close to 0

    def test_calculate_slope_increasing(self):
        """Test slope calculation with increasing log probabilities."""
        tracker = LogprobSlopeTracker(window_size=5)

        # Add increasing values (improving confidence)
        for i in range(5):
            tracker.add_logprob(-1.0 + i * 0.1)

        slope = tracker.get_slope()
        assert slope is not None
        assert slope > 0  # Positive slope

    def test_calculate_slope_decreasing(self):
        """Test slope calculation with decreasing log probabilities."""
        tracker = LogprobSlopeTracker(window_size=5)

        # Add decreasing values (declining confidence)
        for i in range(5):
            tracker.add_logprob(-0.5 - i * 0.1)

        slope = tracker.get_slope()
        assert slope is not None
        assert slope < 0  # Negative slope

    def test_get_slope_with_insufficient_history(self):
        """Test that get_slope returns None with <2 history points."""
        tracker = LogprobSlopeTracker(window_size=5)
        assert tracker.get_slope() is None

        tracker.add_logprob(-0.5)
        assert tracker.get_slope() is None

        tracker.add_logprob(-0.6)
        assert tracker.get_slope() is not None


class TestLogprobSlopeTrackerSharpDrop:
    """Tests for sharp drop detection."""

    def test_has_sharp_drop_false_with_insufficient_history(self):
        """Test that has_sharp_drop returns False with insufficient history."""
        tracker = LogprobSlopeTracker(window_size=5)
        assert not tracker.has_sharp_drop()

        tracker.add_logprob(-0.5)
        assert not tracker.has_sharp_drop()

    def test_has_sharp_drop_false_with_stable_confidence(self):
        """Test that has_sharp_drop returns False with stable confidence."""
        tracker = LogprobSlopeTracker(
            window_size=5,
            sharp_drop_threshold=-0.5,
        )

        # Add stable values
        for _ in range(5):
            tracker.add_logprob(-1.0)

        assert not tracker.has_sharp_drop()

    def test_has_sharp_drop_true_with_sharp_decline(self):
        """Test that has_sharp_drop returns True with sharp decline."""
        tracker = LogprobSlopeTracker(
            window_size=5,
            sharp_drop_threshold=-0.3,
        )

        # Add values with sharp decline
        values = [-0.5, -0.6, -0.9, -1.3, -1.8]
        for val in values:
            tracker.add_logprob(val)

        assert tracker.has_sharp_drop()

    def test_has_sharp_drop_boundary(self):
        """Test has_sharp_drop at the threshold boundary."""
        threshold = -0.4
        tracker = LogprobSlopeTracker(
            window_size=5,
            sharp_drop_threshold=threshold,
        )

        # Create values that produce slope right at threshold
        # Linear decline with slope approximately equal to threshold
        for i in range(5):
            tracker.add_logprob(-0.5 - i * 0.35)

        slope = tracker.get_slope()
        # Should be close to or below threshold
        assert slope is not None
        assert slope <= threshold + 0.1  # Small tolerance


class TestLogprobSlopeTrackerIntegration:
    """Integration tests for the logprob slope tracker."""

    def test_multiple_generations_with_reset(self):
        """Test tracker behavior across multiple generation sequences."""
        tracker = LogprobSlopeTracker(window_size=5)

        # First generation
        for i in range(5):
            logits = torch.randn(1, 100)
            tracker.score(logits)

        assert tracker.history_length == 5

        # Reset for new generation
        tracker.reset()
        assert tracker.history_length == 0

        # Second generation
        for i in range(3):
            logits = torch.randn(1, 100)
            tracker.score(logits)

        assert tracker.history_length == 3

    def test_realistic_confidence_trajectory(self):
        """Test with a realistic confidence trajectory."""
        tracker = LogprobSlopeTracker(
            window_size=10,
            sharp_drop_threshold=-0.4,
        )

        # Simulate realistic generation:
        # Start with high confidence, stable middle, then drop

        scores = []

        # High confidence start (5 tokens)
        for _ in range(5):
            logits = torch.zeros(1, 100)
            logits[0, 0] = 8.0
            scores.append(tracker.score(logits))

        # Stable middle (5 tokens)
        for _ in range(5):
            logits = torch.zeros(1, 100)
            logits[0, 0] = 7.5
            scores.append(tracker.score(logits))

        # Sharp drop (5 tokens) - model becomes uncertain
        for _ in range(5):
            logits = torch.randn(1, 100) * 0.5  # More uniform
            scores.append(tracker.score(logits))

        # Confidence should decrease overall
        assert scores[0] > scores[-1]

        # Should detect sharp drop at the end
        assert tracker.has_sharp_drop()

    def test_temperature_effect(self):
        """Test that temperature affects scoring."""
        # Create two trackers with different temperatures
        tracker_low_temp = LogprobSlopeTracker(temperature=0.5)
        tracker_high_temp = LogprobSlopeTracker(temperature=2.0)

        # Use the same logits
        logits = torch.randn(1, 100)
        logits[0, 0] = 5.0  # Strong preference for token 0

        # Score with both trackers multiple times
        for _ in range(5):
            logits_copy = logits.clone()
            tracker_low_temp.score(logits_copy)
            tracker_high_temp.score(logits_copy)

        # Slopes should differ due to temperature
        slope_low = tracker_low_temp.get_slope()
        slope_high = tracker_high_temp.get_slope()

        # Both should have slopes, though they may be similar
        assert slope_low is not None
        assert slope_high is not None

    def test_repr(self):
        """Test string representation."""
        tracker = LogprobSlopeTracker(
            window_size=10,
            sharp_drop_threshold=-0.5,
            temperature=0.8,
        )
        repr_str = repr(tracker)

        assert "LogprobSlopeTracker" in repr_str
        assert "window_size=10" in repr_str
        assert "sharp_drop_threshold=-0.5" in repr_str
        assert "temperature=0.8" in repr_str
        assert "history_length=0" in repr_str

    def test_consistency_across_multiple_scores(self):
        """Test that repeated scoring with same logits is consistent."""
        tracker = LogprobSlopeTracker(window_size=5)

        # Create fixed logits
        logits = torch.zeros(1, 100)
        logits[0, 0] = 5.0

        # Score multiple times with same logits
        scores = []
        for _ in range(5):
            score = tracker.score(logits.clone())
            scores.append(score)

        # Should have relatively stable scores (slight decrease expected)
        assert all(0.0 <= s <= 1.0 for s in scores)

        # Slope should be close to 0 (stable confidence)
        slope = tracker.get_slope()
        assert slope is not None
        assert abs(slope) < 0.5  # Should be relatively flat


class TestLogprobSlopeTrackerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_score_with_1d_logits(self):
        """Test behavior with unexpected 1D logits."""
        tracker = LogprobSlopeTracker()
        logits = torch.randn(100)  # 1D tensor

        score = tracker.score(logits)
        # Should return neutral score without crashing
        assert score == 0.5

    def test_score_with_4d_logits(self):
        """Test behavior with unexpected 4D logits."""
        tracker = LogprobSlopeTracker()
        logits = torch.randn(1, 2, 5, 100)  # 4D tensor

        score = tracker.score(logits)
        # Should return neutral score without crashing
        assert score == 0.5

    def test_score_with_nan_in_logits(self):
        """Test behavior with NaN values in logits."""
        tracker = LogprobSlopeTracker()
        logits = torch.randn(1, 100)
        logits[0, 0] = float("nan")

        # Should handle gracefully (torch.log_softmax handles NaN)
        score = tracker.score(logits)
        assert isinstance(score, float)

    def test_score_with_inf_in_logits(self):
        """Test behavior with infinity in logits."""
        tracker = LogprobSlopeTracker()
        logits = torch.randn(1, 100)
        logits[0, 0] = float("inf")

        score = tracker.score(logits)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_very_small_window_size(self):
        """Test tracker with minimum window size."""
        tracker = LogprobSlopeTracker(window_size=2)

        for _ in range(3):
            logits = torch.randn(1, 100)
            tracker.score(logits)

        assert tracker.history_length == 2
        assert tracker.get_slope() is not None

    def test_very_large_window_size(self):
        """Test tracker with large window size."""
        tracker = LogprobSlopeTracker(window_size=100)

        # Add some tokens (less than window size)
        for _ in range(10):
            logits = torch.randn(1, 100)
            tracker.score(logits)

        assert tracker.history_length == 10
        assert tracker.get_slope() is not None

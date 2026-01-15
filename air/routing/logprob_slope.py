"""
Logprob Slope Tracker for confidence estimation.

This module implements logprob slope tracking for detecting confidence drops
in token generation. It monitors the trajectory of log probabilities over a
sliding window to identify uncertainty spikes and declining confidence patterns.

The slope tracker is a key component of the routing system, helping determine
when to escalate from a small model to a large model based on confidence drops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from air.interfaces.router import BaseConfidenceScorer

if TYPE_CHECKING:
    from air.types import Logits


class LogprobSlopeTracker(BaseConfidenceScorer):
    """
    Tracks log probability slopes to detect confidence drops.

    This scorer analyzes the trajectory of log probabilities over a sliding
    window of recent tokens. It computes the slope (rate of change) of
    confidence and detects sharp drops that indicate uncertainty spikes.

    A negative slope indicates declining confidence, which may trigger
    escalation to a larger model. The magnitude of the slope indicates
    how quickly confidence is dropping.

    Key features:
    - Sliding window implementation for recent history
    - Linear regression-based slope calculation
    - Sharp drop detection for uncertainty spikes
    - Configurable window size and sensitivity

    Attributes:
        window_size: Number of recent tokens to track for slope calculation.
        sharp_drop_threshold: Threshold for detecting sharp confidence drops.
            More negative values indicate steeper drops.
        temperature: Temperature parameter for softmax normalization.

    Example:
        >>> tracker = LogprobSlopeTracker(window_size=10)
        >>> logits = torch.randn(1, 32000)  # Mock logits
        >>> confidence = tracker.score(logits)
        >>> confidence  # Returns value in [0.0, 1.0]
        0.75
    """

    def __init__(
        self,
        window_size: int = 20,
        sharp_drop_threshold: float = -0.3,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize the logprob slope tracker.

        Args:
            window_size: Number of recent tokens to track. Larger windows
                provide more stable slope estimates but are less responsive
                to recent changes. Default: 20.
            sharp_drop_threshold: Threshold for detecting sharp drops in
                log probability. More negative values detect steeper drops.
                Typical range: [-1.0, 0.0]. Default: -0.3.
            temperature: Temperature for softmax normalization. Higher values
                smooth the distribution. Default: 1.0.

        Raises:
            ValueError: If window_size < 2, sharp_drop_threshold > 0, or
                temperature <= 0.
        """
        if window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {window_size}")
        if sharp_drop_threshold > 0.0:
            raise ValueError(f"sharp_drop_threshold must be negative, got {sharp_drop_threshold}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self._window_size = window_size
        self._sharp_drop_threshold = sharp_drop_threshold
        self._temperature = temperature
        self._logprob_history: list[float] = []

    @property
    def name(self) -> str:
        """Get the name of this confidence scorer."""
        return "logprob_slope"

    @property
    def window_size(self) -> int:
        """Get the window size for tracking."""
        return self._window_size

    @property
    def sharp_drop_threshold(self) -> float:
        """Get the sharp drop threshold."""
        return self._sharp_drop_threshold

    @property
    def temperature(self) -> float:
        """Get the temperature parameter."""
        return self._temperature

    @property
    def history_length(self) -> int:
        """Get the current length of the logprob history."""
        return len(self._logprob_history)

    def add_logprob(self, logprob: float) -> None:
        """
        Add a log probability to the tracking window.

        This method should be called after each token generation to
        maintain the sliding window of recent log probabilities.

        Args:
            logprob: The log probability of the most recent token.
                Should be in range (-inf, 0].
        """
        self._logprob_history.append(logprob)

        # Maintain sliding window by removing oldest entries
        if len(self._logprob_history) > self._window_size:
            self._logprob_history.pop(0)

    def score(self, logits: Logits) -> float:
        """
        Compute a confidence score from logits based on slope trajectory.

        This method computes the log probability of the most likely token,
        adds it to the history, and calculates a confidence score based on
        the slope of recent log probabilities.

        The score is in [0.0, 1.0] where:
        - 1.0 indicates stable or increasing confidence (positive/zero slope)
        - 0.0 indicates rapidly declining confidence (steep negative slope)
        - Values between reflect the magnitude of confidence decline

        Args:
            logits: The model output logits. Shape should be
                (batch_size, vocab_size) or (batch_size, seq_len, vocab_size).
                If 3D, uses the last sequence position.

        Returns:
            A confidence score in [0.0, 1.0] based on logprob slope.

        Note:
            If the history has fewer than 2 tokens, returns a default score
            of 0.8 (assuming high confidence initially).
        """
        # Handle different logits shapes
        if logits.dim() == 3:
            # Shape: (batch_size, seq_len, vocab_size)
            # Take the last position
            logits = logits[:, -1, :]
        elif logits.dim() == 2:
            # Shape: (batch_size, vocab_size)
            pass
        else:
            # Unexpected shape, return neutral score
            return 0.5

        # Apply temperature and compute probabilities
        if self._temperature != 1.0:
            logits = logits / self._temperature

        # Compute log probabilities using log_softmax for numerical stability
        log_probs = torch.log_softmax(logits, dim=-1)

        # Get the log probability of the most likely token
        max_logprob = log_probs.max(dim=-1).values.item()

        # Add to history
        self.add_logprob(max_logprob)

        # Need at least 2 points to compute slope
        if len(self._logprob_history) < 2:
            return 0.8  # High confidence by default for early tokens

        # Calculate slope using linear regression
        slope = self._calculate_slope()

        # Convert slope to confidence score
        # Positive or zero slope -> high confidence (1.0)
        # Negative slope -> lower confidence based on magnitude
        confidence = self._slope_to_confidence(slope)

        return confidence

    def _calculate_slope(self) -> float:
        """
        Calculate the slope of log probabilities using linear regression.

        Uses ordinary least squares regression to fit a line through the
        recent log probability history and returns the slope.

        Returns:
            The slope of the fitted line. Negative values indicate
            declining confidence.
        """
        n = len(self._logprob_history)
        if n < 2:
            return 0.0

        # X values: 0, 1, 2, ..., n-1 (time steps)
        x = torch.arange(n, dtype=torch.float32)
        y = torch.tensor(self._logprob_history, dtype=torch.float32)

        # Calculate slope using least squares: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope.item()

    def _slope_to_confidence(self, slope: float) -> float:
        """
        Convert slope value to confidence score.

        Maps the slope of log probabilities to a confidence score in [0.0, 1.0].
        Uses a sigmoid-like transformation to smoothly map slopes to confidence.

        Args:
            slope: The slope of the log probability trajectory.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        # Positive or zero slope -> high confidence
        if slope >= 0:
            return 1.0

        # Normalize slope relative to sharp drop threshold
        # If slope is at or below threshold, confidence is very low
        normalized_slope = slope / self._sharp_drop_threshold

        # Clamp to [0, 1] and invert (larger negative slope -> lower confidence)
        confidence = max(0.0, min(1.0, 1.0 - normalized_slope))

        return confidence

    def has_sharp_drop(self) -> bool:
        """
        Check if there's a sharp drop in confidence.

        Returns:
            True if the current slope indicates a sharp drop in confidence
            (slope is below the sharp drop threshold).
        """
        if len(self._logprob_history) < 2:
            return False

        slope = self._calculate_slope()
        return slope < self._sharp_drop_threshold

    def get_slope(self) -> float | None:
        """
        Get the current slope of log probabilities.

        Returns:
            The current slope value, or None if insufficient history.
        """
        if len(self._logprob_history) < 2:
            return None
        return self._calculate_slope()

    def reset(self) -> None:
        """
        Reset the tracker, clearing all history.

        This should be called when starting a new generation sequence
        or when model switches occur.
        """
        self._logprob_history.clear()

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"LogprobSlopeTracker(window_size={self._window_size}, "
            f"sharp_drop_threshold={self._sharp_drop_threshold}, "
            f"temperature={self._temperature}, "
            f"history_length={self.history_length})"
        )

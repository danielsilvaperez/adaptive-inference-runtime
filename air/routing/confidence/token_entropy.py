"""
Confidence scoring implementations for model routing decisions.

This module provides various confidence scorers that analyze model outputs
(logits) to estimate confidence levels. These scores are used by the router
to decide when to escalate from small to large models.

Confidence scorers implement the ConfidenceScorer protocol defined in
air.interfaces.router and provide normalized scores in [0.0, 1.0] where:
- 0.0 indicates very low confidence (high uncertainty)
- 1.0 indicates very high confidence (low uncertainty)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from air.interfaces.router import BaseConfidenceScorer

if TYPE_CHECKING:
    from air.types import Logits


class EntropyScorer(BaseConfidenceScorer):
    """
    Confidence scorer based on Shannon entropy of the token distribution.

    Computes the Shannon entropy of the softmax probability distribution
    over tokens. High entropy indicates uncertainty (many tokens have
    similar probabilities), while low entropy indicates confidence
    (one or few tokens dominate the distribution).

    The entropy is normalized to a confidence score in [0.0, 1.0] where:
    - High entropy (uncertainty) → low confidence score
    - Low entropy (peaked distribution) → high confidence score

    Attributes:
        temperature: Temperature parameter for softmax. Higher values
            increase randomness and entropy. Default: 1.0.
            Range: (0.0, inf), typical values [0.5, 2.0]
        max_entropy_threshold: Maximum entropy value used for normalization.
            Default: 8.0 (reasonable for typical vocabulary sizes).
            This can be estimated as log(vocab_size) for perfect normalization.

    Example:
        >>> scorer = EntropyScorer(temperature=1.0)
        >>> logits = torch.tensor([[2.0, 0.5, 0.1, 0.05]])  # Peaked distribution
        >>> score = scorer.score(logits)
        >>> assert 0.0 <= score <= 1.0
        >>> assert score > 0.7  # Should be high confidence (low entropy)

        >>> uniform_logits = torch.ones(1, 1000)  # Uniform distribution
        >>> uniform_score = scorer.score(uniform_logits)
        >>> assert uniform_score < 0.3  # Should be low confidence (high entropy)
    """

    def __init__(self, temperature: float = 1.0, max_entropy_threshold: float = 8.0) -> None:
        """
        Initialize the entropy scorer.

        Args:
            temperature: Temperature parameter for softmax scaling.
                Higher values increase entropy. Must be positive.
            max_entropy_threshold: Maximum entropy value for normalization.
                Used to map entropy values to [0.0, 1.0] confidence range.

        Raises:
            ValueError: If temperature <= 0 or max_entropy_threshold <= 0.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if max_entropy_threshold <= 0:
            raise ValueError(f"max_entropy_threshold must be positive, got {max_entropy_threshold}")

        self._temperature = temperature
        self._max_entropy_threshold = max_entropy_threshold

    @property
    def name(self) -> str:
        """Get the name of this confidence scorer."""
        return "entropy"

    @property
    def temperature(self) -> float:
        """Get the temperature parameter."""
        return self._temperature

    @property
    def max_entropy_threshold(self) -> float:
        """Get the maximum entropy threshold."""
        return self._max_entropy_threshold

    def score(self, logits: Logits) -> float:
        """
        Compute confidence score from logits using Shannon entropy.

        The score is computed as:
        1. Apply temperature scaling: logits / temperature
        2. Compute softmax probabilities
        3. Calculate Shannon entropy: H = -sum(p * log(p))
        4. Normalize to confidence: confidence = 1 - (H / max_entropy_threshold)
        5. Clamp to [0.0, 1.0]

        Args:
            logits: Model output logits. Shape can be:
                - (vocab_size,): Single prediction
                - (batch_size, vocab_size): Batch of predictions
                - (batch_size, seq_len, vocab_size): Sequence predictions

                For multi-dimensional inputs, entropy is computed per position
                and averaged.

        Returns:
            Confidence score in [0.0, 1.0]. Higher values indicate higher
            confidence (lower entropy/uncertainty).

        Note:
            - Returns 0.5 (neutral) for empty tensors or invalid inputs
            - Handles NaN and inf values by replacing them with neutral score
            - Uses small epsilon (1e-10) to avoid log(0)
        """
        try:
            # Handle empty or invalid tensors
            if logits.numel() == 0:
                return 0.5

            # Ensure we're working with the right shape
            # If logits is 1D, add batch dimension
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # If logits is 3D (batch, seq, vocab), flatten to (batch*seq, vocab)
            if logits.dim() == 3:
                batch_size, seq_len, vocab_size = logits.shape
                logits = logits.view(-1, vocab_size)
            elif logits.dim() != 2:
                # Unexpected shape, return neutral
                return 0.5

            # Apply temperature scaling
            scaled_logits = logits / self._temperature

            # Compute softmax probabilities
            # Use log_softmax for numerical stability
            log_probs = torch.nn.functional.log_softmax(scaled_logits, dim=-1)
            probs = torch.exp(log_probs)

            # Compute Shannon entropy: H = -sum(p * log(p))
            # Note: log_probs already contains log(p), so we use that directly
            # H = -sum(p * log(p))
            entropy = -torch.sum(probs * log_probs, dim=-1)

            # Average entropy across all positions (if multiple)
            mean_entropy = entropy.mean().item()

            # Handle NaN or inf
            if not math.isfinite(mean_entropy):
                return 0.5

            # Convert entropy to confidence score
            # High entropy (uncertainty) → low confidence
            # Low entropy (certainty) → high confidence
            confidence = 1.0 - (mean_entropy / self._max_entropy_threshold)

            # Clamp to [0.0, 1.0]
            confidence = max(0.0, min(1.0, confidence))

            return float(confidence)

        except (RuntimeError, ValueError, TypeError):
            # On tensor operation errors, return neutral score
            return 0.5

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"EntropyScorer(temperature={self._temperature}, "
            f"max_entropy_threshold={self._max_entropy_threshold})"
        )

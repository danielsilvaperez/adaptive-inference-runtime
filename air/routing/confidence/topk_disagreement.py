"""
Top-k Disagreement Confidence Scorer.

This module implements the top-k disagreement metric for measuring prediction
consensus. It compares the overlap in top-k predictions across different model
outputs or probability distributions to quantify confidence.

High consensus (high overlap) indicates high confidence, while low consensus
(low overlap) indicates uncertainty and potential need for escalation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from air.interfaces.router import BaseConfidenceScorer

if TYPE_CHECKING:
    import torch
    from air.types import Logits


class TopKDisagreementScorer(BaseConfidenceScorer):
    """
    Confidence scorer based on top-k prediction disagreement.

    This scorer measures the consensus level among the top-k most likely
    predictions by analyzing the distribution of probability mass. Higher
    concentration in fewer top predictions indicates higher confidence.

    The disagreement is quantified by comparing how the probability mass is
    distributed among the top-k tokens. If the top-k tokens have very similar
    probabilities, there's high disagreement (low confidence). If one token
    dominates, there's low disagreement (high confidence).

    Attributes:
        k: Number of top predictions to consider. Default: 5.
        temperature: Temperature for softmax scaling. Higher values make
            the distribution more uniform. Default: 1.0.

    Example:
        >>> import torch
        >>> scorer = TopKDisagreementScorer(k=5)
        >>> logits = torch.tensor([[10.0, 2.0, 1.0, 0.5, 0.1]])
        >>> confidence = scorer.score(logits)
        >>> # High confidence since first logit dominates
        >>> assert confidence > 0.8

        >>> # Uniform distribution has low confidence
        >>> uniform_logits = torch.ones((1, 100))
        >>> confidence = scorer.score(uniform_logits)
        >>> assert confidence < 0.3
    """

    def __init__(self, k: int = 5, temperature: float = 1.0) -> None:
        """
        Initialize the Top-k Disagreement Scorer.

        Args:
            k: Number of top predictions to consider. Must be positive.
                Typical values: 3-10. Default: 5.
            temperature: Temperature parameter for softmax. Must be positive.
                Lower values sharpen the distribution, higher values smooth it.
                Default: 1.0.

        Raises:
            ValueError: If k <= 0 or temperature <= 0.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self._k = k
        self._temperature = temperature

    @property
    def name(self) -> str:
        """Get the name of this confidence scorer."""
        return f"topk_disagreement_k{self._k}"

    @property
    def k(self) -> int:
        """Get the k parameter (number of top predictions to consider)."""
        return self._k

    @property
    def temperature(self) -> float:
        """Get the temperature parameter for softmax scaling."""
        return self._temperature

    def score(self, logits: Logits) -> float:
        """
        Compute confidence score from logits based on top-k disagreement.

        The score measures how concentrated the probability mass is among
        the top-k predictions. Higher concentration indicates higher confidence.

        The algorithm:
        1. Apply temperature-scaled softmax to get probabilities
        2. Extract top-k probabilities
        3. Compute normalized entropy of top-k distribution
        4. Convert entropy to confidence (inverse relationship)

        We use normalized entropy as the disagreement metric. Low entropy
        (one dominant token) = low disagreement = high confidence.
        High entropy (uniform distribution) = high disagreement = low confidence.

        Args:
            logits: Model output logits. Shape: (batch_size, vocab_size) or
                (batch_size, seq_len, vocab_size). If multiple dimensions,
                uses the last token's logits.

        Returns:
            Confidence score in [0.0, 1.0], where:
            - 1.0 indicates perfect confidence (one token dominates top-k)
            - 0.0 indicates maximum uncertainty (uniform distribution in top-k)

        Note:
            If vocab_size < k, uses all available tokens.
            If logits contain NaN or inf, returns 0.5 (neutral confidence).
        """
        import torch

        try:
            # Handle different logits shapes
            if logits.dim() == 3:
                # Shape: (batch_size, seq_len, vocab_size)
                # Take last token in sequence
                logits = logits[:, -1, :]
            elif logits.dim() == 1:
                # Shape: (vocab_size,) - add batch dimension
                logits = logits.unsqueeze(0)
            elif logits.dim() != 2:
                # Invalid shape
                return 0.5

            # Check for invalid values
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return 0.5

            # Apply temperature scaling and softmax
            scaled_logits = logits / self._temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Get top-k probabilities for each batch element
            vocab_size = probs.shape[-1]
            effective_k = min(self._k, vocab_size)

            # topk returns (values, indices)
            topk_probs, _ = torch.topk(probs, k=effective_k, dim=-1)

            # Average across batch dimension if needed
            if topk_probs.shape[0] > 1:
                topk_probs = topk_probs.mean(dim=0)
            else:
                topk_probs = topk_probs.squeeze(0)

            # Compute disagreement metric using normalized entropy
            # We measure how concentrated the probability mass is in the top-k
            
            # Normalize top-k probabilities to sum to 1
            topk_probs_normalized = topk_probs / topk_probs.sum()
            
            # Compute entropy of the normalized top-k distribution
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            topk_entropy = -(topk_probs_normalized * torch.log(topk_probs_normalized + eps)).sum()
            
            # Normalize entropy by max possible entropy for k items
            # Max entropy = log(k) when all k items are equally likely
            max_entropy = torch.log(torch.tensor(float(effective_k)))
            normalized_entropy = topk_entropy / (max_entropy + eps)
            
            # Convert normalized entropy to confidence
            # Low entropy (concentrated distribution) = high confidence
            # High entropy (uniform distribution) = low confidence
            confidence = 1.0 - normalized_entropy.item()
            
            # Clamp to valid range
            return max(0.0, min(1.0, confidence))

        except Exception:
            # On any error, return neutral confidence
            return 0.5

    def __repr__(self) -> str:
        """Get string representation."""
        return f"TopKDisagreementScorer(k={self._k}, temperature={self._temperature})"

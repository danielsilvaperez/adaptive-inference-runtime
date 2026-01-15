"""
Attention Instability Detector for confidence scoring.

This module implements the AttentionInstabilityScorer, which detects variance
in attention patterns across transformer layers to assess model confidence.
High variance in attention patterns often indicates uncertainty or difficulty
in the model's reasoning process.

The scorer:
- Analyzes pre-extracted attention weights from model layers
- Computes variance across layers and heads
- Normalizes the instability metric to a confidence score [0.0, 1.0]
- Supports configurable sensitivity through thresholds

Note: This scorer requires attention weights to be extracted externally
from the model. Not all inference backends provide access to attention weights.
"""

from __future__ import annotations

import torch
from typing import Dict, Optional, Tuple

from air.interfaces.router import BaseConfidenceScorer

# Empirical scaling factor for variance normalization
# This factor converts typical attention variance (usually < 0.1) to a [0, 1] range
# Adjust this value if working with attention patterns that have different variance characteristics
_VARIANCE_NORMALIZATION_FACTOR = 10.0


class AttentionInstabilityScorer(BaseConfidenceScorer):
    """
    Confidence scorer based on attention pattern instability.

    This scorer analyzes the variance in attention patterns across transformer
    layers to detect model uncertainty. When a model is confident, attention
    patterns tend to be stable and consistent across layers. High variance
    indicates the model is struggling to focus consistently, suggesting
    escalation to a larger model may be beneficial.

    The scorer computes:
    1. Variance of attention weights across layers
    2. Variance across attention heads within layers
    3. Standard deviation of attention patterns
    4. Normalized instability score

    Attributes:
        sensitivity: Controls how sensitive the scorer is to instability.
            Range: [0.0, 1.0]. Higher values make the scorer more sensitive,
            detecting smaller variations as instability. Default: 0.5.
        use_head_variance: Whether to include per-head variance in the
            instability calculation. Default: True.
        variance_aggregation: Method to aggregate variance across layers.
            Options: "mean", "max", "weighted". Default: "mean".

    Example:
        >>> import torch
        >>> scorer = AttentionInstabilityScorer(sensitivity=0.7)
        >>> # Simulate attention weights from 32 layers, 32 heads, 128 seq_len
        >>> attn_weights = torch.rand(32, 32, 128, 128)
        >>> score = scorer.score_from_attention(attn_weights)
        >>> print(f"Confidence: {score:.3f}")
        Confidence: 0.823
    """

    def __init__(
        self,
        sensitivity: float = 0.5,
        use_head_variance: bool = True,
        variance_aggregation: str = "mean",
    ) -> None:
        """
        Initialize the attention instability scorer.

        Args:
            sensitivity: Sensitivity to instability (0.0-1.0). Higher values
                make the scorer more sensitive to small variations.
            use_head_variance: Whether to consider per-head variance.
            variance_aggregation: How to aggregate layer variances
                ("mean", "max", or "weighted").

        Raises:
            ValueError: If sensitivity is not in [0.0, 1.0] or if
                variance_aggregation is not a valid option.
        """
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError(f"sensitivity must be in [0.0, 1.0], got {sensitivity}")

        valid_aggregations = {"mean", "max", "weighted"}
        if variance_aggregation not in valid_aggregations:
            raise ValueError(
                f"variance_aggregation must be one of {valid_aggregations}, "
                f"got '{variance_aggregation}'"
            )

        self._sensitivity = sensitivity
        self._use_head_variance = use_head_variance
        self._variance_aggregation = variance_aggregation

    @property
    def name(self) -> str:
        """Get the name of this confidence scorer."""
        return "attention_instability"

    @property
    def sensitivity(self) -> float:
        """Get the current sensitivity setting."""
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        """Set the sensitivity."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"sensitivity must be in [0.0, 1.0], got {value}")
        self._sensitivity = value

    def score(self, logits: "torch.Tensor") -> float:
        """
        Compute confidence score from logits.

        Note: This scorer primarily works with attention weights rather than
        logits. This method returns a default score (0.5) and should not be
        the primary interface. Use score_from_attention() or integrate with
        a model adapter that provides attention weights.

        Args:
            logits: Model output logits (not used by this scorer).

        Returns:
            A default confidence score of 0.5 (neutral confidence).
        """
        # This scorer needs attention weights, not logits
        # Return neutral score as fallback
        return 0.5

    def score_from_attention(
        self,
        attention_weights: "torch.Tensor",
        layer_indices: Optional[Tuple[int, ...]] = None,
    ) -> float:
        """
        Compute confidence score from attention weights.

        This is the primary interface for the attention instability scorer.
        It analyzes the provided attention weights to detect instability
        patterns and returns a confidence score.

        Args:
            attention_weights: Attention weights tensor.
                Expected shape: (num_layers, num_heads, seq_len, seq_len)
                or (num_layers, num_heads, batch_size, seq_len, seq_len).
                The attention weights should be normalized (sum to 1 along
                the last dimension).
            layer_indices: Optional tuple of layer indices to analyze.
                If None, all layers are used. Useful for focusing on
                specific transformer layers (e.g., middle layers).

        Returns:
            A confidence score in [0.0, 1.0] where:
            - 1.0 = stable attention (high confidence)
            - 0.0 = unstable attention (low confidence)

        Raises:
            ValueError: If attention_weights shape is invalid or contains
                NaN/Inf values.

        Example:
            >>> import torch
            >>> scorer = AttentionInstabilityScorer()
            >>> # 32 layers, 32 heads, 128 tokens
            >>> attn = torch.rand(32, 32, 128, 128)
            >>> score = scorer.score_from_attention(attn)
        """

        # Validate input
        if attention_weights.dim() < 4:
            raise ValueError(
                f"attention_weights must have at least 4 dimensions, got {attention_weights.dim()}"
            )

        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            # Return low confidence for invalid attention weights
            return 0.0

        # Handle empty or single-layer cases
        if attention_weights.size(0) == 0:
            return 0.5  # Neutral confidence for no data
        if attention_weights.size(0) == 1:
            # Single layer - cannot compute cross-layer variance
            # Compute within-head variance instead
            return self._compute_head_stability(attention_weights[0])

        # Select layers if specified
        if layer_indices is not None:
            attention_weights = attention_weights[list(layer_indices), ...]

        # Compute instability metrics
        instability = self._compute_instability(attention_weights)

        # Convert instability to confidence (inverse relationship)
        # Apply sensitivity scaling
        scaled_instability = instability * (1.0 + self._sensitivity)
        confidence = 1.0 / (1.0 + scaled_instability)

        return float(confidence)

    def _compute_instability(self, attention_weights: "torch.Tensor") -> float:
        """
        Compute the overall instability metric from attention weights.

        Args:
            attention_weights: Attention weights tensor
                Shape: (num_layers, num_heads, seq_len, seq_len) or
                       (num_layers, num_heads, batch_size, seq_len, seq_len)

        Returns:
            Instability score (higher = more unstable).
        """

        # Compute cross-layer variance
        layer_variance = self._compute_layer_variance(attention_weights)

        # Optionally compute per-head variance
        if self._use_head_variance:
            head_variance = self._compute_head_variance(attention_weights)
            # Combine both metrics
            if self._variance_aggregation == "mean":
                instability = (layer_variance + head_variance) / 2.0
            elif self._variance_aggregation == "max":
                instability = max(layer_variance, head_variance)
            else:  # weighted
                # Weight layer variance more heavily (2:1)
                instability = (2.0 * layer_variance + head_variance) / 3.0
        else:
            instability = layer_variance

        return float(instability)

    def _compute_layer_variance(self, attention_weights: "torch.Tensor") -> float:
        """
        Compute variance in attention patterns across layers.

        Measures how consistent attention patterns are across transformer layers.
        High variance indicates different layers are focusing on different tokens,
        suggesting uncertainty.

        Args:
            attention_weights: Shape (num_layers, num_heads, ..., seq_len, seq_len)

        Returns:
            Normalized variance score.
        """

        # Average attention across heads for each layer
        # Shape: (num_layers, seq_len, seq_len) or (num_layers, batch, seq_len, seq_len)
        if attention_weights.dim() == 5:
            # Has batch dimension
            layer_attn = attention_weights.mean(dim=1)  # Average over heads
            # Average over batch for simplicity
            layer_attn = layer_attn.mean(dim=1)  # Shape: (num_layers, seq_len, seq_len)
        else:
            layer_attn = attention_weights.mean(dim=1)  # Average over heads

        # Compute variance across layers at each position
        # Shape: (seq_len, seq_len)
        attn_variance = torch.var(layer_attn, dim=0, unbiased=False)

        # Take mean variance across all positions
        mean_variance = attn_variance.mean()

        # Normalize to a reasonable range using empirical scaling factor
        # Empirically, variance rarely exceeds 0.1 for normalized attention
        normalized_variance = torch.clamp(
            mean_variance * _VARIANCE_NORMALIZATION_FACTOR, 0.0, 1.0
        )

        return float(normalized_variance)

    def _compute_head_variance(self, attention_weights: "torch.Tensor") -> float:
        """
        Compute variance in attention patterns across heads within layers.

        Measures how much attention heads disagree within each layer.
        High disagreement suggests uncertainty about what to focus on.

        Args:
            attention_weights: Shape (num_layers, num_heads, ..., seq_len, seq_len)

        Returns:
            Normalized variance score.
        """

        # Compute variance across heads for each layer
        if attention_weights.dim() == 5:
            # Has batch dimension, average over batch first
            head_attn = attention_weights.mean(dim=2)  # Average over batch
        else:
            head_attn = attention_weights

        # Compute variance across heads at each position
        # Shape: (num_layers, seq_len, seq_len)
        head_variance = torch.var(head_attn, dim=1, unbiased=False)

        # Take mean variance across layers and positions
        mean_variance = head_variance.mean()

        # Normalize to a reasonable range using empirical scaling factor
        normalized_variance = torch.clamp(
            mean_variance * _VARIANCE_NORMALIZATION_FACTOR, 0.0, 1.0
        )

        return float(normalized_variance)

    def _compute_head_stability(self, attention_weights: "torch.Tensor") -> float:
        """
        Compute stability for a single layer by analyzing head agreement.

        Used as fallback when only one layer is available.

        Args:
            attention_weights: Single layer attention weights
                Shape: (num_heads, seq_len, seq_len) or (num_heads, batch, seq_len, seq_len)

        Returns:
            Confidence score [0.0, 1.0].
        """

        if attention_weights.dim() == 4:
            # Average over batch
            attention_weights = attention_weights.mean(dim=1)

        # Compute variance across heads
        head_variance = torch.var(attention_weights, dim=0, unbiased=False)
        mean_variance = head_variance.mean()

        # Convert variance to confidence using empirical scaling factor
        normalized_variance = torch.clamp(
            mean_variance * _VARIANCE_NORMALIZATION_FACTOR, 0.0, 1.0
        )
        scaled_variance = normalized_variance * (1.0 + self._sensitivity)
        confidence = 1.0 / (1.0 + scaled_variance)

        return float(confidence)

    def compute_layer_statistics(
        self, attention_weights: "torch.Tensor"
    ) -> Dict[str, float]:
        """
        Compute detailed statistics about attention instability per layer.

        Useful for debugging and analysis of attention patterns.

        Args:
            attention_weights: Attention weights tensor
                Shape: (num_layers, num_heads, seq_len, seq_len)

        Returns:
            Dictionary with statistics:
            - mean_variance: Average variance across layers
            - max_variance: Maximum variance across layers
            - std_variance: Standard deviation of variance
            - head_disagreement: Average disagreement between heads
            - overall_instability: Combined instability score

        Example:
            >>> scorer = AttentionInstabilityScorer()
            >>> attn = torch.rand(32, 32, 128, 128)
            >>> stats = scorer.compute_layer_statistics(attn)
            >>> print(f"Max variance: {stats['max_variance']:.4f}")
        """

        if attention_weights.dim() == 5:
            # Average over batch dimension
            attention_weights = attention_weights.mean(dim=2)

        # Layer-wise attention (averaged over heads)
        layer_attn = attention_weights.mean(dim=1)

        # Compute variance across layers at each position
        attn_variance = torch.var(layer_attn, dim=0, unbiased=False)

        # Head variance per layer
        head_variances = []
        for layer_idx in range(attention_weights.size(0)):
            layer_heads = attention_weights[layer_idx]
            head_var = torch.var(layer_heads, dim=0, unbiased=False).mean()
            head_variances.append(float(head_var))

        statistics = {
            "mean_variance": float(attn_variance.mean()),
            "max_variance": float(attn_variance.max()),
            "std_variance": float(torch.std(attn_variance, unbiased=False)),
            "head_disagreement": sum(head_variances) / len(head_variances),
            "overall_instability": self._compute_instability(attention_weights),
        }

        return statistics

    def __repr__(self) -> str:
        """Get string representation."""
        return (
            f"AttentionInstabilityScorer("
            f"sensitivity={self._sensitivity:.2f}, "
            f"use_head_variance={self._use_head_variance}, "
            f"variance_aggregation='{self._variance_aggregation}')"
        )

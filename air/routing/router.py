"""
Adaptive Router implementation for AIR.

This module implements the main routing logic that combines confidence scorers
to make intelligent decisions about which model (small or large) should handle
generation at any point in time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from air.routing.confidence import (
    AttentionInstabilityScorer,
    EntropyScorer,
    TopKDisagreementScorer,
)
from air.routing.logprob_slope import LogprobSlopeTracker
from air.types import ModelSelection, RoutingThresholds

if TYPE_CHECKING:
    from air.state import InferenceState
    from air.types import Logits


class AdaptiveRouter:
    """
    Main adaptive routing engine for AIR.

    Combines multiple confidence scorers (entropy, logprob slope, top-k
    disagreement, attention instability) to make routing decisions between
    small and large models. Uses configurable thresholds and weighted
    score combination.

    The router is designed to be conservative by default - it prefers to
    stay on the small model unless there's clear evidence of uncertainty.
    This maximizes the speedup from using the small model while ensuring
    quality doesn't suffer.

    Attributes:
        thresholds: Routing threshold configuration.
        entropy_scorer: Token entropy confidence scorer.
        topk_scorer: Top-k disagreement scorer.
        logprob_tracker: Log probability slope tracker.
        attention_scorer: Attention instability scorer (optional).

    Example:
        >>> router = AdaptiveRouter(
        ...     small_model_id="llama-7b",
        ...     large_model_id="llama-70b",
        ...     thresholds=RoutingThresholds.balanced()
        ... )
        >>> selection = router.route(state)
        >>> print(f"Using {selection.model_id} (confidence: {selection.confidence_score:.2f})")
    """

    def __init__(
        self,
        small_model_id: str,
        large_model_id: str,
        thresholds: RoutingThresholds | None = None,
        score_weights: dict[str, float] | None = None,
        use_attention_scorer: bool = False,
    ) -> None:
        """
        Initialize the adaptive router.

        Args:
            small_model_id: Identifier for the small (fast) model.
            large_model_id: Identifier for the large (accurate) model.
            thresholds: Routing thresholds configuration. Uses balanced defaults if None.
            score_weights: Weights for combining scorer outputs. Keys are scorer names.
                Defaults to equal weights if None.
            use_attention_scorer: Whether to use attention instability scorer.
                Requires attention weights from model, so disabled by default.
        """
        self._small_model_id = small_model_id
        self._large_model_id = large_model_id
        self._thresholds = thresholds or RoutingThresholds.balanced()

        # Initialize confidence scorers
        self._entropy_scorer = EntropyScorer()
        self._topk_scorer = TopKDisagreementScorer()
        self._logprob_tracker = LogprobSlopeTracker()
        self._attention_scorer = AttentionInstabilityScorer() if use_attention_scorer else None

        # Default score weights (equal weighting)
        default_weights = {
            "entropy": 1.0,
            "topk_disagreement": 1.0,
            "logprob_slope": 1.0,
        }
        if use_attention_scorer:
            default_weights["attention_instability"] = 1.0

        self._score_weights = score_weights or default_weights

        # Normalize weights to sum to 1.0
        total_weight = sum(self._score_weights.values())
        self._score_weights = {k: v / total_weight for k, v in self._score_weights.items()}

        # Track last escalation decision for cooldown
        self._tokens_since_transition = 0
        self._current_model = small_model_id

    @property
    def thresholds(self) -> RoutingThresholds:
        """Return the routing thresholds."""
        return self._thresholds

    @property
    def small_model_id(self) -> str:
        """Return the small model identifier."""
        return self._small_model_id

    @property
    def large_model_id(self) -> str:
        """Return the large model identifier."""
        return self._large_model_id

    def route(self, state: InferenceState) -> ModelSelection:
        """
        Make a routing decision based on current inference state.

        Args:
            state: Current inference state.

        Returns:
            ModelSelection indicating which model to use.
        """
        # If no logits available, stay with current model
        if state.last_logits is None:
            return ModelSelection(
                model_id=self._current_model,
                confidence_score=0.5,
                reason="No logits available, keeping current model",
            )

        # Get all confidence scores
        scores = self.get_confidence_scores(state.last_logits)

        # Combine scores with weights
        combined_confidence = self._combine_scores(scores)

        # Make routing decision
        if self.should_escalate(state, combined_confidence, scores):
            self._current_model = self._large_model_id
            self._tokens_since_transition = 0
            return ModelSelection(
                model_id=self._large_model_id,
                confidence_score=combined_confidence,
                reason=self._explain_escalation(scores),
            )
        else:
            # Stay on or return to small model
            if self._current_model != self._small_model_id:
                self._tokens_since_transition = 0
            self._current_model = self._small_model_id
            self._tokens_since_transition += 1
            return ModelSelection(
                model_id=self._small_model_id,
                confidence_score=combined_confidence,
                reason="Confidence sufficient for small model",
            )

    def get_confidence_scores(self, logits: Logits) -> dict[str, float]:
        """
        Compute all confidence scores from logits.

        Args:
            logits: Model output logits.

        Returns:
            Dictionary of scorer names to scores.
        """
        scores = {
            "entropy": self._entropy_scorer.score(logits),
            "topk_disagreement": self._topk_scorer.score(logits),
            "logprob_slope": self._logprob_tracker.score(logits),
        }

        if self._attention_scorer is not None:
            # Note: Attention scorer needs attention weights, not just logits
            # For now, we skip it unless specifically provided
            scores["attention_instability"] = 0.5

        return scores

    def should_escalate(
        self,
        state: InferenceState,  # noqa: ARG002
        combined_confidence: float,
        scores: dict[str, float],
    ) -> bool:
        """
        Determine if escalation to large model is needed.

        Args:
            state: Current inference state (reserved for future extensions).
            combined_confidence: Combined confidence score.
            scores: Individual scorer outputs.

        Returns:
            True if should escalate to large model.
        """
        # Already on large model? Stay there based on cooldown
        if self._current_model == self._large_model_id:
            # Only de-escalate after cooldown period with high confidence
            if self._tokens_since_transition < self._thresholds.cooldown_tokens:
                return True
            # De-escalate if confidence is high
            return combined_confidence < self._thresholds.min_confidence_for_small_model

        # Check cooldown for staying on small model
        if self._tokens_since_transition < self._thresholds.cooldown_tokens:
            # During cooldown, only escalate on very low confidence
            return combined_confidence < 0.3

        # Check individual threshold conditions
        if scores["entropy"] < (1.0 - self._thresholds.entropy_threshold / 10.0):
            return True

        if scores.get("topk_disagreement", 1.0) < (
            1.0 - self._thresholds.top_k_disagreement_threshold
        ):
            return True

        # Check combined confidence against threshold
        return combined_confidence < self._thresholds.min_confidence_for_small_model

    def _combine_scores(self, scores: dict[str, float]) -> float:
        """
        Combine individual confidence scores into overall confidence.

        Args:
            scores: Individual scorer outputs.

        Returns:
            Combined confidence score in [0.0, 1.0].
        """
        combined = 0.0
        for name, score in scores.items():
            weight = self._score_weights.get(name, 0.0)
            combined += weight * score

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, combined))

    def _explain_escalation(self, scores: dict[str, float]) -> str:
        """
        Generate human-readable explanation for escalation decision.

        Args:
            scores: Individual scorer outputs.

        Returns:
            Explanation string.
        """
        reasons = []

        if scores["entropy"] < 0.5:
            reasons.append(f"high entropy (score: {scores['entropy']:.2f})")

        if scores.get("topk_disagreement", 1.0) < 0.6:
            reasons.append(f"top-k disagreement (score: {scores['topk_disagreement']:.2f})")

        if scores["logprob_slope"] < 0.5:
            reasons.append(f"declining logprobs (score: {scores['logprob_slope']:.2f})")

        if reasons:
            return f"Escalating due to: {', '.join(reasons)}"
        else:
            return "Low overall confidence"

    def reset(self) -> None:
        """Reset router state to initial conditions."""
        self._current_model = self._small_model_id
        self._tokens_since_transition = 0
        self._logprob_tracker.reset()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AdaptiveRouter("
            f"small={self._small_model_id}, "
            f"large={self._large_model_id}, "
            f"current={self._current_model})"
        )


__all__ = ["AdaptiveRouter"]

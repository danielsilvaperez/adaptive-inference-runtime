"""
Router interface definitions for the Adaptive Inference Runtime.

This module defines the protocols for routing decisions, including:
- Router: Main interface for model routing decisions
- ConfidenceScorer: Interface for computing confidence scores from logits

The routing system determines which model (small or large) should handle
each token or span of generation based on uncertainty signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Dict,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from air.state import InferenceState
    from air.types import Logits, ModelSelection


@runtime_checkable
class ConfidenceScorer(Protocol):
    """
    Protocol for confidence scoring algorithms.

    A ConfidenceScorer computes a confidence score from model logits,
    which is used by the Router to make escalation decisions. Different
    implementations can use different signals (entropy, logprob slope,
    top-k disagreement, etc.).

    The protocol requires both a scoring method and a name property
    to identify the scorer for logging and debugging.

    Example:
        >>> class EntropyScorer:
        ...     @property
        ...     def name(self) -> str:
        ...         return "entropy"
        ...
        ...     def score(self, logits: Logits) -> float:
        ...         probs = torch.softmax(logits, dim=-1)
        ...         entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        ...         # Convert entropy to confidence (inverse relationship)
        ...         return 1.0 / (1.0 + entropy.mean().item())
    """

    @property
    def name(self) -> str:
        """
        Get the name of this confidence scorer.

        The name should be a short, descriptive identifier used for
        logging, debugging, and configuration purposes.

        Returns:
            A string identifier for this scorer (e.g., "entropy", "logprob_slope").
        """
        ...

    def score(self, logits: "Logits") -> float:
        """
        Compute a confidence score from logits.

        The score should be in the range [0.0, 1.0], where:
        - 0.0 indicates very low confidence (high uncertainty)
        - 1.0 indicates very high confidence (low uncertainty)

        Args:
            logits: The model output logits. Shape typically
                (batch_size, vocab_size) or (batch_size, seq_len, vocab_size).

        Returns:
            A confidence score in [0.0, 1.0].

        Note:
            Implementations should handle edge cases gracefully, such as
            empty tensors or NaN values, returning a default score (e.g., 0.5)
            rather than raising exceptions.
        """
        ...


@runtime_checkable
class Router(Protocol):
    """
    Protocol for model routing decisions.

    The Router is the central decision-making component that determines
    which model should handle generation at any given point. It uses
    confidence scores from various signals to decide whether to:
    - Continue with the current (typically small) model
    - Escalate to a larger model
    - De-escalate back to the small model

    Implementations should be stateless - all state should be tracked
    in the InferenceState object passed to methods.

    Example:
        >>> class AdaptiveRouter:
        ...     def route(self, state: InferenceState) -> ModelSelection:
        ...         scores = self.get_confidence_scores(state.last_logits)
        ...         combined = self._combine_scores(scores)
        ...         if combined < self.threshold:
        ...             return ModelSelection("llama-70b", combined, "Low confidence")
        ...         return ModelSelection("llama-7b", combined, "High confidence")
    """

    def route(self, state: "InferenceState") -> "ModelSelection":
        """
        Make a routing decision based on current inference state.

        Analyzes the current state including recent tokens, logits,
        and statistics to determine which model should generate the
        next token(s).

        Args:
            state: The current inference state containing model info,
                token history, and statistics.

        Returns:
            A ModelSelection indicating which model to use, along with
            confidence score and reasoning.

        Note:
            The router should respect cooldown periods to avoid rapid
            switching between models. Check state.stats.model_transitions
            and use the routing thresholds appropriately.
        """
        ...

    def get_confidence_scores(self, logits: "Logits") -> Dict[str, float]:
        """
        Compute all confidence scores from logits.

        Runs all configured ConfidenceScorer instances and returns
        their scores in a dictionary keyed by scorer name.

        Args:
            logits: The model output logits.

        Returns:
            A dictionary mapping scorer names to their scores.
            Example: {"entropy": 0.85, "logprob_slope": 0.72, "top_k": 0.90}

        Note:
            Scores should all be normalized to [0.0, 1.0] for consistent
            combination and threshold comparison.
        """
        ...

    def should_escalate(self, state: "InferenceState") -> bool:
        """
        Determine if the model should be escalated.

        A convenience method that checks if the current state warrants
        escalation from a small model to a large model. This is typically
        called before generating each token or batch of tokens.

        Args:
            state: The current inference state.

        Returns:
            True if escalation is recommended, False otherwise.

        Note:
            This method should consider:
            - Current model tier (no escalation if already on large model)
            - Cooldown periods since last transition
            - Combined confidence scores vs thresholds
            - Task-specific escalation policies
        """
        ...


class BaseRouter(ABC):
    """
    Abstract base class for Router implementations.

    Provides a concrete base for implementing routers with common
    functionality. Subclasses must implement the abstract methods
    but can reuse helper methods.

    This class is provided as an alternative to the Router protocol
    for implementations that prefer inheritance over duck typing.

    Attributes:
        scorers: Dictionary of registered confidence scorers.
        thresholds: Routing thresholds for escalation decisions.

    Example:
        >>> class MyRouter(BaseRouter):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.register_scorer(EntropyScorer())
        ...
        ...     def route(self, state: InferenceState) -> ModelSelection:
        ...         # Custom routing logic
        ...         pass
    """

    def __init__(self) -> None:
        """Initialize the base router."""
        from air.types import RoutingThresholds

        self._scorers: Dict[str, ConfidenceScorer] = {}
        self._thresholds: RoutingThresholds = RoutingThresholds()

    @property
    def scorers(self) -> Dict[str, ConfidenceScorer]:
        """Get registered confidence scorers."""
        return self._scorers.copy()

    @property
    def thresholds(self) -> "RoutingThresholds":
        """Get routing thresholds."""
        from air.types import RoutingThresholds

        return self._thresholds

    @thresholds.setter
    def thresholds(self, value: "RoutingThresholds") -> None:
        """Set routing thresholds."""
        self._thresholds = value

    def register_scorer(self, scorer: ConfidenceScorer) -> None:
        """
        Register a confidence scorer.

        Args:
            scorer: The scorer to register. Will be keyed by its name.
        """
        self._scorers[scorer.name] = scorer

    def unregister_scorer(self, name: str) -> None:
        """
        Unregister a confidence scorer.

        Args:
            name: The name of the scorer to remove.
        """
        self._scorers.pop(name, None)

    def get_confidence_scores(self, logits: "Logits") -> Dict[str, float]:
        """
        Compute all confidence scores from logits.

        Args:
            logits: The model output logits.

        Returns:
            Dictionary mapping scorer names to their scores.
        """
        return {name: scorer.score(logits) for name, scorer in self._scorers.items()}

    def combine_scores(self, scores: Dict[str, float]) -> float:
        """
        Combine multiple confidence scores into a single value.

        Default implementation uses simple averaging. Subclasses can
        override for weighted or more sophisticated combination.

        Args:
            scores: Dictionary of scorer names to scores.

        Returns:
            Combined confidence score in [0.0, 1.0].
        """
        if not scores:
            return 0.5  # Default when no scores available
        return sum(scores.values()) / len(scores)

    @abstractmethod
    def route(self, state: "InferenceState") -> "ModelSelection":
        """
        Make a routing decision based on current inference state.

        Args:
            state: The current inference state.

        Returns:
            A ModelSelection indicating which model to use.
        """
        ...

    @abstractmethod
    def should_escalate(self, state: "InferenceState") -> bool:
        """
        Determine if the model should be escalated.

        Args:
            state: The current inference state.

        Returns:
            True if escalation is recommended.
        """
        ...


class BaseConfidenceScorer(ABC):
    """
    Abstract base class for ConfidenceScorer implementations.

    Provides a concrete base for implementing confidence scorers.
    Subclasses must implement the abstract methods.

    Example:
        >>> class EntropyScorer(BaseConfidenceScorer):
        ...     @property
        ...     def name(self) -> str:
        ...         return "entropy"
        ...
        ...     def score(self, logits: Logits) -> float:
        ...         # Compute entropy-based confidence
        ...         pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this confidence scorer."""
        ...

    @abstractmethod
    def score(self, logits: "Logits") -> float:
        """
        Compute a confidence score from logits.

        Args:
            logits: The model output logits.

        Returns:
            A confidence score in [0.0, 1.0].
        """
        ...

    def __repr__(self) -> str:
        """Get string representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"

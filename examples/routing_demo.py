#!/usr/bin/env python3
"""
Simple routing demonstration for Adaptive Inference Runtime (AIR).

This example shows how the AdaptiveRouter makes decisions based on
confidence scores from multiple signals. It uses mock data to demonstrate
the routing logic without requiring actual model loading.

Run this to understand how AIR decides when to escalate from small to large models.
"""

import torch

from air.routing import AdaptiveRouter
from air.state import InferenceState, ModelTier, TokenStats
from air.types import GenerationConfig, RoutingThresholds, Token


def create_mock_logits(entropy_level: str = "low") -> torch.Tensor:
    """
    Create mock logits with different entropy levels.

    Args:
        entropy_level: "low" (peaked), "medium", or "high" (flat).

    Returns:
        Logits tensor of shape (1, vocab_size).
    """
    vocab_size = 1000

    if entropy_level == "low":
        # Peaked distribution - high confidence
        logits = torch.randn(1, vocab_size) * 0.5
        logits[0, 42] = 10.0  # One clear winner
    elif entropy_level == "medium":
        # Moderate entropy
        logits = torch.randn(1, vocab_size) * 2.0
    else:  # high
        # Flat distribution - high uncertainty
        logits = torch.randn(1, vocab_size) * 0.1

    return logits


def demonstrate_routing(strategy: str = "balanced") -> None:
    """
    Demonstrate routing with different confidence scenarios.

    Args:
        strategy: "conservative", "balanced", or "aggressive".
    """
    print(f"\n{'='*70}")
    print(f"Routing Strategy: {strategy.upper()}")
    print(f"{'='*70}\n")

    # Create router with specified strategy
    if strategy == "conservative":
        thresholds = RoutingThresholds.conservative()
    elif strategy == "aggressive":
        thresholds = RoutingThresholds.aggressive()
    else:
        thresholds = RoutingThresholds.balanced()

    router = AdaptiveRouter(
        small_model_id="llama-7b",
        large_model_id="llama-70b",
        thresholds=thresholds,
    )

    # Test scenarios
    scenarios = [
        ("Low entropy (high confidence)", "low"),
        ("Medium entropy (moderate confidence)", "medium"),
        ("High entropy (low confidence)", "high"),
    ]

    for i, (description, entropy_level) in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {description}")
        print("-" * 70)

        # Create mock state
        logits = create_mock_logits(entropy_level)
        state = InferenceState(
            model_id="llama-7b",
            model_tier=ModelTier.SMALL,
            generation_config=GenerationConfig(),
        )
        state.last_logits = logits

        # Make routing decision
        selection = router.route(state)

        # Get confidence scores for display
        scores = router.get_confidence_scores(logits)

        # Display results
        print(f"  Selected Model: {selection.model_id}")
        print(f"  Combined Confidence: {selection.confidence_score:.3f}")
        print(f"  Reason: {selection.reason}")
        print(f"\n  Individual Confidence Scores:")
        for scorer_name, score in scores.items():
            print(f"    • {scorer_name:20s}: {score:.3f}")


def demonstrate_cooldown() -> None:
    """Demonstrate cooldown mechanism to prevent rapid switching."""
    print(f"\n{'='*70}")
    print("Cooldown Mechanism Demonstration")
    print(f"{'='*70}\n")

    router = AdaptiveRouter(
        small_model_id="llama-7b",
        large_model_id="llama-70b",
        thresholds=RoutingThresholds.balanced(),
    )

    state = InferenceState(
        model_id="llama-7b",
        model_tier=ModelTier.SMALL,
        generation_config=GenerationConfig(),
    )

    print("Simulating token generation with alternating confidence levels:\n")

    for i in range(10):
        # Alternate between high and low confidence
        entropy_level = "high" if i % 2 == 0 else "low"
        state.last_logits = create_mock_logits(entropy_level)

        selection = router.route(state)

        print(
            f"  Token {i+1:2d}: Entropy={entropy_level:6s} → "
            f"Model={selection.model_id:10s} "
            f"(Confidence: {selection.confidence_score:.3f})"
        )

    print("\nNotice: The router doesn't switch on every token due to cooldown period.")
    print("This prevents instability and reduces switching overhead.")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Adaptive Inference Runtime (AIR) - Routing Demonstration")
    print("=" * 70)
    print("\nThis demo shows how AIR intelligently routes between small and large models")
    print("based on confidence signals like entropy, top-k disagreement, and logprob.")
    print("\nNo actual models are loaded - this uses mock data for demonstration.")

    # Demonstrate different strategies
    for strategy in ["conservative", "balanced", "aggressive"]:
        demonstrate_routing(strategy)

    # Demonstrate cooldown
    demonstrate_cooldown()

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}\n")
    print("• Conservative: Prefers small model, rarely escalates")
    print("• Balanced: Good tradeoff between quality and efficiency")
    print("• Aggressive: Escalates often for maximum quality")
    print("\nThe routing system combines multiple confidence signals to make")
    print("intelligent decisions about when to use which model, optimizing for")
    print("both speed and quality.\n")


if __name__ == "__main__":
    main()

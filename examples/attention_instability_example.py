"""
Example: Using the Attention Instability Scorer

This example demonstrates how to use the AttentionInstabilityScorer to assess
model confidence based on attention pattern variance. This is useful for
deciding when to escalate from a small model to a larger model during inference.

The attention instability scorer analyzes variance in attention patterns across
transformer layers to detect model uncertainty. High variance (instability)
suggests the model is uncertain and may benefit from using a larger model.
"""

import torch
from air.routing.confidence.attention_instability import AttentionInstabilityScorer


def create_stable_attention(num_layers: int, num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Create simulated stable attention weights.
    
    Stable patterns have low variance across layers - all layers
    focus on similar positions.
    """
    # Create a base attention pattern (e.g., attending to recent tokens)
    base_attn = torch.zeros(seq_len, seq_len)
    # Attend to last 10 tokens
    base_attn[:, -10:] = torch.softmax(torch.randn(seq_len, 10), dim=-1)
    
    # Repeat across layers with small noise
    attn = base_attn.unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1, 1)
    attn = attn + torch.randn_like(attn) * 0.001
    
    # Normalize to ensure valid attention distributions
    attn = torch.softmax(attn.view(num_layers, num_heads, seq_len, -1), dim=-1)
    attn = attn.view(num_layers, num_heads, seq_len, seq_len)
    
    return attn


def create_unstable_attention(num_layers: int, num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Create simulated unstable attention weights.
    
    Unstable patterns have high variance - each layer focuses on
    different positions, indicating uncertainty.
    """
    attn = torch.zeros(num_layers, num_heads, seq_len, seq_len)
    
    # Each layer focuses on different tokens
    for i in range(num_layers):
        # Focus shifts across the sequence for each layer
        focus_start = (i * seq_len) // num_layers
        focus_end = ((i + 1) * seq_len) // num_layers
        attn[i, :, :, focus_start:focus_end] = torch.softmax(
            torch.randn(num_heads, seq_len, focus_end - focus_start), dim=-1
        )
    
    return attn


def main():
    print("=" * 70)
    print("Attention Instability Scorer Example")
    print("=" * 70)
    
    # Configuration
    num_layers = 32
    num_heads = 32
    seq_len = 128
    
    # Initialize the scorer with default settings
    scorer = AttentionInstabilityScorer(sensitivity=0.5)
    print(f"\nScorer: {scorer}")
    
    # Example 1: Stable attention pattern (confident model)
    print("\n" + "-" * 70)
    print("Example 1: Stable Attention Pattern (High Confidence)")
    print("-" * 70)
    
    stable_attn = create_stable_attention(num_layers, num_heads, seq_len)
    stable_score = scorer.score_from_attention(stable_attn)
    
    print(f"Attention shape: {stable_attn.shape}")
    print(f"Confidence score: {stable_score:.4f}")
    print(f"Interpretation: {'Stay on small model' if stable_score > 0.7 else 'Consider escalation'}")
    
    # Get detailed statistics
    stats = scorer.compute_layer_statistics(stable_attn)
    print("\nDetailed statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    
    # Example 2: Unstable attention pattern (uncertain model)
    print("\n" + "-" * 70)
    print("Example 2: Unstable Attention Pattern (Lower Confidence)")
    print("-" * 70)
    
    unstable_attn = create_unstable_attention(num_layers, num_heads, seq_len)
    unstable_score = scorer.score_from_attention(unstable_attn)
    
    print(f"Attention shape: {unstable_attn.shape}")
    print(f"Confidence score: {unstable_score:.4f}")
    print(f"Interpretation: {'Stay on small model' if unstable_score > 0.7 else 'Consider escalation'}")
    
    stats = scorer.compute_layer_statistics(unstable_attn)
    print("\nDetailed statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.6f}")
    
    # Example 3: Comparing different sensitivity settings
    print("\n" + "-" * 70)
    print("Example 3: Effect of Sensitivity Settings")
    print("-" * 70)
    
    sensitivities = [0.2, 0.5, 0.8]
    print(f"\nUsing unstable attention from Example 2:")
    print(f"{'Sensitivity':<15} {'Score':<10} {'Recommendation'}")
    print("-" * 50)
    
    for sens in sensitivities:
        scorer_sens = AttentionInstabilityScorer(sensitivity=sens)
        score = scorer_sens.score_from_attention(unstable_attn)
        recommendation = "Stay on small" if score > 0.7 else "Escalate to large"
        print(f"{sens:<15.1f} {score:<10.4f} {recommendation}")
    
    # Example 4: Using different variance aggregation methods
    print("\n" + "-" * 70)
    print("Example 4: Different Variance Aggregation Methods")
    print("-" * 70)
    
    aggregations = ["mean", "max", "weighted"]
    print(f"\nUsing unstable attention from Example 2:")
    print(f"{'Aggregation':<15} {'Score':<10} {'With Head Variance'}")
    print("-" * 50)
    
    for agg in aggregations:
        scorer_agg = AttentionInstabilityScorer(
            variance_aggregation=agg,
            use_head_variance=True
        )
        score = scorer_agg.score_from_attention(unstable_attn)
        print(f"{agg:<15} {score:<10.4f} Yes")
    
    # Example 5: Analyzing specific layers
    print("\n" + "-" * 70)
    print("Example 5: Analyzing Specific Layers")
    print("-" * 70)
    
    print("\nComparing early, middle, and late layers:")
    layer_groups = {
        "Early (0-10)": tuple(range(0, 11)),
        "Middle (11-21)": tuple(range(11, 22)),
        "Late (22-31)": tuple(range(22, 32)),
    }
    
    for group_name, layer_indices in layer_groups.items():
        score = scorer.score_from_attention(unstable_attn, layer_indices=layer_indices)
        print(f"{group_name:<20} Score: {score:.4f}")
    
    # Usage recommendation
    print("\n" + "=" * 70)
    print("Usage Recommendation for Routing")
    print("=" * 70)
    print("""
The attention instability scorer should be used alongside other confidence
metrics (entropy, logprob slope, top-k disagreement) for robust routing decisions.

Typical routing logic:
    1. Get attention weights from the model adapter
    2. Compute instability score
    3. If score < threshold (e.g., 0.7), consider escalation
    4. Combine with other confidence signals for final decision
    
The sensitivity parameter controls how reactive the scorer is:
    - Low sensitivity (0.2-0.4): Only escalate on severe instability
    - Medium sensitivity (0.4-0.6): Balanced approach (default)
    - High sensitivity (0.6-0.8): Escalate more aggressively
    
Note: This scorer requires access to attention weights from the model,
which may not be available in all inference backends.
""")
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

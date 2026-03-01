# Confidence Scoring Module

This module provides confidence scoring algorithms for the Adaptive Inference Runtime's routing system. Confidence scorers analyze model outputs to determine when to escalate from a small model to a larger model.

## Available Scorers

### AttentionInstabilityScorer

Detects variance in attention patterns across transformer layers to assess model confidence.

**Principle**: When a model is confident, attention patterns tend to be stable and consistent across layers. High variance indicates the model is struggling to focus consistently, suggesting escalation may be beneficial.

**Features**:
- Computes cross-layer variance in attention patterns
- Supports per-head variance analysis
- Configurable sensitivity (0.0-1.0)
- Multiple variance aggregation methods (mean, max, weighted)
- Handles edge cases gracefully (NaN, Inf, empty tensors)
- Provides detailed layer statistics for debugging

**Usage**:

```python
from air.routing.confidence import AttentionInstabilityScorer
import torch

# Initialize with default settings
scorer = AttentionInstabilityScorer(sensitivity=0.5)

# Get attention weights from your model
# Shape: (num_layers, num_heads, seq_len, seq_len)
attention_weights = model.get_attention_weights()

# Compute confidence score (0.0 = low confidence, 1.0 = high confidence)
confidence = scorer.score_from_attention(attention_weights)

if confidence < 0.7:
    print("Low confidence detected - consider escalating to larger model")
else:
    print("High confidence - continue with current model")
```

**Configuration Options**:

- `sensitivity` (float, default=0.5): Controls how sensitive the scorer is to instability
  - Low (0.2-0.4): Only escalate on severe instability
  - Medium (0.4-0.6): Balanced approach
  - High (0.6-0.8): Escalate more aggressively

- `use_head_variance` (bool, default=True): Whether to include per-head variance in the calculation

- `variance_aggregation` (str, default="mean"): Method to aggregate variance across layers
  - "mean": Average variance across all layers
  - "max": Use maximum variance
  - "weighted": Weight layer variance more heavily (2:1 ratio)

**Advanced Features**:

```python
# Analyze specific layers only
confidence = scorer.score_from_attention(
    attention_weights,
    layer_indices=(10, 11, 12, 13, 14)  # Analyze middle layers only
)

# Get detailed statistics
stats = scorer.compute_layer_statistics(attention_weights)
print(f"Mean variance: {stats['mean_variance']:.4f}")
print(f"Head disagreement: {stats['head_disagreement']:.4f}")
print(f"Overall instability: {stats['overall_instability']:.4f}")
```

**Requirements**:
- Access to attention weights from the model (may not be available in all backends)
- Attention weights should be normalized (sum to 1 along last dimension)

**Integration with Routing**:

The attention instability scorer should be used alongside other confidence metrics for robust routing decisions:

```python
from air.routing.confidence import AttentionInstabilityScorer
from air.interfaces.router import BaseRouter

class AdaptiveRouter(BaseRouter):
    def __init__(self):
        super().__init__()
        # Register the attention instability scorer
        self.register_scorer(AttentionInstabilityScorer(sensitivity=0.5))
        # Also register other scorers (entropy, logprob_slope, etc.)
    
    def route(self, state):
        # Get all confidence scores
        attention_weights = state.get_attention_weights()
        
        if attention_weights is not None:
            # Use attention-based scoring if available
            attn_scorer = self._scorers.get("attention_instability")
            confidence = attn_scorer.score_from_attention(attention_weights)
        else:
            # Fallback to logits-based scoring
            scores = self.get_confidence_scores(state.last_logits)
            confidence = self.combine_scores(scores)
        
        # Make routing decision based on confidence
        if confidence < self.thresholds.min_confidence_for_small_model:
            return ModelSelection("large-model", confidence, "Low confidence")
        return ModelSelection("small-model", confidence, "High confidence")
```

## Future Scorers

The following confidence scorers are planned for future implementation:

- **EntropyScorer**: Token entropy calculation (Shannon entropy from softmax distribution)
- **LogprobSlopeScorer**: Confidence trajectory tracking over last N tokens
- **TopKDisagreementScorer**: Consensus level in top-k predictions

See `ROADMAP.md` Phase 1, Task 1.2 for details.

## Testing

Run the confidence scorer tests:

```bash
pytest tests/unit/routing/test_attention_instability.py -v
```

## Examples

See `examples/attention_instability_example.py` for a complete usage example.

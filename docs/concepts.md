# Core Concepts

This document explains the key concepts behind AIR: intelligent routing, speculative decoding, and KV-cache compression.

## Routing: When to Use Which Model

The core insight behind routing is that **not all tokens require the same compute**. Simple continuations like "The capital of France is" need minimal reasoning, while complex chains like multi-step math require deeper processing.

### Confidence Signals

AIR uses multiple signals to detect when the small model is uncertain:

#### 1. Token Entropy

Entropy measures the "spread" of the probability distribution over possible next tokens.

**Calculation:**
```python
def token_entropy(logits: Tensor, temperature: float = 1.0) -> float:
    """
    Shannon entropy of the token probability distribution.

    H = -sum(p * log(p)) for all tokens

    High entropy = uncertain (flat distribution)
    Low entropy = confident (peaked distribution)
    """
    probs = softmax(logits / temperature)
    log_probs = torch.log(probs + 1e-10)  # avoid log(0)
    entropy = -torch.sum(probs * log_probs)
    return entropy.item()
```

**Interpretation:**
- Low entropy (< 1.0): Model is confident, one token dominates
- Medium entropy (1.0 - 3.0): Some uncertainty, few viable options
- High entropy (> 3.0): Model uncertain, many plausible tokens

**When to escalate:** Entropy exceeds a configurable threshold (default: 2.5)

#### 2. Logprob Slope

Tracks how confidence changes over recent tokens. A sudden drop indicates the model hit something unexpected.

**Calculation:**
```python
def logprob_slope(recent_logprobs: List[float], window: int = 5) -> float:
    """
    Linear regression slope of log probabilities over last N tokens.

    Negative slope = declining confidence
    Positive slope = increasing confidence
    """
    if len(recent_logprobs) < window:
        return 0.0

    window_probs = recent_logprobs[-window:]
    x = torch.arange(window, dtype=torch.float)
    y = torch.tensor(window_probs)

    # Simple linear regression
    slope = (window * (x * y).sum() - x.sum() * y.sum()) / \
            (window * (x * x).sum() - x.sum() ** 2)
    return slope.item()
```

**Interpretation:**
- Steep negative slope: Confidence dropping, may need help
- Flat or positive: Model handling content well

**When to escalate:** Slope drops below threshold (default: -0.5)

#### 3. Top-k Disagreement

Measures how much the top predictions "agree" with each other. When the top choices are all close in probability, the model is less certain.

**Calculation:**
```python
def topk_disagreement(logits: Tensor, k: int = 5) -> float:
    """
    Ratio of top-1 probability to sum of top-k probabilities.

    High ratio = top-1 dominates (agreement)
    Low ratio = top-k are close (disagreement)
    """
    probs = softmax(logits)
    topk_probs, _ = torch.topk(probs, k)

    top1 = topk_probs[0]
    topk_sum = topk_probs.sum()

    # 1.0 = complete agreement (top-1 is everything)
    # 1/k = complete disagreement (uniform)
    agreement = top1 / topk_sum
    return 1.0 - agreement  # Return disagreement
```

**Interpretation:**
- Low disagreement (< 0.3): Clear winner, confident
- High disagreement (> 0.6): No clear winner, uncertain

**When to escalate:** Disagreement exceeds threshold (default: 0.5)

### Combining Signals

The router combines signals using a weighted formula:

```python
def should_escalate(
    entropy: float,
    slope: float,
    disagreement: float,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> bool:
    """
    Combine confidence signals into escalation decision.

    Each signal is normalized to [0, 1] and weighted.
    """
    # Normalize signals
    entropy_norm = min(entropy / 5.0, 1.0)
    slope_norm = max(0, -slope)  # Convert negative slope to positive
    disagree_norm = disagreement

    # Weighted combination
    w_e, w_s, w_d = weights
    score = w_e * entropy_norm + w_s * slope_norm + w_d * disagree_norm

    return score > 0.5  # Configurable threshold
```

## Speculative Decoding: Speed Without Sacrifice

Speculative decoding exploits the asymmetry between generation and verification.

### The Draft-Verify Paradigm

**Key insight:** A language model can verify k tokens in roughly the same time it takes to generate 1 token (due to parallel matrix operations).

**Process:**
1. **Draft phase**: Small model quickly generates k candidate tokens
2. **Verify phase**: Large model processes all k tokens in one forward pass
3. **Accept phase**: Accept tokens until first disagreement
4. **Continue**: Resume from rejection point

```
Draft (Small Model):          Verify (Large Model):
"The" -> "answer"            "The" -> verify "answer" ✓
"answer" -> "is"             "answer" -> verify "is" ✓
"is" -> "42"                 "is" -> verify "42" ✓
"42" -> "because"            "42" -> verify "because" ✗ (would have said ".")

Result: Accept "answer", "is", "42" (3 tokens from 1 large model call)
```

### Why This Works

- **Generation is sequential**: Each token depends on all previous tokens
- **Verification is parallel**: All positions can be checked simultaneously
- **Small model is fast**: Low cost to generate draft candidates
- **Acceptance is common**: For many sequences, small and large models agree

### Acceptance Logic

```python
def verify_and_accept(
    draft_tokens: List[int],
    draft_logprobs: List[float],
    large_logits: Tensor  # Shape: [k, vocab_size]
) -> Tuple[List[int], int]:
    """
    Verify draft tokens against large model predictions.

    Returns accepted tokens and position of first rejection.
    """
    accepted = []

    for i, (token, draft_lp) in enumerate(zip(draft_tokens, draft_logprobs)):
        large_probs = softmax(large_logits[i])
        large_prob = large_probs[token].item()

        # Rejection sampling
        # Accept with probability min(1, large_prob / draft_prob)
        draft_prob = math.exp(draft_lp)
        accept_prob = min(1.0, large_prob / draft_prob)

        if random.random() < accept_prob:
            accepted.append(token)
        else:
            # Reject this and all subsequent tokens
            return accepted, i

    return accepted, len(draft_tokens)
```

### Adaptive Draft Length

The optimal draft length k depends on acceptance rate:

- **High acceptance rate**: Increase k (get more tokens per verification)
- **Low acceptance rate**: Decrease k (don't waste draft computation)

```python
def adapt_draft_length(
    current_k: int,
    recent_acceptance_rates: List[float],
    min_k: int = 2,
    max_k: int = 8
) -> int:
    """Adjust draft length based on recent acceptance rates."""
    avg_rate = sum(recent_acceptance_rates) / len(recent_acceptance_rates)

    if avg_rate > 0.8 and current_k < max_k:
        return current_k + 1
    elif avg_rate < 0.4 and current_k > min_k:
        return current_k - 1
    return current_k
```

## KV-Cache Compression: Memory Efficiency

### The Memory Problem

Transformer KV-caches grow linearly with sequence length:

```
Memory = 2 * num_layers * hidden_dim * seq_len * bytes_per_element

For Llama-70B with 4K context:
Memory = 2 * 80 * 8192 * 4096 * 2 bytes = ~10.7 GB just for KV cache
```

This limits:
- Maximum context length
- Concurrent requests
- Deployment on memory-constrained devices

### Eviction Strategies

#### 1. Sliding Window

Keep only the most recent N tokens in cache.

```python
def sliding_window_evict(
    kv_cache: KVCache,
    window_size: int
) -> KVCache:
    """Keep only the last window_size tokens."""
    if kv_cache.seq_len <= window_size:
        return kv_cache

    # Slice to keep last window_size positions
    return kv_cache[:, :, -window_size:, :]
```

**Pros:** Simple, predictable memory usage
**Cons:** Loses early context entirely

#### 2. Heavy Hitter Retention

Keep tokens that receive high attention scores (they're important for generation).

```python
def heavy_hitter_evict(
    kv_cache: KVCache,
    attention_scores: Tensor,  # Shape: [layers, heads, seq_len]
    budget: int,
    keep_recent: int = 64  # Always keep most recent tokens
) -> Tuple[KVCache, List[int]]:
    """
    Keep tokens with highest cumulative attention.

    Based on H2O: Heavy-Hitter Oracle paper.
    """
    seq_len = kv_cache.shape[2]

    # Sum attention across layers and heads
    importance = attention_scores.sum(dim=(0, 1))  # Shape: [seq_len]

    # Always keep recent tokens
    recent_mask = torch.zeros(seq_len, dtype=torch.bool)
    recent_mask[-keep_recent:] = True

    # Select top-k by importance (excluding recent)
    old_importance = importance.clone()
    old_importance[recent_mask] = float('-inf')

    remaining_budget = budget - keep_recent
    _, top_indices = torch.topk(old_importance, remaining_budget)

    # Combine indices
    keep_indices = torch.cat([
        top_indices,
        torch.arange(seq_len - keep_recent, seq_len)
    ]).sort().values

    # Select from cache
    compressed_cache = kv_cache[:, :, keep_indices, :]
    return compressed_cache, keep_indices.tolist()
```

**Pros:** Preserves semantically important tokens
**Cons:** Requires attention score tracking

#### 3. Hybrid: Sliding Window + Heavy Hitters

Combine both approaches:

```python
def hybrid_evict(
    kv_cache: KVCache,
    attention_scores: Tensor,
    total_budget: int,
    recent_ratio: float = 0.5  # 50% for recent, 50% for heavy hitters
) -> KVCache:
    """
    Keep recent tokens AND important historical tokens.
    """
    recent_budget = int(total_budget * recent_ratio)
    hh_budget = total_budget - recent_budget

    # Implementation combines both strategies
    ...
```

### Safety Guards

Some tasks require full context (no compression):

```python
COMPRESSION_SAFE_TASKS = {
    "chat": True,           # Conversation can lose old context
    "summarization": True,  # Can compress source document
    "code_generation": False,  # Needs precise context
    "qa_retrieval": False,     # Needs to reference specific passages
}
```

## Putting It Together

The three techniques combine synergistically:

1. **Routing** reduces compute by using small model when possible
2. **Speculation** speeds up large model usage when needed
3. **Compression** enables longer contexts and more concurrency

```
Without AIR:
- All tokens through 70B
- Sequential generation
- Memory limits context to 4K

With AIR:
- 60-80% of tokens through 7B
- 2-3x speedup via speculation
- 4-8x more context via compression
- Result: 70B quality at fraction of cost
```

## Related Documentation

- [Architecture](architecture.md) - System design and components
- [Getting Started](getting-started.md) - Developer setup guide

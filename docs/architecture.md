# Architecture Overview

This document describes the high-level architecture of the Adaptive Inference Runtime (AIR).

## System Components

AIR consists of four primary components that work together to optimize inference:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ADAPTIVE INFERENCE RUNTIME                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Router     │    │    Model     │    │  Speculation │                  │
│  │              │───►│   Adapters   │───►│   Pipeline   │                  │
│  │ - Entropy    │    │              │    │              │                  │
│  │ - Logprob    │    │ - llama.cpp  │    │ - Draft Gen  │                  │
│  │ - Top-k      │    │ - vLLM       │    │ - Verify     │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         └───────────────────┴───────────────────┘                          │
│                             │                                              │
│                    ┌────────▼────────┐                                     │
│                    │  KV Cache       │                                     │
│                    │  Manager        │                                     │
│                    │                 │                                     │
│                    │ - Compression   │                                     │
│                    │ - Eviction      │                                     │
│                    │ - Memory Mgmt   │                                     │
│                    └─────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Router

The Router is the decision engine that determines which model should handle each token or sequence.

**Responsibilities:**
- Calculate confidence signals from model outputs
- Make routing decisions (small vs. large model)
- Handle mid-generation escalation
- Maintain routing statistics

**Key Interfaces:**
```python
class Router(Protocol):
    def route(self, state: InferenceState) -> ModelSelection:
        """Decide which model should generate the next token(s)."""
        ...

    def calculate_confidence(self, logits: Tensor) -> ConfidenceScore:
        """Compute confidence from model output logits."""
        ...
```

### 2. Model Adapters

Model Adapters provide a unified interface to different inference backends.

**Responsibilities:**
- Load and manage model instances
- Generate tokens and expose logits
- Provide KV cache access
- Handle tokenization/detokenization

**Supported Backends:**
- `llama.cpp` - Primary backend for local inference, especially on Apple Silicon
- `vLLM` - Secondary backend for GPU server deployments

**Key Interfaces:**
```python
class ModelAdapter(Protocol):
    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        """Generate tokens from prompt."""
        ...

    def get_logits(self, tokens: List[int]) -> Tensor:
        """Get logits for given token sequence."""
        ...

    def verify(self, draft_tokens: List[int], prompt: str) -> VerificationResult:
        """Verify draft tokens (for speculative decoding)."""
        ...
```

### 3. Speculation Pipeline

The Speculation Pipeline implements speculative decoding for faster inference.

**Responsibilities:**
- Generate draft token sequences using the small model
- Coordinate parallel verification with the large model
- Handle acceptance/rejection logic
- Adapt draft length based on acceptance rates

**Flow:**
1. Small model generates k draft tokens
2. Large model verifies all k tokens in one forward pass
3. Accept tokens until first rejection
4. Continue from rejection point

### 4. KV Cache Manager

The KV Cache Manager handles memory optimization through intelligent caching.

**Responsibilities:**
- Track KV cache memory usage
- Apply compression/eviction policies
- Manage cache transfers between models
- Monitor and report memory pressure

**Eviction Strategies:**
- Sliding window (keep last N tokens)
- Heavy hitter retention (keep high-attention tokens)
- Hybrid approaches

## Data Flow

### Standard Generation Flow

```
1. User submits prompt
2. Router evaluates prompt complexity
3. If simple → Small model generates directly
4. If complex → Escalate to speculation pipeline
5. Speculation: Small drafts → Large verifies
6. KV Cache Manager compresses as needed
7. Output streamed to user
```

### Mid-Generation Escalation

```
1. Small model generating tokens
2. Router detects uncertainty spike (entropy > threshold)
3. State transferred to large model
4. Large model continues generation
5. Router may de-escalate when confidence returns
```

## Key Design Decisions

### Why Runtime-Only (No Retraining)?

- **Immediate applicability**: Works with existing models today
- **Model agnostic**: Any compatible model pair works
- **Lower barrier**: Users don't need training infrastructure
- **Flexibility**: Swap models without retraining

### Why Small-to-Large Routing?

- **Cost efficiency**: Most tokens don't need 70B compute
- **Latency reduction**: Small model is faster for easy tokens
- **Resource optimization**: Save large model for complex reasoning
- **Empirical validation**: Studies show 60-80% of tokens are "easy"

### Why Speculative Decoding?

- **Mathematically lossless**: Identical output distribution to large model alone
- **Parallelization**: Large model verifies k tokens in time of 1
- **Synergy with routing**: Small model already loaded for routing

### Why KV Cache Compression?

- **Memory bottleneck**: KV cache grows linearly with context length
- **Enables long context**: Fit more history in limited memory
- **Enables concurrency**: Serve more requests simultaneously
- **Quality tradeoff is acceptable**: 5-10% degradation for 4-8x memory savings

## State Management

The `InferenceState` object tracks all runtime state:

```python
@dataclass
class InferenceState:
    # Current context
    tokens: List[int]
    prompt: str

    # Model state
    active_model: ModelId
    kv_cache: Optional[KVCache]

    # Generation state
    generated_tokens: List[int]
    generation_params: GenerationParams

    # Routing state
    confidence_history: List[float]
    routing_decisions: List[RoutingDecision]
```

## Extension Points

AIR is designed for extensibility:

1. **Custom Routers**: Implement `Router` protocol with custom logic
2. **New Backends**: Add adapters for other inference engines
3. **Eviction Policies**: Plug in custom KV cache strategies
4. **Confidence Signals**: Add new metrics for routing decisions

## Related Documentation

- [Core Concepts](concepts.md) - Deep dive into routing, speculation, compression
- [Getting Started](getting-started.md) - Developer setup guide

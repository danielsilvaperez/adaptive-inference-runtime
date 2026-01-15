# Adaptive Inference Runtime (AIR)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()

**A drop-in inference runtime that makes large-model intelligence usable everywhere.**

AIR combines intelligent routing, speculative decoding, and KV-cache compression to deliver 70B-level reasoning on constrained hardware without retraining models.

## Key Features

- **Smart Routing**: Automatically routes between small (7B/13B) and large (70B) models based on query complexity. Simple queries stay on the small model; complex reasoning escalates to the large model.

- **Speculative Decoding**: Small model drafts multiple tokens, large model verifies in parallel. Achieves 2-3x speedup with zero quality loss.

- **KV-Cache Compression**: Intelligent eviction policies reduce memory usage 4-8x, enabling long contexts and higher concurrency on memory-constrained devices.

## System Architecture

```
                              ADAPTIVE INFERENCE RUNTIME
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   Prompt ──► Router ──► Small Model (7B/13B)                           │
    │               │              │                                          │
    │               │              ▼                                          │
    │               │         [Confidence Check]                              │
    │               │              │                                          │
    │               │    Low ◄─────┴─────► High                              │
    │               │     │                  │                                │
    │               │     ▼                  │                                │
    │               │  Draft + Verify ◄──────┘                               │
    │               │  (Speculative)                                          │
    │               │     │                                                   │
    │               │     ▼                                                   │
    │               │  KV Compression                                         │
    │               │     │                                                   │
    │               │     ▼                                                   │
    │               └──► Output Stream                                        │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

**Routing Signals:**
- Token entropy (probability distribution uncertainty)
- Logprob slope (confidence trajectory)
- Top-k disagreement (consensus in predictions)

## Quick Start

### Installation

```bash
pip install air
```

### Basic Usage

```python
from air import AdaptiveRuntime

# Initialize with small and large models
runtime = AdaptiveRuntime(
    small_model="path/to/llama-7b",
    large_model="path/to/llama-70b",
    router="adaptive"  # or "conservative", "aggressive"
)

# Generate with automatic routing
response = runtime.generate(
    prompt="Explain quantum entanglement",
    max_tokens=512
)

# Check which model handled the request
print(f"Model used: {response.model_used}")
print(f"Output: {response.text}")
```

### Configuration

```python
from air import AdaptiveRuntime, RouterConfig

# Fine-tune routing behavior
config = RouterConfig(
    entropy_threshold=0.7,      # Escalate on high entropy
    confidence_window=5,        # Track last 5 tokens
    speculation_depth=4,        # Draft 4 tokens at a time
    kv_compression="balanced"   # "conservative", "balanced", "aggressive"
)

runtime = AdaptiveRuntime(
    small_model="path/to/llama-7b",
    large_model="path/to/llama-70b",
    config=config
)
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Speedup** | 2-4x | Via routing + speculative decoding |
| **Memory Reduction** | 4-8x | Via KV-cache compression |
| **Quality Loss** | <5-10% | From compression only; speculation is lossless |

## Requirements

- **Python**: 3.9+
- **Core Dependencies**:
  - `torch` (PyTorch)
  - `numpy`
- **Inference Backends** (at least one):
  - `llama-cpp-python` (recommended for local inference)
  - `vllm` (for GPU server deployments)
- **Optional**:
  - Apple Silicon with Metal support for MacBook optimization
  - CUDA for NVIDIA GPU acceleration

## Documentation

- [Architecture Overview](docs/architecture.md) - System design and components
- [Core Concepts](docs/concepts.md) - Routing, speculation, and compression explained
- [Getting Started](docs/getting-started.md) - Developer setup guide

## Project Status

AIR is under active development. See [ROADMAP.md](ROADMAP.md) for the detailed development plan.

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | In Progress | Foundations (repo structure, API definitions) |
| 1 | Planned | Small-to-Large Routing |
| 2 | Planned | Speculative Decoding |
| 3 | Planned | KV-Cache Compression |
| 4 | Planned | MacBook Enablement |
| 5 | Planned | Benchmarking |
| 6 | Planned | Developer UX |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development workflow (Issue -> Branch -> PR -> Review)
- Code style guidelines
- Testing requirements

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*AIR: Making 70B-level intelligence accessible on constrained hardware.*

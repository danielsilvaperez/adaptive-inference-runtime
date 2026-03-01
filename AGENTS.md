# AGENTS.md

This file provides guidance to OpenAI coding agents (including ChatGPT, Codex, and GPT-based assistants) when working with code in this repository.

## Project Overview

**Adaptive Inference Runtime (AIR)** is a drop-in inference runtime that makes large-model intelligence usable everywhere by combining:
- **Small→Large model routing**: Run 7B/13B by default, escalate to 70B only when uncertainty is detected
- **Speculative decoding**: 2-3× faster decoding via small model draft + large model verification
- **KV-cache compression**: 4-8× memory reduction through eviction policies and quantization
- **Optional quantization awareness**

**Core Goal**: "Achieve 70B-level reasoning behavior on constrained hardware by spending compute only when intelligence is needed."

**Target**: Make 70B models usable on MacBooks (M3 Pro with 18GB RAM) while maintaining quality.

## Tech Stack

- **Language**: Python + C++ bindings
- **Inference Backends**: llama.cpp (primary), vLLM (secondary)
- **Target Models**: Small (7B/13B), Large (70B local quantized or remote)
- **Testing**: pytest
- **Platform Focus**: Apple Silicon (Metal), with Linux/CUDA support

## Development Workflow

This project uses a professional Issue → Branch → PR → Review → Merge workflow.

"pytest" to test and also ensure it's passing lint checks.

Conda env name: adaptive-inference-runtime

### Branch Naming
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests

### Code Style
- PEP 8 compliance
- Type hints required on function signatures
- Maximum line length: 100 characters
- Docstrings for all public functions/classes

### Commands (once package is set up)
```bash
# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/
pytest tests/unit/test_router.py  # single file

# Run with coverage
pytest --cov=air tests/
```

## Agent Cadence

- Start by pulling `origin/main`, then merge or cherry-pick any agent-created branches
  into `main`.
- Delete merged or empty branches locally and on origin to keep a single-branch repo.
- Push updates promptly with clean, descriptive commits.

## Architecture

```
Prompt
  ↓
Router ── decides token difficulty (entropy, logprob slope, top-k disagreement)
  ↓
Small Model (fast, cheap)
  ↓ (if uncertainty spike)
Speculative Draft + Big Model Verification
  ↓
KV Cache Compression (memory control)
  ↓
Output Stream
```

### Key Interfaces (to be implemented)
- `InferenceState`: State tracking for current model, KV cache, generation params
- `Router`: Routing decisions via `route(state) -> ModelSelection`
- `ModelAdapter`: Unified interface for llama.cpp/vLLM (`generate()`, `verify()`, `get_logits()`)

### Routing Signals
- Token entropy (Shannon entropy from softmax)
- Logprob slope (confidence trajectory over sliding window)
- Top-k disagreement (consensus in top predictions)
- Attention instability (attention pattern variance)

## Project Structure (Planned)

```
/air           # Core package
/tests         # Unit and integration tests
  /unit
  /integration
/benchmarks    # Performance benchmarks
/examples      # Example applications
/configs       # Configuration presets
```

## Development Phases

The project follows a 6-phase roadmap (see ROADMAP.md):

| Phase | Focus | Key Dependencies |
|-------|-------|------------------|
| 0 | Foundations | None (start here) |
| 1 | Small→Large Routing | Phase 0.2 (Core APIs) |
| 2 | Speculative Decoding | Phase 1.1 (Model Adapters) |
| 3 | KV Cache Compression | Phase 1.1 (Model Adapters) |
| 4 | MacBook Enablement | Phases 1, 3 |
| 5 | Benchmarking | Phases 1, 2, 3 |
| 6 | Developer UX | Phase 1 |

### Critical Path
1. Core API Definitions (0.2) → blocks all implementation
2. Model Adapters (1.1) → blocks routing and speculation
3. Confidence Scoring (1.2) → Router Logic (1.3) → Integration (1.4)
4. KV Infrastructure (3.1) → Eviction Policies (3.2) → Evaluation (3.4)

### Parallelizable Tasks
- Confidence scoring metrics (1.2.x) - all independent
- KV eviction policies (3.2.x) - all independent
- Baseline benchmarks (5.2.x) - all independent
- Phase 6 tasks (config, visualization, docs, packaging) - mostly independent

## Key Files

- `ROADMAP.md`: Authoritative 6-phase development plan with 22 tasks
- `CONTRIBUTING.md`: Workflow and code standards
- `ISSUES.md`: Pre-generated GitHub issue templates
- `.github/TEMPLATES/`: PR and issue templates

## Performance Targets

- **Speedup**: 2-4× via routing + speculation
- **Memory**: 4-8× reduction via KV compression
- **Quality**: <5-10% degradation from compression, zero loss from speculation

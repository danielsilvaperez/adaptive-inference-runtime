# Project Name (working): Adaptive Inference Runtime (AIR)

## What it is

A drop-in inference runtime that makes large-model intelligence usable everywhere by combining:

- Small→Large model routing
- Speculative decoding
- KV-cache compression
- (Optional) quantization awareness — but not required

This is runtime-only, model-agnostic, and works today.

⸻

## Why this is the best choice (no fluff)

| Criterion | Why AIR wins |
|-----------|--------------|
| 🔥 Novelty | No unified runtime does all of this together |
| 🚀 Performance | 2–4× speedups + memory wins |
| 💻 Local-first | Makes 70B usable on laptops |
| 🏢 Frontier labs | Direct infra cost savings |
| 🧩 No retraining | Works with existing models |
| 🧠 Research-grade | Publishable + benchmarkable |
| 🌍 OSS-friendly | llama.cpp / vLLM / MLX hooks |

This will get attention if executed cleanly.

⸻

## 🎯 CORE GOAL

"Achieve 70B-level reasoning behavior on constrained hardware by spending compute only when intelligence is needed."

⸻

## 🧠 SYSTEM OVERVIEW (Mental Model)

```
Prompt
  ↓
Router ── decides token difficulty
  ↓
Small Model (fast, cheap)
  ↓ (if uncertainty spike)
Speculative Draft + Big Model Verification
  ↓
KV Cache Compression (memory control)
  ↓
Output Stream
```

⸻

## 🛠️ IMPLEMENTATION PLAN (COMPREHENSIVE TODO)

⸻

### 📖 TASK LEGEND

**Dependency Markers:**
- `[PARALLEL]` - Can be worked on independently, suitable for parallel agent execution
- `[SEQUENTIAL]` - Must be completed in order, depends on previous tasks
- `[BLOCKED BY: Task-ID]` - Cannot start until specified task is complete

**Priority Markers:**
- `[P0]` - Critical path, blocks other work
- `[P1]` - High priority, important for core functionality
- `[P2]` - Medium priority, enhances functionality
- `[P3]` - Low priority, nice to have

**Task Status:**
- `[ ]` - Not started
- `[→]` - In progress
- `[✓]` - Completed

⸻

## PHASE 0 — Foundations (Week 0)

**Goal:** Repo + architecture + scope locked

**Phase Dependencies:** None (starting point)

### ✅ Key Decisions Made
- Language: Python + C++ bindings
- Backends (v1): llama.cpp (primary), vLLM (secondary)
- Target models: Small (7B/13B), Large (70B local quant OR remote)

### Actionable Tasks

#### Task 0.1: Repository Structure Setup `[P0]` `[PARALLEL]`
- [ ] **0.1.1** Create core directory structure (`/air`, `/tests`, `/benchmarks`, `/examples`)
- [ ] **0.1.2** Set up Python package structure (`setup.py`, `pyproject.toml`)
- [ ] **0.1.3** Initialize logging configuration and utilities
- [ ] **0.1.4** Create `.gitignore` for Python, C++, and model artifacts
- [ ] **0.1.5** Set up CI/CD workflows (linting, testing, type checking)

**Acceptance Criteria:** Clean repo structure, installable package, CI runs pass

#### Task 0.2: Core API Definitions `[P0]` `[SEQUENTIAL]`
These tasks define contracts that other phases depend on.

- [ ] **0.2.1** Define `InferenceState` API
  - State tracking (current model, KV cache, generation params)
  - Serialization/deserialization methods
  - State transition methods
- [ ] **0.2.2** Define `Router` interface
  - `route(state) -> ModelSelection` method
  - Confidence scoring interface
  - Escalation decision interface
- [ ] **0.2.3** Define `ModelAdapter` interface
  - `generate()`, `verify()`, `get_logits()` methods
  - Unified interface for llama.cpp and vLLM
  - KV cache access methods
- [ ] **0.2.4** Define common types and data structures (`types.py`)
  - Token, Logits, Entropy types
  - ModelSelection dataclass (model_id, confidence_score, reason)
  - Configuration dataclasses

**Acceptance Criteria:** Well-documented interfaces with type hints, stub implementations pass type checking

#### Task 0.3: Documentation Foundation `[P1]` `[PARALLEL]`
- [ ] **0.3.1** Create API documentation structure
- [ ] **0.3.2** Write architecture overview document
- [ ] **0.3.3** Document core concepts (routing, speculation, compression)
- [ ] **0.3.4** Create developer setup guide

**Acceptance Criteria:** Clear documentation for new contributors

⸻

## PHASE 1 — Small → Large Routing (Week 1)

**Goal:** Intelligent routing between small and large models

**Phase Dependencies:** `[BLOCKED BY: Task 0.2]` (requires core APIs)

**This is the backbone** - enables selective compute allocation

### Core Concept
Run small model by default, escalate to big model only when needed based on confidence signals.

### Routing Signals (cheap + effective)
- Token entropy (probability distribution uncertainty)
- Logprob slope (confidence trajectory)
- Top-k disagreement (consensus in top predictions)
- Attention instability (attention pattern variance)

### Actionable Tasks

#### Task 1.1: Model Adapters Implementation `[P0]` `[SEQUENTIAL]`
Must be completed before routing logic.

- [ ] **1.1.1** Implement llama.cpp adapter
  - Load model, tokenizer
  - Generate tokens, get logits
  - Extract attention weights (if available)
- [ ] **1.1.2** Implement vLLM adapter (basic)
  - API integration
  - Logits extraction
- [ ] **1.1.3** Write adapter tests (mock models for speed)
- [ ] **1.1.4** Benchmark adapter overhead

**Acceptance Criteria:** Both adapters generate tokens, expose logits, tests pass

#### Task 1.2: Confidence Scoring System `[P0]` `[PARALLEL]`
These can be developed independently and combined later.

- [ ] **1.2.1** Implement token entropy calculation
  - Shannon entropy from softmax distribution
  - Configurable temperature parameter
  - Unit tests with known distributions
- [ ] **1.2.2** Implement logprob slope tracker
  - Track confidence over last N tokens
  - Detect sharp drops (uncertainty spikes)
  - Sliding window implementation
- [ ] **1.2.3** Implement top-k disagreement metric
  - Compare top-k predictions overlap
  - Quantify consensus level
- [ ] **1.2.4** Implement attention instability detector (optional)
  - Variance in attention patterns
  - Requires attention weight extraction

**Acceptance Criteria:** Each metric produces sensible scores, validated with test cases

#### Task 1.3: Router Logic `[P0]` `[SEQUENTIAL]` `[BLOCKED BY: Task 1.2]`
Requires confidence scoring to be functional.

- [ ] **1.3.1** Implement routing decision engine
  - Combine confidence signals with weighted formula
  - Configurable thresholds per signal
  - Decision logging for debugging
- [ ] **1.3.2** Define escalation thresholds (empirical tuning)
  - Conservative (rarely escalate)
  - Balanced (default)
  - Aggressive (escalate often)
- [ ] **1.3.3** Implement fallback mechanism mid-generation
  - State transfer from small to large model
  - Seamless continuation
  - Handle KV cache handoff
- [ ] **1.3.4** Implement streaming output continuity
  - Buffer tokens during model switch
  - Maintain consistent output stream

**Acceptance Criteria:** Router correctly selects model based on confidence, seamless transitions

#### Task 1.4: Integration & Testing `[P1]` `[SEQUENTIAL]` `[BLOCKED BY: Task 1.3]`
- [ ] **1.4.1** End-to-end integration test
  - Simple questions → small model only
  - Hard questions → escalation to large model
- [ ] **1.4.2** Create test prompts dataset (easy, medium, hard)
- [ ] **1.4.3** Measure routing accuracy vs ground truth
- [ ] **1.4.4** Profile performance overhead of routing logic

**Deliverable:** "7B answers easy questions, 70B activates only on hard spans."

⸻

## PHASE 2 — Speculative Decoding (Week 2)

**Goal:** 2–3× faster decoding with zero quality loss

**Phase Dependencies:** `[BLOCKED BY: Task 1.1]` (requires model adapters)

**This is the speed multiplier** - amortizes large model cost across multiple tokens

### Core Concept
Small model drafts multiple tokens → large model verifies multiple tokens at once in parallel.

### Actionable Tasks

#### Task 2.1: Draft Generation `[P0]` `[PARALLEL]`
- [ ] **2.1.1** Implement draft window generation (k tokens, configurable)
  - Small model generates k candidate tokens
  - Fast, greedy or sampling-based
  - Store logprobs for each candidate
- [ ] **2.1.2** Implement draft quality heuristics
  - Predict which drafts likely to be accepted
  - Adaptive draft length based on confidence
- [ ] **2.1.3** Test draft generation performance
  - Measure tokens/second for small model
  - Validate draft quality on test prompts

**Acceptance Criteria:** Small model efficiently generates draft sequences

#### Task 2.2: Parallel Verification `[P0]` `[SEQUENTIAL]` `[BLOCKED BY: Task 2.1]`
- [ ] **2.2.1** Implement big model parallel verification
  - Verify all k draft tokens in single forward pass
  - Extract logits for each position
  - Compare with draft model predictions
- [ ] **2.2.2** Implement token acceptance logic
  - Accept tokens while agreement holds
  - Rejection sampling at first disagreement
  - Handle partial acceptance gracefully
- [ ] **2.2.3** Implement rejection handling
  - Fallback to big model generation from rejection point
  - Maintain generation quality
  - Log rejection statistics

**Acceptance Criteria:** Big model verifies drafts correctly, handles rejections

#### Task 2.3: Speculation Pipeline `[P0]` `[SEQUENTIAL]` `[BLOCKED BY: Task 2.2]`
- [ ] **2.3.1** Integrate draft + verify into unified pipeline
  - Coordinate small and large model execution
  - Manage state between draft and verify phases
  - Handle edge cases (EOS in draft, empty drafts)
- [ ] **2.3.2** Implement adaptive draft length tuning
  - Increase k when acceptance rate high
  - Decrease k when acceptance rate low
  - Per-prompt or global tuning
- [ ] **2.3.3** Add speculation metrics tracking
  - Acceptance rate, draft length, speedup
  - Per-session and aggregate statistics

**Acceptance Criteria:** Speculation pipeline runs end-to-end

#### Task 2.4: Benchmarking `[P1]` `[PARALLEL]`
- [ ] **2.4.1** Latency benchmarking suite
  - Measure tokens/second with/without speculation
  - Various prompt types and lengths
  - Compare against baseline (big model only)
- [ ] **2.4.2** Quality validation
  - Verify output identical to big model solo
  - Check for any quality degradation
  - Statistical analysis of differences (should be zero)
- [ ] **2.4.3** Acceptance rate analysis
  - Which prompts have high/low acceptance
  - Correlation with draft quality heuristics

**Deliverable:** 2–3× faster decoding with zero quality loss (verified by benchmarks)

⸻

## PHASE 3 — KV Cache Compression (Week 3)

**Goal:** 4–8× KV memory reduction with <5–10% quality drop

**Phase Dependencies:** `[BLOCKED BY: Task 1.1]` (requires model adapters with KV cache access)

**This unlocks long context + concurrency** - memory is the bottleneck

### Techniques (pick 1–2 initially)
- Sliding window + heavy hitter retention
- Attention-weight eviction (H2O-style)
- Optional: int8 KV quantization

### Actionable Tasks

#### Task 3.1: KV Cache Infrastructure `[P0]` `[SEQUENTIAL]`
Foundation for compression techniques.

- [ ] **3.1.1** Implement KV cache introspection
  - Access KV tensors from model adapters
  - Track cache size and memory usage
  - Monitor per-layer cache statistics
- [ ] **3.1.2** Create KV cache manager abstraction
  - Wrap native cache implementations
  - Provide eviction hook interface
  - Support cache serialization/restore
- [ ] **3.1.3** Implement memory profiling hooks
  - Real-time memory usage tracking
  - Peak memory detection
  - Per-request memory attribution

**Acceptance Criteria:** Can inspect, measure, and manage KV cache programmatically

#### Task 3.2: Eviction Policies `[P0]` `[PARALLEL]`
These compression techniques can be developed independently.

- [ ] **3.2.1** Implement sliding window eviction
  - Keep last N tokens always
  - Configurable window size
  - Test on long sequences
- [ ] **3.2.2** Implement heavy hitter retention
  - Track attention scores for each cached token
  - Retain tokens with high cumulative attention
  - Evict low-attention tokens first
- [ ] **3.2.3** Implement H2O-style eviction (optional)
  - Attention-weight based eviction
  - Per-layer or global policy
  - Configurable retention ratio
- [ ] **3.2.4** Implement int8 KV quantization (optional)
  - Quantize KV tensors to int8
  - Dequantize on attention computation
  - Measure quality impact

**Acceptance Criteria:** Each policy reduces memory, measurable impact on quality

#### Task 3.3: Safety & Quality Guards `[P1]` `[PARALLEL]`
- [ ] **3.3.1** Implement per-task safety guards
  - Disable compression for code generation (precise context needed)
  - Disable for retrieval/question answering (context sensitivity requirements)
  - Configurable per use case
- [ ] **3.3.2** Implement quality monitoring
  - Detect quality degradation automatically
  - Fallback to no compression if quality drops
  - Adaptive compression ratio
- [ ] **3.3.3** Create compression presets
  - Conservative (minimal compression, high quality)
  - Balanced (moderate compression, good quality)
  - Aggressive (maximum compression, acceptable quality)

**Acceptance Criteria:** Guards prevent quality issues, presets make compression easy to use

#### Task 3.4: Evaluation `[P1]` `[SEQUENTIAL]` `[BLOCKED BY: Task 3.2]`
- [ ] **3.4.1** Memory benchmarks
  - Measure memory reduction per policy
  - Peak memory vs average memory
  - Memory over sequence length
- [ ] **3.4.2** Quality benchmarks
  - Perplexity on held-out data
  - Task-specific metrics (accuracy, F1, etc.)
  - Compare compressed vs uncompressed
- [ ] **3.4.3** Latency impact analysis
  - Eviction overhead measurement
  - End-to-end latency with compression

**Deliverable:** 4–8× KV memory reduction with <5–10% quality drop (validated by benchmarks)

⸻

## PHASE 4 — MacBook Enablement (Week 4)

**Goal:** Make 70B usable on M3 Pro (18GB RAM)

**Phase Dependencies:** `[BLOCKED BY: Phase 1, Phase 3]` (requires routing + compression)

**This is the viral demo phase** - tangible, shareable result

### Strategy
- Small model always local
- Big model: ultra-sparse local (paged/quant optional) OR remote fallback

### Actionable Tasks

#### Task 4.1: Metal Backend Support `[P0]` `[SEQUENTIAL]`
- [ ] **4.1.1** Implement Metal backend compatibility
  - Verify llama.cpp Metal support
  - Test model loading on Metal
  - Benchmark Metal vs CPU performance
- [ ] **4.1.2** Optimize for Unified Memory Architecture
  - Leverage shared CPU/GPU memory
  - Minimize memory copies
  - Profile memory bandwidth usage
- [ ] **4.1.3** Implement Metal-specific optimizations
  - Use Metal Performance Shaders if beneficial
  - Optimize kernel execution

**Acceptance Criteria:** Models run efficiently on Apple Silicon

#### Task 4.2: Memory Management `[P0]` `[PARALLEL]`
- [ ] **4.2.1** Implement unified memory pressure detection
  - Monitor system memory availability
  - Detect low memory conditions
  - Trigger eviction or swap proactively
- [ ] **4.2.2** Implement SSD paging fallback
  - Swap KV cache to disk when memory low
  - Asynchronous paging for minimal latency
  - LRU eviction for cache pages
- [ ] **4.2.3** Optimize memory allocation strategy
  - Pre-allocate buffers where possible
  - Reuse memory across requests
  - Minimize fragmentation

**Acceptance Criteria:** System gracefully handles memory pressure, doesn't crash

#### Task 4.3: Hybrid Local/Remote Execution `[P1]` `[SEQUENTIAL]` `[BLOCKED BY: Task 4.1, 4.2]`
- [ ] **4.3.1** Implement seamless local↔remote escalation
  - Detect when local execution infeasible
  - Fallback to remote API (OpenAI, Anthropic, or custom)
  - Serialize state for remote execution
- [ ] **4.3.2** Implement remote model adapter
  - REST API client for remote inference
  - Handle authentication and rate limits
  - Error handling and retries
- [ ] **4.3.3** Optimize local/remote decision logic
  - Cost/latency tradeoffs
  - Network availability checks
  - User preferences (local-only, hybrid, remote-ok)

**Acceptance Criteria:** Seamlessly uses remote when local insufficient

#### Task 4.4: MacBook Demo & Optimization `[P1]` `[SEQUENTIAL]` `[BLOCKED BY: Task 4.3]`
- [ ] **4.4.1** Create demo application
  - Simple CLI or web UI
  - Show which model answering in real-time
  - Display memory usage and performance stats
- [ ] **4.4.2** Test on M3 Pro (18GB)
  - Measure end-to-end latency
  - Verify memory stays within limits
  - Test long conversations (context management)
- [ ] **4.4.3** Create video demo
  - Record demo showcasing 70B on MacBook
  - Show memory usage, latency, quality
  - Shareable for marketing

**Deliverable:** "My Mac feels like it's running a 70B." (with proof)

⸻

## PHASE 5 — Benchmarking + Proof (Week 5)

**Goal:** Hard numbers, no vibes

**Phase Dependencies:** `[BLOCKED BY: Phase 1, Phase 2, Phase 3]` (need all features working)

**This is where attention comes from** - credibility through rigorous evaluation

### Actionable Tasks

#### Task 5.1: Benchmark Infrastructure `[P0]` `[SEQUENTIAL]`
- [ ] **5.1.1** Create reproducible benchmark harness
  - Standardized evaluation protocol
  - Fixed random seeds for reproducibility
  - Configurable evaluation parameters
- [ ] **5.1.2** Implement automated dataset loading
  - GSM8K (reasoning)
  - HumanEval (coding)
  - Long-context QA datasets (e.g., LongBench)
  - Custom internal test sets
- [ ] **5.1.3** Implement evaluation metrics
  - Accuracy, F1, exact match
  - Pass@k for code
  - Perplexity, ROUGE for generation
  - Latency (tokens/second, TTFT, ITL)
  - Memory (peak, average, per-token)
- [ ] **5.1.4** Create result logging and aggregation
  - Store results in structured format (JSON, CSV)
  - Statistical significance tests
  - Automatic report generation

**Acceptance Criteria:** Benchmark harness runs reliably, produces reproducible results

#### Task 5.2: Baseline Comparisons `[P0]` `[PARALLEL]`
These baselines can be evaluated independently.

- [ ] **5.2.1** Benchmark small model only (7B/13B)
  - All datasets
  - Record quality, latency, memory
- [ ] **5.2.2** Benchmark large model only (70B)
  - All datasets
  - Record quality, latency, memory
- [ ] **5.2.3** Benchmark AIR (hybrid system)
  - All configurations (routing only, +speculation, +compression)
  - Record quality, latency, memory, routing decisions

**Acceptance Criteria:** Complete baseline numbers for comparison

#### Task 5.3: Analysis & Visualization `[P1]` `[SEQUENTIAL]` `[BLOCKED BY: Task 5.2]`
- [ ] **5.3.1** Generate comparison charts
  - Quality vs latency tradeoffs
  - Memory usage over time
  - Speedup vs acceptance rate (speculation)
  - Routing decision distributions
- [ ] **5.3.2** Statistical analysis
  - Significance testing
  - Confidence intervals
  - Outlier analysis
- [ ] **5.3.3** Error analysis
  - Where does AIR fail vs baselines?
  - When does routing make wrong decisions?
  - Quality degradation patterns
- [ ] **5.3.4** Create summary report
  - Executive summary with key findings
  - Detailed methodology
  - Full results tables
  - Recommendations and future work

**Acceptance Criteria:** Publication-ready results with clear insights

#### Task 5.4: Publication Prep `[P2]` `[PARALLEL]`
- [ ] **5.4.1** Write technical blog post
  - Explain approach clearly
  - Show impressive results
  - Include visuals and demos
- [ ] **5.4.2** Create GitHub showcase
  - Update README with results
  - Add badges (performance, memory savings)
  - Link to benchmarks and demo
- [ ] **5.4.3** Prepare academic paper (optional)
  - Position as systems contribution
  - Target systems conferences (MLSys, OSDI)
  - Coordinate with co-authors

**Deliverable:** Hard numbers demonstrating 2–4× speedup, 4–8× memory reduction, comparable quality

⸻

## PHASE 6 — Developer UX (Week 6)

**Goal:** Make AIR easy to adopt

**Phase Dependencies:** `[BLOCKED BY: Phase 1]` (core functionality needed)

**Adoption matters** - great tech needs great UX

### Actionable Tasks

#### Task 6.1: Configuration System `[P0]` `[PARALLEL]`
- [ ] **6.1.1** Design configuration schema
  - YAML or TOML format
  - Model paths, routing params, compression settings
  - Presets for common scenarios
- [ ] **6.1.2** Implement config parsing and validation
  - Schema validation
  - Sensible defaults
  - Error messages for invalid configs
- [ ] **6.1.3** Create example configurations
  - Local-only setup
  - Hybrid local/remote
  - Maximum performance vs maximum memory savings
- [ ] **6.1.4** Implement runtime config override
  - Command-line flags override config file
  - Environment variable support

**Acceptance Criteria:** Easy-to-use configuration with good defaults

#### Task 6.2: Command-Line Interface `[P0]` `[SEQUENTIAL]` `[BLOCKED BY: Task 6.1]`
- [ ] **6.2.1** Implement `air run` command
  - Load configuration
  - Initialize models and router
  - Start interactive session or batch processing
- [ ] **6.2.2** Implement `air benchmark` command
  - Run standard benchmarks
  - Generate reports
- [ ] **6.2.3** Implement `air serve` command (optional)
  - Start HTTP API server
  - OpenAI-compatible API
  - Load balancing across multiple instances
- [ ] **6.2.4** Add helpful CLI features
  - `--help` for all commands
  - `--verbose` for debugging
  - `--dry-run` to validate config

**Example CLI:**
```bash
# Using model paths or names (supports various model formats)
air run --small /path/to/model-7b --big /path/to/model-70b --router adaptive
air run --small llama-7b --big llama-70b --router adaptive  # or model registry names

# Benchmarking and serving
air benchmark --dataset gsm8k --config configs/balanced.yaml
air serve --port 8080 --workers 4
```

**Acceptance Criteria:** Intuitive CLI that works for common use cases

#### Task 6.3: Runtime Visualization `[P1]` `[PARALLEL]`
- [ ] **6.3.1** Implement runtime visualization
  - Show which model answered each token
  - Display confidence scores in real-time
  - Show memory usage and latency
- [ ] **6.3.2** Create debug traces
  - Detailed logs of routing decisions
  - Speculation acceptance/rejection
  - Cache eviction events
- [ ] **6.3.3** Implement telemetry (optional)
  - Prometheus metrics export
  - Grafana dashboards
  - Performance monitoring

**Acceptance Criteria:** Users can understand what AIR is doing

#### Task 6.4: Documentation & Examples `[P1]` `[PARALLEL]`
- [ ] **6.4.1** Write comprehensive user guide
  - Installation instructions
  - Quick start tutorial
  - Configuration reference
  - Troubleshooting guide
- [ ] **6.4.2** Create example applications
  - Simple chatbot
  - Code completion tool
  - Question answering system
- [ ] **6.4.3** Create video tutorials
  - Setup and installation
  - Running first inference
  - Configuring for different use cases
- [ ] **6.4.4** API documentation
  - Python API reference
  - REST API documentation (if applicable)

**Acceptance Criteria:** New users can get started in <10 minutes

#### Task 6.5: Packaging & Distribution `[P2]` `[PARALLEL]`
- [ ] **6.5.1** Create pip package
  - Publish to PyPI
  - Include all dependencies
  - Test installation on clean systems
- [ ] **6.5.2** Create Docker images
  - CPU and GPU variants
  - Include model download scripts
  - Optimize image size
- [ ] **6.5.3** Create conda package (optional)
  - Publish to conda-forge
  - Easy installation with conda
- [ ] **6.5.4** Create homebrew formula (optional, for macOS)

**Deliverable:** AIR is easy to install, configure, and use for new developers

⸻

## 📈 WHAT THIS ENABLES (Big Picture)
- Local 70B experience
- Cheaper frontier inference
- Mobile / edge intelligence
- Agent systems that don't waste tokens
- Future research into conditional compute

⸻

## 🧠 WHY OPENAI / ANTHROPIC WOULD CARE
- Cuts infra costs
- Improves latency
- Plays well with existing APIs
- No retraining
- Clear theoretical grounding

⸻

## 🎯 PARALLELIZATION SUMMARY

### Tasks That Can Be Run in Parallel (Agent-Friendly):

**Phase 0:**
- Task 0.1 (Repository Structure Setup) - can run independently
- Task 0.3 (Documentation Foundation) - can run alongside API definitions

**Phase 1:**
- All subtasks in Task 1.2 (Confidence Scoring) - independent metrics

**Phase 2:**
- Task 2.1 (Draft Generation) - can start once adapters ready
- Task 2.4 (Benchmarking) - can be prepared in parallel with pipeline development

**Phase 3:**
- All subtasks in Task 3.2 (Eviction Policies) - independent implementations
- Task 3.3 (Safety Guards) - can develop alongside eviction policies

**Phase 4:**
- Task 4.2 (Memory Management) - can develop alongside Metal backend

**Phase 5:**
- All subtasks in Task 5.2 (Baseline Comparisons) - independent benchmarks
- Task 5.4 (Publication Prep) - can start once results are ready

**Phase 6:**
- Task 6.1 (Configuration System) - independent of other Phase 6 tasks initially
- Task 6.3 (Runtime Visualization) - can develop independently
- Task 6.4 (Documentation & Examples) - can develop independently
- Task 6.5 (Packaging & Distribution) - can develop independently

### Critical Path (Must Be Sequential):
**Note:** Task IDs referenced below should be kept synchronized if tasks are reorganized.

1. Phase 0: Task 0.2 (Core API Definitions) - blocks all implementation work
2. Phase 1: Task 1.1 (Model Adapters) - blocks routing and speculation
3. Phase 1: Task 1.2 → 1.3 → 1.4 (Confidence → Router → Testing)
4. Phase 2: Task 2.1 → 2.2 → 2.3 (Draft → Verify → Pipeline)
5. Phase 3: Task 3.1 → 3.2 → 3.4 (Infrastructure → Policies → Evaluation)
6. Phase 4: Task 4.1 → 4.3 → 4.4 (Metal → Hybrid → Demo)
7. Phase 5: Task 5.1 → 5.2 → 5.3 (Infrastructure → Benchmarks → Analysis)
8. Phase 6: Task 6.1 → 6.2 (Config → CLI)

**Recommendation:** Focus agent parallelization on confidence scoring metrics (Phase 1), eviction policies (Phase 3), and benchmarking/documentation tasks (Phase 5-6) for maximum efficiency.

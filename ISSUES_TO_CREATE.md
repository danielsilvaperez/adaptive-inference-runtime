# Parallel Tasks - GitHub Issues to Create

This document contains all 22 parallel tasks identified from ROADMAP.md that can be worked on independently.

## Summary

| Phase | Task Count | Priority Distribution |
|-------|------------|----------------------|
| Phase 0 | 2 | P0: 1, P1: 1 |
| Phase 1 | 4 | P0: 4 |
| Phase 2 | 2 | P0: 1, P1: 1 |
| Phase 3 | 5 | P0: 4, P1: 1 |
| Phase 4 | 1 | P0: 1 |
| Phase 5 | 4 | P0: 3, P2: 1 |
| Phase 6 | 4 | P0: 1, P1: 2, P2: 1 |
| **Total** | **22** | **P0: 15, P1: 5, P2: 2** |

## How to Create Issues

### Option 1: Using the Python Script (Recommended)

```bash
# Set your GitHub token (create one at https://github.com/settings/tokens)
export GITHUB_TOKEN='your_token_here'

# Run the script
python3 create_parallel_issues.py
```

### Option 2: Using GitHub CLI

```bash
# Ensure gh CLI is authenticated
gh auth login

# Run the issues creation script with gh
./create_issues_with_gh.sh
```

### Option 3: Manual Creation

Use the issue templates below to manually create each issue through the GitHub web interface.

---

## Issue Templates

### Phase 0: Foundations

#### Issue 1: [Phase 0] Task 0.1: Repository Structure Setup

**Labels:** `phase-0`, `parallel`, `P0`, `infrastructure`

**Body:**
```markdown
## Phase 0 - Task 0.1: Repository Structure Setup
**Priority:** P0 (Critical)
**Type:** PARALLEL - Can be worked on independently
**Phase Dependencies:** None (starting point)

### Description
Set up the foundational repository structure and development infrastructure for the Adaptive Inference Runtime project.

### Subtasks
- [ ] **0.1.1** Create core directory structure (`/air`, `/tests`, `/benchmarks`, `/examples`)
- [ ] **0.1.2** Set up Python package structure (`setup.py`, `pyproject.toml`)
- [ ] **0.1.3** Initialize logging configuration and utilities
- [ ] **0.1.4** Create `.gitignore` for Python, C++, and model artifacts
- [ ] **0.1.5** Set up CI/CD workflows (linting, testing, type checking)

### Acceptance Criteria
- Clean repo structure
- Installable package
- CI runs pass

### Related Documentation
See ROADMAP.md Phase 0 for full context.
```

---

#### Issue 2: [Phase 0] Task 0.3: Documentation Foundation

**Labels:** `phase-0`, `parallel`, `P1`, `documentation`

**Body:**
```markdown
## Phase 0 - Task 0.3: Documentation Foundation
**Priority:** P1 (High)
**Type:** PARALLEL - Can run alongside API definitions
**Phase Dependencies:** None (starting point)

### Description
Create the foundational documentation structure to support contributors and users.

### Subtasks
- [ ] **0.3.1** Create API documentation structure
- [ ] **0.3.2** Write architecture overview document
- [ ] **0.3.3** Document core concepts (routing, speculation, compression)
- [ ] **0.3.4** Create developer setup guide

### Acceptance Criteria
Clear documentation for new contributors

### Related Documentation
See ROADMAP.md Phase 0 for full context.
```

---

### Phase 1: Small → Large Routing

#### Issue 3: [Phase 1] Task 1.2.1: Implement Token Entropy Calculation

**Labels:** `phase-1`, `parallel`, `P0`, `routing`, `confidence-scoring`

**Body:**
```markdown
## Phase 1 - Task 1.2.1: Implement Token Entropy Calculation
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent confidence metric
**Phase Dependencies:** Blocked by Task 0.2 (Core API Definitions)

### Description
Implement token entropy calculation for confidence scoring in the routing system.

### Requirements
- Shannon entropy from softmax distribution
- Configurable temperature parameter
- Unit tests with known distributions

### Acceptance Criteria
Metric produces sensible scores, validated with test cases

### Related Documentation
See ROADMAP.md Phase 1, Task 1.2 for full context.
```

---

#### Issue 4: [Phase 1] Task 1.2.2: Implement Logprob Slope Tracker

**Labels:** `phase-1`, `parallel`, `P0`, `routing`, `confidence-scoring`

**Body:**
```markdown
## Phase 1 - Task 1.2.2: Implement Logprob Slope Tracker
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent confidence metric
**Phase Dependencies:** Blocked by Task 0.2 (Core API Definitions)

### Description
Implement logprob slope tracking for detecting confidence drops in generation.

### Requirements
- Track confidence over last N tokens
- Detect sharp drops (uncertainty spikes)
- Sliding window implementation

### Acceptance Criteria
Metric produces sensible scores, validated with test cases

### Related Documentation
See ROADMAP.md Phase 1, Task 1.2 for full context.
```

---

#### Issue 5: [Phase 1] Task 1.2.3: Implement Top-k Disagreement Metric

**Labels:** `phase-1`, `parallel`, `P0`, `routing`, `confidence-scoring`

**Body:**
```markdown
## Phase 1 - Task 1.2.3: Implement Top-k Disagreement Metric
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent confidence metric
**Phase Dependencies:** Blocked by Task 0.2 (Core API Definitions)

### Description
Implement top-k disagreement metric for measuring prediction consensus.

### Requirements
- Compare top-k predictions overlap
- Quantify consensus level
- Efficient computation

### Acceptance Criteria
Metric produces sensible scores, validated with test cases

### Related Documentation
See ROADMAP.md Phase 1, Task 1.2 for full context.
```

---

#### Issue 6: [Phase 1] Task 1.2.4: Implement Attention Instability Detector (Optional)

**Labels:** `phase-1`, `parallel`, `P0`, `routing`, `confidence-scoring`, `optional`

**Body:**
```markdown
## Phase 1 - Task 1.2.4: Implement Attention Instability Detector
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent confidence metric
**Phase Dependencies:** Blocked by Task 0.2 (Core API Definitions)

### Description
Implement attention instability detection for confidence scoring (optional feature).

### Requirements
- Variance in attention patterns
- Requires attention weight extraction
- Configurable sensitivity

### Acceptance Criteria
Metric produces sensible scores, validated with test cases

### Related Documentation
See ROADMAP.md Phase 1, Task 1.2 for full context.
```

---

### Phase 2: Speculative Decoding

#### Issue 7: [Phase 2] Task 2.1: Draft Generation for Speculative Decoding

**Labels:** `phase-2`, `parallel`, `P0`, `speculative-decoding`

**Body:**
```markdown
## Phase 2 - Task 2.1: Draft Generation
**Priority:** P0 (Critical)
**Type:** PARALLEL - Can start once adapters ready
**Phase Dependencies:** Blocked by Task 1.1 (Model Adapters)

### Description
Implement draft generation for speculative decoding using the small model.

### Subtasks
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

### Acceptance Criteria
Small model efficiently generates draft sequences

### Related Documentation
See ROADMAP.md Phase 2 for full context.
```

---

#### Issue 8: [Phase 2] Task 2.4: Speculative Decoding Benchmarking

**Labels:** `phase-2`, `parallel`, `P1`, `benchmarking`, `speculative-decoding`

**Body:**
```markdown
## Phase 2 - Task 2.4: Benchmarking
**Priority:** P1 (High)
**Type:** PARALLEL - Can be prepared alongside pipeline development
**Phase Dependencies:** Blocked by Task 2.3 (Speculation Pipeline)

### Description
Create comprehensive benchmarking suite for speculative decoding.

### Subtasks
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

### Acceptance Criteria
2–3× faster decoding with zero quality loss (verified by benchmarks)

### Related Documentation
See ROADMAP.md Phase 2 for full context.
```

---

### Phase 3: KV Cache Compression

#### Issue 9: [Phase 3] Task 3.2.1: Implement Sliding Window Eviction

**Labels:** `phase-3`, `parallel`, `P0`, `kv-compression`

**Body:**
```markdown
## Phase 3 - Task 3.2.1: Implement Sliding Window Eviction
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent compression technique
**Phase Dependencies:** Blocked by Task 3.1 (KV Cache Infrastructure)

### Description
Implement sliding window eviction policy for KV cache compression.

### Requirements
- Keep last N tokens always
- Configurable window size
- Test on long sequences

### Acceptance Criteria
Policy reduces memory, measurable impact on quality

### Related Documentation
See ROADMAP.md Phase 3, Task 3.2 for full context.
```

---

#### Issue 10: [Phase 3] Task 3.2.2: Implement Heavy Hitter Retention

**Labels:** `phase-3`, `parallel`, `P0`, `kv-compression`

**Body:**
```markdown
## Phase 3 - Task 3.2.2: Implement Heavy Hitter Retention
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent compression technique
**Phase Dependencies:** Blocked by Task 3.1 (KV Cache Infrastructure)

### Description
Implement heavy hitter retention policy for KV cache compression.

### Requirements
- Track attention scores for each cached token
- Retain tokens with high cumulative attention
- Evict low-attention tokens first

### Acceptance Criteria
Policy reduces memory, measurable impact on quality

### Related Documentation
See ROADMAP.md Phase 3, Task 3.2 for full context.
```

---

#### Issue 11: [Phase 3] Task 3.2.3: Implement H2O-style Eviction (Optional)

**Labels:** `phase-3`, `parallel`, `P0`, `kv-compression`, `optional`

**Body:**
```markdown
## Phase 3 - Task 3.2.3: Implement H2O-style Eviction
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent compression technique
**Phase Dependencies:** Blocked by Task 3.1 (KV Cache Infrastructure)

### Description
Implement H2O-style eviction policy for KV cache compression (optional).

### Requirements
- Attention-weight based eviction
- Per-layer or global policy
- Configurable retention ratio

### Acceptance Criteria
Policy reduces memory, measurable impact on quality

### Related Documentation
See ROADMAP.md Phase 3, Task 3.2 for full context.
```

---

#### Issue 12: [Phase 3] Task 3.2.4: Implement int8 KV Quantization (Optional)

**Labels:** `phase-3`, `parallel`, `P0`, `kv-compression`, `optional`

**Body:**
```markdown
## Phase 3 - Task 3.2.4: Implement int8 KV Quantization
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent compression technique
**Phase Dependencies:** Blocked by Task 3.1 (KV Cache Infrastructure)

### Description
Implement int8 quantization for KV cache tensors (optional).

### Requirements
- Quantize KV tensors to int8
- Dequantize on attention computation
- Measure quality impact

### Acceptance Criteria
Quantization reduces memory, measurable impact on quality

### Related Documentation
See ROADMAP.md Phase 3, Task 3.2 for full context.
```

---

#### Issue 13: [Phase 3] Task 3.3: Safety & Quality Guards

**Labels:** `phase-3`, `parallel`, `P1`, `kv-compression`, `safety`

**Body:**
```markdown
## Phase 3 - Task 3.3: Safety & Quality Guards
**Priority:** P1 (High)
**Type:** PARALLEL - Can develop alongside eviction policies
**Phase Dependencies:** Blocked by Task 3.1 (KV Cache Infrastructure)

### Description
Implement safety guards and quality monitoring for KV cache compression.

### Subtasks
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

### Acceptance Criteria
Guards prevent quality issues, presets make compression easy to use

### Related Documentation
See ROADMAP.md Phase 3 for full context.
```

---

### Phase 4: MacBook Enablement

#### Issue 14: [Phase 4] Task 4.2: Memory Management for MacBook

**Labels:** `phase-4`, `parallel`, `P0`, `memory-management`, `macos`

**Body:**
```markdown
## Phase 4 - Task 4.2: Memory Management
**Priority:** P0 (Critical)
**Type:** PARALLEL - Can develop alongside Metal backend
**Phase Dependencies:** Blocked by Phase 1 and Phase 3 (routing + compression)

### Description
Implement unified memory management for running on constrained hardware like MacBook.

### Subtasks
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

### Acceptance Criteria
System gracefully handles memory pressure, doesn't crash

### Related Documentation
See ROADMAP.md Phase 4 for full context.
```

---

### Phase 5: Benchmarking + Proof

#### Issue 15: [Phase 5] Task 5.2.1: Benchmark Small Model Only

**Labels:** `phase-5`, `parallel`, `P0`, `benchmarking`

**Body:**
```markdown
## Phase 5 - Task 5.2.1: Benchmark Small Model Only
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent baseline evaluation
**Phase Dependencies:** Blocked by Task 5.1 (Benchmark Infrastructure)

### Description
Evaluate small model (7B/13B) alone as a baseline for comparison.

### Requirements
- Run on all datasets (GSM8K, HumanEval, Long-context QA)
- Record quality, latency, memory metrics
- Generate reproducible results

### Acceptance Criteria
Complete baseline numbers for small model

### Related Documentation
See ROADMAP.md Phase 5 for full context.
```

---

#### Issue 16: [Phase 5] Task 5.2.2: Benchmark Large Model Only

**Labels:** `phase-5`, `parallel`, `P0`, `benchmarking`

**Body:**
```markdown
## Phase 5 - Task 5.2.2: Benchmark Large Model Only
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent baseline evaluation
**Phase Dependencies:** Blocked by Task 5.1 (Benchmark Infrastructure)

### Description
Evaluate large model (70B) alone as a baseline for comparison.

### Requirements
- Run on all datasets (GSM8K, HumanEval, Long-context QA)
- Record quality, latency, memory metrics
- Generate reproducible results

### Acceptance Criteria
Complete baseline numbers for large model

### Related Documentation
See ROADMAP.md Phase 5 for full context.
```

---

#### Issue 17: [Phase 5] Task 5.2.3: Benchmark AIR (Hybrid System)

**Labels:** `phase-5`, `parallel`, `P0`, `benchmarking`

**Body:**
```markdown
## Phase 5 - Task 5.2.3: Benchmark AIR (Hybrid System)
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent evaluation
**Phase Dependencies:** Blocked by Task 5.1 (Benchmark Infrastructure) and all Phase 1-3 features

### Description
Evaluate the complete AIR system with all features enabled.

### Requirements
- Run on all datasets (GSM8K, HumanEval, Long-context QA)
- Test all configurations (routing only, +speculation, +compression)
- Record quality, latency, memory, routing decisions
- Generate reproducible results

### Acceptance Criteria
Complete benchmark results for AIR system

### Related Documentation
See ROADMAP.md Phase 5 for full context.
```

---

#### Issue 18: [Phase 5] Task 5.4: Publication Prep

**Labels:** `phase-5`, `parallel`, `P2`, `documentation`, `publication`

**Body:**
```markdown
## Phase 5 - Task 5.4: Publication Prep
**Priority:** P2 (Medium)
**Type:** PARALLEL - Can start once results are ready
**Phase Dependencies:** Blocked by Task 5.3 (Analysis & Visualization)

### Description
Prepare materials for publication and community sharing.

### Subtasks
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

### Acceptance Criteria
Hard numbers demonstrating 2–4× speedup, 4–8× memory reduction, comparable quality

### Related Documentation
See ROADMAP.md Phase 5 for full context.
```

---

### Phase 6: Developer UX

#### Issue 19: [Phase 6] Task 6.1: Configuration System

**Labels:** `phase-6`, `parallel`, `P0`, `configuration`, `developer-ux`

**Body:**
```markdown
## Phase 6 - Task 6.1: Configuration System
**Priority:** P0 (Critical)
**Type:** PARALLEL - Independent of other Phase 6 tasks initially
**Phase Dependencies:** Blocked by Phase 1 (core functionality needed)

### Description
Design and implement a comprehensive configuration system for AIR.

### Subtasks
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

### Acceptance Criteria
Easy-to-use configuration with good defaults

### Related Documentation
See ROADMAP.md Phase 6 for full context.
```

---

#### Issue 20: [Phase 6] Task 6.3: Runtime Visualization

**Labels:** `phase-6`, `parallel`, `P1`, `visualization`, `developer-ux`

**Body:**
```markdown
## Phase 6 - Task 6.3: Runtime Visualization
**Priority:** P1 (High)
**Type:** PARALLEL - Can develop independently
**Phase Dependencies:** Blocked by Phase 1 (core functionality needed)

### Description
Implement runtime visualization and debugging tools.

### Subtasks
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

### Acceptance Criteria
Users can understand what AIR is doing

### Related Documentation
See ROADMAP.md Phase 6 for full context.
```

---

#### Issue 21: [Phase 6] Task 6.4: Documentation & Examples

**Labels:** `phase-6`, `parallel`, `P1`, `documentation`, `developer-ux`

**Body:**
```markdown
## Phase 6 - Task 6.4: Documentation & Examples
**Priority:** P1 (High)
**Type:** PARALLEL - Can develop independently
**Phase Dependencies:** Blocked by Phase 1 (core functionality needed)

### Description
Create comprehensive documentation and example applications.

### Subtasks
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

### Acceptance Criteria
New users can get started in <10 minutes

### Related Documentation
See ROADMAP.md Phase 6 for full context.
```

---

#### Issue 22: [Phase 6] Task 6.5: Packaging & Distribution

**Labels:** `phase-6`, `parallel`, `P2`, `packaging`, `developer-ux`

**Body:**
```markdown
## Phase 6 - Task 6.5: Packaging & Distribution
**Priority:** P2 (Medium)
**Type:** PARALLEL - Can develop independently
**Phase Dependencies:** Blocked by Phase 1 (core functionality needed)

### Description
Create packaging and distribution mechanisms for easy installation.

### Subtasks
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

### Acceptance Criteria
AIR is easy to install, configure, and use for new developers

### Related Documentation
See ROADMAP.md Phase 6 for full context.
```

---

## Notes

- All issues are tagged with their phase, type (`parallel`), priority (P0-P2), and relevant technical areas
- Dependencies are clearly stated in each issue description
- Each issue includes detailed subtasks and acceptance criteria
- Issues can be assigned to different agents/developers for concurrent work

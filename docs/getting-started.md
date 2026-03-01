# Getting Started

This guide covers developer setup for contributing to AIR.

## Prerequisites

### Required

- **Python 3.9+**
- **Git**
- **pip** (or conda)

### Recommended

- **Apple Silicon Mac** (M1/M2/M3) for Metal acceleration testing
- **NVIDIA GPU** with CUDA for vLLM backend testing
- **16GB+ RAM** for running model tests

### Optional (for full testing)

- Small model (7B-13B parameters) in GGUF format
- Access to large model (70B) locally or via API

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/danielsilva010/Adaptive-Inference-Runtime.git
cd Adaptive-Inference-Runtime
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Or using conda
conda create -n air python=3.9
conda activate air
```

### 3. Install Dependencies

```bash
# Core dependencies (when available)
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt

# Install AIR in development mode
pip install -e .
```

### 4. Install Inference Backend

Choose at least one backend:

**llama.cpp (recommended for local development):**
```bash
pip install llama-cpp-python

# For Apple Silicon with Metal support:
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**vLLM (for GPU servers):**
```bash
pip install vllm
```

## Running Tests

### All Tests

```bash
pytest tests/
```

### Unit Tests Only

```bash
pytest tests/unit/
```

### Specific Test File

```bash
pytest tests/unit/test_router.py
```

### With Coverage

```bash
pytest --cov=air tests/
```

### With Verbose Output

```bash
pytest -v tests/
```

## Project Structure

```
Adaptive-Inference-Runtime/
├── air/                    # Core package (to be created)
│   ├── __init__.py
│   ├── router/             # Routing logic
│   │   ├── __init__.py
│   │   ├── confidence.py   # Confidence signal calculations
│   │   └── router.py       # Main router implementation
│   ├── adapters/           # Model adapters
│   │   ├── __init__.py
│   │   ├── base.py         # Base adapter interface
│   │   ├── llamacpp.py     # llama.cpp adapter
│   │   └── vllm.py         # vLLM adapter
│   ├── speculation/        # Speculative decoding
│   │   ├── __init__.py
│   │   ├── draft.py        # Draft generation
│   │   └── verify.py       # Verification logic
│   ├── compression/        # KV cache compression
│   │   ├── __init__.py
│   │   ├── eviction.py     # Eviction policies
│   │   └── manager.py      # Cache manager
│   ├── types.py            # Common type definitions
│   └── runtime.py          # Main runtime class
├── tests/
│   ├── unit/               # Fast, isolated tests
│   │   ├── test_router.py
│   │   ├── test_confidence.py
│   │   └── ...
│   ├── integration/        # Cross-component tests
│   └── benchmarks/         # Performance tests
├── benchmarks/             # Benchmark harness
├── examples/               # Example applications
├── configs/                # Configuration presets
├── docs/                   # Documentation
│   ├── architecture.md
│   ├── concepts.md
│   └── getting-started.md  # This file
├── .github/
│   └── TEMPLATES/          # Issue and PR templates
├── README.md
├── ROADMAP.md
├── CONTRIBUTING.md
├── CLAUDE.md               # AI assistant guidance
├── pyproject.toml          # Package configuration
└── requirements.txt
```

## Development Workflow

AIR follows a structured workflow. See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

### Quick Reference

1. **Find or create an issue** for your work
2. **Create a branch** from main:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** with clear, focused commits
4. **Run tests** to ensure nothing breaks:
   ```bash
   pytest tests/
   ```
5. **Push and create a PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring
- `test/` - Test additions

## Code Style

- Follow PEP 8
- Use type hints on all function signatures
- Maximum line length: 100 characters
- Docstrings for all public functions

### Example

```python
def calculate_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Calculate Shannon entropy of token probability distribution.

    Args:
        logits: Raw model output logits of shape (vocab_size,)
        temperature: Softmax temperature (default: 1.0)

    Returns:
        Entropy value in bits

    Raises:
        ValueError: If temperature is non-positive
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    probs = torch.softmax(logits / temperature, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs)
    return entropy.item()
```

## Common Tasks

### Adding a New Confidence Signal

1. Create function in `air/router/confidence.py`
2. Add unit tests in `tests/unit/test_confidence.py`
3. Integrate with router in `air/router/router.py`
4. Update documentation in `docs/concepts.md`

### Adding a New Model Adapter

1. Implement `ModelAdapter` protocol in `air/adapters/`
2. Add unit tests with mock model responses
3. Test integration with router and speculation pipeline
4. Document backend-specific setup

### Running Benchmarks

```bash
# When benchmark harness is implemented
air benchmark --dataset gsm8k --config configs/balanced.yaml
```

## Troubleshooting

### llama-cpp-python Installation Issues

**macOS (Metal):**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
```

**Linux (CUDA):**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall
```

### Import Errors

Ensure AIR is installed in development mode:
```bash
pip install -e .
```

### Test Failures

1. Check Python version: `python --version` (need 3.9+)
2. Verify dependencies: `pip install -r requirements-dev.txt`
3. Run specific failing test with verbose output:
   ```bash
   pytest -v tests/path/to/test.py::test_function_name
   ```

## Next Steps

- Read [Architecture Overview](architecture.md) to understand system design
- Read [Core Concepts](concepts.md) for technical deep dives
- Check [ROADMAP.md](../ROADMAP.md) to see current development priorities
- Find a good first issue to work on

## Getting Help

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Create an issue using the bug template
- **Feature requests**: Create an issue using the feature template

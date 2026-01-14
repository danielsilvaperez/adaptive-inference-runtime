# Contributing to Adaptive Inference Runtime (AIR)

Thank you for your interest in contributing to AIR! This document will guide you through the professional development workflow we use.

## Table of Contents

1. [Understanding Professional Development Workflows](#understanding-professional-development-workflows)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Style and Standards](#code-style-and-standards)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)

## Understanding Professional Development Workflows

### What is a "Pro Dev Workflow"?

Professional development workflows are structured processes that help teams build software efficiently while maintaining quality. Here's what they typically include:

1. **Issue Tracking**: Every piece of work starts as an issue (bug, feature, task)
2. **Branching Strategy**: Organized way of managing code changes
3. **Code Reviews**: Team members review each other's code before merging
4. **Automated Testing**: Tests run automatically to catch bugs
5. **Documentation**: Keep docs updated with code changes
6. **Version Control Best Practices**: Clear commit messages, logical commits

### Why Use This Workflow?

- **Transparency**: Everyone knows what's being worked on
- **Quality**: Code reviews catch bugs and improve design
- **History**: Issues and PRs document why changes were made
- **Collaboration**: Multiple people can work without conflicts
- **Accountability**: Clear ownership of tasks

## Getting Started

### Prerequisites

- Git installed locally
- GitHub account
- Python 3.8+ (for this project)
- Familiarity with command line

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Adaptive-Inference-Runtime.git
   cd Adaptive-Inference-Runtime
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/danielsilva010/Adaptive-Inference-Runtime.git
   ```

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when available)
pip install -r requirements.txt

# Install development dependencies (when available)
pip install -r requirements-dev.txt
```

## Development Workflow

### Step 1: Create or Find an Issue

**Before writing code, create an issue!** This is crucial for professional workflows.

1. Check if an issue already exists for your idea
2. If not, create a new issue using the appropriate template
3. Wait for discussion/approval before starting major work
4. Assign the issue to yourself when you start working

### Step 2: Create a Feature Branch

Always create a new branch for your work. Never commit directly to `main`.

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new branch with a descriptive name
git checkout -b feature/router-implementation
# or
git checkout -b fix/memory-leak-in-cache
# or
git checkout -b docs/update-readme
```

**Branch Naming Convention:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests

### Step 3: Make Your Changes

1. **Write code in small, logical commits**
   ```bash
   git add <files>
   git commit -m "Add token entropy calculation for router"
   ```

2. **Follow commit message guidelines:**
   - Use present tense: "Add feature" not "Added feature"
   - Be descriptive but concise
   - Reference issue numbers: "Fix memory leak (#42)"
   - First line should be 50 chars or less
   - Add detailed description after blank line if needed

3. **Keep commits focused**
   - Each commit should do one thing
   - Makes it easier to review and revert if needed

### Step 4: Write Tests

- Add tests for new features
- Ensure existing tests still pass
- Run tests locally before pushing:
  ```bash
  pytest tests/
  ```

### Step 5: Update Documentation

- Update README.md if you changed functionality
- Add docstrings to new functions/classes
- Update relevant documentation files

### Step 6: Push Your Branch

```bash
git push origin feature/router-implementation
```

### Step 7: Create a Pull Request (PR)

1. Go to GitHub and create a Pull Request
2. Use the PR template (auto-populated)
3. Fill out all sections completely
4. Link the related issue: "Closes #42" or "Fixes #42"
5. Request reviews from maintainers

### Step 8: Code Review Process

1. Maintainers will review your code
2. They may request changes
3. Address feedback by making new commits
4. Push changes to the same branch
5. PR updates automatically
6. Once approved, maintainer will merge

### Step 9: After Merging

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Delete your feature branch
git branch -d feature/router-implementation
git push origin --delete feature/router-implementation
```

## Code Style and Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use meaningful variable names
- Add type hints to function signatures
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

```python
def calculate_token_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """Calculate entropy of token probability distribution.
    
    Args:
        logits: Raw model output logits
        temperature: Temperature for softmax (default: 1.0)
    
    Returns:
        Entropy value in bits
    
    Raises:
        ValueError: If temperature is <= 0
    """
    # Implementation here
    pass
```

### Documentation Style

- Use Markdown for all documentation
- Keep language clear and concise
- Include code examples where helpful
- Update docs with code changes

## Testing

### Test Structure

```
tests/
├── unit/          # Fast, isolated tests
├── integration/   # Tests combining multiple components
└── benchmarks/    # Performance tests
```

### Writing Tests

```python
import pytest
from air.router import Router

def test_router_entropy_calculation():
    """Test that router correctly calculates token entropy."""
    router = Router()
    # Arrange
    mock_logits = torch.tensor([1.0, 2.0, 3.0])
    
    # Act
    entropy = router.calculate_entropy(mock_logits)
    
    # Assert
    assert entropy > 0
    assert isinstance(entropy, float)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_router.py

# Run with coverage
pytest --cov=air tests/
```

## Submitting Changes

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Checklist

When you submit a PR, ensure:

1. **Title is descriptive**: "Add adaptive routing based on token entropy"
2. **Description explains**:
   - What changes were made
   - Why they were made
   - How to test them
3. **Links to issue**: "Closes #42"
4. **Screenshots included** (if UI changes)
5. **Breaking changes noted** (if any)

## Questions?

- **General questions**: Open a discussion on GitHub
- **Bug reports**: Create an issue with bug template
- **Feature requests**: Create an issue with feature template
- **Security issues**: Email maintainer directly (don't create public issue)

## Recognition

All contributors will be recognized in our README and release notes. Thank you for helping make AIR better!

---

## Additional Resources

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- [Code Review Best Practices](https://google.github.io/eng-practices/review/)

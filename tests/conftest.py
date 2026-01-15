"""
Pytest configuration and shared fixtures for AIR tests.
"""

import logging
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def reset_logging() -> Generator[None, None, None]:
    """Reset logging configuration between tests."""
    yield

    # Clear all handlers from the air logger after each test
    air_logger = logging.getLogger("air")
    air_logger.handlers.clear()
    air_logger.setLevel(logging.NOTSET)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir: Path) -> Path:
    """Create a sample configuration file for testing."""
    config_path = temp_dir / "test_config.yaml"
    config_content = """
runtime:
  device: cpu
  num_workers: 2

routing:
  enabled: true
  confidence_threshold: 0.85

speculation:
  enabled: false

compression:
  enabled: false

generation:
  max_tokens: 100
  temperature: 0.7
"""
    config_path.write_text(config_content)
    return config_path

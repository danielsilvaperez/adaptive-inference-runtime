"""
AIR Utilities Module

Common utilities shared across AIR components including logging,
configuration management, and helper functions.

Key Components:
    - logging: Configurable logging with colored output
    - config: Configuration loading and validation
"""

from air.utils.logging import get_logger, setup_logging

__all__ = [
    "get_logger",
    "setup_logging",
]

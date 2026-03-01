"""
Base adapter interface for AIR model backends.

This module preserves the historical import path for adapter implementations
while delegating to the canonical BaseModelAdapter interface.
"""

from __future__ import annotations

from air.interfaces.adapter import BaseModelAdapter


class ModelAdapter(BaseModelAdapter):
    """
    Base adapter for model backends.

    Subclasses should implement the abstract methods defined by BaseModelAdapter.
    """

    pass


__all__ = ["ModelAdapter"]

"""
Pooling strategies for KV cache compression.

This module provides various pooling strategies that can be used to merge
or aggregate KV cache entries, reducing memory while preserving information.
Pooling is an alternative to eviction for certain compression scenarios.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class PoolingStrategy(Enum):
    """
    Available pooling strategies for KV cache compression.

    Attributes:
        MEAN: Average pooling - compute mean of key/value vectors.
        MAX: Max pooling - take element-wise maximum.
        WEIGHTED: Weighted pooling based on attention scores.
        NONE: No pooling, just eviction.
    """

    MEAN = "mean"
    MAX = "max"
    WEIGHTED = "weighted"
    NONE = "none"


def mean_pool(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute mean pooling of KV tensors.

    Args:
        tensors: List of tensors to pool.

    Returns:
        Mean-pooled tensor.
    """
    import torch

    if not tensors:
        raise ValueError("Cannot pool empty tensor list")

    return torch.stack(tensors).mean(dim=0)


def max_pool(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute max pooling of KV tensors.

    Args:
        tensors: List of tensors to pool.

    Returns:
        Max-pooled tensor.
    """
    import torch

    if not tensors:
        raise ValueError("Cannot pool empty tensor list")

    return torch.stack(tensors).max(dim=0)[0]


def weighted_pool(tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """
    Compute weighted pooling of KV tensors.

    Args:
        tensors: List of tensors to pool.
        weights: Weights for each tensor (e.g., attention scores).

    Returns:
        Weighted-pooled tensor.
    """
    import torch

    if not tensors:
        raise ValueError("Cannot pool empty tensor list")
    if len(tensors) != len(weights):
        raise ValueError("Number of tensors must match number of weights")

    # Normalize weights
    weights_tensor = torch.tensor(weights, dtype=tensors[0].dtype)
    weights_tensor = weights_tensor / weights_tensor.sum()

    # Weighted sum
    stacked = torch.stack(tensors)
    weights_expanded = weights_tensor.view(-1, *([1] * (stacked.dim() - 1)))
    return (stacked * weights_expanded).sum(dim=0)


def apply_pooling(
    tensors: list[torch.Tensor],
    strategy: PoolingStrategy,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """
    Apply pooling strategy to a list of tensors.

    Args:
        tensors: Tensors to pool.
        strategy: Pooling strategy to use.
        weights: Weights for weighted pooling (optional).

    Returns:
        Pooled tensor.

    Raises:
        ValueError: If strategy is invalid or weights are missing for weighted pooling.
    """
    if strategy == PoolingStrategy.MEAN:
        return mean_pool(tensors)
    elif strategy == PoolingStrategy.MAX:
        return max_pool(tensors)
    elif strategy == PoolingStrategy.WEIGHTED:
        if weights is None:
            raise ValueError("Weights required for weighted pooling")
        return weighted_pool(tensors, weights)
    elif strategy == PoolingStrategy.NONE:
        # No pooling, return first tensor
        return tensors[0] if tensors else None
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


__all__ = [
    "PoolingStrategy",
    "mean_pool",
    "max_pool",
    "weighted_pool",
    "apply_pooling",
]

"""
KV Cache implementation utilities for compression operations.

This module provides utilities for manipulating KV caches during compression,
including token eviction and cache reconstruction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from air.types import KVCache


class SimpleKVCache:
    """
    Simple in-memory KV cache implementation for testing and demonstration.

    This is a concrete implementation of the KVCache protocol that stores
    key and value tensors for each layer.

    Attributes:
        _keys: List of key tensors per layer.
        _values: List of value tensors per layer.
        _size: Current number of tokens in the cache.
        _max_size: Maximum capacity of the cache.
    """

    def __init__(
        self,
        num_layers: int,
        max_size: int,
        keys: list[torch.Tensor] | None = None,
        values: list[torch.Tensor] | None = None,
    ) -> None:
        """
        Initialize the KV cache.

        Args:
            num_layers: Number of transformer layers.
            max_size: Maximum number of tokens the cache can hold.
            keys: Optional pre-initialized key tensors.
            values: Optional pre-initialized value tensors.
        """
        self._num_layers = num_layers
        self._max_size = max_size

        if keys is not None and values is not None:
            self._keys = keys
            self._values = values
            # Infer size from tensor shape (assuming shape [batch, heads, seq_len, dim])
            if len(keys) > 0 and keys[0] is not None:
                self._size = keys[0].shape[2] if keys[0].ndim >= 3 else 0
            else:
                self._size = 0
        else:
            self._keys = [None] * num_layers  # type: ignore
            self._values = [None] * num_layers  # type: ignore
            self._size = 0

    @property
    def size(self) -> int:
        """Return the current number of tokens in the cache."""
        return self._size

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        return self._num_layers

    @property
    def max_size(self) -> int:
        """Return the maximum capacity of the cache."""
        return self._max_size

    def get_kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the key and value tensors for a specific layer.

        Args:
            layer: The layer index (0-indexed).

        Returns:
            A tuple of (keys, values) tensors for the specified layer.

        Raises:
            IndexError: If layer is out of bounds.
        """
        if not 0 <= layer < self._num_layers:
            raise IndexError(f"Layer {layer} out of bounds (0-{self._num_layers - 1})")
        return self._keys[layer], self._values[layer]

    def set_kv(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Set the key and value tensors for a specific layer.

        Args:
            layer: The layer index (0-indexed).
            keys: The key tensor to store.
            values: The value tensor to store.

        Raises:
            IndexError: If layer is out of bounds.
        """
        if not 0 <= layer < self._num_layers:
            raise IndexError(f"Layer {layer} out of bounds (0-{self._num_layers - 1})")
        self._keys[layer] = keys
        self._values[layer] = values

        # Update size based on key tensor shape
        if keys is not None and keys.ndim >= 3:
            self._size = keys.shape[2]

    def clear(self) -> None:
        """Clear all cached key-value pairs."""
        self._keys = [None] * self._num_layers  # type: ignore
        self._values = [None] * self._num_layers  # type: ignore
        self._size = 0

    def clone(self) -> SimpleKVCache:
        """
        Create a deep copy of the cache.

        Returns:
            A new SimpleKVCache with copied tensors.
        """
        # Clone tensors if they exist
        cloned_keys = [k.clone() if k is not None else None for k in self._keys]
        cloned_values = [v.clone() if v is not None else None for v in self._values]

        return SimpleKVCache(
            num_layers=self._num_layers,
            max_size=self._max_size,
            keys=cloned_keys,
            values=cloned_values,
        )


def evict_tokens_from_cache(cache: KVCache, positions_to_evict: list[int]) -> KVCache:
    """
    Evict specified token positions from the KV cache.

    This function creates a new cache with the specified tokens removed.
    Token positions are shifted to maintain sequential ordering.

    Args:
        cache: The original KV cache.
        positions_to_evict: List of token positions to remove (0-indexed).

    Returns:
        A new KVCache with specified tokens evicted.

    Note:
        This is a reference implementation. Production implementations
        should be optimized for the specific backend (llama.cpp, vLLM, etc.).
    """
    if not positions_to_evict:
        return cache

    # Import torch here to avoid import errors if not available
    try:
        import torch
    except ImportError as e:
        # If torch is not available, we cannot perform the eviction
        # This is a critical error as the function is expected to work
        raise ImportError(
            "PyTorch is required for KV cache eviction operations. "
            "Please install torch: pip install torch"
        ) from e

    # Sort positions for consistent processing
    positions_to_evict = sorted(set(positions_to_evict))

    # Create a new cache
    new_cache = SimpleKVCache(num_layers=cache.num_layers, max_size=cache.max_size)

    # Create a mask of positions to keep
    positions_to_keep = [i for i in range(cache.size) if i not in positions_to_evict]

    # For each layer, select and copy the tokens we want to keep
    for layer in range(cache.num_layers):
        keys, values = cache.get_kv(layer)

        if keys is None or values is None:
            continue

        # Assuming shape: [batch, num_heads, seq_len, head_dim]
        # Select the positions we want to keep along the sequence dimension (dim=2)
        if len(positions_to_keep) > 0:
            new_keys = torch.index_select(
                keys, dim=2, index=torch.tensor(positions_to_keep, device=keys.device)
            )
            new_values = torch.index_select(
                values,
                dim=2,
                index=torch.tensor(positions_to_keep, device=values.device),
            )
        else:
            # All tokens evicted, create empty tensors
            new_keys = keys[:, :, :0, :]
            new_values = values[:, :, :0, :]

        new_cache.set_kv(layer, new_keys, new_values)

    return new_cache

"""
Base compressor interface for KV cache compression.

This module defines the base Compressor class and common utilities for
implementing KV cache compression strategies. All compression implementations
(sliding window, heavy hitter, H2O, etc.) inherit from this base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from air.types import CompressionConfig, KVCache


@dataclass
class CompressionStats:
    """
    Statistics from cache compression operations.

    Attributes:
        original_size: Original cache size in tokens before compression.
        compressed_size: Cache size after compression.
        compression_ratio: Ratio of compressed to original size.
        tokens_evicted: Number of tokens evicted.
        memory_saved_bytes: Estimated memory saved in bytes.
    """

    original_size: int
    compressed_size: int
    compression_ratio: float
    tokens_evicted: int
    memory_saved_bytes: int


class Compressor(ABC):
    """
    Abstract base class for KV cache compressors.

    Compressors reduce the memory footprint of the KV cache by selectively
    evicting less important tokens while preserving generation quality.
    Different strategies (sliding window, attention-based, etc.) implement
    different eviction policies.

    All compressors must implement:
        - compress(): Apply compression to a cache
        - get_compression_stats(): Return statistics about compression
    """

    def __init__(self, config: CompressionConfig) -> None:
        """
        Initialize the compressor with configuration.

        Args:
            config: Compression configuration parameters.
        """
        self._config = config

    @abstractmethod
    def compress(self, cache: KVCache) -> None:
        """
        Compress the KV cache in-place.

        Args:
            cache: KV cache to compress.
        """
        pass

    @abstractmethod
    def get_compression_stats(self, cache: KVCache) -> CompressionStats:
        """
        Get statistics about compression.

        Args:
            cache: KV cache to analyze.

        Returns:
            Compression statistics.
        """
        pass

    @property
    def config(self) -> CompressionConfig:
        """Return the compression configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Return whether compression is enabled."""
        return self._config.enabled

    def should_compress(self, cache_size: int) -> bool:
        """
        Determine if compression should be applied based on cache size.

        Args:
            cache_size: Current size of the cache in tokens.

        Returns:
            True if compression should be applied.
        """
        return self._config.enabled and cache_size >= self._config.min_tokens_before_compression

    def compute_target_size(self, current_size: int) -> int:
        """
        Compute target cache size after compression.

        Args:
            current_size: Current cache size in tokens.

        Returns:
            Target size after applying compression ratio.
        """
        target = int(current_size * self._config.target_ratio)
        # Ensure we don't compress below protected token count
        return max(target, self._config.protected_token_count)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"policy={self._config.eviction_policy}, "
            f"enabled={self._config.enabled})"
        )


__all__ = ["Compressor", "CompressionStats"]

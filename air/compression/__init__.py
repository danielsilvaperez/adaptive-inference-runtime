"""
AIR KV Cache Compression Module

Implements memory-efficient KV cache compression strategies to reduce
memory footprint during inference while maintaining generation quality.

Key Components:
    - KVCacheCompressor: Main compression interface
    - HeavyHitterCompressor: Heavy hitter retention policy
    - AttentionPooling: Attention-based cache compression
    - QuantizedCache: Quantized KV cache for memory efficiency
    - SlidingWindowCompressor: Sliding window retention policy
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from air.compression.compressor import KVCacheCompressor
    from air.compression.heavy_hitter import HeavyHitterCompressor
    from air.compression.pooling import AttentionPooling
    from air.compression.quantized import QuantizedCache
    from air.compression.sliding_window import SlidingWindowCompressor

__all__ = [
    "KVCacheCompressor",
    "HeavyHitterCompressor",
    "AttentionPooling",
    "QuantizedCache",
    "SlidingWindowCompressor",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for compression components."""
    if name == "KVCacheCompressor":
        from air.compression import compressor

        return compressor.KVCacheCompressor
    elif name == "HeavyHitterCompressor":
        from air.compression import heavy_hitter
        return heavy_hitter.HeavyHitterCompressor
    elif name == "AttentionPooling":
        from air.compression import pooling

        return pooling.AttentionPooling
    elif name == "QuantizedCache":
        from air.compression import quantized

        return quantized.QuantizedCache
    elif name == "SlidingWindowCompressor":
        from air.compression import sliding_window
        return sliding_window.SlidingWindowCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

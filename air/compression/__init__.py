"""
AIR KV Cache Compression Module

Implements memory-efficient KV cache compression strategies to reduce
memory footprint during inference while maintaining generation quality.

Key Components:
    - KVCacheCompressor: Main compression interface
    - AttentionPooling: Attention-based cache compression
    - QuantizedCache: Quantized KV cache for memory efficiency
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from air.compression.compressor import KVCacheCompressor
    from air.compression.pooling import AttentionPooling
    from air.compression.quantized import QuantizedCache

__all__ = [
    "KVCacheCompressor",
    "AttentionPooling",
    "QuantizedCache",
]


def __getattr__(name: str):
    """Lazy import mechanism for compression components."""
    if name == "KVCacheCompressor":
        from air.compression import compressor
        return compressor.KVCacheCompressor
    elif name == "AttentionPooling":
        from air.compression import pooling
        return pooling.AttentionPooling
    elif name == "QuantizedCache":
        from air.compression import quantized
        return quantized.QuantizedCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

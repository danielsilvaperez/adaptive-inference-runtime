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
    - H2OCompressor: H2O (Heavy Hitter Oracle) eviction policy
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from air.compression.compressor import Compressor as KVCacheCompressor
    from air.compression.h2o import H2OCompressor
    from air.compression.heavy_hitter import HeavyHitterCompressor
    from air.compression.pooling import PoolingStrategy as AttentionPooling
    from air.compression.quantized import QuantizedKVCache as QuantizedCache
    from air.compression.safety import (
        CompressionDecision,
        CompressionQualityMonitor,
        CompressionSafetyGuard,
        CompressionSafetyManager,
        CompressionUseCase,
        QualityMonitorConfig,
        SafetyGuardConfig,
    )
    from air.compression.sliding_window import SlidingWindowCompressor

__all__ = [
    "KVCacheCompressor",
    "HeavyHitterCompressor",
    "AttentionPooling",
    "QuantizedCache",
    "SlidingWindowCompressor",
    "H2OCompressor",
    "CompressionDecision",
    "CompressionQualityMonitor",
    "CompressionSafetyGuard",
    "CompressionSafetyManager",
    "CompressionUseCase",
    "QualityMonitorConfig",
    "SafetyGuardConfig",
]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for compression components."""
    if name == "KVCacheCompressor":
        from air.compression import compressor

        return compressor.Compressor
    elif name == "HeavyHitterCompressor":
        from air.compression import heavy_hitter

        return heavy_hitter.HeavyHitterCompressor
    elif name == "AttentionPooling":
        from air.compression import pooling

        return pooling.PoolingStrategy
    elif name == "QuantizedCache":
        from air.compression import quantized

        return quantized.QuantizedKVCache
    elif name == "SlidingWindowCompressor":
        from air.compression import sliding_window

        return sliding_window.SlidingWindowCompressor
    elif name == "H2OCompressor":
        from air.compression import h2o

        return h2o.H2OCompressor
    elif name in {
        "CompressionDecision",
        "CompressionQualityMonitor",
        "CompressionSafetyGuard",
        "CompressionSafetyManager",
        "CompressionUseCase",
        "QualityMonitorConfig",
        "SafetyGuardConfig",
    }:
        from air.compression import safety

        return getattr(safety, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

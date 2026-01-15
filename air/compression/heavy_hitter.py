"""
Heavy Hitter Retention Policy for KV Cache Compression.

This module implements a compression strategy that tracks attention scores
for each cached token and retains tokens with high cumulative attention
while evicting low-attention tokens first.

The heavy hitter policy is based on the observation that certain tokens
receive significantly more attention during generation than others. By
tracking cumulative attention scores and retaining these "heavy hitters",
we can maintain generation quality while reducing memory usage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    from air.types import CompressionConfig, KVCache

from air.interfaces.compressor import BaseKVCompressor

logger = logging.getLogger(__name__)


class HeavyHitterCompressor(BaseKVCompressor):
    """
    Heavy Hitter Retention Policy for KV cache compression.

    This compressor tracks cumulative attention scores for each token and
    retains tokens with the highest attention scores while evicting tokens
    with low attention scores first. This approach preserves important
    context while reducing memory usage.

    The policy maintains:
    - Cumulative attention scores per token
    - Protected recent tokens (never evicted)
    - Heavy hitter ratio to determine retention count

    Attributes:
        config: Compression configuration with heavy_hitter_ratio setting.

    Example:
        >>> config = CompressionConfig(
        ...     eviction_policy="heavy_hitter",
        ...     heavy_hitter_ratio=0.2,
        ...     target_ratio=0.5,
        ...     protected_token_count=32
        ... )
        >>> compressor = HeavyHitterCompressor(config)
        >>> compressed_cache = compressor.compress(cache)
    """

    def __init__(self, config: CompressionConfig) -> None:
        """
        Initialize the Heavy Hitter Compressor.

        Args:
            config: Compression configuration. Must have eviction_policy
                set to "heavy_hitter" or "h2o".

        Raises:
            ValueError: If eviction policy is not compatible.
        """
        super().__init__(config)
        if config.eviction_policy not in ("heavy_hitter", "h2o"):
            raise ValueError(
                f"HeavyHitterCompressor requires eviction_policy to be "
                f"'heavy_hitter' or 'h2o', got '{config.eviction_policy}'"
            )

        # Track cumulative attention scores per token
        self._attention_scores: dict[int, dict[int, float]] = {}

    def update_attention_scores(
        self, layer: int, token_positions: list[int], attention_scores: list[float]
    ) -> None:
        """
        Update cumulative attention scores for tokens.

        This method should be called during generation to track which tokens
        receive attention. The scores are accumulated over time to identify
        heavy hitters.

        Args:
            layer: The transformer layer index.
            token_positions: List of token positions in the cache.
            attention_scores: Corresponding attention scores (higher = more important).

        Note:
            Scores are accumulated (added) across calls for the same token.
        """
        if layer not in self._attention_scores:
            self._attention_scores[layer] = {}

        layer_scores = self._attention_scores[layer]
        for pos, score in zip(token_positions, attention_scores):
            layer_scores[pos] = layer_scores.get(pos, 0.0) + score

    def get_cumulative_attention(self, token_position: int) -> float:
        """
        Get the cumulative attention score for a token across all layers.

        Args:
            token_position: The position of the token in the cache.

        Returns:
            Sum of attention scores across all layers for this token.
        """
        total_score = 0.0
        for layer_scores in self._attention_scores.values():
            total_score += layer_scores.get(token_position, 0.0)
        return total_score

    def _get_eviction_candidates(
        self, cache: KVCache, target_size: int
    ) -> list[int]:
        """
        Determine which tokens should be evicted based on attention scores.

        Args:
            cache: The KV cache to analyze.
            target_size: The desired cache size after eviction.

        Returns:
            List of token positions to evict, sorted by increasing attention
            (least important first).
        """
        current_size = cache.size
        protected_count = min(self._config.protected_token_count, current_size)

        # Calculate number of tokens to evict
        num_to_evict = max(0, current_size - target_size)

        if num_to_evict == 0:
            return []

        # Build list of (position, cumulative_attention) for evictable tokens
        # Recent tokens (last protected_count) are never evicted
        evictable_positions = range(0, current_size - protected_count)
        token_scores = [
            (pos, self.get_cumulative_attention(pos)) for pos in evictable_positions
        ]

        # Sort by attention score (ascending) - lowest attention first
        token_scores.sort(key=lambda x: x[1])

        # Return the positions of tokens to evict (lowest attention scores)
        eviction_candidates = [pos for pos, _ in token_scores[:num_to_evict]]

        logger.debug(
            f"Selected {len(eviction_candidates)} tokens for eviction "
            f"(target: {num_to_evict}, protected: {protected_count})"
        )

        return eviction_candidates

    def compress(self, cache: KVCache) -> KVCache:
        """
        Compress the KV cache using heavy hitter retention.

        This method analyzes cumulative attention scores and removes tokens
        with the lowest attention, keeping tokens that have received the
        most attention during generation.

        Args:
            cache: The KV cache to compress.

        Returns:
            A compressed KVCache with low-attention tokens evicted.
            If compression is not needed, returns the original cache.
        """
        if not self._should_compress(cache):
            return cache

        # Calculate target size based on heavy_hitter_ratio
        # Keep: heavy_hitter_ratio * original + protected tokens
        base_retention = int(cache.size * self._config.heavy_hitter_ratio)
        target_size = max(
            base_retention + self._config.protected_token_count,
            int(cache.size * self._config.target_ratio),
        )

        # Ensure target size doesn't exceed current size
        target_size = min(target_size, cache.size)

        logger.info(
            f"Compressing cache from {cache.size} to {target_size} tokens "
            f"(heavy_hitter_ratio={self._config.heavy_hitter_ratio})"
        )

        return self.evict(cache, target_size)

    def evict(self, cache: KVCache, target_size: int) -> KVCache:
        """
        Evict low-attention tokens from the cache to reach target size.

        This method removes tokens with the lowest cumulative attention
        scores while protecting recent tokens.

        Args:
            cache: The KV cache to evict from.
            target_size: The desired cache size in tokens.

        Returns:
            A new KVCache with low-attention tokens removed.

        Raises:
            ValueError: If target_size is invalid.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")

        if target_size >= cache.size:
            # No eviction needed
            return cache

        # Get tokens to evict
        eviction_candidates = self._get_eviction_candidates(cache, target_size)

        if not eviction_candidates:
            return cache

        # Create a new cache by removing evicted tokens
        from air.compression.cache_impl import evict_tokens_from_cache

        new_cache = evict_tokens_from_cache(cache, eviction_candidates)

        # Update attention scores after eviction
        self._update_scores_after_eviction(eviction_candidates)

        logger.debug(
            f"Evicted {len(eviction_candidates)} tokens, "
            f"new cache size: {new_cache.size}"
        )

        return new_cache

    def _update_scores_after_eviction(self, evicted_positions: list[int]) -> None:
        """
        Update internal attention scores after evicting tokens.

        When tokens are evicted, we need to update the position indices
        in our attention score tracking.

        Args:
            evicted_positions: List of positions that were evicted (sorted).
        """
        if not evicted_positions:
            return

        # Sort evicted positions for consistent processing
        evicted_set = set(evicted_positions)

        # Update all layers
        for layer in self._attention_scores:
            old_scores = self._attention_scores[layer]
            new_scores: dict[int, float] = {}

            # Remap positions after eviction
            for old_pos, score in old_scores.items():
                if old_pos not in evicted_set:
                    # Calculate new position (how many evicted tokens were before this)
                    num_evicted_before = sum(
                        1 for evicted in evicted_positions if evicted < old_pos
                    )
                    new_pos = old_pos - num_evicted_before
                    new_scores[new_pos] = score

            self._attention_scores[layer] = new_scores

    def reset_attention_scores(self) -> None:
        """
        Reset all tracked attention scores.

        This should be called when starting a new generation session
        or when the cache is cleared.
        """
        self._attention_scores.clear()
        logger.debug("Reset attention scores for heavy hitter tracking")

    def get_compression_stats(
        self, original: KVCache, compressed: KVCache
    ) -> dict[str, Any]:
        """
        Get detailed statistics about a compression operation.

        Extends the base implementation with heavy hitter specific metrics.

        Args:
            original: The original cache before compression.
            compressed: The compressed cache.

        Returns:
            Dictionary with compression statistics including average attention
            scores for retained vs evicted tokens.
        """
        stats = super().get_compression_stats(original, compressed)

        # Add heavy hitter specific stats
        if self._attention_scores:
            all_scores = [
                self.get_cumulative_attention(pos) for pos in range(original.size)
            ]
            if all_scores:
                stats["avg_attention_score"] = sum(all_scores) / len(all_scores)
                stats["max_attention_score"] = max(all_scores)
                stats["min_attention_score"] = min(all_scores)

        stats["heavy_hitter_ratio"] = self._config.heavy_hitter_ratio

        return stats

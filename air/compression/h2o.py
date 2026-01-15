"""
H2O (Heavy Hitter Oracle) style KV cache eviction for AIR.

This module implements the H2O eviction policy which prioritizes keeping
tokens with high cumulative attention scores. The algorithm tracks attention
weights during generation and evicts tokens with lower attention scores
when memory limits are reached.

Key features:
- Attention-weight based eviction (keeps "heavy hitters")
- Configurable retention ratio
- Support for per-layer or global eviction policies
- Protected recent tokens (sliding window)

Reference:
H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
(Zhang et al., 2023)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from air.interfaces.compressor import BaseKVCompressor

if TYPE_CHECKING:
    from air.types import CompressionConfig, KVCache


class H2OCompressor(BaseKVCompressor):
    """
    H2O-style KV cache compressor using attention-weight based eviction.

    The H2O algorithm maintains a running count of attention weights for each
    cached token. When eviction is needed, it retains the tokens with the
    highest cumulative attention scores (heavy hitters) and evicts the rest.

    This approach is based on the observation that a small subset of tokens
    receive most of the attention weight during generation, and these tokens
    are critical for maintaining generation quality.

    Attributes:
        config: Compression configuration with H2O-specific parameters.
        _attention_scores: Accumulated attention scores per layer and token.
        _per_layer: Whether to use per-layer eviction policy.

    Example:
        >>> config = CompressionConfig(
        ...     eviction_policy="h2o",
        ...     target_ratio=0.5,
        ...     heavy_hitter_ratio=0.1,
        ...     protected_token_count=32
        ... )
        >>> compressor = H2OCompressor(config)
        >>> compressed_cache = compressor.compress(cache)
    """

    def __init__(self, config: CompressionConfig) -> None:
        """
        Initialize the H2O compressor.

        Args:
            config: Compression configuration. Must have eviction_policy="h2o".

        Raises:
            ValueError: If eviction_policy is not "h2o".
        """
        super().__init__(config)
        if config.eviction_policy != "h2o":
            raise ValueError(
                f"H2OCompressor requires eviction_policy='h2o', got '{config.eviction_policy}'"
            )

        # Store attention scores per layer: {layer_idx: {token_pos: score}}
        self._attention_scores: dict[int, dict[int, float]] = {}

        # Track whether to use per-layer eviction policy
        self._per_layer = config.per_layer_policy

    def update_attention_scores(
        self,
        layer: int,
        attention_weights: torch.Tensor,
        token_positions: list[int] | None = None,
    ) -> None:
        """
        Update cumulative attention scores for cached tokens.

        This method should be called during generation to track which tokens
        are receiving attention. The attention weights are accumulated to
        identify heavy hitters.

        Args:
            layer: The layer index (0-indexed).
            attention_weights: Attention weights tensor of shape
                (batch, num_heads, seq_len, seq_len) or (num_heads, seq_len, seq_len).
                The last dimension represents attention over cached tokens.
            token_positions: Optional list of token positions in the cache.
                If None, assumes positions 0 to seq_len-1.

        Note:
            This method accumulates scores across multiple calls. Use
            reset_attention_scores() to clear accumulated scores.
        """
        # Initialize layer tracking if needed
        if layer not in self._attention_scores:
            self._attention_scores[layer] = {}

        # Handle different attention weight tensor shapes
        if attention_weights.dim() == 4:
            # Shape: (batch, num_heads, seq_len, seq_len)
            # Average over batch and heads, sum over query positions
            attn = attention_weights.mean(dim=(0, 1)).sum(dim=0)  # (seq_len,)
        elif attention_weights.dim() == 3:
            # Shape: (num_heads, seq_len, seq_len)
            # Average over heads, sum over query positions
            attn = attention_weights.mean(dim=0).sum(dim=0)  # (seq_len,)
        else:
            raise ValueError(
                f"Expected attention_weights with 3 or 4 dimensions, got {attention_weights.dim()}"
            )

        # Determine token positions
        seq_len = attn.shape[0]
        if token_positions is None:
            token_positions = list(range(seq_len))
        elif len(token_positions) != seq_len:
            raise ValueError(
                f"token_positions length ({len(token_positions)}) "
                f"must match sequence length ({seq_len})"
            )

        # Accumulate attention scores
        for pos, score in zip(token_positions, attn.tolist()):
            if pos in self._attention_scores[layer]:
                self._attention_scores[layer][pos] += score
            else:
                self._attention_scores[layer][pos] = score

    def reset_attention_scores(self, layer: int | None = None) -> None:
        """
        Reset accumulated attention scores.

        Args:
            layer: If specified, only reset scores for this layer.
                If None, reset all layers.
        """
        if layer is not None:
            self._attention_scores.pop(layer, None)
        else:
            self._attention_scores.clear()

    def compress(self, cache: KVCache) -> KVCache:
        """
        Compress the KV cache using H2O eviction.

        Applies H2O eviction if the cache size exceeds the target based on
        the configured retention ratio. Protected recent tokens are never
        evicted.

        Args:
            cache: The KV cache to compress.

        Returns:
            A compressed KV cache with heavy hitters retained.
        """
        if not self._should_compress(cache):
            return cache

        # Calculate target size based on retention ratio
        target_size = int(cache.size * self._config.target_ratio)

        # Ensure we keep at least the protected tokens
        target_size = max(target_size, self._config.protected_token_count)

        return self.evict(cache, target_size)

    def evict(self, cache: KVCache, target_size: int) -> KVCache:
        """
        Evict tokens from the cache to reach target size using H2O policy.

        The H2O eviction policy:
        1. Always protect the most recent N tokens (sliding window)
        2. For remaining tokens, keep those with highest attention scores
        3. Evict tokens with lowest attention scores

        Args:
            cache: The cache to evict from.
            target_size: The desired cache size in tokens.

        Returns:
            A new KVCache with tokens evicted according to H2O policy.

        Raises:
            ValueError: If target_size is invalid.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")

        # If already below target, no eviction needed
        if cache.size <= target_size:
            return cache

        # Determine which tokens to keep
        tokens_to_keep = self._select_tokens_to_keep(cache, target_size)

        # Create new cache with only the selected tokens
        return self._create_evicted_cache(cache, tokens_to_keep)

    def _select_tokens_to_keep(self, cache: KVCache, target_size: int) -> list[int]:
        """
        Select which token positions to keep based on H2O policy.

        Args:
            cache: The cache to select from.
            target_size: How many tokens to keep.

        Returns:
            List of token positions to retain, sorted by position.
        """
        current_size = cache.size
        protected_count = min(self._config.protected_token_count, current_size)

        # Always keep the most recent protected_count tokens
        protected_positions = list(range(current_size - protected_count, current_size))

        # If target size equals or is less than protected count, only keep protected
        if target_size <= protected_count:
            # Return the most recent target_size tokens
            return protected_positions[-target_size:]

        # Calculate how many additional tokens to keep beyond protected
        additional_to_keep = target_size - protected_count

        # Get evictable positions (all except protected)
        evictable_positions = list(range(current_size - protected_count))

        if additional_to_keep >= len(evictable_positions):
            # Keep all tokens
            return list(range(current_size))

        # Select heavy hitters from evictable positions
        if self._per_layer:
            # Per-layer policy: aggregate scores across layers
            heavy_hitters = self._select_heavy_hitters_per_layer(
                cache, evictable_positions, additional_to_keep
            )
        else:
            # Global policy: use global attention scores
            heavy_hitters = self._select_heavy_hitters_global(
                cache, evictable_positions, additional_to_keep
            )

        # Combine heavy hitters with protected tokens, sorted by position
        tokens_to_keep = sorted(heavy_hitters + protected_positions)

        return tokens_to_keep

    def _select_heavy_hitters_global(
        self, _cache: KVCache, evictable_positions: list[int], count: int
    ) -> list[int]:
        """
        Select heavy hitters using global attention scores across all layers.

        Args:
            _cache: The cache being compressed (unused, for interface consistency).
            evictable_positions: Positions that can be evicted.
            count: How many positions to select.

        Returns:
            List of selected token positions.
        """
        # Aggregate scores across all layers
        global_scores: dict[int, float] = {}

        for layer_scores in self._attention_scores.values():
            for pos, score in layer_scores.items():
                if pos in evictable_positions:
                    global_scores[pos] = global_scores.get(pos, 0.0) + score

        # If we don't have scores yet (e.g., first generation step), keep oldest tokens
        if not global_scores:
            return evictable_positions[:count]

        # Sort positions by score (descending) and take top count
        sorted_positions = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)

        return [pos for pos, _ in sorted_positions[:count]]

    def _select_heavy_hitters_per_layer(
        self, cache: KVCache, evictable_positions: list[int], count: int
    ) -> list[int]:
        """
        Select heavy hitters using per-layer attention scores.

        This method applies eviction independently per layer and takes the
        union of heavy hitters across layers.

        Args:
            cache: The cache being compressed.
            evictable_positions: Positions that can be evicted.
            count: How many positions to select.

        Returns:
            List of selected token positions.
        """
        # Calculate per-layer heavy hitters
        per_layer_hitters: list[list[int]] = []

        for layer in range(cache.num_layers):
            if layer not in self._attention_scores:
                continue

            layer_scores = self._attention_scores[layer]

            # Filter to evictable positions
            evictable_scores = {
                pos: score for pos, score in layer_scores.items() if pos in evictable_positions
            }

            if not evictable_scores:
                continue

            # Sort by score and take top heavy_hitter_ratio
            sorted_positions = sorted(evictable_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep top heavy_hitter_ratio of tokens per layer
            keep_count = int(len(sorted_positions) * self._config.heavy_hitter_ratio)
            if keep_count > 0:
                layer_hitters = [pos for pos, _ in sorted_positions[:keep_count]]
                per_layer_hitters.append(layer_hitters)

        # Take union of heavy hitters across layers
        if not per_layer_hitters:
            # No scores available, keep oldest tokens
            return evictable_positions[:count]

        # Union of all layer heavy hitters
        all_hitters = sorted({pos for layer_hitters in per_layer_hitters for pos in layer_hitters})

        # If we have more than needed, prioritize by global score
        if len(all_hitters) > count:
            # Aggregate global scores for ranking
            global_scores: dict[int, float] = {}
            for layer_scores in self._attention_scores.values():
                for pos, score in layer_scores.items():
                    if pos in all_hitters:
                        global_scores[pos] = global_scores.get(pos, 0.0) + score

            # Sort by global score and take top count
            sorted_hitters = sorted(global_scores.items(), key=lambda x: x[1], reverse=True)
            return [pos for pos, _ in sorted_hitters[:count]]

        # If we have fewer than needed, add more based on global scores
        if len(all_hitters) < count:
            remaining_positions = [p for p in evictable_positions if p not in all_hitters]
            additional_needed = count - len(all_hitters)

            # Get global scores for remaining positions
            remaining_scores: dict[int, float] = {}
            for layer_scores in self._attention_scores.values():
                for pos, score in layer_scores.items():
                    if pos in remaining_positions:
                        remaining_scores[pos] = remaining_scores.get(pos, 0.0) + score

            if remaining_scores:
                sorted_remaining = sorted(
                    remaining_scores.items(), key=lambda x: x[1], reverse=True
                )
                additional = [pos for pos, _ in sorted_remaining[:additional_needed]]
            else:
                # No scores, just take first positions
                additional = remaining_positions[:additional_needed]

            all_hitters.extend(additional)

        return sorted(all_hitters)

    def _create_evicted_cache(
        self, original_cache: KVCache, _positions_to_keep: list[int]
    ) -> KVCache:
        """
        Create a new cache containing only the specified token positions.

        Args:
            original_cache: The original cache.
            _positions_to_keep: Token positions to retain (unused, for interface).

        Returns:
            A new KVCache with only the retained tokens.

        Note:
            This is a placeholder implementation. The actual implementation
            depends on the concrete KVCache class being used. Subclasses or
            adapters should override this method to handle cache manipulation.
        """
        # This is a simplified implementation. In practice, this would need to:
        # 1. Create a new cache instance
        # 2. Copy only the selected token positions from each layer
        # 3. Update cache metadata (size, position mappings, etc.)
        #
        # Since we're working with a Protocol, we can't instantiate directly.
        # The actual implementation would be in adapter-specific code.

        # For now, return the original cache as a placeholder
        # Real implementations should override this method
        warnings.warn(
            "H2OCompressor._create_evicted_cache is using placeholder implementation. "
            "This should be overridden by adapter-specific code.",
            RuntimeWarning,
            stacklevel=2,
        )

        return original_cache

    def get_attention_score(self, layer: int, position: int) -> float:
        """
        Get the accumulated attention score for a token at a specific position.

        Args:
            layer: The layer index.
            position: The token position.

        Returns:
            The accumulated attention score, or 0.0 if not tracked.
        """
        return self._attention_scores.get(layer, {}).get(position, 0.0)

    def get_heavy_hitter_positions(
        self, layer: int | None = None, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """
        Get the top-k heavy hitter positions by attention score.

        Args:
            layer: If specified, get heavy hitters for this layer only.
                If None, aggregate across all layers.
            top_k: Number of top positions to return.

        Returns:
            List of (position, score) tuples, sorted by score descending.
        """
        if layer is not None:
            # Single layer
            if layer not in self._attention_scores:
                return []

            scores = self._attention_scores[layer]
        else:
            # Aggregate across all layers
            scores = {}
            for layer_scores in self._attention_scores.values():
                for pos, score in layer_scores.items():
                    scores[pos] = scores.get(pos, 0.0) + score

        # Sort by score descending
        sorted_positions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_positions[:top_k]

    def __repr__(self) -> str:
        """Get string representation."""
        policy_type = "per-layer" if self._per_layer else "global"
        tracked_layers = len(self._attention_scores)

        return (
            f"H2OCompressor("
            f"policy='{policy_type}', "
            f"target_ratio={self._config.target_ratio}, "
            f"heavy_hitter_ratio={self._config.heavy_hitter_ratio}, "
            f"tracked_layers={tracked_layers}, "
            f"enabled={self.is_enabled})"
        )

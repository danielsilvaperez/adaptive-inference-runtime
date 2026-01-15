"""
Example: Using Sliding Window Compression for KV Cache

This example demonstrates how to use the SlidingWindowCompressor to reduce
memory usage during long-sequence generation while maintaining good quality.
"""

from air.compression.sliding_window import SlidingWindowCompressor
from air.types import CompressionConfig


def main():
    """Demonstrate sliding window compression configuration and usage."""

    # Example 1: Conservative compression (minimal memory savings, high quality)
    print("Example 1: Conservative Compression")
    print("-" * 50)
    conservative_config = CompressionConfig.conservative()
    print(f"Window size: {conservative_config.sliding_window_size}")
    print(f"Target ratio: {conservative_config.target_ratio}")
    print(f"Protected tokens: {conservative_config.protected_token_count}")
    print()

    # Example 2: Balanced compression (default)
    print("Example 2: Balanced Compression (Default)")
    print("-" * 50)
    balanced_config = CompressionConfig.balanced()
    compressor = SlidingWindowCompressor(balanced_config)
    print(f"Configuration: {compressor}")
    print(f"Window size: {compressor.window_size}")
    print(f"Protected count: {compressor.protected_count}")
    print()

    # Example 3: Custom sliding window configuration
    print("Example 3: Custom Configuration")
    print("-" * 50)
    custom_config = CompressionConfig(
        enabled=True,
        eviction_policy="sliding_window",
        sliding_window_size=1024,  # Keep last 1024 tokens
        protected_token_count=64,  # Protect last 64 tokens from eviction
        min_tokens_before_compression=256,  # Start compression after 256 tokens
    )
    custom_compressor = SlidingWindowCompressor(custom_config)
    print(f"Configuration: {custom_compressor}")
    print()

    # Example 4: Simulating compression statistics
    print("Example 4: Compression Statistics")
    print("-" * 50)
    print("For a hypothetical cache with 5000 tokens:")

    class MockCache:
        """Simple mock cache for demonstration."""

        def __init__(self, size):
            self.size = size
            self.num_layers = 32
            self.max_size = 8192

    mock_cache = MockCache(size=5000)
    stats = custom_compressor.get_eviction_stats(mock_cache)

    print(f"  Current size: {stats['current_size']} tokens")
    print(f"  Window size: {stats['window_size']} tokens")
    print(f"  Compression needed: {stats['compression_needed']}")
    print(f"  Would evict: {stats['would_evict']} tokens")
    print(f"  Tokens after compression: {stats['tokens_after_compression']} tokens")
    print(f"  Memory savings: ~{(stats['would_evict'] / stats['current_size']) * 100:.1f}%")
    print()

    # Example 5: Different use cases
    print("Example 5: Use Case Recommendations")
    print("-" * 50)
    print("Chatbot (long conversations):")
    print("  - Window size: 2048-4096 tokens")
    print("  - Protects recent context")
    print()
    print("Code completion:")
    print("  - Window size: 512-1024 tokens")
    print("  - Focus on recent code context")
    print()
    print("Document Q&A (long documents):")
    print("  - Window size: 4096-8192 tokens")
    print("  - May need larger window to maintain context")
    print()
    print("Real-time chat:")
    print("  - Window size: 256-512 tokens")
    print("  - Low latency, minimal memory")
    print()


if __name__ == "__main__":
    main()

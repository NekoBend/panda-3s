"""
Cache module for panda-3s.
Provides optimized caching system with safetensors, schema-awareness, and sharding.
"""

from .optimized import SafeOptimizedEmbeddingCache

__all__ = ["SafeOptimizedEmbeddingCache"]

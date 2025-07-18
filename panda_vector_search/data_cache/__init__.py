"""
Cache module for panda-vector-search.
Provides optimized caching system with safetensors, schema-awareness, and sharding.
"""

from .optimized import EmbeddingCacheManager

__all__ = ["EmbeddingCacheManager"]

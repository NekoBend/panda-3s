"""
panda-3s: Pandas-focused Semantic Search Suite

A simple and powerful semantic search suite for pandas DataFrames using sentence transformers and FAISS.
"""

from .cache import SafeOptimizedEmbeddingCache
from .embedding import Embedding
from .indexer import FaissIndex
from .search import EmbeddingConfig, PandaSearch

__version__ = "1.0.0"

__all__ = [
    "PandaSearch",
    "Embedding",
    "FaissIndex",
    "SafeOptimizedEmbeddingCache",
    "EmbeddingConfig",
]

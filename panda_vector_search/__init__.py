"""
panda-3s: Pandas-focused Semantic Search Suite

A simple and powerful semantic search suite for pandas DataFrames using sentence transformers and FAISS.
"""

from .common_utilities import ConfigurationError
from .data_cache import EmbeddingCacheManager
from .document_indexer import FaissVectorIndexer
from .embedding_service import SentenceEmbeddingService
from .query_processor import EmbeddingConfig, PandaSearch

__version__ = "1.0.0"

__all__ = [
    "PandaSearch",
    "SentenceEmbeddingService",
    "FaissVectorIndexer",
    "EmbeddingCacheManager",
    "EmbeddingConfig",
    "ConfigurationError",
]

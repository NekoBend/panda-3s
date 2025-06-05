"""
panda-3s: Pandas-focused Semantic Search Suite

A simple and powerful semantic search suite for pandas DataFrames using sentence transformers and FAISS.
"""

from .core import PandaSearch
from .embedding import Embedding
from .indexer import FaissIndex

__version__ = "0.1.0"

__all__ = ["PandaSearch", "Embedding", "FaissIndex"]

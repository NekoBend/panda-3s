"""Tests for the EmbeddingCacheManager module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panda_vector_search.data_cache import EmbeddingCacheManager


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a test sentence for caching.",
        "Another sample text for cache testing.",
        "Third example text for verification.",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    np.random.seed(42)
    return [
        np.random.rand(384).astype(np.float32),
        np.random.rand(384).astype(np.float32),
        np.random.rand(384).astype(np.float32),
    ]


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "text": ["Doc 1", "Doc 2", "Doc 3"],
        "category": ["A", "B", "A"],
    })


def test_cache_manager_initialization():
    """Test proper initialization of EmbeddingCacheManager."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = EmbeddingCacheManager(cache_directory=tmp_dir)

        assert cache_manager._cache_root_directory == Path(tmp_dir)
        assert cache_manager._cache_root_directory.exists()


def test_persist_and_retrieve_single_embedding(sample_df):
    """Test persisting and retrieving a single embedding."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = EmbeddingCacheManager(cache_directory=tmp_dir)

        text = "Test text for embedding"
        embedding = np.random.rand(384).astype(np.float32)
        model_name = "test-model"

        # Persist embedding
        success = cache_manager.persist_embedding(
            text, embedding, model_name, sample_df
        )
        assert success is True

        # Retrieve embedding
        retrieved = cache_manager.retrieve_embedding(text, model_name, sample_df)

        assert retrieved is not None
        np.testing.assert_array_equal(embedding, retrieved)


def test_persist_and_retrieve_batch_embeddings(
    sample_texts, sample_embeddings, sample_df
):
    """Test batch persist and retrieve operations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = EmbeddingCacheManager(cache_directory=tmp_dir)

        model_name = "test-model"

        # Persist batch
        success = cache_manager.persist_embeddings_batch(
            sample_texts, sample_embeddings, model_name, sample_df
        )
        assert success is True

        # Retrieve batch
        results, missing = cache_manager.retrieve_embeddings_batch(
            sample_texts, model_name, sample_df
        )

        assert len(results) == len(sample_texts)
        assert len(missing) == 0  # All should be found

        for i, retrieved in enumerate(results):
            assert retrieved is not None
            np.testing.assert_array_equal(sample_embeddings[i], retrieved)


def test_cache_miss_handling(sample_df):
    """Test handling of cache misses."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = EmbeddingCacheManager(cache_directory=tmp_dir)

        # Try to retrieve non-existent embedding
        result = cache_manager.retrieve_embedding(
            "non-existent text", "test-model", sample_df
        )

        assert result is None


def test_partial_cache_hit(sample_texts, sample_embeddings, sample_df):
    """Test scenario with partial cache hits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = EmbeddingCacheManager(cache_directory=tmp_dir)

        model_name = "test-model"

        # Cache only first two embeddings
        success = cache_manager.persist_embeddings_batch(
            sample_texts[:2], sample_embeddings[:2], model_name, sample_df
        )
        assert success is True

        # Try to retrieve all three
        results, missing = cache_manager.retrieve_embeddings_batch(
            sample_texts, model_name, sample_df
        )

        assert len(results) == len(sample_texts)
        assert len(missing) == 1  # Third text should be missing
        assert missing[0] == sample_texts[2]

        # First two should be found
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None


def test_cache_statistics():
    """Test cache statistics functionality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = EmbeddingCacheManager(cache_directory=tmp_dir)

        stats = cache_manager.get_cache_statistics()

        assert isinstance(stats, dict)
        assert "cache_version" in stats
        assert "cache_directory" in stats
        assert "memory_cache_items" in stats
        assert "disk_cache_files" in stats

        # Initial state should have empty caches
        assert stats["memory_cache_items"] == 0
        assert stats["disk_cache_files"] == 0


def test_clear_all_cached_data():
    """Test clearing all cached data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_manager = EmbeddingCacheManager(cache_directory=tmp_dir)

        # Add some data
        text = "Test text"
        embedding = np.random.rand(384).astype(np.float32)
        model_name = "test-model"

        cache_manager.persist_embedding(text, embedding, model_name)

        # Verify data exists
        retrieved = cache_manager.retrieve_embedding(text, model_name)
        assert retrieved is not None

        # Clear all data
        cache_manager.clear_all_cached_data()

        # Verify data is cleared
        retrieved_after_clear = cache_manager.retrieve_embedding(text, model_name)
        assert retrieved_after_clear is None


def test_in_memory_cache_lru():
    """Test LRU behavior of in-memory cache."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create cache manager with small memory cache size
        cache_manager = EmbeddingCacheManager(
            cache_directory=tmp_dir, max_in_memory_cache_size=2
        )

        model_name = "test-model"

        # Add three embeddings (exceeds cache size)
        for i in range(3):
            text = f"text_{i}"
            embedding = np.random.rand(384).astype(np.float32)
            cache_manager.persist_embedding(text, embedding, model_name)

        # Check memory cache size doesn't exceed limit
        stats = cache_manager.get_cache_statistics()
        assert stats["memory_cache_items"] <= 2

"""Tests for the SentenceEmbeddingService module."""

import numpy as np
import pandas as pd
import pytest

from panda_vector_search.embedding_service import SentenceEmbeddingService


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a test sentence.",
        "Another sample text for embedding.",
        "Third example text here.",
    ]


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "text": [
            "First document content",
            "Second document text",
            "Third document example",
        ],
        "category": ["A", "B", "A"],
    })


def test_embedding_service_initialization():
    """Test proper initialization of SentenceEmbeddingService."""
    service = SentenceEmbeddingService(
        model_name_or_path="all-MiniLM-L6-v2",
        device="cpu",
        enable_embedding_cache=False,
    )

    assert service.model_name_or_path == "all-MiniLM-L6-v2"
    assert service.cache is None
    assert service.get_embedding_dimension() > 0


def test_embedding_generation(sample_texts):
    """Test embedding generation for a list of texts."""
    service = SentenceEmbeddingService(
        model_name_or_path="all-MiniLM-L6-v2",
        device="cpu",
        enable_embedding_cache=False,
    )

    embeddings = service.embed(sample_texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] == service.get_embedding_dimension()
    assert embeddings.dtype == np.float32


def test_embedding_with_cache(sample_texts, tmp_path):
    """Test embedding generation with caching enabled."""
    service = SentenceEmbeddingService(
        model_name_or_path="all-MiniLM-L6-v2",
        device="cpu",
        enable_embedding_cache=True,
        embedding_cache_directory=str(tmp_path / "test_cache"),
    )

    # First call should generate and cache embeddings
    embeddings1 = service.embed(sample_texts)

    # Second call should retrieve from cache
    embeddings2 = service.embed(sample_texts)

    # Results should be identical
    np.testing.assert_array_equal(embeddings1, embeddings2)


def test_empty_text_list():
    """Test handling of empty text list."""
    service = SentenceEmbeddingService(
        model_name_or_path="all-MiniLM-L6-v2",
        device="cpu",
        enable_embedding_cache=False,
    )

    embeddings = service.embed([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, service.get_embedding_dimension())


def test_embedding_with_dataframe_context(sample_texts, sample_df):
    """Test embedding generation with DataFrame context."""
    service = SentenceEmbeddingService(
        model_name_or_path="all-MiniLM-L6-v2",
        device="cpu",
        enable_embedding_cache=False,
    )

    embeddings = service.embed(sample_texts, cache_context_df=sample_df)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] == service.get_embedding_dimension()

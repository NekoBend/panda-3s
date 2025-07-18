"""Tests for the FaissVectorIndexer module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panda_vector_search.document_indexer import FaissVectorIndexer


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "First test document for indexing",
        "Second document with different content",
        "Third document for vector search testing",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings corresponding to test documents."""
    np.random.seed(42)  # For reproducible tests
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "text": [
            "First DataFrame text content",
            "Second DataFrame text example",
            "Third DataFrame text sample",
        ],
        "category": ["A", "B", "A"],
        "score": [0.9, 0.8, 0.7],
    })


def test_indexer_initialization():
    """Test proper initialization of FaissVectorIndexer."""
    indexer = FaissVectorIndexer(vector_dimension=384)

    assert indexer.get_vector_dimension() == 384
    assert indexer.get_document_count() == 0
    assert indexer.get_indexed_vector_count() == 0


def test_add_documents(sample_documents, sample_embeddings):
    """Test adding documents and embeddings to the index."""
    indexer = FaissVectorIndexer()
    indexer.add_documents(sample_documents, sample_embeddings)

    assert indexer.get_document_count() == len(sample_documents)
    assert indexer.get_indexed_vector_count() == len(sample_documents)
    assert indexer.get_vector_dimension() == sample_embeddings.shape[1]


def test_add_dataframe(sample_df):
    """Test adding DataFrame data to the index."""
    indexer = FaissVectorIndexer()

    # Generate random embeddings for the DataFrame
    np.random.seed(42)
    embeddings = np.random.rand(len(sample_df), 384).astype(np.float32)

    indexer.add_dataframe(sample_df, "text", embeddings)

    assert indexer.get_document_count() == len(sample_df)
    assert indexer.get_indexed_vector_count() == len(sample_df)


def test_search_functionality(sample_documents, sample_embeddings):
    """Test search functionality of the indexer."""
    indexer = FaissVectorIndexer()
    indexer.add_documents(sample_documents, sample_embeddings)

    # Use first embedding as query (should return itself as top result)
    query_embedding = sample_embeddings[0]
    results = indexer.search(query_embedding, k=2, return_scores=False)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert sample_documents[0] in results


def test_search_with_scores(sample_documents, sample_embeddings):
    """Test search functionality returning scores."""
    indexer = FaissVectorIndexer()
    indexer.add_documents(sample_documents, sample_embeddings)

    query_embedding = sample_embeddings[0]
    results, scores = indexer.search(query_embedding, k=2, return_scores=True)

    assert isinstance(results, list)
    assert isinstance(scores, list)
    assert len(results) == len(scores)
    assert all(isinstance(score, float) for score in scores)


def test_save_and_load_index(sample_documents, sample_embeddings):
    """Test saving and loading index functionality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_path = Path(tmp_dir) / "test_index.faiss"
        metadata_path = Path(tmp_dir) / "test_metadata.json"

        # Create and populate indexer
        indexer1 = FaissVectorIndexer()
        indexer1.add_documents(sample_documents, sample_embeddings)

        # Save index
        indexer1.save_vector_index(str(index_path), str(metadata_path))

        # Load into new indexer
        indexer2 = FaissVectorIndexer()
        indexer2.load_vector_index(str(index_path), str(metadata_path))

        # Verify loaded data
        assert indexer2.get_document_count() == len(sample_documents)
        assert indexer2.get_indexed_vector_count() == len(sample_documents)
        assert indexer2.get_vector_dimension() == sample_embeddings.shape[1]


def test_clear_index_data(sample_documents, sample_embeddings):
    """Test clearing index data."""
    indexer = FaissVectorIndexer()
    indexer.add_documents(sample_documents, sample_embeddings)

    # Verify data is present
    assert indexer.get_document_count() > 0

    # Clear data
    indexer.clear_index_data()

    # Verify data is cleared
    assert indexer.get_document_count() == 0
    assert indexer.get_indexed_vector_count() == 0
    assert indexer.get_vector_dimension() is None


def test_dimension_mismatch_error():
    """Test that dimension mismatch raises appropriate error."""
    indexer = FaissVectorIndexer(vector_dimension=384)

    # Add initial documents with correct dimension
    documents = ["test doc"]
    embeddings = np.random.rand(1, 384).astype(np.float32)
    indexer.add_documents(documents, embeddings)

    # Try to add documents with wrong dimension
    wrong_embeddings = np.random.rand(1, 512).astype(np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        indexer.add_documents(["another doc"], wrong_embeddings)


def test_empty_search():
    """Test search behavior on empty index."""
    indexer = FaissVectorIndexer()
    query_embedding = np.random.rand(384).astype(np.float32)

    results = indexer.search(query_embedding, k=5)

    assert isinstance(results, list)
    assert len(results) == 0

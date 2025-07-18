import pandas as pd
import pytest

from panda_vector_search.query_processor.core import EmbeddingConfig, PandaSearch


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "title": [
            "AI Research Papers",
            "Machine Learning Guide",
            "Data Science Basics",
        ],
        "content": [
            "Latest developments in AI",
            "Comprehensive ML tutorial",
            "Introduction to data analysis",
        ],
        "category": ["Research", "Education", "Tutorial"],
    })


def test_search_e2e(sample_df):
    search = PandaSearch(df=sample_df, text_columns=["title", "content"])

    # Use a known fast model for testing
    embedding_config = EmbeddingConfig(
        model_name_or_path="all-MiniLM-L6-v2", device="cpu"
    )

    search.embedding(config=embedding_config)

    assert search.is_indexed
    assert search.indexed_item_count == len(sample_df)

    results = search.search("machine learning tutorial", k=3)

    assert isinstance(results, pd.DataFrame)
    assert "score" in results.columns
    assert len(results) > 0
    assert results.iloc[0]["title"] == "Machine Learning Guide"

    # Test with threshold
    results_with_threshold = search.search(
        "machine learning tutorial", k=3, threshold=0.5
    )
    assert len(results_with_threshold) <= len(results)

    # Test with dict query
    dict_query_results = search.search({"title": "AI", "content": "developments"}, k=1)
    assert len(dict_query_results) == 1
    assert dict_query_results.iloc[0]["title"] == "AI Research Papers"

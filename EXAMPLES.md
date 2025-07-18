# Example Usage of panda-vector-search

This file demonstrates various usage patterns for the panda-vector-search library.

## Basic Usage

```python
import pandas as pd
from panda_vector_search import PandaSearch, EmbeddingConfig

# Sample data
df = pd.DataFrame({
    "title": [
        "AI Research Papers",
        "Machine Learning Guide",
        "Data Science Basics",
        "Python Programming Tutorial",
        "Deep Learning Applications"
    ],
    "content": [
        "Latest developments in artificial intelligence",
        "Comprehensive machine learning tutorial",
        "Introduction to data analysis",
        "Learn Python programming fundamentals",
        "Neural networks and deep learning methods"
    ],
    "category": ["Research", "Education", "Tutorial", "Programming", "Research"]
})

# Initialize search engine
search = PandaSearch(df=df, text_columns=["title", "content"])

# Configure embeddings
config = EmbeddingConfig(
    model_name_or_path="all-MiniLM-L6-v2",
    device="cpu"
)

# Build embeddings and index
search.embedding(config=config)

# Perform searches
results = search.search("machine learning tutorial", k=3)
print(results)
```

## Advanced Configuration

```python
# Custom configuration with caching
advanced_config = EmbeddingConfig(
    model_name_or_path="all-mpnet-base-v2",
    device="cuda",  # Use GPU if available
    enable_embedding_artifact_cache=True,
    embedding_artifact_cache_dir="./custom_cache",
    trust_remote_code=False
)

# Multi-column search with weights
weighted_search = PandaSearch(
    df=df,
    text_columns=["title", "content"],
    column_weights={"title": 2.0, "content": 1.0}  # Title is more important
)

weighted_search.embedding(config=advanced_config)

# Dictionary-based query
results = weighted_search.search({
    "title": "machine learning",
    "content": "tutorial guide"
}, k=5, threshold=0.3)
```

## Error Handling

```python
from panda_vector_search import ConfigurationError

try:
    # Invalid configuration
    invalid_config = EmbeddingConfig(
        model_name_or_path="",  # Empty model name
        device="invalid_device"
    )
    search.embedding(config=invalid_config)
except ConfigurationError as e:
    print(f"Configuration error: {e}")

try:
    # Search without building index
    unindexed_search = PandaSearch(df=df, text_columns=["title"])
    results = unindexed_search.search("test query")
except ValueError as e:
    print(f"Search error: {e}")
```

## Performance Monitoring

```python
# Check if indexed
print(f"Is indexed: {search.is_indexed}")
print(f"Indexed items: {search.indexed_item_count}")

# Cache statistics
cache_stats = search.get_cache_statistics()
if cache_stats:
    print(f"Cache version: {cache_stats['cache_version']}")
    print(f"Memory cache items: {cache_stats['memory_cache_items']}")
    print(f"Disk cache files: {cache_stats['disk_cache_files']}")
```

## Working with Large Datasets

```python
import numpy as np

# Generate larger dataset
large_df = pd.DataFrame({
    "document_id": range(10000),
    "title": [f"Document {i}" for i in range(10000)],
    "content": [f"Content for document {i} with various keywords" for i in range(10000)],
    "category": np.random.choice(["A", "B", "C"], 10000)
})

# Use caching for better performance
large_search = PandaSearch(df=large_df, text_columns=["title", "content"])

efficient_config = EmbeddingConfig(
    model_name_or_path="all-MiniLM-L6-v2",
    enable_embedding_artifact_cache=True,
    embedding_artifact_cache_dir="./large_dataset_cache"
)

# First run will generate embeddings
large_search.embedding(config=efficient_config)

# Subsequent runs will use cache
results = large_search.search("document keywords", k=10)
```

## Custom Column Processing

```python
# Prepare data with custom text processing
df_custom = pd.DataFrame({
    "product_name": ["Laptop Pro", "Gaming Mouse", "Mechanical Keyboard"],
    "description": ["High-performance laptop", "Precision gaming mouse", "Tactile keyboard"],
    "features": ["Intel i7, 16GB RAM", "RGB lighting, DPI", "Blue switches, backlit"],
    "price": [1200, 50, 150]
})

# Combine multiple text columns
multi_column_search = PandaSearch(
    df=df_custom,
    text_columns=["product_name", "description", "features"],
    column_separator=" | ",  # Custom separator
    column_weights={
        "product_name": 3.0,
        "description": 2.0,
        "features": 1.0
    }
)

multi_column_search.embedding(config=config)
results = multi_column_search.search("gaming laptop RGB", k=3)
```

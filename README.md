# panda-3s: Pandas-focused Semantic Search Suite

Simple and powerful semantic search library for pandas DataFrames using Sentence Transformers and FAISS.

## Installation

### From PyPI (when available)

```bash
pip install panda-3s
```

### From Git Repository

```bash
pip install git+https://github.com/NekoBend/panda-3s.git
# or for development
pip install -e git+https://github.com/NekoBend/panda-3s.git#egg=panda-3s
```

## Basic Usage

### Simple semantic search with DataFrame

```python
import panda_3s
import pandas as pd

# Test data
df = pd.DataFrame({
    "text": ["AI is useful", "Hello world", "Machine learning study"],
    "category": ["AI", "greeting", "ML"]
})

# Create search engine using PandaSearch class
search_engine = panda_3s.PandaSearch(df, "text")

# Setup embedding model (handles model, device, and cache automatically)
search_engine.embedding()

# Perform semantic search
results = search_engine.search("Hello World", k=3)
print(results)
```

### Advanced usage with embedding configuration

```python
# Customize embedding model and settings
search_engine = panda_3s.PandaSearch(df, "text")
search_engine.embedding(
    model_name_or_path="all-mpnet-base-v2",
    device="cpu",  # or "cuda"
    enable_cache=True,
    embedding_cache="./my_embedding_cache",
    model_cache="./my_model_cache"
)

# Search with custom parameters
results = search_engine.search(
    "artificial intelligence",
    k=5,
    threshold=0.5
)
```

### Multiple column search

```python
# Search across multiple columns with flexible options
df_multi = pd.DataFrame({
    'title': ['AI Research', 'Data Science', 'Machine Learning'],
    'content': ['Neural networks', 'Statistical analysis', 'Deep learning']
})

# Basic multiple column search
search_engine = panda_3s.PandaSearch(df_multi, ["title", "content"])
search_engine.embedding()
results = search_engine.search("machine learning AI", k=3)

# Advanced: Custom separator and column weights
weights = {"title": 2.0, "content": 1.0}  # Title weighted 2x more important
search_engine = panda_3s.PandaSearch(
    df_multi,
    ["title", "content"],
    column_separator=" | ",
    column_weights=weights
)
search_engine.embedding()
results = search_engine.search("research", k=3)
```

## API Reference

### PandaSearch Class

#### `panda_3s.PandaSearch(df, text_columns, column_separator=" ", column_weights=None)`

Create a semantic search engine for pandas DataFrame.

**Parameters:**

- `df` (pandas.DataFrame): DataFrame to search
- `text_columns` (str or list): Column name(s) to search in
- `column_separator` (str): Separator for joining multiple columns (default: " ")
- `column_weights` (dict): Optional weights for different columns when combining

**Returns:**

- PandaSearch: Search engine instance

#### `embedding(model_name_or_path="all-MiniLM-L6-v2", device=None, ...)`

Configure and build embeddings for the DataFrame.

**Parameters:**

- `model_name_or_path` (str): Sentence transformer model name
- `device` (str): Device to use ("cpu" or "cuda", auto-detected if None)
- `enable_cache` (bool): Enable embedding cache (default: True)
- `cache_dir` (str): Cache directory path (default: "./panda_3s_cache")

#### `search(query, k=10, threshold=0.0)`

Perform semantic search.

**Parameters:**

- `query` (str): Search query
- `k` (int): Number of results to return (default: 10)
- `threshold` (float): Minimum similarity threshold (default: 0.0)

**Returns:**

- pandas.DataFrame with columns: original data + 'similarity_score'

### Output Format

Search results are returned as a pandas DataFrame with similarity scores added:

```python
   index           text  category  similarity_score
0      1    Hello world  greeting          0.834567
1      2  Machine learning study        ML          0.756432
```

The results are automatically sorted by similarity score in descending order, with the most relevant matches first.

## Features

- ✅ Designed specifically for pandas DataFrames
- ✅ Simple unified class-based API (`panda_3s.PandaSearch`) focused on search conditions
- ✅ Automatic embedding model management and optimization
- ✅ High-performance caching with safetensors (managed internally)
- ✅ Hash-based sharding for large datasets
- ✅ Support for multiple columns
- ✅ Configurable similarity thresholds
- ✅ GPU and CPU support (automatically detected)
- ✅ Type hints and error handling

## Architecture

The library follows a clean separation of concerns:

- **Search API**: Simple interface for specifying search conditions (`query`, `k`, `threshold`)
- **Embedding Layer**: Handles model selection, device management, and performance optimization
- **Cache System**: Transparent caching for embeddings and indexes without user intervention
- **Index Management**: Efficient FAISS-based similarity search with automatic optimization

## Cache System

The library includes an advanced caching system that operates transparently:

- Uses safetensors format for safe and fast storage
- Implements hash-based sharding for large datasets
- Supports parallel loading for improved performance
- Automatically avoids recomputation of embeddings for identical data
- Stores both embeddings and FAISS indexes efficiently

Cache files are automatically managed in the default cache directory (`./panda_3s_cache`) with no user configuration required. The system is designed to be invisible to the user while providing significant performance benefits.

## Requirements

- Python >= 3.13
- pandas >= 2.3.0
- sentence-transformers >= 4.1.0
- faiss-cpu >= 1.11.0
- safetensors >= 0.5.3

## License

This project is open source and available under the MIT License.

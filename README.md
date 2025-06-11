# 🐼 panda-3s: Pandas-focused Semantic Search Suite

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/NekoBend/panda-3s)

Simple and powerful semantic search library for pandas DataFrames using Sentence Transformers and FAISS with advanced caching capabilities.

## ✨ Features

- 🎯 **DataFrame-native**: Designed specifically for pandas DataFrames
- 🚀 **High Performance**: Advanced caching system with safetensors and hash-based sharding
- 🔧 **Simple API**: Clean, intuitive interface with `PandaSearch` class
- 📊 **Multi-column Support**: Search across multiple columns with weights
- 💾 **Smart Caching**: Transparent embedding and index caching for optimal performance
- 🤖 **Auto-optimization**: Automatic device detection and model management
- 🔒 **Type Safe**: Full type hints and robust error handling

## 📦 Installation

### Development Installation

```bash
git clone https://github.com/NekoBend/panda-3s.git
cd panda-3s
pip install -e .
```

### From Git Repository

```bash
pip install git+https://github.com/NekoBend/panda-3s.git
```

## 🚀 Quick Start

### Basic Usage

```python
import pandas as pd
from panda_3s import PandaSearch

# Sample data
df = pd.DataFrame({
    "title": ["AI Research Papers", "Machine Learning Guide", "Data Science Basics"],
    "content": ["Latest developments in AI", "Comprehensive ML tutorial", "Introduction to data analysis"],
    "category": ["Research", "Education", "Tutorial"]
})

# Create search engine
search = PandaSearch(df=df, text_columns=["title", "content"])

# Build embeddings (with automatic caching)
# Configure embedding settings using EmbeddingConfig
from panda_3s import EmbeddingConfig
embedding_config = EmbeddingConfig(
    model_name_or_path="all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu", # Example, adjust as needed
    # Add other relevant EmbeddingConfig parameters here
)
search.embedding(config=embedding_config)

# Perform search
results = search.search("machine learning tutorial", k=3)
print(results)
```

### Advanced Configuration

```python
# Custom model and cache settings
# Configure embedding settings using EmbeddingConfig
embedding_config_advanced = EmbeddingConfig(
    model_name_or_path="all-mpnet-base-v2",
    device="cuda",  # Use GPU if available
    cache_dir="./my_cache",
    # Add other relevant EmbeddingConfig parameters here
)
search.embedding(config=embedding_config_advanced)

# Search with threshold
results = search.search("AI research", k=5, threshold=0.7)
```

## 📖 API Reference

### PandaSearch

Main class for semantic search on pandas DataFrames.

#### Constructor

```python
PandaSearch(df, text_columns, column_separator=" ", column_weights=None)
```

**Parameters:**

- `df` (pd.DataFrame): DataFrame to search
- `text_columns` (str | List[str]): Column name(s) for text search
- `column_separator` (str): Separator for joining multiple columns (default: " ")
- `column_weights` (Dict[str, float]): Column importance weights (optional)

#### Methods

##### `embedding(config: EmbeddingConfig)`

Configure and build embeddings for the DataFrame.

**Parameters:**

- `config` (EmbeddingConfig): Configuration object for embedding generation.
  - `model_name_or_path` (str): Sentence transformer model (default: "all-MiniLM-L6-v2").
  - `device` (str | None): Device ("cpu"/"cuda", auto-detected if None).
  - `enable_cache` (bool): Enable embedding cache (default: True).
  - `cache_dir` (str): Cache directory (default: "./.panda_3s_cache").
  - `trust_remote_code` (bool): Trust remote model code (default: False).
  - `batch_size` (int): Batch size for embedding generation (default: 32).
  - `show_progress_bar` (bool): Show progress bar during embedding (default: False).
  - `normalize_embeddings` (bool): Normalize embeddings after generation (default: False).
  - `force_rebuild` (bool): Force rebuild of embeddings, ignoring cache (default: False).

##### `search(query, k=10, threshold=0.0)`

Perform semantic search.

**Parameters:**

- `query` (str | Dict[str, str]): Search query or column-specific queries
- `k` (int): Number of results (default: 10)
- `threshold` (float): Minimum similarity score (default: 0.0)

**Returns:**

- `pd.DataFrame`: Results with similarity scores

### Output Format

```python
# Example output
   title                content           category  similarity_score
0  Machine Learning Guide  Comprehensive ML tutorial  Education     0.892
1  AI Research Papers      Latest developments in AI  Research      0.743
```

## 🔧 Requirements

- **Python**: >= 3.13
- **pandas**: >= 2.3.0
- **sentence-transformers**: >= 4.1.0
- **faiss-cpu**: >= 1.11.0
- **safetensors**: >= 0.5.3

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

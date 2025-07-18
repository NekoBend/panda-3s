# 🐼 Panda-Semantic-Search-Suite

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/NekoBend/panda-3s)

Simple and powerful semantic search suite for pandas DataFrames using Sentence Transformers and FAISS with advanced caching capabilities.

## ✨ Features

- 🎯 **DataFrame-native**: Designed specifically for pandas DataFrames
- 🚀 **High Performance**: Advanced caching system with safetensors and hash-based sharding
- 🔧 **Simple API**: Clean, intuitive interface with `PandaSearch` class
- 📊 **Multi-column Support**: Search across multiple columns with weights
- 💾 **Smart Caching**: Transparent embedding and index caching for optimal performance
- 🤖 **Auto-optimization**: Automatic device detection and model management
- 🔒 **Type Safe**: Full type hints and robust error handling

## 📦 Installation

### Prerequisites

First, install [uv](https://docs.astral.sh/uv/) for modern Python dependency management:

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Development Installation

```bash
git clone https://github.com/NekoBend/panda-3s.git
cd panda-3s
uv sync
```

### From Git Repository

```bash
uv add git+https://github.com/NekoBend/panda-3s.git
```

## 🚀 Quick Start

```python
import pandas as pd
from panda_vector_search import PandaSearch, EmbeddingConfig

# Sample data
df = pd.DataFrame({
    "title": ["AI Research Papers", "Machine Learning Guide", "Data Science Basics"],
    "content": ["Latest developments in AI", "Comprehensive ML tutorial", "Introduction to data analysis"],
    "category": ["Research", "Education", "Tutorial"]
})

# Create search engine
search = PandaSearch(df=df, text_columns=["title", "content"])

# Configure embeddings
config = EmbeddingConfig(
    model_name_or_path="all-MiniLM-L6-v2",
    device="cpu"
)

# Build embeddings and index
search.embedding(config=config)

# Perform search
results = search.search("machine learning tutorial", k=3)
print(results)
```

## 📖 Documentation

For detailed usage examples and API documentation, see:

- **[EXAMPLES.md](./EXAMPLES.md)** - Comprehensive usage examples
- **[scripts/performance_test.py](./scripts/performance_test.py)** - Performance benchmarking

## 📊 Performance

The library is designed for high performance with built-in caching:

- **Embedding Cache**: Persistent storage with safetensors
- **Fast Search**: FAISS-powered vector similarity search
- **Memory Efficient**: LRU cache with configurable limits

See `scripts/performance_test.py` for benchmark results.

## 🧪 Testing

Run the test suite:

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run performance benchmark
uv run python scripts/performance_test.py
```

## 🔧 Requirements

- **Python**: >= 3.9
- **pandas**: >= 2.3.0
- **sentence-transformers**: >= 4.1.0
- **faiss-cpu**: >= 1.11.0
- **safetensors**: >= 0.5.3

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

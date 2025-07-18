# Performance Test for panda-vector-search

import time

import numpy as np
import pandas as pd

from panda_vector_search import EmbeddingConfig, PandaSearch


def generate_sample_data(num_rows: int) -> pd.DataFrame:
    """Generate sample data for performance testing."""
    np.random.seed(42)

    categories = ["Technology", "Science", "Business", "Health", "Sports"]

    data = {
        "title": [
            f"Document {i}: {np.random.choice(['AI', 'ML', 'Data', 'Research', 'Analysis'])} Study"
            for i in range(num_rows)
        ],
        "content": [
            f"This is sample content for document {i}. "
            + " ".join([f"keyword_{j}" for j in range(10)])
            for i in range(num_rows)
        ],
        "category": [np.random.choice(categories) for _ in range(num_rows)],
        "score": np.random.uniform(0.1, 1.0, num_rows),
    }

    return pd.DataFrame(data)


def benchmark_search_performance(num_rows: int, num_searches: int = 10):
    """Benchmark search performance with different dataset sizes."""
    print(f"\n=== Performance Test: {num_rows} rows ===")

    # Generate test data
    print("Generating test data...")
    df = generate_sample_data(num_rows)

    # Initialize search engine
    print("Initializing search engine...")
    search = PandaSearch(df=df, text_columns=["title", "content"])

    # Configure embedding
    embedding_config = EmbeddingConfig(
        model_name_or_path="all-MiniLM-L6-v2",
        device="cpu",
        enable_embedding_artifact_cache=True,
    )

    # Measure embedding time
    print("Building embeddings and index...")
    start_time = time.time()
    search.embedding(config=embedding_config)
    embedding_time = time.time() - start_time

    print(f"Embedding time: {embedding_time:.2f} seconds")
    print(f"Indexed items: {search.indexed_item_count}")

    # Measure search performance
    print(f"Running {num_searches} search queries...")
    search_queries = [
        "artificial intelligence research",
        "machine learning applications",
        "data science methodology",
        "performance optimization",
        "technology innovation",
    ]

    search_times = []
    for i in range(num_searches):
        query = search_queries[i % len(search_queries)]
        start_time = time.time()
        results = search.search(query, k=10)
        search_time = time.time() - start_time
        search_times.append(search_time)

        if i == 0:  # Show first result details
            print(f"First search returned {len(results)} results")

    avg_search_time = np.mean(search_times)
    print(f"Average search time: {avg_search_time:.4f} seconds")
    print(f"Search throughput: {1 / avg_search_time:.2f} queries/second")

    # Show cache statistics
    cache_stats = search.get_cache_statistics()
    if cache_stats:
        print("Cache statistics:")
        print(f"  Memory cache items: {cache_stats['memory_cache_items']}")
        print(f"  Disk cache files: {cache_stats['disk_cache_files']}")
        print(f"  Disk cache size: {cache_stats['disk_cache_total_size_bytes']} bytes")

    return {
        "num_rows": num_rows,
        "embedding_time": embedding_time,
        "avg_search_time": avg_search_time,
        "search_throughput": 1 / avg_search_time,
    }


def main():
    """Run performance benchmarks."""
    print("🐼 panda-vector-search Performance Benchmark")
    print("=" * 50)

    # Test different dataset sizes
    test_sizes = [100, 500, 1000, 5000]
    results = []

    for size in test_sizes:
        try:
            result = benchmark_search_performance(size)
            results.append(result)
        except Exception as e:
            print(f"Error testing {size} rows: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"{'Rows':<8} {'Embed Time':<12} {'Search Time':<12} {'Throughput':<12}")
    print("-" * 50)

    for result in results:
        print(
            f"{result['num_rows']:<8} "
            f"{result['embedding_time']:<12.2f} "
            f"{result['avg_search_time']:<12.4f} "
            f"{result['search_throughput']:<12.2f}"
        )


if __name__ == "__main__":
    main()

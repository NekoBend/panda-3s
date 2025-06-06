"""
Advanced caching system using safetensors for embeddings and FAISS indexes.
Features sharding for large datasets and parallel loading for performance.
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

logger = logging.getLogger(__name__)


def _compute_hash(data: str, truncate: int | None = None) -> str:
    """Compute SHA256 hash for given data.

    Args:
        data: String data to hash
        truncate: Optional length to truncate hash to

    Returns:
        Hexadecimal hash string
    """
    hash_value = hashlib.sha256(data.encode()).hexdigest()
    return hash_value[:truncate] if truncate else hash_value


def _compute_cache_key(texts: list[str], model_name: str) -> str:
    """Compute cache key for a list of texts and model."""
    combined = f"{model_name}::" + "::".join(texts)
    return _compute_hash(combined)


def _compute_shard_hash(text: str) -> str:
    """Compute shard hash for a single text to enable hash-based sharding."""
    return _compute_hash(text, truncate=8)


def _compute_row_hash(row_data: str) -> str:
    """Compute hash for a single DataFrame row."""
    return _compute_hash(row_data)


def _compute_row_shard_path(row_hash: str, cache_dir: Path) -> Path:
    """Compute shard directory path for a row hash."""
    # Use first 2 characters for directory structure: aa/, ab/, ac/, etc.
    shard_dir = row_hash[:2]
    return cache_dir / shard_dir


def _compute_index_cache_key(df_hash: str, column: str, model_name: str) -> str:
    """Compute cache key for FAISS index."""
    combined = f"{model_name}::{column}::{df_hash}"
    return _compute_hash(combined)


class EmbeddingCache:
    """Cache for embeddings using safetensors with hash-based sharding support."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory to store cache files (default: ~/.panda_3s_cache/embeddings)
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".panda_3s_cache" / "embeddings"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"EmbeddingCache initialized: {self.cache_dir}")

    def get_embeddings(self, texts: list[str], model_name: str) -> Optional[np.ndarray]:
        """Retrieve embeddings from cache with hash-based sharding and parallel loading."""
        return self._get_embeddings_hash_sharded(texts, model_name)

    def _get_embeddings_hash_sharded(
        self, texts: list[str], model_name: str
    ) -> Optional[np.ndarray]:
        """Get embeddings using hash-based sharding."""
        cache_key = _compute_cache_key(texts, model_name)

        # Group texts by shard hash
        shard_groups: dict[str, list[str]] = {}
        for text in texts:
            shard_hash = _compute_shard_hash(text)
            if shard_hash not in shard_groups:
                shard_groups[shard_hash] = []
            shard_groups[shard_hash].append(text)        # Check if all required shards exist
        missing_shards = []
        for shard_hash in shard_groups:
            shard_file = self.cache_dir / f"{cache_key}_{shard_hash}.safetensors"
            if not shard_file.exists():
                missing_shards.append(shard_hash)

        if missing_shards:
            logger.debug(f"Missing shards for cache key {cache_key}: {missing_shards}")
            return None

        try:
            result_embeddings = {}

            def load_shard(shard_hash: str) -> tuple[str, dict[str, np.ndarray]]:
                """Load a single shard file."""
                shard_file = self.cache_dir / f"{cache_key}_{shard_hash}.safetensors"

                with safe_open(shard_file, framework="numpy") as f:
                    embeddings = f.get_tensor("embeddings")
                    # Get texts from the shard_groups for this shard_hash
                    shard_texts = shard_groups[shard_hash]

                return shard_hash, {
                    text: embeddings[i] for i, text in enumerate(shard_texts)
                }

            # Load shards in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_shard = {
                    executor.submit(load_shard, shard_hash): shard_hash
                    for shard_hash in shard_groups
                }

                for future in as_completed(future_to_shard):
                    try:
                        shard_hash, shard_embeddings = future.result()
                        result_embeddings.update(shard_embeddings)
                    except Exception as e:
                        logger.error(
                            f"Error loading shard {future_to_shard[future]}: {e}"
                        )
                        return None            # Build result array in the same order as input texts
            if not all(text in result_embeddings for text in texts):
                logger.debug("Not all texts found in cache")
                return None

            embedding_dim = next(iter(result_embeddings.values())).shape[0]
            final_embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)

            for i, text in enumerate(texts):
                final_embeddings[i] = result_embeddings[text]

            logger.info(
                f"Successfully loaded {len(texts)} embeddings from hash-sharded cache"
            )
            return final_embeddings

        except Exception as e:
            logger.error(f"Error loading embeddings from hash-sharded cache: {e}")
            return None

    def save_embeddings(
        self, texts: list[str], model_name: str, embeddings: np.ndarray
    ) -> None:
        """Save embeddings to cache with hash-based sharding."""
        self._save_embeddings_hash_sharded(texts, model_name, embeddings)

    def _save_embeddings_hash_sharded(
        self, texts: list[str], model_name: str, embeddings: np.ndarray
    ) -> None:
        """Save embeddings to cache using hash-based sharding for better distribution."""
        cache_key = _compute_cache_key(texts, model_name)

        try:            # Group texts by shard hash
            shard_groups: dict[str, list[tuple[str, np.ndarray]]] = {}
            for i, text in enumerate(texts):
                shard_hash = _compute_shard_hash(text)
                if shard_hash not in shard_groups:
                    shard_groups[shard_hash] = []
                shard_groups[shard_hash].append((
                    text,
                    embeddings[i],
                ))

            # Save each shard
            for shard_hash, text_embedding_pairs in shard_groups.items():
                shard_embeddings = np.array([pair[1] for pair in text_embedding_pairs])

                # Save embeddings using safetensors with simple naming
                shard_file = self.cache_dir / f"{cache_key}_{shard_hash}.safetensors"
                save_file({"embeddings": shard_embeddings}, shard_file)

            logger.info(
                f"Saved {len(texts)} embeddings to hash-based cache in {len(shard_groups)} shards"
            )

        except Exception as e:
            logger.error(f"Error saving embeddings to hash-based cache: {e}")


class IndexCache:
    """Cache for FAISS indexes."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize index cache.

        Args:
            cache_dir: Directory to store cache files (default: ~/.panda_3s_cache/indexes)
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".panda_3s_cache" / "indexes"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"IndexCache initialized: {self.cache_dir}")

    def get_index(
        self, df_hash: str, column: str, model_name: str
    ) -> Optional[tuple[faiss.Index, list[int]]]:
        """Retrieve FAISS index from cache."""
        cache_key = _compute_index_cache_key(df_hash, column, model_name)

        index_file = self.cache_dir / f"{cache_key}.faiss"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

        if not (index_file.exists() and metadata_file.exists()):
            return None

        try:
            # Load FAISS index
            index = faiss.read_index(str(index_file))

            # Load metadata
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            row_indices = metadata["row_indices"]

            logger.info(f"Successfully loaded index from cache: {cache_key}")
            return index, row_indices

        except Exception as e:
            logger.error(f"Error loading index from cache: {e}")
            return None

    def save_index(
        self,
        df_hash: str,
        column: str,
        model_name: str,
        index: faiss.Index,
        row_indices: list[int],
    ) -> None:
        """Save FAISS index to cache."""
        cache_key = _compute_index_cache_key(df_hash, column, model_name)

        try:
            # Save FAISS index
            index_file = self.cache_dir / f"{cache_key}.faiss"
            faiss.write_index(index, str(index_file))  # Save metadata
            metadata = {
                "df_hash": df_hash,
                "column": column,
                "model_name": model_name,
                "row_indices": row_indices,
                "index_type": type(index).__name__,
                "total_count": len(row_indices),
            }

            metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved index to cache: {cache_key}")

        except Exception as e:
            logger.error(f"Error saving index to cache: {e}")

    def get_row_embedding(self, row_hash: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve embedding for a single row from cache."""
        shard_dir = _compute_row_shard_path(row_hash, self.cache_dir)
        shard_file = shard_dir / f"{model_name}_{row_hash[:8]}.safetensors"

        if not shard_file.exists():
            return None

        try:
            with safe_open(str(shard_file), framework="np") as f:
                if row_hash in f.keys():
                    embedding = f.get_tensor(row_hash)
                    logger.debug(f"Row embedding cache hit: {row_hash[:8]}...")
                    return embedding
            return None
        except Exception as e:
            logger.error(f"Error loading row embedding from cache: {e}")
            return None

    def save_row_embedding(
        self, row_hash: str, model_name: str, embedding: np.ndarray
    ) -> None:
        """Save embedding for a single row to cache."""
        shard_dir = _compute_row_shard_path(row_hash, self.cache_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)

        shard_file = shard_dir / f"{model_name}_{row_hash[:8]}.safetensors"

        try:
            # Load existing embeddings if file exists
            existing_embeddings = {}
            if shard_file.exists():
                with safe_open(str(shard_file), framework="np") as f:
                    for key in f.keys():
                        existing_embeddings[key] = f.get_tensor(key)

            # Add new embedding
            existing_embeddings[row_hash] = embedding

            # Save all embeddings back to file
            save_file(existing_embeddings, str(shard_file))
            logger.debug(f"Saved row embedding: {row_hash[:8]}...")

        except Exception as e:
            logger.error(f"Error saving row embedding to cache: {e}")

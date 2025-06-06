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


def _compute_shard_path(text: str, cache_dir: Path) -> tuple[Path, str]:
    """Compute shard directory path and filename for a text using directory-based sharding.

    Uses full SHA256 hash for collision resistance:
    - First 2 chars: Level 1 directory (256 possibilities)
    - Next 2 chars: Level 2 directory (256 possibilities per level 1)
    - Remaining chars: Filename (full collision resistance)

    Returns:
        tuple[Path, str]: (shard_directory, filename)
    """
    full_hash = _compute_hash(text)
    # Use 2-level directory structure for better distribution
    level1 = full_hash[:2]  # 00-ff (256 dirs)
    level2 = full_hash[2:4]  # 00-ff (256 subdirs each)
    filename = full_hash[4:]  # Remaining 56 chars for filename

    shard_dir = cache_dir / level1 / level2
    return shard_dir, filename


def _compute_shard_hash(text: str) -> str:
    """Compute shard identifier for a single text (backwards compatibility)."""
    full_hash = _compute_hash(text)
    return f"{full_hash[:2]}/{full_hash[2:4]}/{full_hash[4:]}"


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

    def __init__(self, cache_dir: Optional[str] = None):
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
        """Get embeddings using directory-based sharding with full collision resistance."""
        cache_key = _compute_cache_key(texts, model_name)

        # Group texts by shard directory with their original indices
        shard_groups: dict[str, list[tuple[int, str, Path]]] = {}
        for i, text in enumerate(texts):
            shard_dir, filename = _compute_shard_path(text, self.cache_dir)
            shard_key = str(shard_dir.relative_to(self.cache_dir))

            if shard_key not in shard_groups:
                shard_groups[shard_key] = []
            shard_groups[shard_key].append((
                i,
                text,
                shard_dir / f"{cache_key}_{filename}.safetensors",
            ))

        # Check if all required shard files exist
        missing_shards = []
        for shard_key, shard_data in shard_groups.items():
            for _, _, shard_file in shard_data:
                if not shard_file.exists():
                    missing_shards.append(str(shard_file))
                    break

        if missing_shards:
            logger.debug(f"Missing shard files: {missing_shards}")
            return None

        try:
            # Get first shard file to determine embedding dimension
            first_shard_data = next(iter(shard_groups.values()))
            first_shard_file = first_shard_data[0][2]  # Get first file path

            with safe_open(first_shard_file, framework="numpy") as f:
                sample_embedding = f.get_tensor("embeddings")
                embedding_dim = sample_embedding.shape[1]

            final_embeddings = np.zeros((len(texts), embedding_dim), dtype=np.float32)

            def load_shard_file(
                shard_file: Path, original_indices: list[int]
            ) -> tuple[Path, list[tuple[int, np.ndarray]] | None]:
                """Load a single shard file and return embeddings with their indices."""
                try:
                    with safe_open(shard_file, framework="numpy") as f:
                        embeddings = f.get_tensor("embeddings")

                        return shard_file, [
                            (original_idx, embeddings[shard_idx])
                            for shard_idx, original_idx in enumerate(original_indices)
                        ]
                except Exception as e:
                    logger.error(f"Error loading shard file {shard_file}: {e}")
                    return shard_file, None

            # Prepare file-to-indices mapping for parallel loading
            file_to_indices: dict[Path, list[int]] = {}
            for shard_data in shard_groups.values():
                for original_idx, _, shard_file in shard_data:
                    if shard_file not in file_to_indices:
                        file_to_indices[shard_file] = []
                    file_to_indices[shard_file].append(original_idx)

            # Load shard files in parallel
            successful_files = 0
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(load_shard_file, shard_file, indices): shard_file
                    for shard_file, indices in file_to_indices.items()
                }

                for future in as_completed(future_to_file):
                    shard_file, shard_embeddings = future.result()
                    if shard_embeddings is None:
                        logger.error(f"Failed to load shard file {shard_file}")
                        return None

                    # Directly assign to final array
                    for original_idx, embedding in shard_embeddings:
                        final_embeddings[original_idx] = embedding

                    successful_files += 1

            if successful_files != len(file_to_indices):
                logger.error(
                    f"Only loaded {successful_files}/{len(file_to_indices)} shard files successfully"
                )
                return None

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
        """Save embeddings to cache using directory-based sharding with full collision resistance."""
        cache_key = _compute_cache_key(texts, model_name)

        try:
            # Group texts by shard directory
            shard_files: dict[Path, list[tuple[str, np.ndarray]]] = {}
            for i, text in enumerate(texts):
                shard_dir, filename = _compute_shard_path(text, self.cache_dir)
                shard_file = shard_dir / f"{cache_key}_{filename}.safetensors"

                if shard_file not in shard_files:
                    shard_files[shard_file] = []
                shard_files[shard_file].append((text, embeddings[i]))

            # Save each shard file
            for shard_file, text_embedding_pairs in shard_files.items():
                # Ensure directory exists
                shard_file.parent.mkdir(parents=True, exist_ok=True)

                shard_embeddings = np.array([pair[1] for pair in text_embedding_pairs])

                # Save embeddings using safetensors
                save_file({"embeddings": shard_embeddings}, shard_file)

            logger.info(
                f"Saved {len(texts)} embeddings to directory-based cache in {len(shard_files)} shard files"
            )

        except Exception as e:
            logger.error(f"Error saving embeddings to hash-based cache: {e}")


class IndexCache:
    """Cache for FAISS indexes with row-level embedding support."""

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
            faiss.write_index(index, str(index_file))

            # Save metadata
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

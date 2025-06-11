"""
Safe and robust cache system for large datasets.
Uses safetensors exclusively, hash-based sharding, and DataFrame schema-aware keys.
"""

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from safetensors import safe_open  # type: ignore
from safetensors.numpy import save_file  # type: ignore

from panda_3s.utils.hashing import (  # Updated import
    compute_dataframe_schema_hash,
    compute_hash,
)

logger = logging.getLogger(__name__)


class SafeOptimizedEmbeddingCache:
    """Safe and robust embedding cache with safetensors, hash sharding, and schema awareness."""

    CACHE_VERSION = "3.0"  # Version for individual file caching

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        memory_cache_size: int = 1000,
    ):
        """Initialize safe optimized embedding cache.

        Args:
            cache_dir: Directory to store cache files
            memory_cache_size: Number of embeddings to keep in memory cache
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "panda_3s_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir = self.cache_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        self.memory_cache: Dict[str, np.ndarray] = {}
        self.memory_cache_order: List[str] = []
        self.memory_cache_size = memory_cache_size
        self.cache_lock = threading.RLock()

        self.cache_version_file = self.cache_dir / "cache_version.json"
        self.ensure_cache_version()

        logger.info(
            f"SafeOptimizedEmbeddingCache initialized at {self.cache_dir} (Version: {self.CACHE_VERSION})"
        )
        logger.info(f"Config: memory_cache_size={memory_cache_size}")

    def ensure_cache_version(self):
        """Ensure cache version compatibility."""
        version_data = {"version": self.CACHE_VERSION, "created": time.time()}

        if self.cache_version_file.exists():
            try:
                with open(self.cache_version_file, "r") as f:
                    existing_version_data = json.load(f)
                if existing_version_data.get("version") != self.CACHE_VERSION:
                    logger.warning(
                        f"Cache version mismatch. Expected {self.CACHE_VERSION}, "
                        f"found {existing_version_data.get('version')}. "
                        f"Clearing the old cache directory: {self.cache_dir}"
                    )
                    self.clear_all_cache(recreate_version_file=False)  # Avoid recursion
                    # Re-initialize basic dirs after clearing
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self.shards_dir.mkdir(parents=True, exist_ok=True)

                else:  # Versions match, preserve original creation time
                    version_data["created"] = existing_version_data.get(
                        "created", time.time()
                    )

            except json.JSONDecodeError:
                logger.error(
                    f"Failed to decode cache version file: {self.cache_version_file}. Re-initializing cache."
                )
                self.clear_all_cache(recreate_version_file=False)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.shards_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(
                    f"Error reading cache version file: {e}. Re-initializing cache."
                )
                self.clear_all_cache(recreate_version_file=False)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.shards_dir.mkdir(parents=True, exist_ok=True)

        # Write current or existing version data
        try:
            temp_file_path = self.cache_version_file.with_suffix(".json.tmp")
            with open(temp_file_path, "w") as f:
                json.dump(version_data, f, indent=4)
            shutil.move(str(temp_file_path), str(self.cache_version_file))
        except Exception as e:
            logger.error(f"Failed to write cache version file: {e}")

    def clear_all_cache(self, recreate_version_file: bool = True):
        """Clear all cache files safely."""
        with self.cache_lock:
            try:
                if self.shards_dir.exists():
                    shutil.rmtree(self.shards_dir)
                # Clear other potential old files/dirs at cache_dir root if necessary,
                # but be careful not to delete the version file if we are about to recreate it.
                for item in self.cache_dir.iterdir():
                    if item.is_file() and item.name != "cache_version.json":
                        item.unlink()
                    elif item.is_dir() and item.name != "shards":
                        shutil.rmtree(item)

                self.shards_dir.mkdir(
                    parents=True, exist_ok=True
                )  # Recreate shards dir
                self.clear_memory_cache()
                logger.info("All disk and memory cache cleared successfully.")
                if recreate_version_file:
                    # If version file was deleted or needs recreation
                    if self.cache_version_file.exists():
                        self.cache_version_file.unlink()
                    self.ensure_cache_version()  # Recreate version file with current version
            except Exception as e:
                logger.error(f"Error clearing all cache: {e}")

    def _compute_text_hash(
        self,
        text: str,
        model_name: str,
        dataframe_schema_hash: str,  # dataframe_schema_hash is now passed directly
    ) -> str:
        """Compute hash for text, model, and DataFrame schema combination."""
        # The dataframe_schema_hash is now computed externally and passed in.
        combined = f"{model_name}::{dataframe_schema_hash}::{text}"
        # Use the generic compute_hash from utils for consistency, though it's similar to hashlib.sha256().hexdigest()
        return compute_hash(combined)

    def _get_cache_file_path(self, text_hash: str, model_name: str) -> Path:
        """Get sharded cache file path based on text_hash and model_name."""
        shard_prefix = text_hash[:2]  # Use first 2 chars for sharding
        model_safe_name = model_name.replace("/", "_").replace(
            "\\\\", "_"
        )  # Make model name path-safe

        shard_dir = self.shards_dir / model_safe_name / shard_prefix
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{text_hash}.safetensors"

    def _add_to_memory_cache(self, cache_key: str, embedding: np.ndarray):
        with self.cache_lock:
            if cache_key in self.memory_cache:
                self.memory_cache_order.remove(cache_key)  # Move to end
            elif len(self.memory_cache) >= self.memory_cache_size:
                oldest_key = self.memory_cache_order.pop(0)  # Remove oldest
                del self.memory_cache[oldest_key]

            self.memory_cache[cache_key] = embedding
            self.memory_cache_order.append(cache_key)

    def _load_embedding_from_file(self, file_path: Path) -> Optional[np.ndarray]:
        """Load a single embedding from a safetensors file."""
        try:
            with safe_open(file_path, framework="numpy", device="cpu") as f:  # type: ignore
                # Assuming the embedding is stored with a consistent key, e.g., "embedding"
                tensor_keys = f.keys()
                if "embedding" in tensor_keys:
                    return f.get_tensor("embedding")
                elif tensor_keys:  # Fallback if only one tensor exists
                    logger.debug(
                        f"No 'embedding' key in {file_path}, using first key: {tensor_keys[0]}"
                    )
                    return f.get_tensor(tensor_keys[0])
                else:
                    logger.warning(f"No tensors found in cache file: {file_path}")
                    return None
        except FileNotFoundError:
            logger.debug(f"Cache file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading embedding from file {file_path}: {e}")
            return None

    def get_embedding(
        self, text: str, model_name: str, df: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """Retrieve a single embedding, checking memory and disk cache."""
        with self.cache_lock:
            # Use the imported utility function for schema hash
            dataframe_schema_hash = compute_dataframe_schema_hash(df)
            cache_key = self._compute_text_hash(text, model_name, dataframe_schema_hash)

            # Check memory cache
            if cache_key in self.memory_cache:
                self._add_to_memory_cache(
                    cache_key, self.memory_cache[cache_key]
                )  # Update LRU
                return self.memory_cache[cache_key]

            # Check disk cache
            cache_file_path = self._get_cache_file_path(cache_key, model_name)
            embedding = self._load_embedding_from_file(cache_file_path)

            if embedding is not None:
                self._add_to_memory_cache(cache_key, embedding)  # Add to memory cache
                return embedding

            return None

    def store_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: str,
        df: Optional[Any] = None,
    ) -> bool:
        """Store a single embedding to memory and disk cache."""
        temp_file_path: Optional[Path] = None  # Initialize temp_file_path
        with self.cache_lock:
            # Use the imported utility function for schema hash
            dataframe_schema_hash = compute_dataframe_schema_hash(df)
            cache_key = self._compute_text_hash(text, model_name, dataframe_schema_hash)
            cache_file_path = self._get_cache_file_path(cache_key, model_name)

            try:
                tensor_dict = {"embedding": embedding}
                # Atomic write: write to a temporary file then rename
                temp_file_path = cache_file_path.with_suffix(".safetensors.tmp")
                save_file(tensor_dict, temp_file_path)  # type: ignore
                shutil.move(str(temp_file_path), str(cache_file_path))

                self._add_to_memory_cache(cache_key, embedding)
                logger.debug(
                    f"Stored embedding for hash {cache_key} at {cache_file_path}"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Failed to store embedding for hash {cache_key} at {cache_file_path}: {e}"
                )
                # Attempt to clean up temp file if it exists
                if (
                    temp_file_path and temp_file_path.exists()
                ):  # Check if temp_file_path was assigned
                    try:
                        temp_file_path.unlink()
                    except Exception as unlink_e:
                        logger.error(
                            f"Failed to remove temporary cache file {temp_file_path}: {unlink_e}"
                        )
                return False

    def get_embeddings_batch(
        self, texts: List[str], model_name: str, df: Optional[Any] = None
    ) -> Tuple[List[Optional[np.ndarray]], List[str]]:
        """Retrieve embeddings for a batch of texts."""
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_texts_values: List[
            str
        ] = []  # Store actual text values that are missing

        for i, text in enumerate(texts):
            # Pass the pre-computed schema_hash to avoid re-calculating it for each text in the batch
            # For get_embedding, it will re-calculate if df is passed, so we ensure it's consistent
            # by passing the same df or relying on its internal call to compute_dataframe_schema_hash.
            # To be more direct, we can modify get_embedding to accept an optional precomputed schema hash,
            # or compute the cache_key directly here.
            # For now, let's keep it simple and rely on get_embedding's df handling.
            # The schema hash is primarily used in _compute_text_hash.
            # Let's adjust get_embedding and store_embedding to optionally take schema_hash.

            embedding = self.get_embedding(
                text, model_name, df
            )  # df is passed, schema hash computed inside
            if embedding is not None:
                results[i] = embedding
            else:
                missing_texts_values.append(text)

        # The 'missing_texts_values' list now correctly contains the texts that were not found.
        # The 'results' list maintains the original order and length, with None for missing embeddings.
        return results, missing_texts_values

    def store_embeddings_batch(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        model_name: str,
        df: Optional[Any] = None,
    ) -> bool:
        """Store a batch of embeddings."""
        if not texts or not embeddings or len(texts) != len(embeddings):
            logger.error(
                "Invalid input for store_embeddings_batch: texts and embeddings must be non-empty and of equal length."
            )
            return False

        # Compute schema hash once for the batch if df is provided
        # dataframe_schema_hash = compute_dataframe_schema_hash(df) # Not strictly needed here if store_embedding handles it

        all_stored_successfully = True
        for i, text in enumerate(texts):
            # store_embedding will compute the schema hash internally using the passed df
            if not self.store_embedding(text, embeddings[i], model_name, df):
                all_stored_successfully = False
                logger.warning(
                    f"Failed to store embedding for text: '{text[:50]}...'"
                )  # Log part of text

        return all_stored_successfully

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        with self.cache_lock:
            num_memory_items = len(self.memory_cache)

            num_disk_files = 0
            total_disk_size = 0
            if self.shards_dir.exists():
                for item in self.shards_dir.glob("**/*.safetensors"):
                    if item.is_file():
                        num_disk_files += 1
                        try:
                            total_disk_size += item.stat().st_size
                        except OSError as e:
                            logger.warning(f"Could not stat file {item}: {e}")

            return {
                "cache_version": self.CACHE_VERSION,
                "cache_directory": str(self.cache_dir),
                "memory_cache_configured_size": self.memory_cache_size,
                "memory_cache_items": num_memory_items,
                "disk_cache_files": num_disk_files,
                "disk_cache_total_size_bytes": total_disk_size,
            }

    def clear_memory_cache(self):
        """Clear the in-memory cache."""
        with self.cache_lock:
            self.memory_cache.clear()
            self.memory_cache_order.clear()
            logger.info("In-memory cache cleared.")

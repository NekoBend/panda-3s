import logging
from typing import Any

import faiss
import numpy as np
import pandas as pd

from .cache import IndexCache, _compute_hash, _compute_row_hash


def _compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute hash for DataFrame content."""
    df_str = df.to_string()
    return _compute_hash(df_str)


def _compute_dataframe_row_hashes(df: pd.DataFrame, text_column: str) -> dict[int, str]:
    """Compute hash for each row in DataFrame."""
    row_hashes = {}
    for i in range(len(df)):
        row = df.iloc[i]
        # Create a consistent string representation of the row
        row_str = f"{text_column}:{row[text_column]}"
        # Add other columns for more complete hash
        for col in df.columns:
            if col != text_column:
                row_str += f"|{col}:{row[col]}"
        row_hashes[i] = _compute_row_hash(row_str)
    return row_hashes


logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS-based vector index with caching support."""

    def __init__(
        self,
        dimension: int | None = None,
        enable_cache: bool = True,
        index_cache: str | None = None,
    ):
        """Initialize FaissIndex.

        Args:
            dimension: Vector dimension for the index
            enable_cache: Whether to enable index caching
            index_cache: Directory for index cache files
        """
        self.dimension = dimension
        self.index: faiss.Index | None = None
        self.documents: list[str] = []
        self.document_data: list[dict[str, Any]] = []
        self.enable_cache = enable_cache
        self.cache: IndexCache | None = None

        if self.enable_cache:
            self.cache = IndexCache(index_cache)
            logger.info("Index cache enabled")
        else:
            logger.info("Index cache disabled")

    def _ensure_index_initialized(self, dimension: int) -> None:
        """Initialize the FAISS index if not already done."""
        if self.index is None:
            self.dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Faiss index initialized: dimension={dimension}")

    def add_documents(
        self, documents: list[str], embeddings: np.ndarray[Any, np.dtype[np.float32]]
    ) -> None:
        """Add documents to the index."""
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings do not match")

        self._ensure_index_initialized(embeddings.shape[1])
        logger.debug(f"Adding documents to index: {len(documents)} items")

        if self.index is not None:
            self.index.add(embeddings.astype(np.float32))  # type: ignore
        self.documents.extend(documents)

    def add_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        embeddings: np.ndarray[Any, np.dtype[np.float32]],
        model_name: str = "default",
    ) -> None:
        """Add DataFrame to the index with caching support."""
        if len(df) != embeddings.shape[0]:
            raise ValueError("Number of DataFrame rows and embeddings do not match")

        self._ensure_index_initialized(embeddings.shape[1])
        logger.debug(f"Adding DataFrame to index: {len(df)} rows")

        # Generate hash for DataFrame
        df_hash = _compute_dataframe_hash(df)
        row_indices = list(df.index)

        # Try to load from cache first
        if self.enable_cache and self.cache is not None:
            cached_result = self.cache.get_index(df_hash, text_column, model_name)

            if cached_result is not None:
                cached_index, cached_row_indices = cached_result
                self.index = cached_index
                logger.info("Loaded index from cache")

                # Restore document data from cached row indices
                for row_idx in cached_row_indices:
                    if row_idx in df.index:
                        row = df.loc[row_idx]
                        self.documents.append(str(row[text_column]))
                        row_dict = {str(k): v for k, v in row.to_dict().items()}
                        self.document_data.append(row_dict)
                return

        # Build new index if not cached
        if self.index is not None:
            self.index.add(embeddings.astype(np.float32))  # type: ignore

            # Save to cache
            if self.enable_cache and self.cache is not None:
                self.cache.save_index(
                    df_hash, text_column, model_name, self.index, row_indices
                )
                logger.debug("Index saved to cache")

        # Add document data
        for _, row in df.iterrows():
            self.documents.append(str(row[text_column]))
            row_dict = {str(k): v for k, v in row.to_dict().items()}
            self.document_data.append(row_dict)

    def add_dataframe_with_row_caching(
        self,
        df: pd.DataFrame,
        text_column: str,
        embeddings: np.ndarray[Any, np.dtype[np.float32]],
        model_name: str = "default",
    ) -> None:
        """Add DataFrame to the index with row-level caching support."""
        if len(df) != embeddings.shape[0]:
            raise ValueError("Number of DataFrame rows and embeddings do not match")

        self._ensure_index_initialized(embeddings.shape[1])
        logger.debug(f"Adding DataFrame with row caching: {len(df)} rows")

        # Compute row hashes for efficient caching
        row_hashes = _compute_dataframe_row_hashes(
            df, text_column
        )  # Check for cached embeddings at row level
        cache_hits = 0
        cached_embeddings = []
        new_embeddings = []
        new_rows = []

        if self.enable_cache and self.cache is not None:
            for i in range(len(df)):
                row_hash = row_hashes[i]

                # Try to get cached embedding for this row
                cached_embedding = self.cache.get_row_embedding(row_hash, model_name)

                if cached_embedding is not None:
                    cached_embeddings.append(cached_embedding)
                    cache_hits += 1
                else:
                    # Add to new embeddings to cache
                    new_embeddings.append(embeddings[i])
                    new_rows.append((i, df.iloc[i], row_hash))
                    cached_embeddings.append(embeddings[i])

        else:  # No caching enabled, use all embeddings
            cached_embeddings = list(embeddings)

        # Add all embeddings to the index
        embeddings_array = np.array(cached_embeddings, dtype=np.float32)
        if self.index is not None:
            self.index.add(embeddings_array)  # type: ignore

        # Cache new embeddings
        if self.enable_cache and self.cache is not None and new_rows:
            for (idx, row, row_hash), embedding in zip(new_rows, new_embeddings):
                self.cache.save_row_embedding(row_hash, model_name, embedding)

        logger.info(f"Row-level caching: {cache_hits} cache hits out of {len(df)} rows")

        # Add document data
        for _, row in df.iterrows():
            self.documents.append(str(row[text_column]))
            row_dict = {str(k): v for k, v in row.to_dict().items()}
            self.document_data.append(row_dict)

    def search(
        self,
        query_embedding: np.ndarray[Any, Any],
        k: int = 5,
        return_scores: bool = False,
    ) -> list[str] | tuple[list[str], list[float]]:
        """Search for similar documents."""
        if self.index is None or len(self.documents) == 0:
            raise ValueError("Index is empty. Please add documents first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(k, len(self.documents))
        logger.debug(f"Executing search: k={k}")
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)  # type: ignore

        results = []
        valid_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append(self.documents[idx])
                valid_scores.append(float(score))

        if return_scores:
            return results, valid_scores
        return results

    def search_dataframe(
        self,
        query_embedding: np.ndarray[Any, Any],
        k: int = 5,
        return_dataframe: bool = True,
    ) -> pd.DataFrame | list[dict[str, Any]]:
        """Search DataFrame with similarity scores."""
        if self.index is None or len(self.documents) == 0:
            raise ValueError("Index is empty. Please add documents first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(k, len(self.document_data))
        logger.debug(f"Executing DataFrame search: k={k}")
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)  # type: ignore

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.document_data):
                result_data = self.document_data[idx].copy()
                result_data["index"] = int(idx)
                result_data["score"] = float(score)
                results.append(result_data)

        if return_dataframe:
            return pd.DataFrame(results)
        return results

    def save_index(self, filepath: str) -> None:
        """Save the index to file."""
        if self.index is None:
            raise ValueError("No index to save.")
        faiss.write_index(self.index, filepath)
        logger.info(f"Index saved: {filepath}")

    def load_index(self, filepath: str, documents: list[str]) -> None:
        """Load index from file."""
        self.index = faiss.read_index(filepath)
        self.documents = documents.copy()
        if hasattr(self.index, "d"):
            self.dimension = self.index.d  # type: ignore
        logger.info(f"Index loaded: {filepath}")

    def clear(self) -> None:
        """Clear the index."""
        self.index = None
        self.documents.clear()
        self.document_data.clear()
        self.dimension = None
        logger.debug("Index cleared")

    def get_document_count(self) -> int:
        """Get number of documents in index."""
        return len(self.documents)

    def get_dimension(self) -> int | None:
        """Get vector dimension."""
        return self.dimension

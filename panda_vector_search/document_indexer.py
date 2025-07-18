import json
import logging
from pathlib import Path
from typing import (
    Any,
)

import faiss  # type: ignore
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FaissVectorIndexer:
    """FAISS-based vector indexer.

    This class manages a FAISS index and associated document metadata.
    Persistence of the index is handled by `save_vector_index` and `load_vector_index` methods.

    Attributes:
        _vector_dimension: The dimensionality of the vectors in the index.
        index: The FAISS index object.
        _document_texts: A list of text documents corresponding to the indexed embeddings.
        _document_metadata: A list of dictionaries containing metadata for each document.
    """

    def __init__(
        self,
        vector_dimension: int | None = None,
    ):
        """Initialize FaissVectorIndexer.

        Args:
            vector_dimension: Vector dimension for the index. Can be set later if the index is built
                       or loaded.
        """
        self._vector_dimension: int | None = vector_dimension
        self.index: faiss.Index | None = None
        self._document_texts: list[str] = []
        self._document_metadata: list[dict[str, Any]] = []
        logger.info(
            f"FaissVectorIndexer initialized. Dimension: {vector_dimension if vector_dimension else 'Not set'}."
        )

    def _ensure_index_initialized(self, dimension: int) -> None:
        """Initialize the FAISS index if not already done, or validate existing dimension.

        Args:
            dimension: The dimension of the embeddings to be added.

        Raises:
            ValueError: If there's a dimension mismatch with an existing index.
        """
        if self.index is None:
            if self._vector_dimension is None:
                self._vector_dimension = dimension
            elif self._vector_dimension != dimension:
                logger.error(
                    f"Dimension mismatch. Index was configured with {self._vector_dimension}, "
                    f"but new data has dimension {dimension}."
                )
                raise ValueError(
                    f"Dimension mismatch: Index is {self._vector_dimension}D, new data is {dimension}D."
                )

            if (
                self._vector_dimension is None
            ):  # Should not happen if logic is correct, but as a safeguard
                logger.error("Cannot initialize FAISS index without a dimension.")
                raise ValueError("Dimension must be set to initialize FAISS index.")

            logger.info(
                f"Initializing new Faiss index (IndexFlatIP) with dimension={self._vector_dimension}"
            )
            self.index = faiss.IndexFlatIP(self._vector_dimension)
            logger.debug("Faiss index (IndexFlatIP) created.")
        elif self._vector_dimension != dimension:  # Index exists, validate dimension
            logger.error(
                f"Dimension mismatch. Existing index has dimension {self._vector_dimension}, "
                f"but new data has dimension {dimension}."
            )
            raise ValueError(
                f"Dimension mismatch: Index is {self._vector_dimension}D, new data is {dimension}D."
            )

    def add_documents(
        self,
        documents: list[str],
        embeddings: np.ndarray,  # Using np.ndarray for broader compatibility, ensure float32 before add
    ) -> None:
        """Add documents and their embeddings to the index.

        Args:
            documents: A list of text documents.
            embeddings: A 2D NumPy array of embeddings corresponding to the documents.
                        Shape: (n_documents, embedding_dimension). Must be float32 for FAISS.

        Raises:
            ValueError: If the number of documents and embeddings do not match,
                        or if embeddings are not a 2D array.
            RuntimeError: If the FAISS index is not initialized.
        """
        if len(documents) != embeddings.shape[0]:
            msg = "Number of documents and embeddings do not match."
            logger.error(f"Failed to add documents: {msg}")
            raise ValueError(msg)
        if embeddings.ndim != 2:
            msg = f"Embeddings must be a 2D array, got {embeddings.ndim}D shape {embeddings.shape}."
            logger.error(f"Failed to add documents: {msg}")
            raise ValueError(msg)

        self._ensure_index_initialized(embeddings.shape[1])

        if self.index is not None:
            logger.info(
                f"Adding {len(documents)} documents with embeddings of shape {embeddings.shape} to index."
            )
            logger.debug(
                f"Embeddings shape before add: {embeddings.shape}, dtype: {embeddings.dtype}"
            )  # 追加
            self.index.add(embeddings.astype(np.float32))  # Pass only embeddings
            self._document_texts.extend(documents)
            logger.debug(
                f"{len(documents)} documents and their embeddings added. Total documents: {len(self._document_texts)}."
            )
        else:
            msg = "FAISS index is not initialized even after attempting to ensure it."
            logger.error(f"Failed to add documents: {msg}")
            raise RuntimeError(msg)

    def add_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        embeddings: np.ndarray,  # Using np.ndarray, ensure float32 before add
    ) -> None:
        """Add data from a DataFrame to the index.

        Embeddings are assumed to be pre-computed and provided.

        Args:
            df: The pandas DataFrame containing the data.
            text_column: The name of the column in `df` that contains the text for documents.
            embeddings: A 2D NumPy array of embeddings corresponding to the DataFrame rows.
                        Shape: (n_rows, embedding_dimension). Must be float32 for FAISS.

        Raises:
            ValueError: If dimensions mismatch, text column not found, or embeddings are not 2D.
            RuntimeError: If the FAISS index is not initialized.
        """
        if len(df) != embeddings.shape[0]:
            msg = "Number of DataFrame rows and embeddings do not match."
            logger.error(f"Failed to add DataFrame: {msg}")
            raise ValueError(msg)
        if embeddings.ndim != 2:
            msg = f"Embeddings must be a 2D array, got {embeddings.ndim}D shape {embeddings.shape}."
            logger.error(f"Failed to add DataFrame: {msg}")
            raise ValueError(msg)
        if text_column not in df.columns:
            msg = f"Text column '{text_column}' not found in DataFrame columns: {df.columns.tolist()}."
            logger.error(f"Failed to add DataFrame: {msg}")
            raise ValueError(msg)

        self._ensure_index_initialized(embeddings.shape[1])

        if self.index is not None:
            logger.info(
                f"Adding {len(df)} rows from DataFrame to index. Text column: '{text_column}'. Embeddings shape: {embeddings.shape}."
            )
            logger.debug(
                f"Embeddings shape before add: {embeddings.shape}, dtype: {embeddings.dtype}"
            )  # 追加
            self.index.add(embeddings.astype(np.float32))  # Pass only embeddings

            for _, row in df.iterrows():
                self._document_texts.append(str(row[text_column]))
                row_dict = {str(k): v for k, v in row.to_dict().items()}
                self._document_metadata.append(row_dict)
            logger.debug(
                f"{len(df)} rows processed. Total documents: {len(self._document_texts)}, Total document_data entries: {len(self._document_metadata)}."
            )
        else:
            msg = "FAISS index is not initialized."
            logger.error(f"Failed to add DataFrame: {msg}")
            raise RuntimeError(msg)

    def _perform_faiss_search(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Internal method to perform search in FAISS index.

        Args:
            query_embedding: A 1D or 2D NumPy array for the query embedding.
                             If 2D, expects shape (1, embedding_dimension). Must be float32 for FAISS.
            k: The number of nearest neighbors to retrieve.

        Returns:
            A tuple containing:
                - indices: A 2D NumPy array of indices of the nearest neighbors.
                - scores: A 2D NumPy array of similarity scores (distances for L2, inner products for IP).

        Raises:
            ValueError: If the index is not built/loaded or k is invalid.
            RuntimeError: If FAISS search fails.
        """
        _query_embedding = query_embedding.astype(np.float32)
        if _query_embedding.ndim == 1:
            _query_embedding = np.expand_dims(_query_embedding, axis=0)

        if self.index is None:
            logger.error("Search attempted on a non-existent or unloaded index.")
            raise ValueError(
                "Index is not built or loaded. Call build_index or load_index first."
            )

        # Ensure k is not greater than the number of items in the index
        actual_k = min(k, self.index.ntotal)

        if actual_k == 0:
            logger.warning(
                "Search attempted with k=0 or empty index (ntotal=0). Returning empty results."
            )
            # Return empty arrays with correct dimensions if possible, or generic empty ones
            # For indices, shape is (num_queries, 0)
            # For distances, shape is (num_queries, 0)
            num_queries = _query_embedding.shape[0]
            return np.array([]).reshape(num_queries, 0), np.array([]).reshape(
                num_queries, 0
            )

        logger.debug(
            f"Executing FAISS search: actual_k={actual_k}, ntotal={self.index.ntotal}, query_embedding_shape={_query_embedding.shape}"
        )
        try:
            # Basic call: D, I = index.search(queries, k)
            distances, indices = self.index.search(_query_embedding, actual_k)
            logger.debug(
                f"FAISS search completed. Found {indices.shape[1]} neighbors for {indices.shape[0]} queries. Distances shape: {distances.shape}"
            )
            return indices, distances
        except RuntimeError as e:
            logger.error(f"FAISS search failed: {e}")
            # Consider the case where faiss might return -1 in indices for no results
            # or if actual_k was valid but search still had issues.
            # For now, re-raise or handle as appropriate for the application.
            raise RuntimeError(f"FAISS search operation failed: {e}")

    def search(
        self,
        query_embedding: np.ndarray,  # Using np.ndarray, ensure float32 in _search_faiss
        k: int = 5,
        return_scores: bool = False,
    ) -> list[str] | tuple[list[str], list[float]]:
        """Search for similar documents based on a query embedding.

        Args:
            query_embedding: A 1D NumPy array representing the query embedding. Must be float32 for FAISS.
            k: The number of similar documents to retrieve.
            return_scores: If True, also return the similarity scores.

        Returns:
            If `return_scores` is False, a list of matching document texts.
            If `return_scores` is True, a tuple containing a list of document texts
            and a list of their corresponding scores.
            Returns empty list(s) if the index is empty or no results are found.

        Raises:
            ValueError: If query embedding dimensions mismatch or other search issues.
        """
        if not self._document_texts:
            logger.warning(
                "Search called when no documents are stored. Returning empty results."
            )
            return ([], []) if return_scores else []

        indices_arr, scores_arr = self._perform_faiss_search(query_embedding, k)

        results: list[str] = []
        valid_scores: list[float] = []

        if indices_arr.size > 0:
            for idx, score_val in zip(indices_arr[0], scores_arr[0]):
                if 0 <= idx < len(self._document_texts):
                    results.append(self._document_texts[idx])
                    valid_scores.append(float(score_val))
                elif idx != -1:
                    logger.warning(
                        f"Invalid index {idx} from FAISS search results. Max index: {len(self._document_texts) - 1}. Skipping."
                    )

        logger.debug(
            f"FAISS search processed, returning {len(results)} valid document results."
        )

        if return_scores:
            return results, valid_scores
        return results

    def save_vector_index(self, index_filepath: str, metadata_filepath: str) -> None:
        """Save the FAISS index and associated metadata to disk.

        Args:
            index_filepath: Path to save the FAISS index file.
            metadata_filepath: Path to save the metadata JSON file.

        Raises:
            ValueError: If there is no index to save.
            IOError: If there is an error during file writing.
        """
        if self.index is None:
            logger.error("Failed to save index: No index to save.")
            raise ValueError("No index to save.")

        index_path = Path(index_filepath)
        metadata_path = Path(metadata_filepath)

        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path))  # Ensure path is string
            logger.info(f"FAISS index saved to: {index_path}")
        except Exception as e:
            logger.exception(f"Error saving FAISS index to {index_path}: {e}")
            raise IOError(f"Could not write FAISS index to {index_path}") from e

        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "documents": self._document_texts,
                "document_data": self._document_metadata,
                "dimension": self._vector_dimension,
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Index metadata saved to: {metadata_path}")
        except Exception as e:
            logger.exception(f"Error saving index metadata to {metadata_path}: {e}")
            raise IOError(f"Could not write metadata to {metadata_path}") from e

    def load_vector_index(self, index_filepath: str, metadata_filepath: str) -> None:
        """Load the FAISS index and associated metadata from disk.

        Args:
            index_filepath: Path to the FAISS index file.
            metadata_filepath: Path to the metadata JSON file.

        Raises:
            FileNotFoundError: If index or metadata file is not found.
            IOError: If there is an error during file reading or parsing.
            ValueError: If loaded data is inconsistent.
        """
        index_path = Path(index_filepath)
        metadata_path = Path(metadata_filepath)

        if not index_path.exists():
            logger.error(f"Failed to load index: Index file not found at {index_path}")
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not metadata_path.exists():
            logger.error(
                f"Failed to load index: Metadata file not found at {metadata_path}"
            )
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )  # Corrected: FileNotFoundError

        try:
            self.index = faiss.read_index(str(index_path))  # Ensure path is string
            logger.info(f"FAISS index loaded from: {index_path}")
        except Exception as e:
            logger.exception(f"Error loading FAISS index from {index_path}: {e}")
            raise IOError(f"Could not read FAISS index from {index_path}") from e

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info(f"Index metadata loaded from: {metadata_path}")
        except json.JSONDecodeError as e:
            logger.exception(f"Error decoding JSON metadata from {metadata_path}: {e}")
            raise IOError(f"Could not parse metadata from {metadata_path}") from e
        except Exception as e:
            logger.exception(f"Error loading index metadata from {metadata_path}: {e}")
            raise IOError(f"Could not read metadata from {metadata_path}") from e

        self._document_texts = metadata.get("documents", [])
        self._document_metadata = metadata.get("document_data", [])
        loaded_dimension = metadata.get("dimension")

        if not isinstance(self._document_texts, list) or not all(
            isinstance(doc, str) for doc in self._document_texts
        ):
            logger.error("Loaded 'documents' metadata is not a list of strings.")
            raise ValueError("Invalid format for 'documents' in metadata.")

        if not isinstance(self._document_metadata, list) or not all(
            isinstance(item, dict) for item in self._document_metadata
        ):
            logger.error(
                "Loaded 'document_data' metadata is not a list of dictionaries."
            )
            raise ValueError("Invalid format for 'document_data' in metadata.")

        # Verify consistency of loaded dimension
        if self.index and hasattr(self.index, "d") and self.index.d != 0:
            if loaded_dimension is not None and self.index.d != loaded_dimension:
                logger.warning(
                    f"Dimension mismatch: FAISS index has dimension {self.index.d}, "
                    f"but metadata specifies {loaded_dimension}. Using FAISS index dimension ({self.index.d})."
                )
            self._vector_dimension = self.index.d
        elif loaded_dimension is not None:
            if not isinstance(loaded_dimension, int):
                logger.error(
                    f"Loaded 'dimension' metadata is not an integer: {loaded_dimension}"
                )
                raise ValueError("Invalid format for 'dimension' in metadata.")
            self._vector_dimension = loaded_dimension
            # If index is None or d=0, but we have a dimension from metadata,
            # we might need to re-initialize index if new data is added.
            # _ensure_index_initialized will handle this.
            if self.index is None and self._vector_dimension is not None:
                logger.info(
                    f"Metadata specifies dimension {self._vector_dimension}, but FAISS index is not yet loaded/trained. Will use this dimension."
                )
            elif (
                self.index
                and hasattr(self.index, "d")
                and self.index.d == 0
                and self._vector_dimension is not None
            ):
                logger.info(
                    f"FAISS index is untrained (d=0). Using dimension {self._vector_dimension} from metadata."
                )

        else:  # self.index.d is 0 (untrained) or None, and loaded_dimension is None
            self._vector_dimension = None
            logger.warning(
                "Index dimension could not be determined from loaded FAISS index or metadata. "
                "It must be set explicitly or will be inferred from the first data added."
            )

        if self._vector_dimension is not None:
            logger.info(f"Index dimension after load: {self._vector_dimension}")

        # Validate consistency between documents, document_data and index ntotal
        if self.index and self.index.ntotal > 0:
            if not (
                len(self._document_texts) == self.index.ntotal
                and len(self._document_metadata) == self.index.ntotal
            ):
                logger.warning(
                    f"Inconsistency after load: FAISS index has {self.index.ntotal} items, "
                    f"but loaded documents count is {len(self._document_texts)} and "
                    f"document_data count is {len(self._document_metadata)}. "
                    "This might lead to issues if counts don't align with indexed vectors."
                )
                # Decide on a recovery strategy or raise an error. For now, just warn.
                # If self.documents or self.document_data is the source of truth for size,
                # and index is larger, it's problematic. If index is smaller, also problematic.
                # Simplest assumption: if index.ntotal is the ground truth for vectors,
                # metadata should align. If not, it's a corrupted save.

    def clear_index_data(self) -> None:
        """Clear the index and all associated data (documents, metadata, dimension)."""
        self.index = None
        self._document_texts.clear()
        self._document_metadata.clear()
        self._vector_dimension = None  # Reset dimension
        logger.info(
            "FaissIndex cleared (index, documents, document_data, dimension reset)."
        )

    def get_document_count(self) -> int:
        """Get the number of documents currently stored (text content).

        Returns:
            The number of documents in `self._document_texts`.
        """
        return len(self._document_texts)

    def get_indexed_vector_count(self) -> int:
        """Get the number of items currently in the FAISS index.

        Returns:
            The number of vectors in the FAISS index, or 0 if the index is not initialized.
        """
        if self.index:
            return self.index.ntotal
        return 0

    def get_vector_dimension(self) -> int | None:
        """Get the vector dimension of the index.

        Returns:
            The dimension of the vectors in the index, or None if not set.
        """
        return self._vector_dimension

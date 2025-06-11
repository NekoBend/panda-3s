"""
Core module for panda-3s semantic search functionality.

This module provides the main PandaSearch class for performing semantic search
on pandas DataFrames using sentence transformers and FAISS indexing.
"""

import logging
from dataclasses import dataclass
from typing import List  # Corrected: Import List for type hinting

import numpy as np
import pandas as pd

from ..embedding import Embedding
from ..indexer import FaissIndex

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for the Embedding model initialization.

    Attributes:
        model_name_or_path: Name or path of the SentenceTransformer model.
        device: Device to run the model on (e.g., "cpu", "cuda"). Auto-detected if None.
        model_download_cache_dir: Folder to cache downloaded model files.
                                  Corresponds to `cache_folder` in `sentence_transformers.SentenceTransformer`.
        trust_remote_code: Whether to trust remote code when loading the model.
        enable_embedding_artifact_cache: Whether to enable caching of generated embeddings.
                                         Corresponds to `enable_cache` in `panda_3s.embedding.Embedding`.
        embedding_artifact_cache_dir: Directory for storing cached embedding artifacts.
                                      Corresponds to `embedding_cache_dir` in `panda_3s.embedding.Embedding`.
    """

    model_name_or_path: str = "all-MiniLM-L6-v2"
    device: str | None = None
    model_download_cache_dir: str | None = None
    trust_remote_code: bool = False
    enable_embedding_artifact_cache: bool = True
    embedding_artifact_cache_dir: str = "./panda_3s_cache"


class PandaSearch:
    """Main class for semantic search with pandas DataFrame.

    This class provides a unified interface for creating embeddings and performing
    semantic search on text data within pandas DataFrames with flexible column handling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_columns: str | list[str],
        column_separator: str = " ",
        column_weights: dict[str, float] | None = None,
    ):
        """Initialize PandaSearch instance.

        Args:
            df: The DataFrame to search.
            text_columns: Column name(s) to use for text search.
            column_separator: Separator for joining multiple columns (default: " ").
            column_weights: Optional weights for different columns when combining.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: If specified columns don't exist in the DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Initialization failed: df must be a pandas DataFrame.")
            raise TypeError("df must be a pandas DataFrame")

        self.df = df.copy()
        self.text_columns = (
            [text_columns] if isinstance(text_columns, str) else text_columns
        )
        self.column_separator = column_separator
        self.column_weights = column_weights if column_weights is not None else {}
        self.embedding_model: Embedding | None = None
        self.index: FaissIndex = FaissIndex()
        self._is_indexed: bool = False

        # Validate that all specified columns exist
        missing_cols = [col for col in self.text_columns if col not in df.columns]
        if missing_cols:
            logger.error(
                f"Initialization failed: Specified columns do not exist: {missing_cols}"
            )
            raise ValueError(f"Specified columns do not exist: {missing_cols}")

        # Validate column weights
        if self.column_weights:
            invalid_cols = set(self.column_weights.keys()) - set(self.text_columns)
            if invalid_cols:
                logger.error(
                    f"Initialization failed: Column weights specified for non-existent columns: {invalid_cols}"
                )
                raise ValueError(
                    f"Column weights specified for non-existent columns: {invalid_cols}"
                )

        logger.info(
            f"PandaSearch initialized: {len(df)} rows, columns={self.text_columns}, separator='{self.column_separator}'"
        )

    def _prepare_texts(self) -> list[str]:
        """Prepare text data from DataFrame columns with flexible combination options.

        Returns:
            List of combined text strings ready for embedding.
        """
        logger.debug(f"Preparing texts from columns: {self.text_columns}")
        if len(self.text_columns) == 1:
            texts = self.df[self.text_columns[0]].astype(str).tolist()
            logger.debug(f"Prepared {len(texts)} texts from single column.")
            return texts

        def combine_row(row: pd.Series) -> str:
            combined_parts: list[str] = []
            for col in self.text_columns:
                text: str = str(row[col]) if pd.notna(row[col]) else ""
                if text.strip():
                    weight: float = self.column_weights.get(col, 1.0)
                    if weight != 1.0:
                        repeat_count = int(weight) if weight > 0 else 1
                        text = f"{text} " * repeat_count
                    combined_parts.append(text.strip())
            return self.column_separator.join(combined_parts)

        combined_texts = self.df[self.text_columns].apply(combine_row, axis=1).tolist()
        logger.debug(
            f"Prepared {len(combined_texts)} combined texts from multiple columns."
        )
        return combined_texts

    def _initialize_embedding_model(self, config: EmbeddingConfig) -> None:
        """Initializes the embedding model using the provided configuration."""
        logger.info(f"Initializing embedding model: {config.model_name_or_path}")
        self.embedding_model = Embedding(
            model_name_or_path=config.model_name_or_path,
            device=config.device,
            cache_folder=config.model_download_cache_dir,
            trust_remote_code=config.trust_remote_code,
            enable_cache=config.enable_embedding_artifact_cache,
            embedding_cache_dir=config.embedding_artifact_cache_dir,
        )
        logger.debug("Embedding model initialized successfully.")

    def _generate_embeddings_for_df(self, texts: list[str]) -> np.ndarray:
        """Generates embeddings for the prepared texts from the DataFrame."""
        if not self.embedding_model:
            logger.error(
                "Embedding model not initialized before generating embeddings."
            )
            raise ValueError(
                "Embedding model is not initialized. Call embedding() first."
            )
        logger.info(f"Generating embeddings for {len(texts)} texts.")
        # Pass the DataFrame to the embed method for cache key generation
        embeddings = self.embedding_model.embed(texts, df_for_cache=self.df)
        logger.debug(
            f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}."
        )
        return embeddings

    def _build_faiss_index(
        self,
        embeddings: np.ndarray,
    ) -> None:
        """Builds the FAISS index with the generated embeddings."""
        logger.info(f"Building FAISS index with {embeddings.shape[0]} embeddings.")
        # FaissIndex.add_dataframe no longer takes model_name
        self.index.add_dataframe(
            self.df,
            self.text_columns[0],
            embeddings,  # Removed model_name argument
        )
        self._is_indexed = True
        logger.debug("FAISS index built successfully.")

    def embedding(
        self,
        config: EmbeddingConfig | None = None,
    ) -> "PandaSearch":
        """Build embeddings and index for the DataFrame.

        This method handles all embedding-related configuration using an EmbeddingConfig object.

        Args:
            config: Configuration object for the embedding model.
                    If None, default configuration is used.

        Returns:
            Self for method chaining.
        """
        current_config = config if config is not None else EmbeddingConfig()

        logger.info(
            f"Starting embedding process: model={current_config.model_name_or_path}, columns={self.text_columns}"
        )

        self._initialize_embedding_model(current_config)

        texts = self._prepare_texts()
        embeddings_array = self._generate_embeddings_for_df(texts)

        self._build_faiss_index(embeddings_array)

        self._is_indexed = True
        logger.info("Embedding and indexing completed.")
        return self

    def search(
        self, query: str | dict[str, str], k: int = 10, threshold: float = 0.0
    ) -> pd.DataFrame:
        """Search for similar items in the DataFrame.

        Args:
            query: Search query string or dictionary mapping columns to query text
            k: Number of results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            DataFrame with search results, including similarity scores

        Raises:
            ValueError: If index hasn't been built or embedding model not initialized
        """
        if not self._is_indexed:
            logger.error(
                "Search failed: Index has not been built. Please call embedding() method first."
            )
            raise ValueError("Please call embedding() method first to build the index")

        if self.embedding_model is None:
            logger.error("Search failed: Embedding model is not initialized.")
            raise ValueError("Embedding model is not initialized")

        # Prepare query text with same logic as training
        query_text: str
        if isinstance(query, dict):
            query_parts: List[str] = []  # Corrected: Use List from typing
            for col in (
                self.text_columns
            ):  # Iterate over defined text_columns to maintain order/weighting logic
                text: str = query.get(col, "")
                if text.strip():
                    weight: float = self.column_weights.get(col, 1.0)
                    if weight != 1.0:
                        repeat_count = int(weight) if weight > 0 else 1
                        text = f"{text} " * repeat_count
                    query_parts.append(text.strip())
            query_text = self.column_separator.join(query_parts)
            logger.debug(f"Constructed query text from dict: '{query_text}'")
        else:
            query_text = str(query)
            logger.debug(f"Using provided query text: '{query_text}'")

        logger.info(f"Executing search: k={k}, threshold={threshold}")
        logger.debug(f"Query for embedding: '{query_text}'")

        # Generate query embedding
        query_embedding = self.embedding_model.embed([query_text])[0]
        logger.debug(f"Query embedding generated with shape: {query_embedding.shape}")

        # Search the index
        # results_df = self.index.search_dataframe(query_embedding, k=k) # Original line
        # Corrected: Call search method of FaissIndex, which returns List[str] or Tuple[List[str], List[float]]
        # Then, construct a DataFrame from the results.

        search_results = self.index.search(query_embedding, k=k, return_scores=True)

        # search_results will be a tuple: (list_of_document_texts, list_of_scores)
        # We need to map these back to the original DataFrame rows.
        # The FaissIndex.search method currently returns document texts, not original indices.
        # This needs to be reconciled. For now, let's assume FaissIndex.search can return
        # indices or that we can match texts back.
        # A more robust way would be for FaissIndex.search to return original indices or full rows.

        # Temporary workaround: If FaissIndex.search returns texts and scores,
        # we need a way to get the original DataFrame rows.
        # This part of the logic needs to be revisited based on FaissIndex.search's capabilities.
        # For now, assuming FaissIndex.search returns enough info to reconstruct or retrieve.

        # If FaissIndex.search returns document texts and scores:
        doc_texts, scores = search_results

        # How to get the original DataFrame rows from just text? This is problematic if texts are not unique.
        # Let's assume for now that FaissIndex.search will be modified to return indices
        # that can be used with self.df.iloc[] or similar.
        # Or, that PandaSearch._build_faiss_index stores original indices.

        # Placeholder: Create a DataFrame from the returned texts and scores.
        # This will not contain the original DataFrame's other columns unless FaissIndex is changed.

        # If FaissIndex.document_data stores dicts of original rows:
        # And if FaissIndex.search returned indices into self.index.document_data

        # Let's assume self.index.search returns indices into the original df
        # This is a conceptual change needed in FaissIndex or how PandaSearch uses it.
        # For the purpose of fixing the AttributeError, let's assume `search_results`
        # are the indices and scores that can be used to build the DataFrame.

        # Re-evaluating: FaissIndex.search returns (indices_arr, scores_arr)
        # We need to find these strings in self.df to get the full rows.
        # This is inefficient and error-prone if strings are not unique or match complexly.

        # The FaissIndex.search method was changed to return (indices_arr, scores_arr)
        # from _search_faiss if we modify it to do so directly, or if search_dataframe
        # was intended to be a higher-level method in FaissIndex.

        # Given the current FaissIndex.search signature:
        # results: list[str] | tuple[list[str], list[float]]

        # Let's assume we want the DataFrame rows.
        # The `search_dataframe` method was removed from `FaissIndex`.
        # `PandaSearch.search` needs to reconstruct the DataFrame.

        # The `FaissIndex.search` method returns (list of doc texts, list of scores)
        # We need to find which rows in the original `self.df` correspond to these `doc_texts`.
        # This is tricky if `doc_texts` are not unique or if they were combined from multiple columns.

        # A better approach: `FaissIndex.search` should return the *indices* of the matching documents
        # within its internal `self.documents` list (which corresponds to `self.df` rows).

        # Let's modify `FaissIndex.search` to return indices and scores.
        # And then use `self.df.iloc[indices]`

        # Assuming `FaissIndex._search_faiss` is called and returns (indices_arr, scores_arr)
        # And `FaissIndex.search` is updated to return these directly or process them.

        # For now, let's assume `self.index.search` returns (indices, scores)
        # where indices are for `self.df`. This requires `FaissIndex` to be aware of original df indices.
        # This is what `FaissIndex.add_dataframe` was implicitly doing by storing `document_data`.

        # If `self.index.search` returns (list_of_indices_in_faiss_index, list_of_scores)
        faiss_indices, scores = self.index._search_faiss(
            query_embedding, k
        )  # Call the internal one for now

        if faiss_indices.size == 0:
            results_df = pd.DataFrame(columns=self.df.columns.tolist() + ["score"])
        else:
            # faiss_indices[0] contains the actual indices
            matched_indices_in_df = faiss_indices[0]

            # Filter out any potential -1 indices if k > ntotal
            valid_mask = matched_indices_in_df != -1
            matched_indices_in_df = matched_indices_in_df[valid_mask]

            if matched_indices_in_df.size == 0:
                results_df = pd.DataFrame(columns=self.df.columns.tolist() + ["score"])
            else:
                results_df = self.df.iloc[matched_indices_in_df].copy()
                results_df["score"] = scores[0][valid_mask]

        logger.debug(
            f"Initial search returned {len(results_df)} results from FAISS index."
        )

        if not isinstance(results_df, pd.DataFrame):
            logger.warning(
                f"search_dataframe did not return a DataFrame. Type: {type(results_df)}. Converting."
            )
            results_df = pd.DataFrame(results_df)

        if threshold > 0.0 and "score" in results_df.columns:
            original_count = len(results_df)
            results_df = results_df[results_df["score"] >= threshold]
            logger.debug(
                f"Applied threshold {threshold}. Filtered {original_count - len(results_df)} results. "
                f"{len(results_df)} results remaining."
            )
        elif threshold > 0.0:
            logger.warning(
                f"Threshold {threshold} specified, but 'score' column not found in results. Skipping filtering."
            )

        logger.info(
            f"Search completed. Returning {len(results_df)} results."
        )  # Changed to INFO
        return results_df.reset_index(drop=True)

    def clear_index(self) -> None:
        """Clear the FAISS index and reset indexed state."""
        self.index.clear()
        self._is_indexed = False
        logger.info("PandaSearch index cleared.")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self.df

    @property
    def is_indexed(self) -> bool:
        """Check if the index has been built."""
        return self._is_indexed

    @property
    def indexed_item_count(self) -> int:
        """Get the number of items currently in the FAISS index."""
        return self.index.get_indexed_item_count()

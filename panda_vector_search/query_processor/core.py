"""
Core module for panda-vector-search semantic search functionality.

This module provides the main PandaSearch class for performing semantic search
on pandas DataFrames using sentence transformers and FAISS indexing.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..common_utilities.validation import (
    ConfigurationError,
    validate_dataframe_and_columns,
    validate_embedding_config,
    validate_search_parameters,
)
from ..document_indexer import FaissVectorIndexer
from ..embedding_service import SentenceEmbeddingService

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
                                         Corresponds to `enable_embedding_cache` in `panda_vector_search.embedding_service.SentenceEmbeddingService`.
        embedding_artifact_cache_dir: Directory for storing cached embedding artifacts.
                                      Corresponds to `embedding_cache_directory` in `panda_vector_search.embedding_service.SentenceEmbeddingService`.
    """

    model_name_or_path: str = "all-MiniLM-L6-v2"
    device: str | None = None
    model_download_cache_dir: str | None = None
    trust_remote_code: bool = False
    enable_embedding_artifact_cache: bool = True
    embedding_artifact_cache_dir: str = "./panda_vector_search_cache"


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
            ConfigurationError: If configuration validation fails.
        """
        # Validate DataFrame and columns
        validation_errors = validate_dataframe_and_columns(df, text_columns)
        if validation_errors:
            error_msg = "Configuration validation failed: " + "; ".join(
                validation_errors
            )
            logger.error(f"Initialization failed: {error_msg}")
            raise ConfigurationError(error_msg)

        if not isinstance(df, pd.DataFrame):
            logger.error("Initialization failed: df must be a pandas DataFrame.")
            raise TypeError("df must be a pandas DataFrame")

        self.df = df.copy()
        self.text_columns = (
            [text_columns] if isinstance(text_columns, str) else text_columns
        )
        self.column_separator = column_separator
        self.column_weights = column_weights if column_weights is not None else {}
        self.embedding_model: SentenceEmbeddingService | None = None
        self.index: FaissVectorIndexer = FaissVectorIndexer()
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
        """Prepare text data from DataFrame columns.

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
                    combined_parts.append(text.strip())
            return self.column_separator.join(combined_parts)

        combined_texts = self.df[self.text_columns].apply(combine_row, axis=1).tolist()
        logger.debug(
            f"Prepared {len(combined_texts)} combined texts from multiple columns."
        )
        return combined_texts

    def _prepare_column_specific_texts(self) -> dict[str, list[str]]:
        """Prepare text data for each column separately for weighted embedding.

        Returns:
            Dictionary mapping column names to their text lists.
        """
        if not self.column_weights or len(self.text_columns) == 1:
            return {}

        column_texts = {}
        for col in self.text_columns:
            texts = self.df[col].astype(str).fillna("").tolist()
            column_texts[col] = texts

        logger.debug(f"Prepared column-specific texts for {len(column_texts)} columns.")
        return column_texts

    def _generate_weighted_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings with proper column weighting.

        Args:
            texts: Combined texts (fallback when no weights specified).

        Returns:
            Weighted embeddings array.
        """
        if not self.embedding_model:
            raise ValueError(
                "Embedding model is not initialized. Call embedding() first."
            )

        # If no weights specified or single column, use simple embedding
        if not self.column_weights or len(self.text_columns) == 1:
            return self.embedding_model.embed(texts, cache_context_df=self.df)

        logger.info(
            f"Generating weighted embeddings for {len(self.text_columns)} columns."
        )

        # Generate embeddings for each column separately
        column_embeddings = {}
        column_texts = self._prepare_column_specific_texts()

        for col in self.text_columns:
            if col in column_texts:
                col_texts = column_texts[col]
                embeddings = self.embedding_model.embed(
                    col_texts, cache_context_df=self.df
                )
                column_embeddings[col] = embeddings
                logger.debug(
                    f"Generated embeddings for column '{col}': {embeddings.shape}"
                )

        # Compute weighted average of embeddings
        if not column_embeddings:
            logger.warning(
                "No column embeddings generated, falling back to combined text embedding."
            )
            return self.embedding_model.embed(texts, cache_context_df=self.df)

        weighted_sum = None
        total_weight = 0.0

        for col, embeddings in column_embeddings.items():
            weight = self.column_weights.get(col, 1.0)
            if weight <= 0:
                continue

            if weighted_sum is None:
                weighted_sum = embeddings * weight
            else:
                weighted_sum += embeddings * weight
            total_weight += weight

        if weighted_sum is not None and total_weight > 0:
            final_embeddings = weighted_sum / total_weight
            logger.debug(
                f"Computed weighted embeddings: {final_embeddings.shape}, total_weight={total_weight}"
            )
            return final_embeddings
        else:
            logger.warning(
                "Failed to compute weighted embeddings, falling back to combined text embedding."
            )
            return self.embedding_model.embed(texts, cache_context_df=self.df)

    def _generate_query_embedding_from_dict(
        self, query_dict: dict[str, str]
    ) -> np.ndarray:
        """Generate weighted embedding for dictionary-based queries.

        Args:
            query_dict: Dictionary mapping column names to query text.

        Returns:
            Single query embedding vector.
        """
        if not self.embedding_model:
            raise ValueError(
                "Embedding model is not initialized. Call embedding() first."
            )

        if not self.column_weights or len(self.text_columns) == 1:
            # Fallback to simple concatenation
            query_parts = [
                query_dict.get(col, "").strip()
                for col in self.text_columns
                if query_dict.get(col, "").strip()
            ]
            query_text = self.column_separator.join(query_parts)
            return self.embedding_model.embed(texts=[query_text])[0]

        # Generate embeddings for each query column with weights
        weighted_sum = None
        total_weight = 0.0

        for col in self.text_columns:
            text = query_dict.get(col, "").strip()
            if not text:
                continue

            weight = self.column_weights.get(col, 1.0)
            if weight <= 0:
                continue

            embedding = self.embedding_model.embed(texts=[text])[0]

            if weighted_sum is None:
                weighted_sum = embedding * weight
            else:
                weighted_sum += embedding * weight
            total_weight += weight

        if weighted_sum is not None and total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback
            query_parts = [
                query_dict.get(col, "").strip()
                for col in self.text_columns
                if query_dict.get(col, "").strip()
            ]
            query_text = self.column_separator.join(query_parts)
            return self.embedding_model.embed(texts=[query_text])[0]

    def _initialize_embedding_model(self, config: EmbeddingConfig) -> None:
        """Initializes the embedding model using the provided configuration."""
        logger.info(f"Initializing embedding model: {config.model_name_or_path}")
        self.embedding_model = SentenceEmbeddingService(
            model_name_or_path=config.model_name_or_path,
            device=config.device,
            model_cache_folder=config.model_download_cache_dir,
            trust_remote_code=config.trust_remote_code,
            enable_embedding_cache=config.enable_embedding_artifact_cache,
            embedding_cache_directory=config.embedding_artifact_cache_dir,
        )
        logger.debug("Embedding model initialized successfully.")

    def _generate_embeddings_for_df(self, texts: list[str]) -> np.ndarray:
        """Generates embeddings for the prepared texts from the DataFrame with proper weighting."""
        if not self.embedding_model:
            logger.error(
                "Embedding model not initialized before generating embeddings."
            )
            raise ValueError(
                "Embedding model is not initialized. Call embedding() first."
            )
        logger.info(f"Generating embeddings for {len(texts)} texts.")

        # Use weighted embedding generation if column weights are specified
        embeddings = self._generate_weighted_embeddings(texts)

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

        Raises:
            ConfigurationError: If configuration validation fails.
        """
        current_config = config if config is not None else EmbeddingConfig()

        # Validate configuration
        validation_errors = validate_embedding_config(current_config)
        if validation_errors:
            error_msg = "Embedding configuration validation failed: " + "; ".join(
                validation_errors
            )
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

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
            ConfigurationError: If search parameters are invalid
        """
        if not self._is_indexed or self.embedding_model is None:
            logger.error(
                "Search failed: Index not built or embedding model not initialized. Call embedding() first."
            )
            raise ValueError(
                "Index must be built and embedding model initialized. Call embedding() first."
            )

        # Validate search parameters
        validation_errors = validate_search_parameters(query, k, threshold)
        if validation_errors:
            error_msg = "Search parameter validation failed: " + "; ".join(
                validation_errors
            )
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

        query_text: str
        if isinstance(query, dict):
            # For dictionary queries, we need to generate weighted embeddings properly
            if self.column_weights and len(self.text_columns) > 1:
                query_embedding = self._generate_query_embedding_from_dict(query)
            else:
                # Simple concatenation for non-weighted queries
                query_parts: list[str] = []
                for col in self.text_columns:
                    text = query.get(col, "")
                    if text.strip():
                        query_parts.append(text.strip())
                query_text = self.column_separator.join(query_parts)
                if not self.embedding_model:
                    raise ValueError("Embedding model is not initialized.")
                query_embedding = self.embedding_model.embed(texts=[query_text])[0]
        else:
            query_text = str(query)
            if not self.embedding_model:
                raise ValueError("Embedding model is not initialized.")
            query_embedding = self.embedding_model.embed(texts=[query_text])[0]

        faiss_indices, scores = self.index._perform_faiss_search(query_embedding, k)

        if faiss_indices.size == 0:
            return pd.DataFrame(columns=self.df.columns.tolist() + ["score"])

        matched_indices_in_df = faiss_indices[0]
        valid_mask = matched_indices_in_df != -1
        matched_indices_in_df = matched_indices_in_df[valid_mask]

        if matched_indices_in_df.size == 0:
            return pd.DataFrame(columns=self.df.columns.tolist() + ["score"])

        results_df = self.df.iloc[matched_indices_in_df].copy()
        results_df["score"] = scores[0][valid_mask]

        if threshold > 0.0:
            results_df = results_df[results_df["score"] >= threshold]

        logger.info(f"Search completed. Returning {len(results_df)} results.")
        return results_df.reset_index(drop=True)

    def clear_index(self) -> None:
        """Clear the FAISS index and reset indexed state."""
        self.index.clear_index_data()
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
        return self.index.get_indexed_vector_count()

    def get_cache_statistics(self) -> dict[str, Any] | None:
        """Get cache statistics if caching is enabled.

        Returns:
            Dictionary containing cache statistics, or None if caching is disabled.
        """
        if self.embedding_model and self.embedding_model.cache:
            return self.embedding_model.cache.get_cache_statistics()
        return None

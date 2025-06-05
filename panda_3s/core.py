"""
Core module for panda-3s semantic search functionality.

This module provides the main PandaSearch class for performing semantic search
on pandas DataFrames using sentence transformers and FAISS indexing.
"""

import logging
from typing import Dict, List, Union

import pandas as pd

from .embedding import Embedding
from .indexer import FaissIndex

logger = logging.getLogger(__name__)


class PandaSearch:
    """Main class for semantic search with pandas DataFrame.

    This class provides a unified interface for creating embeddings and performing
    semantic search on text data within pandas DataFrames with flexible column handling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_columns: Union[str, List[str]],
        column_separator: str = " ",
        column_weights: Dict[str, float] | None = None,
    ):
        """Initialize PandaSearch instance.

        Args:
            df: The DataFrame to search
            text_columns: Column name(s) to use for text search
            column_separator: Separator for joining multiple columns (default: " ")
            column_weights: Optional weights for different columns when combining

        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If specified columns don't exist in the DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        self.df = df.copy()
        self.text_columns = (
            [text_columns] if isinstance(text_columns, str) else text_columns
        )
        self.column_separator = column_separator
        self.column_weights = column_weights or {}
        self.embedding_model: Embedding | None = None
        self.index = FaissIndex()
        self._is_indexed = False

        # Validate that all specified columns exist
        missing_cols = [col for col in self.text_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Specified columns do not exist: {missing_cols}")

        # Validate column weights
        if self.column_weights:
            invalid_cols = set(self.column_weights.keys()) - set(self.text_columns)
            if invalid_cols:
                raise ValueError(
                    f"Column weights specified for non-existent columns: {invalid_cols}"
                )

        logger.debug(
            f"PandaSearch initialized: {len(df)} rows, columns={self.text_columns}, separator='{self.column_separator}'"
        )

    def _prepare_texts(self) -> List[str]:
        """Prepare text data from DataFrame columns with flexible combination options.

        Returns:
            List of combined text strings ready for embedding
        """
        if len(self.text_columns) == 1:
            return self.df[self.text_columns[0]].astype(str).tolist()

        # Multiple columns - handle different combination strategies
        def combine_row(row):
            combined_parts = []
            for col in self.text_columns:
                text = str(row[col]) if pd.notna(row[col]) else ""
                if text.strip():  # Only add non-empty text
                    weight = self.column_weights.get(col, 1.0)
                    if weight != 1.0:
                        # Repeat text based on weight (simple weighting approach)
                        text = f"{text} " * int(weight)
                    combined_parts.append(text.strip())

            return self.column_separator.join(combined_parts)

        return self.df[self.text_columns].apply(combine_row, axis=1).tolist()

    def embedding(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        enable_cache: bool = True,
        cache_dir: str = "./panda_3s_cache",
        use_row_caching: bool = False,
    ) -> "PandaSearch":
        """Build embeddings and index for the DataFrame.

        This method handles all embedding-related configuration including
        model selection, device management, and cache settings.

        Args:
            model_name_or_path: Name or path of the SentenceTransformer model
            device: Device to run the model on (auto-detected if None)
            cache_folder: Folder to cache model files
            trust_remote_code: Whether to trust remote code
            enable_cache: Whether to enable embedding caching
            cache_dir: Directory for embedding cache files
            use_row_caching: Whether to use row-level caching (more granular)

        Returns:
            Self for method chaining
        """
        logger.info(
            f"Starting embedding process: model={model_name_or_path}, columns={self.text_columns}"
        )

        # Initialize embedding model
        self.embedding_model = Embedding(
            model_name_or_path=model_name_or_path,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
        )

        # Prepare text data for embedding using the flexible method
        texts = self._prepare_texts()  # Generate embeddings
        embeddings = self.embedding_model.embed(texts)  # Configure index for caching
        if not hasattr(self.index, "cache") or self.index.cache is None:
            self.index = FaissIndex(enable_cache=enable_cache, cache_dir=cache_dir)

        # Choose caching strategy
        if use_row_caching:
            # Use row-level caching for better granularity
            primary_column = (
                self.text_columns[0] if len(self.text_columns) == 1 else "combined_text"
            )
            self.index.add_dataframe_with_row_caching(
                self.df, primary_column, embeddings, model_name_or_path
            )
            logger.info("Used row-level caching for embeddings")
        else:
            # Use traditional DataFrame-level caching
            primary_column = (
                self.text_columns[0] if len(self.text_columns) == 1 else "combined_text"
            )
            self.index.add_dataframe(
                self.df, primary_column, embeddings, model_name_or_path
            )

        logger.info("Index construction completed")
        self._is_indexed = True
        return self

    def search(
        self, query: Union[str, Dict[str, str]], k: int = 10, threshold: float = 0.0
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
            raise ValueError("Please call embedding() method first to build the index")

        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized")

        # Prepare query text with same logic as training
        if isinstance(query, dict):
            query_parts = []
            for col in self.text_columns:
                text = query.get(col, "")
                if text.strip():
                    weight = self.column_weights.get(col, 1.0)
                    if weight != 1.0:
                        text = f"{text} " * int(weight)
                    query_parts.append(text.strip())
            query_text = self.column_separator.join(query_parts)
        else:
            query_text = str(query)

        logger.debug(
            f"Executing search: query='{query_text}', k={k}, threshold={threshold}"
        )

        # Generate query embedding
        query_embedding = self.embedding_model.embed([query_text])[0]

        # Search the index
        results = self.index.search_dataframe(query_embedding, k=k)

        # Ensure results is a DataFrame
        if isinstance(results, list):
            results_df = pd.DataFrame(results)
        else:
            results_df = results

        # Apply threshold filtering if specified
        if threshold > 0 and "score" in results_df.columns:
            results_df = results_df[results_df["score"] >= threshold]
            logger.debug(f"After threshold filtering: {len(results_df)} items")

        return results_df

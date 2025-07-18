import logging
from typing import Any, List, Optional, Tuple  # Added Tuple

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

from .data_cache import EmbeddingCacheManager

logger = logging.getLogger(__name__)


class SentenceEmbeddingService:
    """Service for generating sentence embeddings with caching support."""

    cache: Optional[EmbeddingCacheManager]
    _embedding_dimension: int  # Dimension of the embeddings

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        model_cache_folder: str | None = None,  # For SentenceTransformer model cache
        trust_remote_code: bool = False,
        enable_embedding_cache: bool = True,
        embedding_cache_directory: str | None = None,  # For our embedding cache
    ):
        """Initialize the embedding model.

        Args:
            model_name_or_path: Name or path of the SentenceTransformer model
            device: Device to run the model on (e.g., 'cuda', 'cpu')
            cache_folder: Folder to cache downloaded SentenceTransformer models
            trust_remote_code: Whether to trust remote code for model loading
            enable_cache: Whether to enable embedding caching
            embedding_cache_dir: Directory for storing embedding cache files
        """
        self.model_name_or_path = model_name_or_path
        self.enable_embedding_cache = enable_embedding_cache

        logger.info(f"Initializing embedding model: {model_name_or_path}")

        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            cache_folder=model_cache_folder,  # Pass the model cache folder here
            trust_remote_code=trust_remote_code,
        )

        self._embedding_dimension = self._determine_embedding_dimension(
            model_name_or_path
        )

        if enable_embedding_cache:
            # Ensure embedding_cache_directory is provided if enable_embedding_cache is True
            if embedding_cache_directory is None:
                logger.warning(
                    "Embedding cache is enabled but embedding_cache_directory is not set. "
                    "Cache will be disabled or use a default path if EmbeddingCacheManager handles it."
                )
                # Defaulting behavior for embedding_cache_directory is handled by EmbeddingCacheManager
            self.cache = EmbeddingCacheManager(
                cache_directory=embedding_cache_directory
            )
            logger.info(
                f"Embedding cache enabled at {self.cache._cache_root_directory if self.cache else 'N/A'}"
            )
        else:
            self.cache = None
            logger.info("Embedding cache disabled")

        logger.info("Embedding model initialization completed")

    def _determine_embedding_dimension(self, model_name_or_path: str) -> int:
        """Determines the embedding dimension of the loaded SentenceTransformer model."""
        dimension_value: Optional[int] = None
        try:
            dimension_value = self.model.get_sentence_embedding_dimension()
            if dimension_value is None:  # Some models might return None
                raise AttributeError("get_sentence_embedding_dimension returned None")
            logger.info(f"Embedding dimension retrieved: {dimension_value}")
        except AttributeError:
            logger.warning(
                f"Model {model_name_or_path} does not have get_sentence_embedding_dimension. "
                f"Falling back to generating a dummy embedding."
            )
            try:
                dummy_embedding = self.model.encode(
                    ["test_for_dimension"],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                if dummy_embedding.ndim > 1 and dummy_embedding.shape[0] > 0:
                    dimension_value = dummy_embedding.shape[1]
                    logger.info(
                        f"Embedding dimension determined via fallback: {dimension_value}"
                    )
                else:
                    logger.error(
                        f"Fallback dummy embedding has unexpected shape: {dummy_embedding.shape}"
                    )
            except Exception as e:
                logger.error(
                    f"Error generating dummy embedding for dimension check: {e}"
                )

        if dimension_value is None:
            raise RuntimeError(
                f"Failed to determine embedding dimension for {model_name_or_path}."
            )
        return dimension_value

    def _retrieve_cached_embeddings(
        self, texts: List[str], cache_context_df: Optional[Any]
    ) -> Tuple[List[Optional[np.ndarray]], List[Tuple[int, str]]]:
        """Attempts to retrieve embeddings from cache for the given texts.

        Args:
            texts: List of texts to retrieve embeddings for.
            cache_context_df: Optional DataFrame for cache key generation.

        Returns:
            A tuple containing:
                - A list of embeddings (or None if not found) corresponding to the input texts.
                - A list of tuples (original_index, text_value) for texts not found in cache.
        """
        if not self.enable_embedding_cache or not self.cache:
            # Cache disabled or not initialized, all texts need encoding
            return [None] * len(texts), list(enumerate(texts))

        logger.debug(f"Attempting to retrieve {len(texts)} embeddings from cache...")
        cached_embeddings_list, missing_texts_values = (
            self.cache.retrieve_embeddings_batch(
                texts, self.model_name_or_path, df=cache_context_df
            )
        )

        # The current EmbeddingCacheManager.retrieve_embeddings_batch returns:
        # 1. results: List[Optional[np.ndarray]] (embeddings or None, in order of input texts)
        # 2. missing_texts_values: List[str] (actual text strings that were missing)

        # We need to identify the original indices of the missing texts.
        texts_to_encode: List[Tuple[int, str]] = []
        num_found_in_cache = 0

        for i, text in enumerate(texts):
            if cached_embeddings_list[i] is not None:
                num_found_in_cache += 1
            else:
                # This text was not in the cache (or failed to load), mark for encoding
                texts_to_encode.append((i, text))

        if num_found_in_cache > 0:
            logger.info(
                f"Retrieved {num_found_in_cache}/{len(texts)} embeddings from cache."
            )
        if not texts_to_encode:
            logger.info("All requested embeddings were found in cache.")

        return cached_embeddings_list, texts_to_encode

    def _generate_and_store_embeddings_batch(
        self,
        texts_to_encode_values: List[str],
        batch_size: int,
        cache_context_df: Optional[Any],
    ) -> Optional[np.ndarray]:
        """Encodes a batch of texts and stores them in the cache if enabled.

        Args:
            texts_to_encode_values: List of text strings to encode.
            batch_size: Batch size for model encoding.
            cache_context_df: Optional DataFrame for cache key generation.

        Returns:
            A NumPy array of the newly encoded embeddings, or None if encoding fails.
        """
        if not texts_to_encode_values:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dimension)

        logger.info(
            f"Encoding {len(texts_to_encode_values)} texts not found in cache..."
        )
        try:
            newly_encoded_embeddings_np = self.model.encode(
                texts_to_encode_values,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Assuming normalization is desired
                batch_size=batch_size,
                show_progress_bar=len(texts_to_encode_values)
                > 1000,  # Show progress for large batches
            )
            logger.info(f"Successfully encoded {len(texts_to_encode_values)} texts.")

            if self.enable_embedding_cache and self.cache:
                embeddings_to_store_list = [
                    newly_encoded_embeddings_np[i]
                    for i in range(len(texts_to_encode_values))
                ]
                logger.debug(
                    f"Storing {len(texts_to_encode_values)} newly encoded embeddings to cache..."
                )
                success = self.cache.persist_embeddings_batch(
                    texts_to_encode_values,
                    embeddings_to_store_list,
                    self.model_name_or_path,
                    df=cache_context_df,
                )
                if success:
                    logger.info(
                        f"Successfully stored {len(texts_to_encode_values)} embeddings in cache."
                    )
                else:
                    logger.warning(
                        "Failed to store some or all new embeddings in cache."
                    )
            return newly_encoded_embeddings_np
        except Exception as e:
            logger.error(
                f"Error during model encoding or caching for batch of {len(texts_to_encode_values)} texts: {e}",
                exc_info=True,
            )
            return None

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        cache_context_df: Optional[Any] = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Generate embeddings for given texts with caching support.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for model encoding.
            cache_context_df: Optional DataFrame associated with the texts, used for cache key generation.

        Returns:
            NumPy array of embeddings, in the same order as input texts.
            If some embeddings fail, they might be missing or replaced by zeros (current: missing).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dimension)

        logger.debug(f"Request to embed {len(texts)} texts. Batch size: {batch_size}")

        # Step 1: Try to get embeddings from cache
        # final_embeddings will store results in the original order
        final_embeddings, texts_needing_encoding_with_indices = (
            self._retrieve_cached_embeddings(texts, cache_context_df)
        )

        # texts_needing_encoding_with_indices is List[Tuple[int, str]]
        # where int is the original index and str is the text.

        if not texts_needing_encoding_with_indices:
            logger.info("All embeddings successfully retrieved from cache.")
            # Ensure all are np.ndarray before stacking and they are in correct order
            # _retrieve_cached_embeddings already returns them in order with Nones
            valid_cached_embeddings = [
                emb for emb in final_embeddings if emb is not None
            ]
            if len(valid_cached_embeddings) == len(texts):
                return np.array(valid_cached_embeddings)
            else:
                # This case (all found in cache but counts don't match) should ideally not happen
                # if _retrieve_cached_embeddings is correct.
                logger.error(
                    "Cache retrieval inconsistency. Re-encoding problematic items or failing."
                )
                # Fallback: identify Nones and attempt to re-process or handle error
                # For now, let's assume if texts_needing_encoding_with_indices is empty, all were found.

        # Step 2: Encode texts that were not found in cache
        original_indices_to_encode = [
            item[0] for item in texts_needing_encoding_with_indices
        ]
        texts_to_encode_values = [
            item[1] for item in texts_needing_encoding_with_indices
        ]

        if texts_to_encode_values:
            newly_encoded_batch_np = self._generate_and_store_embeddings_batch(
                texts_to_encode_values, batch_size, cache_context_df
            )

            if newly_encoded_batch_np is not None and len(
                newly_encoded_batch_np
            ) == len(original_indices_to_encode):
                # Place newly encoded embeddings into their correct positions in final_embeddings
                for idx_in_batch, original_list_idx in enumerate(
                    original_indices_to_encode
                ):
                    final_embeddings[original_list_idx] = newly_encoded_batch_np[
                        idx_in_batch
                    ]
            elif newly_encoded_batch_np is None:
                logger.error(
                    f"Encoding failed for a batch of {len(texts_to_encode_values)} texts. "
                    "These embeddings will be missing in the output."
                )
            else:  # Mismatch in length, should not happen if _generate_and_store_embeddings_batch is correct
                logger.error(
                    f"Mismatch in length between encoded batch ({len(newly_encoded_batch_np)}) and "
                    f"expected ({len(original_indices_to_encode)}). Some embeddings may be misplaced or missing."
                )

        # Step 3: Assemble final results
        # final_embeddings list now contains a mix of cached and newly encoded embeddings (or None for failures)

        processed_embeddings: List[np.ndarray] = []
        for i, emb in enumerate(final_embeddings):
            if emb is None:
                logger.warning(  # Changed to warning, as error was logged during encoding failure
                    f"Embedding for text index {i} ('{texts[i][:50]}...') is None after processing. "
                    "This indicates it was not found in cache and encoding failed or was skipped."
                )
                # Option: Fill with zeros of self._embedding_dimension, or skip.
                # Current behavior: skip, leading to a potentially shorter result array than input texts.
                # To maintain length, one might do:
                # processed_embeddings.append(np.zeros(self._embedding_dimension, dtype=np.float32))
            else:
                processed_embeddings.append(emb)

        if not processed_embeddings:
            logger.error(
                "No embeddings could be successfully processed or retrieved for the batch."
            )
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dimension)

        # If we want to ensure the output array has the same number of rows as input texts,
        # and fill failures with zeros:
        # final_result_array = np.zeros((len(texts), self._embedding_dimension), dtype=np.float32)
        # for i, emb in enumerate(final_embeddings):
        #     if emb is not None:
        #         final_result_array[i] = emb
        # return final_result_array
        # For now, returning only successfully processed embeddings:
        return np.array(processed_embeddings)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings generated by this model."""
        return self._embedding_dimension

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
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dimension)

        logger.debug(f"Request to embed {len(texts)} texts. Batch size: {batch_size}")

        # Try cache first if enabled
        cached_results, missing_indices = self._get_cached_embeddings(
            texts, cache_context_df
        )
        if not missing_indices:
            return self._assemble_final_embeddings(cached_results, [])

        # Encode missing texts
        missing_texts = [texts[i] for i in missing_indices]
        new_embeddings = self._encode_texts_batch(missing_texts, batch_size)
        if new_embeddings is None:
            return self._assemble_final_embeddings(cached_results, [])

        # Store new embeddings in cache
        if self.enable_embedding_cache and self.cache:
            self._store_embeddings_batch(
                missing_texts, new_embeddings, cache_context_df
            )

        # Combine cached and new embeddings
        return self._assemble_final_embeddings(
            cached_results, list(zip(missing_indices, new_embeddings))
        )

    def _get_cached_embeddings(
        self, texts: List[str], cache_context_df: Optional[Any]
    ) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """Retrieve embeddings from cache.

        Returns:
            Tuple of (cached_results, missing_indices)
        """
        if not self.enable_embedding_cache or not self.cache:
            return [None] * len(texts), list(range(len(texts)))

        logger.debug(f"Attempting to retrieve {len(texts)} embeddings from cache")
        cached_embeddings, _ = self.cache.retrieve_embeddings_batch(
            texts, self.model_name_or_path, df=cache_context_df
        )

        missing_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]
        found_count = len(texts) - len(missing_indices)

        if found_count > 0:
            logger.info(f"Retrieved {found_count}/{len(texts)} embeddings from cache")

        return cached_embeddings, missing_indices

    def _encode_texts_batch(
        self, texts: List[str], batch_size: int
    ) -> Optional[np.ndarray]:
        """Encode texts using the model.

        Returns:
            Encoded embeddings or None on failure.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dimension)

        logger.info(f"Encoding {len(texts)} texts")
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 1000,
            )
            logger.info(f"Successfully encoded {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Encoding failed for {len(texts)} texts: {e}", exc_info=True)
            return None

    def _store_embeddings_batch(
        self, texts: List[str], embeddings: np.ndarray, cache_context_df: Optional[Any]
    ) -> None:
        """Store embeddings in cache."""
        if not self.cache:
            return

        logger.debug(f"Storing {len(texts)} embeddings in cache")
        embeddings_list = [embeddings[i] for i in range(len(texts))]
        success = self.cache.persist_embeddings_batch(
            texts, embeddings_list, self.model_name_or_path, df=cache_context_df
        )
        if success:
            logger.info(f"Successfully cached {len(texts)} embeddings")
        else:
            logger.warning("Failed to cache some embeddings")

    def _assemble_final_embeddings(
        self,
        cached_results: List[Optional[np.ndarray]],
        new_results: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """Assemble final embeddings from cached and new results."""
        # Create mapping of new embeddings by index
        new_embeddings_map = dict(new_results)

        final_embeddings = []
        for i, cached_emb in enumerate(cached_results):
            if cached_emb is not None:
                final_embeddings.append(cached_emb)
            elif i in new_embeddings_map:
                final_embeddings.append(new_embeddings_map[i])
            else:
                logger.warning(f"No embedding available for text at index {i}")
                # Skip missing embeddings rather than adding zeros
                continue

        if not final_embeddings:
            return np.array([], dtype=np.float32).reshape(0, self._embedding_dimension)

        return np.array(final_embeddings)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings generated by this model."""
        return self._embedding_dimension

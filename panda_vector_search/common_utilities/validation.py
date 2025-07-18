"""Configuration validation utilities for panda-vector-search."""

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


def validate_embedding_config(config: Any) -> List[str]:
    """Validate EmbeddingConfig object.

    Args:
        config: The EmbeddingConfig object to validate.

    Returns:
        List of validation error messages. Empty list if valid.
    """
    errors = []

    if not hasattr(config, "model_name_or_path"):
        errors.append("EmbeddingConfig must have 'model_name_or_path' attribute")
    elif not isinstance(config.model_name_or_path, str):
        errors.append("'model_name_or_path' must be a string")
    elif not config.model_name_or_path.strip():
        errors.append("'model_name_or_path' cannot be empty")

    if hasattr(config, "device") and config.device is not None:
        if not isinstance(config.device, str):
            errors.append("'device' must be a string or None")
        elif config.device not in ["cpu", "cuda", "mps"]:
            errors.append("'device' must be one of: 'cpu', 'cuda', 'mps', or None")

    if hasattr(config, "trust_remote_code"):
        if not isinstance(config.trust_remote_code, bool):
            errors.append("'trust_remote_code' must be a boolean")

    if hasattr(config, "enable_embedding_artifact_cache"):
        if not isinstance(config.enable_embedding_artifact_cache, bool):
            errors.append("'enable_embedding_artifact_cache' must be a boolean")

    if hasattr(config, "embedding_artifact_cache_dir"):
        if not isinstance(config.embedding_artifact_cache_dir, str):
            errors.append("'embedding_artifact_cache_dir' must be a string")

    return errors


def validate_search_parameters(query: Any, k: int, threshold: float) -> List[str]:
    """Validate search parameters.

    Args:
        query: Search query (string or dict).
        k: Number of results to return.
        threshold: Minimum similarity threshold.

    Returns:
        List of validation error messages. Empty list if valid.
    """
    errors = []

    if not isinstance(query, (str, dict)):
        errors.append("Query must be a string or dictionary")
    elif isinstance(query, str) and not query.strip():
        errors.append("Query string cannot be empty")
    elif isinstance(query, dict):
        if not query:
            errors.append("Query dictionary cannot be empty")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in query.items()):
            errors.append("Query dictionary must contain only string keys and values")

    if not isinstance(k, int):
        errors.append("Parameter 'k' must be an integer")
    elif k <= 0:
        errors.append("Parameter 'k' must be positive")
    elif k > 10000:  # Reasonable upper limit
        errors.append("Parameter 'k' is too large (maximum: 10000)")

    if not isinstance(threshold, (int, float)):
        errors.append("Parameter 'threshold' must be a number")
    elif threshold < 0.0:
        errors.append("Parameter 'threshold' must be non-negative")
    elif threshold > 1.0:
        errors.append("Parameter 'threshold' must not exceed 1.0")

    return errors


def validate_dataframe_and_columns(df: Any, text_columns: Any) -> List[str]:
    """Validate DataFrame and text columns.

    Args:
        df: DataFrame to validate.
        text_columns: Text columns specification.

    Returns:
        List of validation error messages. Empty list if valid.
    """
    errors = []

    # Validate DataFrame
    try:
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            errors.append("Input must be a pandas DataFrame")
            return errors  # Can't continue validation without proper DataFrame
    except ImportError:
        errors.append("pandas is required but not available")
        return errors

    if df.empty:
        errors.append("DataFrame cannot be empty")
        return errors

    # Validate text columns
    if isinstance(text_columns, str):
        text_columns = [text_columns]
    elif not isinstance(text_columns, list):
        errors.append("text_columns must be a string or list of strings")
        return errors

    if not text_columns:
        errors.append("text_columns cannot be empty")
        return errors

    if not all(isinstance(col, str) for col in text_columns):
        errors.append("All text_columns must be strings")
        return errors

    # Check if columns exist in DataFrame
    missing_columns = [col for col in text_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Columns not found in DataFrame: {missing_columns}")

    # Check if specified columns contain mostly empty values
    for col in text_columns:
        if col in df.columns:
            non_empty_count = df[col].notna().sum()
            if non_empty_count == 0:
                errors.append(f"Column '{col}' contains no non-null values")
            elif non_empty_count < len(df) * 0.1:  # Less than 10% non-empty
                logger.warning(
                    f"Column '{col}' contains very few non-empty values ({non_empty_count}/{len(df)})"
                )

    return errors

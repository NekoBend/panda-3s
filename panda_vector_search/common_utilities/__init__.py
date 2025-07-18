"""
Common utilities for panda-vector-search.
Provides hashing functions and configuration validation utilities.
"""

from .hashing import compute_hash, generate_dataframe_schema_fingerprint
from .validation import (
    ConfigurationError,
    validate_dataframe_and_columns,
    validate_embedding_config,
    validate_search_parameters,
)

__all__ = [
    "compute_hash",
    "generate_dataframe_schema_fingerprint",
    "ConfigurationError",
    "validate_embedding_config",
    "validate_search_parameters",
    "validate_dataframe_and_columns",
]

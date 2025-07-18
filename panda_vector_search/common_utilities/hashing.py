import hashlib
from typing import Optional

import pandas as pd


def compute_hash(data: str, truncate: int | None = None) -> str:
    """Compute SHA256 hash for given data.

    Args:
        data: The string data to hash.
        truncate: Optional. The number of characters to truncate the hash to.

    Returns:
        The SHA256 hash of the data, optionally truncated.
    """
    hash_value = hashlib.sha256(data.encode()).hexdigest()
    return hash_value[:truncate] if truncate else hash_value


def generate_dataframe_schema_fingerprint(
    df: Optional[pd.DataFrame],
) -> str:
    """Generate a unique fingerprint for a DataFrame's schema (column names and types).

    This fingerprint is consistent across runs and can be used for caching or validation.

    Args:
        df: The pandas DataFrame to generate the fingerprint for. Can be None.

    Returns:
        A SHA256 hash representing the DataFrame's schema. Returns a specific hash
        if the DataFrame is None.
    """
    if df is None:
        return compute_hash("no_schema_provided")  # Consistent with cache logic

    # Sort by column name to ensure consistent hash
    sorted_columns = sorted(df.columns)
    schema_parts = [f"{col}:{str(df[col].dtype)}" for col in sorted_columns]
    schema_str = ";".join(schema_parts)
    return compute_hash(
        f"schema_v1:{schema_str}"
    )  # Added a version prefix for future changes

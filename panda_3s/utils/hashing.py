import hashlib
from typing import Optional  # Added Optional

import pandas as pd


def compute_hash(data: str, truncate: int | None = None) -> str:
    """Compute SHA256 hash for given data."""
    hash_value = hashlib.sha256(data.encode()).hexdigest()
    return hash_value[:truncate] if truncate else hash_value


def compute_row_hash(row_data: str) -> str:
    """Compute hash for a row."""
    return compute_hash(row_data)


def compute_dataframe_schema_hash(
    df: Optional[pd.DataFrame],
) -> str:  # Renamed and signature changed
    """Compute SHA256 hash for DataFrame schema (column names and types)."""
    if df is None:
        return compute_hash("no_schema_provided")  # Consistent with cache logic

    # Sort by column name to ensure consistent hash
    sorted_columns = sorted(df.columns)
    schema_parts = [f"{col}:{str(df[col].dtype)}" for col in sorted_columns]
    schema_str = ";".join(schema_parts)
    return compute_hash(
        f"schema_v1:{schema_str}"
    )  # Added a version prefix for future changes


def compute_dataframe_row_hashes(df: pd.DataFrame, text_column: str) -> dict[int, str]:
    """Compute hash for each row in DataFrame."""
    row_hashes = {}
    for i in range(
        len(df)
    ):  # Consider using df.itertuples() for potentially better performance
        row = df.iloc[i]
        # Create a consistent string representation of the row
        row_str = f"{text_column}:{row[text_column]}"
        # Add other columns for more complete hash
        for col in df.columns:
            if col != text_column:
                row_str += f"|{col}:{row[col]}"
        row_hashes[i] = compute_row_hash(row_str)
    return row_hashes

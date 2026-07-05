"""Local DuckDB SQL layer for the workshop notebook (offline mode)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.simulate import save_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TABLE_FILES = {
    "users": DATA_DIR / "users.parquet",
    "ads": DATA_DIR / "ads.parquet",
    "impressions": DATA_DIR / "impressions.parquet",
    "conversions": DATA_DIR / "conversions.parquet",
}

TABLE_COLUMNS = {
    "users": [
        "user_id", "age_group", "gender", "device_type", "region",
        "account_age_days", "past_purchases", "avg_order_value",
        "loyalty_tier", "sessions_per_week",
    ],
    "ads": [
        "ad_id", "category", "ad_format", "product_price", "discount_pct",
        "creative_quality_score", "headline_clickbait_score", "brand_familiarity",
    ],
    "impressions": [
        "impression_id", "user_id", "ad_id", "timestamp", "page_type",
        "position", "day_of_week", "hour_of_day", "session_depth", "clicked",
    ],
    "conversions": [
        "conversion_id", "impression_id", "user_id", "ad_id",
        "revenue", "time_to_convert_minutes",
    ],
}

_connection: duckdb.DuckDBPyConnection | None = None


def ensure_data(n_impressions: int = 500_000, seed: int = 42) -> None:
    """Generate parquet tables if they are missing."""
    if all(p.exists() for p in TABLE_FILES.values()):
        return
    print("Synthetic data not found — generating (one-time, ~30s)...")
    save_dataset(DATA_DIR, n_impressions=n_impressions, seed=seed)
    reset_connection()
    print("Done.\n")


def reset_connection() -> None:
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def _connect() -> duckdb.DuckDBPyConnection:
    global _connection
    if _connection is None:
        ensure_data()
        _connection = duckdb.connect()
        for table, path in TABLE_FILES.items():
            _connection.execute(
                f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path}')"
            )
    return _connection


def query(sql: str, limit: int = 50_000) -> pd.DataFrame:
    """Run SQL against local synthetic tables (same interface as workshop API)."""
    con = _connect()
    wrapped = f"SELECT * FROM ({sql.strip().rstrip(';')}) AS _q LIMIT {limit}"
    return con.execute(wrapped).df()


def get_schema() -> dict:
    """Schema dict matching the workshop /api/schema response shape."""
    ensure_data()
    tables = {}
    for name, cols in TABLE_COLUMNS.items():
        path = TABLE_FILES[name]
        n_rows = len(pd.read_parquet(path, columns=[cols[0]]))
        tables[name] = {
            "row_count": n_rows,
            "columns": [{"name": c} for c in cols],
        }
    return {"tables": tables}


def print_schema(schema: dict | None = None) -> None:
    schema = schema or get_schema()
    for table, info in schema["tables"].items():
        cols = [c["name"] for c in info["columns"]]
        print(f"\n{table} ({info['row_count']:,} rows):")
        print(f"  Columns: {', '.join(cols)}")

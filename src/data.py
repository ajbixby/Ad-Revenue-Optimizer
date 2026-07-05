"""Load training data from local parquet or the (optional) live workshop API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "synthetic_impressions.parquet"
SERVER = "https://lse-mayday.onrender.com"

JOINED_COLUMNS = [
    "impression_id",
    "age_group", "gender", "device_type", "region",
    "account_age_days", "past_purchases", "avg_order_value",
    "sessions_per_week", "loyalty_tier",
    "category", "ad_format", "product_price", "discount_pct",
    "creative_quality_score", "headline_clickbait_score", "brand_familiarity",
    "page_type", "position", "hour_of_day", "day_of_week", "session_depth",
    "clicked", "converted", "revenue",
]


def query(sql: str, limit: int = 50_000, server: str = SERVER) -> pd.DataFrame:
    resp = requests.post(f"{server}/api/sql", json={"query": sql, "limit": limit}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data["rows"], columns=data["columns"])


def pull_from_server(max_rows: int = 100_000, server: str = SERVER) -> pd.DataFrame:
    batch_size = 50_000
    chunks: list[pd.DataFrame] = []
    offset = 0
    while offset < max_rows:
        batch = min(batch_size, max_rows - offset)
        df = query(f"""
            SELECT
                i.impression_id,
                u.age_group, u.gender, u.device_type, u.region,
                u.account_age_days, u.past_purchases, u.avg_order_value,
                u.sessions_per_week, u.loyalty_tier,
                a.category, a.ad_format, a.product_price, a.discount_pct,
                a.creative_quality_score, a.headline_clickbait_score,
                a.brand_familiarity,
                i.page_type, i.position, i.hour_of_day, i.day_of_week,
                i.session_depth,
                i.clicked,
                CASE WHEN c.conversion_id IS NOT NULL THEN 1 ELSE 0 END AS converted,
                COALESCE(c.revenue, 0) AS revenue
            FROM impressions i
            JOIN users u ON i.user_id = u.user_id
            JOIN ads   a ON i.ad_id   = a.ad_id
            LEFT JOIN conversions c ON i.impression_id = c.impression_id
            ORDER BY i.impression_id
            LIMIT {batch} OFFSET {offset}
        """, limit=batch, server=server)
        if len(df) == 0:
            break
        chunks.append(df)
        offset += len(df)
        print(f"  pulled {offset:,} rows...")
    return pd.concat(chunks, ignore_index=True)


def load_local(path: Path = DEFAULT_DATA_PATH, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No data at {path}. Generate it with:\n"
            f"  python -m src.simulate --rows 500000 --out {path}"
        )
    df = pd.read_parquet(path)
    if max_rows is not None:
        df = df.head(max_rows)
    return df[JOINED_COLUMNS]


def pull_training_data(
    max_rows: int = 100_000,
    *,
    source: str = "local",
    data_path: Path = DEFAULT_DATA_PATH,
    server: str = SERVER,
) -> pd.DataFrame:
    """
    Load joined impression training data.

    source: 'local' (default) reads parquet; 'live' pulls from workshop API.
    """
    if source == "live":
        return pull_from_server(max_rows=max_rows, server=server)
    return load_local(data_path, max_rows=max_rows)

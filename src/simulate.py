"""
Synthetic e-commerce ad funnel calibrated to LSE Mayday workshop statistics.

Generates normalized SQL tables (users, ads, impressions, conversions) plus a
joined training table matching the workshop pull schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUT = DATA_DIR / "synthetic_impressions.parquet"

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS = ["female", "male", "other"]
DEVICES = ["desktop", "mobile", "tablet"]
REGIONS = ["north", "south", "east", "west", "central"]
LOYALTY_TIERS = ["bronze", "silver", "gold", "platinum"]
CATEGORIES = [
    "electronics", "fashion", "home", "beauty", "sports",
    "books", "food", "toys", "automotive", "health",
]
AD_FORMATS = ["banner", "carousel", "video", "native", "sidebar"]
PAGE_TYPES = ["home", "search", "product", "category", "checkout"]

LOYALTY_RANK = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr, dtype=float)
    return (arr - arr.mean()) / std


def generate_users(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    tier_probs = np.array([0.42, 0.32, 0.18, 0.08])
    tiers = rng.choice(LOYALTY_TIERS, size=n_users, p=tier_probs)
    ranks = np.array([LOYALTY_RANK[t] for t in tiers])

    base_purchases = rng.poisson(lam=2, size=n_users) + ranks * rng.integers(3, 12, size=n_users)
    past_purchases = np.clip(base_purchases, 0, 80)

    base_aov = rng.lognormal(mean=3.2, sigma=0.45, size=n_users)
    avg_order_value = np.clip(base_aov * (1 + 0.35 * ranks), 15, 450)

    return pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age_group": rng.choice(AGE_GROUPS, size=n_users, p=[0.18, 0.26, 0.22, 0.18, 0.16]),
        "gender": rng.choice(GENDERS, size=n_users, p=[0.52, 0.46, 0.02]),
        "device_type": rng.choice(DEVICES, size=n_users, p=[0.28, 0.55, 0.17]),
        "region": rng.choice(REGIONS, size=n_users),
        "account_age_days": rng.integers(7, 900, size=n_users),
        "past_purchases": past_purchases,
        "avg_order_value": np.round(avg_order_value, 2),
        "loyalty_tier": tiers,
        "sessions_per_week": np.clip(rng.poisson(lam=2.5, size=n_users) + ranks, 1, 14),
    })


def generate_ads(n_ads: int, rng: np.random.Generator) -> pd.DataFrame:
    product_price = np.round(rng.lognormal(mean=3.6, sigma=0.55, size=n_ads), 2)
    discount_pct = np.clip(rng.beta(2, 8, size=n_ads) * 40, 0, 35)
    # Workshop notebook buckets scores on a 0–10 scale
    creative_quality = np.round(rng.uniform(2, 10, size=n_ads), 2)
    clickbait = np.round(rng.uniform(0, 10, size=n_ads), 2)
    brand_familiarity = np.round(rng.uniform(0.1, 1.0, size=n_ads), 3)

    return pd.DataFrame({
        "ad_id": np.arange(1, n_ads + 1),
        "category": rng.choice(CATEGORIES, size=n_ads),
        "ad_format": rng.choice(AD_FORMATS, size=n_ads),
        "product_price": product_price,
        "discount_pct": np.round(discount_pct, 1),
        "creative_quality_score": creative_quality,
        "headline_clickbait_score": clickbait,
        "brand_familiarity": brand_familiarity,
    })


def _sample_impression_pairs(
    n_impressions: int,
    users: pd.DataFrame,
    ads: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    user_w = np.ones(len(users))
    user_w += (users["loyalty_tier"] == "bronze").to_numpy() * 0.6
    user_w += (users["device_type"] == "mobile").to_numpy() * 0.4
    user_w /= user_w.sum()

    ad_w = np.ones(len(ads))
    ad_w += _zscore(ads["headline_clickbait_score"].to_numpy()) * 0.25
    ad_w = np.clip(ad_w, 0.05, None)
    ad_w /= ad_w.sum()

    user_idx = rng.choice(len(users), size=n_impressions, p=user_w)
    ad_idx = rng.choice(len(ads), size=n_impressions, p=ad_w)
    return user_idx, ad_idx


def _click_logits(
    users: pd.DataFrame,
    ads: pd.DataFrame,
    ctx: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    loyalty_rank = users["loyalty_tier"].map(LOYALTY_RANK).to_numpy()
    clickbait_z = _zscore(ads["headline_clickbait_score"].to_numpy())
    quality_z = _zscore(ads["creative_quality_score"].to_numpy())
    device = users["device_type"].to_numpy()
    device_bonus = np.where(device == "mobile", 0.65, np.where(device == "tablet", 0.18, 0.0))

    return (
        -1.05
        + 0.90 * clickbait_z
        + 0.22 * quality_z
        - 0.38 * loyalty_rank
        + device_bonus
        - 0.07 * ctx["position"].to_numpy()
        + 0.035 * ctx["session_depth"].to_numpy()
        + 0.12 * ctx["is_peak_hour"].to_numpy()
        + 0.08 * ctx["is_weekend"].to_numpy()
        + rng.normal(0, 0.35, size=len(users))
    )


def _convert_logits(users: pd.DataFrame, ads: pd.DataFrame) -> np.ndarray:
    loyalty_rank = users["loyalty_tier"].map(LOYALTY_RANK).to_numpy()
    aov_z = _zscore(users["avg_order_value"].to_numpy())
    brand_z = _zscore(ads["brand_familiarity"].to_numpy())
    quality_z = _zscore(ads["creative_quality_score"].to_numpy())
    clickbait_z = _zscore(ads["headline_clickbait_score"].to_numpy())
    price_z = _zscore(ads["product_price"].to_numpy())
    device = users["device_type"].to_numpy()
    device_bonus = np.where(device == "desktop", 0.32, np.where(device == "tablet", 0.08, -0.12))
    price_fit = -np.abs(price_z - aov_z) * np.where(loyalty_rank <= 1, 0.35, 0.10)

    return (
        -2.65
        + 0.95 * loyalty_rank
        + 0.42 * aov_z
        + 0.38 * brand_z
        + 0.45 * quality_z
        - 0.32 * clickbait_z
        + 0.18 * price_z * (loyalty_rank / 3.0)
        + device_bonus
        + price_fit
    )


def _revenue(users: pd.DataFrame, ads: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    loyalty_rank = users["loyalty_tier"].map(LOYALTY_RANK).to_numpy()
    aov_z = _zscore(users["avg_order_value"].to_numpy())
    net = ads["product_price"].to_numpy() * (1 - ads["discount_pct"].to_numpy() / 100)
    noise = rng.lognormal(mean=0, sigma=0.22, size=len(users))
    multiplier = np.exp(0.28 * loyalty_rank + 0.18 * aov_z)
    return np.round(net * multiplier * noise, 2)


def generate_dataset(
    n_impressions: int,
    n_users: int = 10_000,
    n_ads: int = 200,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    users = generate_users(n_users, rng)
    ads = generate_ads(n_ads, rng)

    user_idx, ad_idx = _sample_impression_pairs(n_impressions, users, ads, rng)
    u = users.iloc[user_idx].reset_index(drop=True)
    a = ads.iloc[ad_idx].reset_index(drop=True)

    ctx = pd.DataFrame({
        "page_type": rng.choice(PAGE_TYPES, size=n_impressions, p=[0.22, 0.28, 0.24, 0.18, 0.08]),
        "position": rng.integers(1, 9, size=n_impressions),
        "hour_of_day": rng.integers(0, 24, size=n_impressions),
        "day_of_week": rng.integers(0, 7, size=n_impressions),
        "session_depth": rng.integers(1, 16, size=n_impressions),
    })
    ctx["is_peak_hour"] = ctx["hour_of_day"].between(18, 22).astype(int)
    ctx["is_weekend"] = ctx["day_of_week"].isin([5, 6]).astype(int)

    click_logit = _click_logits(u, a, ctx, rng)
    clicked = rng.binomial(1, _sigmoid(click_logit))

    conv_logit = _convert_logits(u, a) + rng.normal(0, 0.25, size=n_impressions)
    converted = clicked * rng.binomial(1, _sigmoid(conv_logit))

    rev = _revenue(u, a, rng)
    revenue = converted * rev

    impression_ids = np.arange(1, n_impressions + 1)
    base_ts = pd.Timestamp("2025-03-01")
    timestamps = [base_ts + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 60 * 24 * 30, n_impressions)]

    impressions = pd.DataFrame({
        "impression_id": impression_ids,
        "user_id": u["user_id"].values,
        "ad_id": a["ad_id"].values,
        "timestamp": timestamps,
        "page_type": ctx["page_type"].values,
        "position": ctx["position"].values,
        "day_of_week": ctx["day_of_week"].values,
        "hour_of_day": ctx["hour_of_day"].values,
        "session_depth": ctx["session_depth"].values,
        "clicked": clicked.astype(int),
    })

    conv_mask = converted.astype(bool)
    n_conv = int(conv_mask.sum())
    conversions = pd.DataFrame({
        "conversion_id": np.arange(1, n_conv + 1),
        "impression_id": impression_ids[conv_mask],
        "user_id": u.loc[conv_mask, "user_id"].values,
        "ad_id": a.loc[conv_mask, "ad_id"].values,
        "revenue": revenue[conv_mask],
        "time_to_convert_minutes": rng.integers(1, 180, size=n_conv),
    })

    joined = pd.DataFrame({
        "impression_id": impression_ids,
        "age_group": u["age_group"].values,
        "gender": u["gender"].values,
        "device_type": u["device_type"].values,
        "region": u["region"].values,
        "account_age_days": u["account_age_days"].values,
        "past_purchases": u["past_purchases"].values,
        "avg_order_value": u["avg_order_value"].values,
        "sessions_per_week": u["sessions_per_week"].values,
        "loyalty_tier": u["loyalty_tier"].values,
        "category": a["category"].values,
        "ad_format": a["ad_format"].values,
        "product_price": a["product_price"].values,
        "discount_pct": a["discount_pct"].values,
        "creative_quality_score": a["creative_quality_score"].values,
        "headline_clickbait_score": a["headline_clickbait_score"].values,
        "brand_familiarity": a["brand_familiarity"].values,
        "page_type": ctx["page_type"].values,
        "position": ctx["position"].values,
        "hour_of_day": ctx["hour_of_day"].values,
        "day_of_week": ctx["day_of_week"].values,
        "session_depth": ctx["session_depth"].values,
        "clicked": clicked.astype(int),
        "converted": converted.astype(int),
        "revenue": revenue,
    })

    return {
        "users": users,
        "ads": ads,
        "impressions": impressions,
        "conversions": conversions,
        "joined": joined,
    }


def generate_impressions(
    n_impressions: int,
    n_users: int = 10_000,
    n_ads: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Return joined training table (backward compatible)."""
    return generate_dataset(n_impressions, n_users, n_ads, seed)["joined"]


def save_dataset(
    out_dir: Path = DATA_DIR,
    n_impressions: int = 500_000,
    seed: int = 42,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = generate_dataset(n_impressions, seed=seed)
    paths = {
        "users": out_dir / "users.parquet",
        "ads": out_dir / "ads.parquet",
        "impressions": out_dir / "impressions.parquet",
        "conversions": out_dir / "conversions.parquet",
        "joined": out_dir / "synthetic_impressions.parquet",
    }
    for key, path in paths.items():
        tables[key].to_parquet(path, index=False)
    return paths


def print_summary(df: pd.DataFrame) -> None:
    ctr = df["clicked"].mean() * 100
    cvr = df["converted"].mean() * 100
    rpi = df["revenue"].mean()
    print(f"Rows:        {len(df):,}")
    print(f"CTR:         {ctr:.2f}%")
    print(f"CVR (all):   {cvr:.2f}%")
    print(f"Avg rev/imp: £{rpi:.4f}")

    seg = (
        df.groupby(["loyalty_tier", "device_type"], observed=True)
        .agg(
            impressions=("impression_id", "count"),
            ctr_pct=("clicked", lambda s: round(s.mean() * 100, 2)),
            revenue_per_impression=("revenue", "mean"),
        )
        .reset_index()
        .sort_values("revenue_per_impression", ascending=False)
    )
    print("\nTop segments by revenue/impression:")
    print(seg.head(6).to_string(index=False))
    print("\nBottom segments by revenue/impression:")
    print(seg.tail(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Mayday ad funnel data")
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.rows:,} impressions (seed={args.seed})...")
    paths = save_dataset(args.out.parent, n_impressions=args.rows, seed=args.seed)
    joined = pd.read_parquet(paths["joined"])
    size_mb = paths["joined"].stat().st_size / (1024 * 1024)
    print(f"Saved tables to {args.out.parent}/ ({size_mb:.1f} MB joined)\n")
    print_summary(joined)


if __name__ == "__main__":
    main()

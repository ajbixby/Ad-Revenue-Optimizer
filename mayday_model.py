"""
Full pipeline: load data → engineer features → train 3-stage chain
→ evaluate → benchmark latency → (optional) upload to server.

Usage:
    python mayday_model.py --skip-upload
    python mayday_model.py --rows 100000 --skip-upload
    python mayday_model.py --source live --team your-team-name   # if server is up
    python -m src.simulate --rows 500000                         # regenerate data
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from src.data import DEFAULT_DATA_PATH, SERVER, pull_training_data
from src.simulate import save_dataset

warnings.filterwarnings("ignore")

if sys.version_info[:2] != (3, 12):
    print(
        f"WARNING: workshop server expects Python 3.12 — you are on "
        f"{sys.version_info.major}.{sys.version_info.minor}. "
        f"Pickle may fail on upload."
    )

TEAM_NAME = "The_Meat_Team"

CAT_COLS = [
    "age_group", "gender", "device_type", "region",
    "loyalty_tier", "category", "ad_format", "page_type",
]
ENCODERS: dict = {}

FEATURE_COLS = [
    "age_group", "gender", "device_type", "region",
    "loyalty_tier", "loyalty_rank",
    "account_age_days", "past_purchases", "avg_order_value", "sessions_per_week",
    "category", "ad_format", "product_price", "discount_pct",
    "max_possible_revenue", "price_x_quality", "discount_x_brand",
    "creative_quality_score", "headline_clickbait_score", "brand_familiarity",
    "page_type", "position", "hour_of_day", "day_of_week",
    "session_depth", "is_weekend", "is_peak_hour",
    "aov_x_price", "purchases_x_brand",
]


def prepare_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    d = df.copy()

    loyalty_map = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3}
    d["loyalty_rank"] = d["loyalty_tier"].str.lower().map(loyalty_map).fillna(0)

    d["max_possible_revenue"] = d["product_price"] * (1 - d["discount_pct"] / 100)
    d["price_x_quality"] = d["product_price"] * d["creative_quality_score"]
    d["discount_x_brand"] = d["discount_pct"] * d["brand_familiarity"]
    d["aov_x_price"] = d["avg_order_value"] * d["product_price"]
    d["purchases_x_brand"] = d["past_purchases"] * d["brand_familiarity"]
    d["is_weekend"] = d["day_of_week"].isin([5, 6]).astype(int)
    d["is_peak_hour"] = d["hour_of_day"].between(18, 22).astype(int)

    for col in CAT_COLS:
        d[col] = d[col].astype(str)
        if fit:
            le = LabelEncoder()
            d[col] = le.fit_transform(d[col])
            ENCODERS[col] = le
        else:
            le = ENCODERS[col]
            d[col] = d[col].map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

    return d[FEATURE_COLS].astype(float)


def train_stages(df_train: pd.DataFrame, X_train: pd.DataFrame):
    xgb_shared = dict(
        max_depth=5, learning_rate=0.05, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )

    print("\nStage 1 — P(click)")
    spw1 = (df_train["clicked"] == 0).sum() / (df_train["clicked"] == 1).sum()
    m1 = XGBClassifier(**xgb_shared, scale_pos_weight=spw1)
    m1.fit(X_train, df_train["clicked"], verbose=False)

    print("Stage 2 — P(convert | click)")
    mask_click = df_train["clicked"] == 1
    spw2 = (
        (df_train.loc[mask_click, "converted"] == 0).sum() /
        (df_train.loc[mask_click, "converted"] == 1).sum()
    )
    m2 = XGBClassifier(**xgb_shared, scale_pos_weight=spw2)
    m2.fit(X_train[mask_click], df_train.loc[mask_click, "converted"], verbose=False)

    print("Stage 3 — E(revenue | convert)")
    mask_conv = df_train["converted"] == 1
    m3 = XGBRegressor(**xgb_shared)
    m3.fit(X_train[mask_conv], df_train.loc[mask_conv, "revenue"])

    return m1, m2, m3


def evaluate(m1, m2, m3, df_test: pd.DataFrame, X_test: pd.DataFrame):
    click_auc = roc_auc_score(df_test["clicked"], m1.predict_proba(X_test)[:, 1])

    mask_click = df_test["clicked"] == 1
    conv_auc = roc_auc_score(
        df_test.loc[mask_click, "converted"],
        m2.predict_proba(X_test[mask_click])[:, 1],
    )

    p_click = m1.predict_proba(X_test)[:, 1]
    p_convert = m2.predict_proba(X_test)[:, 1]
    e_revenue = m3.predict(X_test)
    score = p_click * p_convert * e_revenue

    df_eval = df_test.copy()
    df_eval["score"] = score
    threshold = df_eval["score"].quantile(0.9)

    top_rpi = df_eval.loc[df_eval["score"] >= threshold, "revenue"].mean()
    bot_rpi = df_eval.loc[df_eval["score"] < threshold, "revenue"].mean()
    lift = top_rpi / bot_rpi if bot_rpi > 0 else float("inf")

    print(f"\n{'─'*40}")
    print(f"  Stage 1 AUC (click):          {click_auc:.4f}")
    print(f"  Stage 2 AUC (convert|click):  {conv_auc:.4f}")
    print(f"  Top 10% avg revenue:          £{top_rpi:.4f}")
    print(f"  Bottom 90% avg revenue:       £{bot_rpi:.4f}")
    print(f"  Revenue lift:                 {lift:.2f}×")
    print(f"{'─'*40}")

    if lift < 2:
        print("  ⚠  Lift <2× — check target or feature engineering")
    elif lift > 3:
        print("  ✓  Lift >3× — good signal")

    return score


class ScoringModel:
    """Wrapper for deployment: predict(raw_df) → revenue scores."""

    def __init__(self, m1, m2, m3, encoders: dict, feature_cols: list):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.encoders = encoders
        self.feature_cols = feature_cols

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        loyalty_map = {"bronze": 0, "silver": 1, "gold": 2, "platinum": 3}
        d["loyalty_rank"] = d["loyalty_tier"].str.lower().map(loyalty_map).fillna(0)
        d["max_possible_revenue"] = d["product_price"] * (1 - d["discount_pct"] / 100)
        d["price_x_quality"] = d["product_price"] * d["creative_quality_score"]
        d["discount_x_brand"] = d["discount_pct"] * d["brand_familiarity"]
        d["aov_x_price"] = d["avg_order_value"] * d["product_price"]
        d["purchases_x_brand"] = d["past_purchases"] * d["brand_familiarity"]
        d["is_weekend"] = d["day_of_week"].isin([5, 6]).astype(int)
        d["is_peak_hour"] = d["hour_of_day"].between(18, 22).astype(int)
        for col in CAT_COLS:
            le = self.encoders[col]
            d[col] = d[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )
        return d[self.feature_cols].astype(float)

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        X = self._prepare(raw_df)
        p_click = self.m1.predict_proba(X)[:, 1]
        p_convert = self.m2.predict_proba(X)[:, 1]
        e_revenue = self.m3.predict(X)
        return p_click * p_convert * e_revenue


def benchmark_latency(model: ScoringModel, sample_df: pd.DataFrame, n_trials: int = 100):
    batch = sample_df.head(10)
    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        model.predict(batch)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    violations = (latencies > 50).mean()
    print(f"\nLatency over {n_trials} trials (batch=10):")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  Budget violations (>50ms): {violations:.1%}")
    if violations > 0.1:
        print("  ⚠  >10% violations — reduce n_estimators or drop features")
    else:
        print("  ✓  Within latency budget")


def upload_model(model: ScoringModel, team: str):
    path = f"model_{team}.pkl"
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)
    size_kb = Path(path).stat().st_size / 1024
    print(f"\nModel saved: {path} ({size_kb:.1f} KB)")

    with open(path, "rb") as f:
        resp = requests.post(
            f"{SERVER}/api/teams/{team}/model",
            files={"model": (path, f, "application/octet-stream")},
            timeout=60,
        )
    result = resp.json()
    if resp.ok:
        print("✓ Uploaded successfully")
        print(f"  Model type:           {result.get('model_type')}")
        print(f"  Validation score:     {result.get('validation_prediction'):.4f}")
        print(f"  Validation latency:   {result.get('validation_latency_ms'):.2f} ms")
    else:
        print(f"✗ Upload failed: {result}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=TEAM_NAME)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--source", choices=["local", "live"], default="local")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--generate-data", action="store_true",
                        help="Regenerate synthetic parquet before training")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if args.generate_data or (args.source == "local" and not args.data_path.exists()):
        if not args.data_path.exists():
            print(f"No data at {args.data_path} — generating synthetic dataset...")
        else:
            print(f"Regenerating {args.data_path}...")
        args.data_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(args.data_path.parent, n_impressions=500_000, seed=42)
        print(f"Saved tables to {args.data_path.parent}/\n")

    print(
        f"Team: {args.team} | Rows: {args.rows:,} | "
        f"Source: {args.source} | Upload: {not args.skip_upload}"
    )

    print("\n── Loading data ─────────────────────────────────────────────────")
    df = pull_training_data(
        max_rows=args.rows,
        source=args.source,
        data_path=args.data_path,
    )
    print(f"Total rows: {len(df):,}")
    print(f"CTR:        {df['clicked'].mean()*100:.2f}%")
    print(f"CVR:        {df['converted'].mean()*100:.2f}%")
    print(f"Avg rev/imp: £{df['revenue'].mean():.4f}")

    print("\n── Engineering features ─────────────────────────────────────────")
    X = prepare_features(df, fit=True)
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, df_train, df_test = train_test_split(
        X, df, test_size=0.2, random_state=42
    )

    print("\n── Training three-stage chain ───────────────────────────────────")
    m1, m2, m3 = train_stages(df_train, X_train)

    print("\n── Evaluation ───────────────────────────────────────────────────")
    evaluate(m1, m2, m3, df_test, X_test)

    final_model = ScoringModel(m1, m2, m3, ENCODERS, FEATURE_COLS)

    sample_scores = final_model.predict(df.head(5))
    print(f"\nSample scores (raw df): {np.round(sample_scores, 4)}")

    print("\n── Latency benchmark ────────────────────────────────────────────")
    benchmark_latency(final_model, df.head(100))

    if not args.skip_upload:
        print("\n── Uploading ─────────────────────────────────────────────────")
        upload_model(final_model, args.team)
    else:
        print("\n── Skipping upload (--skip-upload) ──────────────────────────")


if __name__ == "__main__":
    main()

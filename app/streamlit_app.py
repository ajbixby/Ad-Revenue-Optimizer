"""Streamlit demo: score and rank ads by predicted revenue per impression."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mayday_model import (  # noqa: E402
    FEATURE_COLS,
    ScoringModel,
    prepare_features,
    train_stages,
)
from src.data import pull_training_data  # noqa: E402
from src.simulate import (  # noqa: E402
    AD_FORMATS,
    AGE_GROUPS,
    CATEGORIES,
    DEVICES,
    GENDERS,
    LOYALTY_TIERS,
    PAGE_TYPES,
    REGIONS,
)

st.set_page_config(
    page_title="Ad Revenue Optimizer",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner="Training 3-stage model on synthetic data…")
def load_model(train_rows: int = 100_000) -> ScoringModel:
    import mayday_model as mm

    mm.ENCODERS.clear()
    df = pull_training_data(max_rows=train_rows, source="local")
    X = prepare_features(df, fit=True)
    m1, m2, m3 = train_stages(df, X)
    return ScoringModel(m1, m2, m3, dict(mm.ENCODERS), FEATURE_COLS)


def score_breakdown(model: ScoringModel, row: pd.DataFrame) -> dict[str, float]:
    X = model._prepare(row)
    p_click = float(model.m1.predict_proba(X)[0, 1])
    p_convert = float(model.m2.predict_proba(X)[0, 1])
    e_revenue = float(model.m3.predict(X)[0])
    return {
        "p_click": p_click,
        "p_convert_given_click": p_convert,
        "e_revenue_given_convert": e_revenue,
        "expected_revenue": p_click * p_convert * e_revenue,
    }


def user_inputs(prefix: str) -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        age_group = st.selectbox("Age group", AGE_GROUPS, key=f"{prefix}_age")
        gender = st.selectbox("Gender", GENDERS, key=f"{prefix}_gender")
        device_type = st.selectbox("Device", DEVICES, key=f"{prefix}_device")
        region = st.selectbox("Region", REGIONS, key=f"{prefix}_region")
    with c2:
        loyalty_tier = st.selectbox("Loyalty tier", LOYALTY_TIERS, index=2, key=f"{prefix}_loyalty")
        past_purchases = st.number_input("Past purchases", 0, 80, 12, key=f"{prefix}_purchases")
        avg_order_value = st.number_input("Avg order value (£)", 15.0, 450.0, 85.0, key=f"{prefix}_aov")
        sessions_per_week = st.number_input("Sessions / week", 1, 14, 4, key=f"{prefix}_sessions")
    with c3:
        account_age_days = st.number_input("Account age (days)", 7, 900, 180, key=f"{prefix}_acct")
    return {
        "age_group": age_group,
        "gender": gender,
        "device_type": device_type,
        "region": region,
        "loyalty_tier": loyalty_tier,
        "past_purchases": past_purchases,
        "avg_order_value": avg_order_value,
        "sessions_per_week": sessions_per_week,
        "account_age_days": account_age_days,
    }


def ad_inputs(prefix: str, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    c1, c2 = st.columns(2)
    with c1:
        category = st.selectbox("Category", CATEGORIES, index=0, key=f"{prefix}_cat")
        ad_format = st.selectbox("Ad format", AD_FORMATS, key=f"{prefix}_fmt")
        product_price = st.number_input(
            "Product price (£)", 5.0, 500.0,
            float(defaults.get("product_price", 49.99)), key=f"{prefix}_price",
        )
        discount_pct = st.slider(
            "Discount %", 0.0, 35.0,
            float(defaults.get("discount_pct", 10.0)), key=f"{prefix}_disc",
        )
    with c2:
        creative_quality = st.slider(
            "Creative quality (0–10)", 0.0, 10.0,
            float(defaults.get("creative_quality_score", 7.0)), key=f"{prefix}_qual",
        )
        clickbait = st.slider(
            "Headline clickbait (0–10)", 0.0, 10.0,
            float(defaults.get("headline_clickbait_score", 5.0)), key=f"{prefix}_cb",
        )
        brand_familiarity = st.slider(
            "Brand familiarity", 0.0, 1.0,
            float(defaults.get("brand_familiarity", 0.6)), key=f"{prefix}_brand",
        )
    return {
        "category": category,
        "ad_format": ad_format,
        "product_price": product_price,
        "discount_pct": discount_pct,
        "creative_quality_score": creative_quality,
        "headline_clickbait_score": clickbait,
        "brand_familiarity": brand_familiarity,
    }


def context_inputs(prefix: str = "ctx") -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        page_type = st.selectbox("Page type", PAGE_TYPES, key=f"{prefix}_page")
        position = st.slider("Ad position", 1, 8, 2, key=f"{prefix}_pos")
    with c2:
        hour_of_day = st.slider("Hour of day", 0, 23, 19, key=f"{prefix}_hour")
        day_of_week = st.slider("Day of week (0=Mon)", 0, 6, 4, key=f"{prefix}_dow")
    with c3:
        session_depth = st.slider("Session depth", 1, 15, 5, key=f"{prefix}_depth")
    return {
        "page_type": page_type,
        "position": position,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "session_depth": session_depth,
    }


def build_row(user: dict, ad: dict, ctx: dict) -> pd.DataFrame:
    return pd.DataFrame([{**user, **ad, **ctx}])


def main() -> None:
    st.title("Ad Revenue Optimizer")
    st.caption(
        "3-stage model: score = P(click) × P(convert|click) × E(revenue|convert). "
        "Synthetic data from the LSE Mayday workshop pipeline."
    )

    model = load_model()

    tab_single, tab_rank = st.tabs(["Single ad scorer", "Rank ad candidates"])

    with tab_single:
        st.subheader("User & context")
        u = user_inputs("single")
        ctx = context_inputs("single")
        st.subheader("Ad creative")
        ad = ad_inputs("single")
        row = build_row(u, ad, ctx)
        scores = score_breakdown(model, row)

        st.subheader("Predicted revenue breakdown")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("P(click)", f"{scores['p_click']:.1%}")
        m2.metric("P(convert | click)", f"{scores['p_convert_given_click']:.1%}")
        m3.metric("E(revenue | convert)", f"£{scores['e_revenue_given_convert']:.2f}")
        m4.metric("Expected revenue / imp", f"£{scores['expected_revenue']:.4f}")

        st.info(
            "Higher clickbait boosts P(click) but often hurts conversion value. "
            "Loyalty tier and AOV drive the revenue stages."
        )

    with tab_rank:
        st.subheader("Same user — which ad wins?")
        u = user_inputs("rank")
        ctx = context_inputs("rank")

        presets = {
            "High clickbait banner": {
                "product_price": 35.0, "discount_pct": 25.0,
                "creative_quality_score": 4.0, "headline_clickbait_score": 9.0,
                "brand_familiarity": 0.3,
            },
            "Premium quality native": {
                "product_price": 120.0, "discount_pct": 5.0,
                "creative_quality_score": 9.0, "headline_clickbait_score": 2.0,
                "brand_familiarity": 0.85,
            },
            "Mid-range carousel": {
                "product_price": 65.0, "discount_pct": 12.0,
                "creative_quality_score": 6.5, "headline_clickbait_score": 5.0,
                "brand_familiarity": 0.55,
            },
        }

        results = []
        for name, defaults in presets.items():
            ad = {
                "category": "electronics",
                "ad_format": "native" if "native" in name else ("banner" if "banner" in name else "carousel"),
                **defaults,
            }
            row = build_row(u, ad, ctx)
            s = score_breakdown(model, row)
            results.append({
                "Ad": name,
                "P(click)": s["p_click"],
                "P(convert|click)": s["p_convert_given_click"],
                "E(revenue|convert)": s["e_revenue_given_convert"],
                "Expected revenue": s["expected_revenue"],
            })

        rank_df = pd.DataFrame(results).sort_values("Expected revenue", ascending=False)
        rank_df["Rank"] = np.arange(1, len(rank_df) + 1)
        rank_df = rank_df[["Rank", "Ad", "P(click)", "P(convert|click)", "E(revenue|convert)", "Expected revenue"]]

        display = rank_df.copy()
        display["P(click)"] = display["P(click)"].map(lambda x: f"{x:.1%}")
        display["P(convert|click)"] = display["P(convert|click)"].map(lambda x: f"{x:.1%}")
        display["E(revenue|convert)"] = display["E(revenue|convert)"].map(lambda x: f"£{x:.2f}")
        display["Expected revenue"] = display["Expected revenue"].map(lambda x: f"£{x:.4f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

        winner = rank_df.iloc[0]["Ad"]
        ctr_winner = rank_df.sort_values("P(click)", ascending=False).iloc[0]["Ad"]
        if winner != ctr_winner:
            st.warning(
                f"**Revenue winner:** {winner} — but **CTR winner:** {ctr_winner}. "
                "Optimizing clicks alone would pick the wrong ad."
            )
        else:
            st.success(f"**{winner}** wins on both expected revenue and CTR for this user.")


if __name__ == "__main__":
    main()

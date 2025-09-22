"""Streamlit dashboard for CRISI resilience analytics."""
from __future__ import annotations

import logging
from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="CRISI Resilience Explorer", layout="wide")
LOGGER = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "features_geo.parquet"


@st.cache_data(show_spinner=False)  # Streamlit caching docs: https://docs.streamlit.io/develop/api-reference/caching-and-state/
def load_features() -> gpd.GeoDataFrame:
    if not DATA_PATH.exists():
        st.error("Processed dataset not found. Run `python -m flows.etl` first.")
        st.stop()
    gdf = gpd.read_parquet(DATA_PATH)
    return gdf


def compute_zscore(series: pd.Series) -> pd.Series:
    if series.isna().all():
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / (series.std(ddof=0) or 1.0)


def impute_within_group(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = df.groupby(["scenario", "year"])[column].transform(
            lambda x: x.fillna(x.median())
        )
    return df


def build_resilience(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = impute_within_group(df, ["heat_tmax_delta", "heatwave_days", "tourism_gdp_share", "seasonality_idx", "readiness_proxy"])

    exposure = compute_zscore(df["heat_tmax_delta"]) + compute_zscore(df["heatwave_days"])
    sensitivity = compute_zscore(df["tourism_gdp_share"]) + compute_zscore(df["seasonality_idx"])
    capacity = compute_zscore(df["readiness_proxy"])

    df["resilience_score"] = 100 - (0.6 * exposure + 0.3 * sensitivity - 0.1 * capacity)
    return df


def make_deck(gdf: gpd.GeoDataFrame) -> pdk.Deck:
    geojson = json.loads(gdf.to_json())
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        opacity=0.5,
        stroked=True,
        filled=True,
        get_fill_color="[255 - resilience_score, 100, resilience_score]",
        get_line_color="[40, 40, 40]",
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=50, longitude=10, zoom=3.5)
    tooltip = {"html": "<b>{region_name}</b><br/>Score: {resilience_score:.1f}"}
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)


def main() -> None:
    st.title("CRISI Tourism & Climate Resilience")
    gdf = load_features()

    scenarios = sorted(gdf["scenario"].dropna().unique())
    years = sorted(gdf["year"].dropna().unique())

    col1, col2 = st.columns(2)
    scenario = col1.selectbox("Scenario", scenarios)
    year = col2.selectbox("Year", years)

    filtered = gdf[(gdf["scenario"] == scenario) & (gdf["year"] == year)].copy()
    if filtered.empty:
        st.warning("No rows for the selected filters.")
        st.stop()

    scored = build_resilience(filtered)
    scored = scored.sort_values("resilience_score", ascending=False)

    st.subheader("Regional ranking")
    st.dataframe(
        scored[["region_id", "region_name", "resilience_score", "heat_tmax_delta", "heatwave_days", "seasonality_idx"]]
        .round({"resilience_score": 2, "heat_tmax_delta": 2, "heatwave_days": 1, "seasonality_idx": 1})
        .reset_index(drop=True)
    )

    st.subheader("Map view")
    deck = make_deck(scored)
    st.pydeck_chart(deck)


if __name__ == "__main__":
    main()

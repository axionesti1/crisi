"""Streamlit app for the CRISI pilot.

This module implements a basic interactive dashboard allowing users to
select a climate scenario, view the resilience scores across regions,
and download the resulting scores as CSV. The map is rendered
using Streamlit's built‑in mapping functionality.

To run the app, navigate to the project root and execute::

    streamlit run app/app.py

Ensure that ``data_proc/features_geo.parquet`` exists (generated
via the data_prep script) and that ``config/indicators.yaml`` is
configured appropriately.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from scoring import compute_scores


def load_data(path: str) -> pd.DataFrame:
    """Load preprocessed features.

    For the pilot we support CSV and Parquet files. If the file
    extension is .csv we load with pandas; if it's .parquet we
    attempt to use pandas.read_parquet (which works without geopandas
    and shapely as long as geometry is not encoded). The resulting
    DataFrame should contain ``lat`` and ``lon`` columns for
    mapping.
    """
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    # Fallback to pandas.read_parquet for other formats
    return pd.read_parquet(p)


def main() -> None:
    st.set_page_config(layout="wide", page_title="CRISI Pilot")
    st.title("CRISI – Climate Resilience Investment Scoring (Pilot)")

    # Sidebar for scenario selection
    with st.sidebar:
        st.header("Options")
        data_path = st.text_input(
            "Features file", value="data_proc/features.csv", help="Path to preprocessed features CSV or Parquet."
        )
        # Load the data lazily
        gdf = load_data(data_path)
        scenarios = sorted(gdf["scenario"].unique())
        scenario = st.selectbox("Scenario", scenarios)

    # Filter by selected scenario
    df = gdf[gdf["scenario"] == scenario].reset_index(drop=True)

    # Compute scores
    scores = compute_scores(df)
    # Merge scores back onto original data
    gdf_scored = df.merge(scores, on=["region_id", "region_name", "scenario"], suffixes=("", "_score"))

    # Display top regions by resilience
    top = (
        gdf_scored.sort_values("resilience_score", ascending=False)[["region_name", "resilience_score"]].head(10)
    )
    st.subheader("Top regions by resilience score")
    st.dataframe(top.reset_index(drop=True))

    # Display map
    st.subheader("Resilience map")
    # Use lat/lon columns for mapping
    if not {"lat", "lon"}.issubset(gdf_scored.columns):
        st.warning("No latitude/longitude columns found for mapping.")
    else:
        map_df = gdf_scored[["lat", "lon"]].copy()
        st.map(map_df)

    # Download button
    st.download_button(
        label="Download scores as CSV",
        data=scores.to_csv(index=False).encode("utf-8"),
        file_name=f"crisi_scores_{scenario}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
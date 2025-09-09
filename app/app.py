import pandas as pd
import numpy as np
import yaml
import streamlit as st
from typing import Dict, List, Any
from sklearn.decomposition import PCA

st.set_page_config(page_title="CRISI: Multi-Scenario, Multi-Horizon Scorer", layout="wide")

# Cache data loading for efficiency
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file, allowing spaces after delimiters."""
    return pd.read_csv(path, skipinitialspace=True)


@st.cache_data
def load_cfg(path: str) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def min_max(series: pd.Series) -> pd.Series:
    """Normalize a series to the [0, 1] range using min-max scaling.

    If all values are equal, returns 0.5 for all entries.
    """
    s = series.astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def normalize_df(df: pd.DataFrame, benefit: Dict[str, bool], indicators: List[str]) -> pd.DataFrame:
    """Add normalized indicator columns to a DataFrame.

    Each indicator gets a corresponding column prefixed with ``norm__``.
    Indicators for which ``benefit`` is False are inverted so that lower raw values become higher normalized values.
    """
    out = df.copy()
    for ind in indicators:
        norm = min_max(out[ind])
        if not benefit.get(ind, True):
            norm = 1 - norm
        out[f"norm__{ind}"] = norm
    return out


def weight_vector(base_weights: Dict[str, float], override: Dict[str, float]) -> Dict[str, float]:
    """Combine base weights with optional overrides and normalize them to sum to 1."""
    w = base_weights.copy()
    if override:
        w.update(override)
    s = sum(w.values())
    if s != 0:
        for k in w:
            w[k] = w[k] / s
    return w


def apply_rcp(
    df: pd.DataFrame,
    scenario_cfg: Dict[str, Any],
    years: int,
    ui_mult: float = None,
) -> pd.DataFrame:
    """Project climate risk forward for RCP scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        The base DataFrame.
    scenario_cfg : Dict[str, Any]
        Scenario configuration from the YAML.
    years : int
        Number of years forward to project.
    ui_mult : float, optional
        UI override multiplier per 10 years. If provided, overrides the YAML multiplier.
    """
    out = df.copy()
    mult_10y = float(scenario_cfg.get("climate_multiplier_per_10y", 1.0))
    if ui_mult is not None:
        mult_10y = ui_mult
    factor = mult_10y ** (years / 10)
    if "climate_risk" in out.columns:
        out["climate_risk"] = out["climate_risk"] * factor
    return out


def apply_foresight(
    df: pd.DataFrame,
    reg: str,
    tech: str,
    years: int,
    REG: Dict[str, Dict[str, float]] = None,
    TECH: Dict[str, Dict[str, float]] = None,
) -> pd.DataFrame:
    """Project multiple indicators forward for foresight scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        The base DataFrame.
    reg : str
        Regulation type ('strict' or 'easy').
    tech : str
        Technology type ('strict' or 'easy').
    years : int
        Number of years forward to project.
    REG : Dict[str, Dict[str, float]], optional
        Per-10-year multipliers for regulation effects.
    TECH : Dict[str, Dict[str, float]], optional
        Per-10-year multipliers for technology effects.
    """
    out = df.copy()
    steps = years / 10

    def apply_mult(col: str, mult_10y: float) -> None:
        if col in out.columns:
            out[col] = out[col] * (mult_10y ** steps)

    # Apply regulation multipliers
    for col, m in REG.get(reg, {}).items():
        apply_mult(col, m)
    # Apply technology multipliers
    for col, m in TECH.get(tech, {}).items():
        apply_mult(col, m)
    return out


def score_once(df: pd.DataFrame, weights: Dict[str, float], indicators: List[str]) -> (pd.DataFrame, pd.Series):
    """Compute the composite score for each row and return contributions.

    Returns a DataFrame with a new 'score' column and the sum of contributions across indicators.
    """
    out = df.copy()
    s = np.zeros(len(df))
    contrib_cols = []
    for ind in indicators:
        cc = f"contrib__{ind}"
        out[cc] = out[f"norm__{ind}"] * weights[ind]
        s += out[cc].values
        contrib_cols.append(cc)
    out["score"] = s
    return out, out[contrib_cols].sum(axis=0)


def run_scenario(
    df_base: pd.DataFrame,
    cfg: Dict[str, Any],
    scenario_name: str,
    horizons: List[int],
    use_pca_weights: bool,
    rcp45_mult: float,
    rcp85_mult: float,
    reg_strict_cr: float,
    reg_strict_inc: float,
    reg_strict_un: float,
    reg_easy_cr: float,
    reg_easy_inc: float,
    reg_easy_un: float,
    tech_strict_infra: float,
    tech_strict_seas: float,
    tech_strict_un: float,
    tech_easy_infra: float,
    tech_easy_seas: float,
    tech_easy_un: float,
) -> pd.DataFrame:
    """Run a scenario across multiple horizons and return a tidy DataFrame."""
    scen = cfg["scenarios"][scenario_name]
    indicators = list(cfg["weights"].keys())
    benefit = cfg["benefit"]
    base_w = cfg["weights"]
    weights_override = scen.get("weights_override", {})
    w = weight_vector(base_w, weights_override)
    # Optionally compute PCA-based weights once per run
    if use_pca_weights:
        norm_base = normalize_df(df_base, benefit, indicators)
        X = norm_base[[f"norm__{i}" for i in indicators]].fillna(0.5)
        pca = PCA(n_components=1).fit(X)
        load = np.abs(pca.components_[0])
        load = load / load.sum()
        w = {ind: float(load[i]) for i, ind in enumerate(indicators)}
    rows = []
    for years in horizons:
        if scen["type"] == "baseline":
            df_proj = df_base.copy()
        elif scen["type"] == "rcp":
            ui = None
            if scenario_name == "rcp45":
                ui = rcp45_mult
            elif scenario_name == "rcp85":
                ui = rcp85_mult
            df_proj = apply_rcp(df_base, scen, years, ui_mult=ui)
        elif scen["type"] == "foresight":
            REG = {
                "strict": {"climate_risk": reg_strict_cr, "income_dependency": reg_strict_inc, "unemployment_rate": reg_strict_un},
                "easy":   {"climate_risk": reg_easy_cr,   "income_dependency": reg_easy_inc,   "unemployment_rate": reg_easy_un},
            }
            TECH = {
                "strict": {"infra_score": tech_strict_infra, "seasonality_index": tech_strict_seas, "unemployment_rate": tech_strict_un},
                "easy":   {"infra_score": tech_easy_infra,   "seasonality_index": tech_easy_seas,   "unemployment_rate": tech_easy_un},
            }
            df_proj = apply_foresight(df_base, scen["regulation"], scen["technology"], years, REG=REG, TECH=TECH)
        else:
            df_proj = df_base.copy()
        df_norm = normalize_df(df_proj, benefit, indicators)
        scored, _ = score_once(df_norm, w, indicators)
        keep = ["region", "nuts3_code", "lat", "lon", "score"] + [f"norm__{i}" for i in indicators] + [f"contrib__{i}" for i in indicators]
        scored = scored[keep]
        scored["scenario"] = scenario_name
        scored["horizon_years"] = years
        rows.append(scored)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Entry point for the Streamlit app."""
    st.title("CRISI: Multi-Scenario, Multi-Horizon Scorer")
    # Sidebar inputs
    with st.sidebar:
        st.header("Inputs")
        data_path = st.text_input("Data CSV path", value="data/regions_demo.csv")
        cfg = load_cfg("indicators.yaml")
        horizons_all = cfg.get("horizons", [5, 10, 15, 20, 25, 30])
        scenarios = list(cfg["scenarios"].keys())
        scen_sel = st.multiselect("Scenarios", options=scenarios, default=scenarios)
        horizons_sel = st.multiselect("Horizons (years)", options=horizons_all, default=horizons_all)
        topn = st.number_input("Top N regions (charts)", min_value=1, max_value=50, value=10, step=1)
        use_pca_weights = st.checkbox("Use PCA-derived weights (auto-learned)", value=False)
        # Dynamic knobs for scenario projections
        with st.expander("Adjust scenario dynamics (advanced)", expanded=False):
            st.caption("Per-10-year multipliers. <1 improves 'bad' indicators; >1 worsens.")
            st.markdown("**RCP climate multipliers (per 10y)**")
            rcp45_mult = st.slider("RCP45 climate risk × per 10y", 0.80, 1.10, 0.95, 0.01)
            rcp85_mult = st.slider("RCP85 climate risk × per 10y", 1.00, 1.50, 1.15, 0.01)
            st.markdown("**Regulation effects (per 10y)**")
            reg_strict_cr = st.slider("Strict: climate_risk ×", 0.80, 1.10, 0.92, 0.01)
            reg_strict_inc = st.slider("Strict: income_dependency ×", 0.80, 1.10, 0.94, 0.01)
            reg_strict_un = st.slider("Strict: unemployment_rate ×", 0.80, 1.10, 0.98, 0.01)
            reg_easy_cr = st.slider("Easy: climate_risk ×", 0.90, 1.20, 1.05, 0.01)
            reg_easy_inc = st.slider("Easy: income_dependency ×", 0.90, 1.10, 0.99, 0.01)
            reg_easy_un = st.slider("Easy: unemployment_rate ×", 0.80, 1.10, 0.96, 0.01)
            st.markdown("**Technology effects (per 10y)**")
            tech_strict_infra = st.slider("Strict tech: infra_score ×", 0.90, 1.30, 1.12, 0.01)
            tech_strict_seas = st.slider("Strict tech: seasonality ×", 0.80, 1.10, 0.94, 0.01)
            tech_strict_un = st.slider("Strict tech: unemployment ×", 0.80, 1.10, 0.92, 0.01)
            tech_easy_infra = st.slider("Easy tech: infra_score ×", 0.90, 1.20, 1.05, 0.01)
            tech_easy_seas = st.slider("Easy tech: seasonality ×", 0.90, 1.10, 0.98, 0.01)
            tech_easy_un = st.slider("Easy tech: unemployment ×", 0.80, 1.10, 0.96, 0.01)
    # Load data
    df = load_csv(data_path)
    if df is None or df.empty:
        st.warning("No data loaded. Check the file path.")
        return
    # Run selected scenarios and horizons
    all_scored = []
    for sn in scen_sel:
        all_scored.append(
            run_scenario(
                df,
                cfg,
                sn,
                horizons_sel,
                use_pca_weights,
                rcp45_mult,
                rcp85_mult,
                reg_strict_cr,
                reg_strict_inc,
                reg_strict_un,
                reg_easy_cr,
                reg_easy_inc,
                reg_easy_un,
                tech_strict_infra,
                tech_strict_seas,
                tech_strict_un,
                tech_easy_infra,
                tech_easy_seas,
                tech_easy_un,
            )
        )
    if not all_scored:
        st.stop()
    scored_long = pd.concat(all_scored, ignore_index=True)
    # Charts
    st.subheader("Scores over Time (line chart)")
    # Determine top regions from latest horizon of first selected scenario
    if scen_sel:
        ref = scored_long[scored_long["scenario"] == scen_sel[0]]
    else:
        ref = pd.DataFrame()
    if not ref.empty:
        latest = max(horizons_sel)
        top_regions = (
            ref[ref["horizon_years"] == latest]
            .nlargest(topn, "score")
            ["region"]
            .unique()
            .tolist()
        )
    else:
        latest = max(horizons_sel)
        top_regions = scored_long["region"].unique().tolist()
    line_df = scored_long[scored_long["region"].isin(top_regions)].copy()
    line_piv = line_df.pivot_table(
        index="horizon_years", columns=["region", "scenario"], values="score"
    )
    # Flatten the MultiIndex columns for Streamlit
    if isinstance(line_piv.columns, pd.MultiIndex):
        line_piv.columns = [f"{r} ({s})" for r, s in line_piv.columns.to_list()]
    st.line_chart(line_piv)
    # Bar chart: top N per scenario at latest horizon
    st.subheader("Top Regions (bar, latest horizon per scenario)")
    bar_scope = scored_long[scored_long["horizon_years"] == latest]
    bar_df = (
        bar_scope.sort_values(["scenario", "score"], ascending=[True, False])
        .groupby("scenario")
        .head(topn)
    )
    bar_piv = bar_df.pivot_table(index="region", columns="scenario", values="score").fillna(0)
    st.bar_chart(bar_piv)
    # Contributions area chart
    st.subheader("Indicator Contributions (stacked for one region)")
    indicators = list(cfg["weights"].keys())
    region_pick = st.selectbox(
        "Region", sorted(scored_long["region"].unique().tolist())
    )
    scen_for_contrib = st.selectbox(
        "Scenario (for contributions)", sorted(scored_long["scenario"].unique().tolist())
    )
    contrib_df = (
        scored_long[
            (scored_long["region"] == region_pick)
            & (scored_long["scenario"] == scen_for_contrib)
        ][["horizon_years"] + [f"contrib__{i}" for i in indicators]]
        .sort_values("horizon_years")
        .set_index("horizon_years")
    )
    contrib_df.columns = [c.replace("contrib__", "") for c in contrib_df.columns]
    st.area_chart(contrib_df)
    # Simple point map (fallback). A choropleth will be used if a GeoJSON is provided and GeoPandas is available.
    st.subheader("Map (points)")
    latest_scope = scored_long[
        (scored_long["scenario"] == scen_sel[0])
        & (scored_long["horizon_years"] == latest)
    ]
    if {"lat", "lon"}.issubset(latest_scope.columns):
        st.map(latest_scope.rename(columns={"lat": "latitude", "lon": "longitude"}))
    # Download sections
    st.subheader("Download results")
    long_cols = [
        "region",
        "nuts3_code",
        "scenario",
        "horizon_years",
        "score",
    ] + [c for c in scored_long.columns if c.startswith("norm__")] + [c for c in scored_long.columns if c.startswith("contrib__")]
    dl_long = scored_long[long_cols].sort_values(
        ["scenario", "horizon_years", "score"], ascending=[True, True, False]
    )
    st.download_button(
        "Download scores (tidy long CSV)",
        data=dl_long.to_csv(index=False).encode("utf-8"),
        file_name="scores_long.csv",
        mime="text/csv",
    )
    summary = (
        scored_long[scored_long["horizon_years"] == latest]
        .sort_values(["scenario", "score"], ascending=[True, False])
        .groupby("scenario")
        .head(topn)[["scenario", "region", "nuts3_code", "score"]]
    )
    st.download_button(
        "Download summary (latest horizon, top N per scenario)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="summary_latest_topN.csv",
        mime="text/csv",
    )
    # Additional ranking with buckets for the latest horizon
    latest_scope = scored_long[scored_long["horizon_years"] == latest].copy()
    latest_scope["rank_within_scenario"] = latest_scope.groupby("scenario")["score"].rank(
        ascending=False, method="min"
    )
    latest_scope["bucket_1_low_5_high"] = (
        (latest_scope["score"] * 5).clip(0, 4).astype(int) + 1
    )
    st.download_button(
        "Download ranked (latest horizon, all regions + buckets)",
        data=(
            latest_scope.sort_values(
                ["scenario", "rank_within_scenario"]
            )[
                [
                    "scenario",
                    "region",
                    "nuts3_code",
                    "score",
                    "rank_within_scenario",
                    "bucket_1_low_5_high",
                ]
            ].to_csv(index=False).encode("utf-8")
        ),
        file_name="ranked_latest.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

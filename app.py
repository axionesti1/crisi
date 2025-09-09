import pandas as pd
import numpy as np
import yaml
import streamlit as st
from typing import Dict, Any, List

st.set_page_config(page_title="CRISI: Multi-Scenario, Multi-Horizon Scorer (with Social Drivers)", layout="wide")

# ---------------------------- Data Loaders ----------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with tolerance for spaces after commas."""
    return pd.read_csv(path, skipinitialspace=True)

@st.cache_data
def load_cfg(yaml_path: str) -> Dict[str, Any]:
    """Load YAML configuration for indicators and scenarios."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------- Normalisation Utilities ----------------------------

def compute_baseline_bounds(df: pd.DataFrame, indicators: List[str]) -> Dict[str, tuple]:
    """
    Compute min and max values for each indicator using the baseline (year 0) data.
    These bounds are reused for all future horizons to preserve changes over time.
    """
    bounds = {}
    for col in indicators:
        s = pd.to_numeric(df[col], errors="coerce")
        lo = float(s.min())
        hi = float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            lo, hi = 0.0, 1.0  # safe defaults if data is missing or constant
        bounds[col] = (lo, hi)
    return bounds


def normalize_with_bounds(df: pd.DataFrame, benefit: Dict[str, bool], indicators: List[str], bounds: Dict[str, tuple]) -> pd.DataFrame:
    """
    Normalise each indicator using fixed baseline bounds.  Values outside [lo, hi]
    are clipped to 0–1.  Invert indicators where benefit is False.
    """
    out = df.copy()
    for col in indicators:
        lo, hi = bounds[col]
        s = pd.to_numeric(out[col], errors="coerce")
        # avoid divide-by-zero; assign 0.5 if hi==lo
        if hi == lo:
            z = pd.Series([0.5] * len(s), index=s.index)
        else:
            z = (s - lo) / (hi - lo)
            z = z.clip(0.0, 1.0)
        # invert if benefit is false
        if not benefit.get(col, True):
            z = 1.0 - z
        out[f"norm__{col}"] = z
    return out


def weight_vector(base_w: Dict[str, float]) -> Dict[str, float]:
    """Normalise weights so they sum to one."""
    w = {k: float(v) for k, v in base_w.items()}
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}

# ---------------------------- Projection Dynamics ----------------------------

def logistic_adoption(t: float, t50: float = 10.0, k: float = 0.35) -> float:
    """Simple logistic adoption curve from 0 to 1."""
    return 1.0 / (1.0 + np.exp(-k * (t - t50)))


def apply_rcp(df: pd.DataFrame, scen_cfg: Dict[str, Any], years: float, ui_mult: float = None) -> pd.DataFrame:
    """
    Forward projection for RCP scenarios.  Modify climate_risk using a per-10-year
    multiplier, with region-specific elasticity based on coastal_exposure and
    heat_stress_exposure.  A UI override (ui_mult) can replace the scenario
    multiplier.
    """
    out = df.copy()
    mult_10y = float(scen_cfg.get("climate_multiplier_per_10y", 1.0))
    if ui_mult is not None:
        mult_10y = ui_mult
    steps = years / 10.0
    # region-specific elasticity: weighted average of exposures
    elast = 0.5
    if "coastal_exposure" in out.columns or "heat_stress_exposure" in out.columns:
        ce = out.get("coastal_exposure", 0.5)
        he = out.get("heat_stress_exposure", 0.5)
        elast = ce * 0.6 + he * 0.4
    factor = (mult_10y ** steps) ** (0.5 + elast)
    if "climate_risk" in out.columns:
        out["climate_risk"] = out["climate_risk"] * factor
    return out


def apply_foresight(df: pd.DataFrame, reg: str, tech: str, years: float,
                    REG: Dict[str, Dict[str, float]], TECH: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Forward projection for foresight scenarios.  Effects from regulation and
    technology follow logistic adoption curves.  Multipliers are applied to
    relevant columns based on per-10-year multipliers provided in REG and TECH.
    Additionally, infrastructure score converges towards a ceiling based on
    technology adoption.
    """
    out = df.copy()
    t = float(years)
    # adoption levels for regulation and technology
    reg_level = logistic_adoption(t, t50=12.0, k=0.25)
    tech_level = logistic_adoption(t, t50=8.0, k=0.45)
    # apply regulation multipliers
    for col, m10 in REG.get(reg, {}).items():
        if col in out.columns:
            exponent = (t / 10.0) * (0.8 + reg_level * 0.4)
            out[col] = out[col] * (m10 ** exponent)
    # apply technology multipliers
    for col, m10 in TECH.get(tech, {}).items():
        if col in out.columns:
            exponent = (t / 10.0) * (0.8 + tech_level * 0.4)
            out[col] = out[col] * (m10 ** exponent)
    # infrastructure score approaches a ceiling (1.0)
    if "infra_score" in out.columns:
        ceiling = 1.0
        out["infra_score"] = out["infra_score"] + (ceiling - out["infra_score"]) * 0.35 * tech_level
    return out


def apply_social_drivers(df: pd.DataFrame, years: float) -> pd.DataFrame:
    """
    Adjust arrivals and seasonality based on social media signals.  Higher
    sentiment and engagement drive growth in arrivals; more balanced buzz
    reduces seasonality.  Topic spikes (heat/fire) reduce arrivals and
    increase climate risk.
    """
    out = df.copy()
    # Arrivals growth tied to sentiment and engagement
    if {"sentiment_mean", "engagement_rate"}.issubset(out.columns):
        growth = 0.02 + 0.05 * (out["sentiment_mean"] - 0.5) + 0.03 * out["engagement_rate"]
        out["arrivals_per_capita"] = out["arrivals_per_capita"] * (1.0 + growth) ** (years / 1.0)
    # Seasonality flattening tied to buzz seasonality and influencer share
    if "buzz_seasonality" in out.columns:
        flatten = 0.10 * (0.5 - out["buzz_seasonality"]) + 0.08 * out.get("influencer_share", 0.0)
        out["seasonality_index"] = out["seasonality_index"] * (1.0 - np.clip(flatten, -0.2, 0.2)) ** (years / 5.0)
    # Shock effects based on heat/fire topics
    if "topic_share_heat" in out.columns or "topic_share_fire" in out.columns:
        shock = out.get("topic_share_heat", 0.0) * 0.3 + out.get("topic_share_fire", 0.0) * 0.5
        out["arrivals_per_capita"] = out["arrivals_per_capita"] * (1.0 - np.clip(shock, 0.0, 0.25)) ** (years / 2.0)
        if "climate_risk" in out.columns:
            out["climate_risk"] = out["climate_risk"] * (1.0 + np.clip(shock, 0.0, 0.4)) ** (years / 2.0)
    return out

# ---------------------------- Main Application ----------------------------

def main():
    st.title("CRISI: Multi-Scenario, Multi-Horizon Scorer (with Social Drivers)")

    # Sidebar inputs
    with st.sidebar:
        st.header("Inputs")
        data_path = st.text_input("Data CSV path", value="data/regions_demo.csv")
        cfg_path = st.text_input("Indicators YAML", value="indicators.yaml")
        topn = st.number_input("Top N regions (charts)", min_value=1, max_value=30, value=5, step=1)

        cfg = load_cfg(cfg_path)
        horizons_all = cfg.get("horizons", [5, 10, 15, 20, 25, 30])
        scenarios_all = list(cfg["scenarios"].keys())
        scen_sel = st.multiselect("Scenarios", options=scenarios_all, default=scenarios_all[:3])
        horizons_sel = st.multiselect("Horizons (years)", options=horizons_all, default=horizons_all)

        # Dynamic UI knobs for RCP and foresight multipliers
        with st.expander("Adjust scenario dynamics (advanced)", expanded=False):
            st.caption("Per-10-year multipliers.  <1 improves 'bad' indicators; >1 worsens.")
            st.markdown("**RCP climate multipliers (per 10y)**")
            rcp45_mult = st.slider("RCP45 climate risk × per 10y", 0.80, 1.10, 0.95, 0.01)
            rcp85_mult = st.slider("RCP85 climate risk × per 10y", 1.00, 1.50, 1.15, 0.01)

            st.markdown("**Regulation effects (per 10y)**")
            reg_strict_cr  = st.slider("Strict: climate_risk ×",        0.80, 1.10, 0.92, 0.01)
            reg_strict_inc = st.slider("Strict: income_dependency ×",    0.80, 1.10, 0.94, 0.01)
            reg_strict_un  = st.slider("Strict: unemployment_rate ×",    0.80, 1.10, 0.98, 0.01)
            reg_easy_cr    = st.slider("Easy: climate_risk ×",           0.90, 1.20, 1.05, 0.01)
            reg_easy_inc   = st.slider("Easy: income_dependency ×",      0.90, 1.10, 0.99, 0.01)
            reg_easy_un    = st.slider("Easy: unemployment_rate ×",      0.80, 1.10, 0.96, 0.01)

            st.markdown("**Technology effects (per 10y)**")
            tech_strict_infra = st.slider("Strict tech: infra_score ×",  0.90, 1.30, 1.12, 0.01)
            tech_strict_seas  = st.slider("Strict tech: seasonality ×",  0.80, 1.10, 0.94, 0.01)
            tech_strict_un    = st.slider("Strict tech: unemployment ×", 0.80, 1.10, 0.92, 0.01)
            tech_easy_infra   = st.slider("Easy tech: infra_score ×",    0.90, 1.20, 1.05, 0.01)
            tech_easy_seas    = st.slider("Easy tech: seasonality ×",    0.90, 1.10, 0.98, 0.01)
            tech_easy_un      = st.slider("Easy tech: unemployment ×",   0.80, 1.10, 0.96, 0.01)

        use_pca_weights = st.checkbox("Use PCA-derived weights (auto-learned)", value=False)

    # Load dataset and configuration
    df = load_csv(data_path)
    # ensure region and nuts3_code exist
    df = df.dropna(subset=["region", "nuts3_code"]).reset_index(drop=True)
    cfg = load_cfg(cfg_path)
    indicators = list(cfg["weights"].keys())
    benefit = cfg["benefit"]

    # baseline bounds for normalisation
    BASE_BOUNDS = compute_baseline_bounds(df, indicators)

    # Build regulation and technology multiplier dictionaries from sliders
    REG = {
        "strict": {"climate_risk": reg_strict_cr, "income_dependency": reg_strict_inc, "unemployment_rate": reg_strict_un},
        "easy":   {"climate_risk": reg_easy_cr,   "income_dependency": reg_easy_inc,  "unemployment_rate": reg_easy_un},
    }
    TECH = {
        "strict": {"infra_score": tech_strict_infra, "seasonality_index": tech_strict_seas, "unemployment_rate": tech_strict_un},
        "easy":   {"infra_score": tech_easy_infra,   "seasonality_index": tech_easy_seas,  "unemployment_rate": tech_easy_un},
    }

    # Collect scored data
    all_rows = []
    for scen_name in scen_sel:
        scen_cfg = cfg["scenarios"][scen_name]
        for y in horizons_sel:
            years = float(y)
            df_proj = df.copy()
            # baseline: only social drivers
            if scen_cfg["type"] == "baseline":
                df_proj = apply_social_drivers(df_proj, years)
            elif scen_cfg["type"] == "rcp":
                ui_mult = None
                if scen_name == "rcp45":
                    ui_mult = rcp45_mult
                elif scen_name == "rcp85":
                    ui_mult = rcp85_mult
                df_proj = apply_rcp(df_proj, scen_cfg, years, ui_mult=ui_mult)
                df_proj = apply_social_drivers(df_proj, years)
            elif scen_cfg["type"] == "foresight":
                df_proj = apply_foresight(df_proj, scen_cfg["regulation"], scen_cfg["technology"], years, REG=REG, TECH=TECH)
                df_proj = apply_social_drivers(df_proj, years)
            else:
                df_proj = df.copy()
            # Normalise using baseline bounds
            norm = normalize_with_bounds(df_proj, benefit, indicators, BASE_BOUNDS)
            # Determine weights (base plus any overrides)
            base_w = cfg.get("weights", {})
            over = scen_cfg.get("weights_override", {})
            w_dict = base_w.copy()
            w_dict.update(over)
            weights = weight_vector(w_dict)
            # Possibly override with PCA weights
            if use_pca_weights:
                from sklearn.decomposition import PCA
                X = norm[[f"norm__{ind}" for ind in indicators]].fillna(0.5)
                pca = PCA(n_components=1).fit(X)
                load = np.abs(pca.components_[0])
                load = load / load.sum()
                weights = {ind: float(load[i]) for i, ind in enumerate(indicators)}
            # Compute contributions and total score
            norm["score"] = 0.0
            for ind in indicators:
                contrib_col = f"contrib__{ind}"
                norm[contrib_col] = weights[ind] * norm[f"norm__{ind}"]
                norm["score"] += norm[contrib_col]
            norm["scenario"] = scen_name
            norm["horizon_years"] = years
            all_rows.append(norm)

    scored_long = pd.concat(all_rows, ignore_index=True)

    # --------------------- Charts and Outputs ---------------------
    if scored_long.empty:
        st.warning("No data to display. Check your selections.")
        return

    # Line chart over time for top regions
    st.subheader("Scores over Time (line chart)")
    first_scen = scen_sel[0] if scen_sel else None
    latest_horiz = max(horizons_sel) if horizons_sel else max(cfg.get("horizons", [5, 10, 15, 20, 25, 30]))
    ref = scored_long[(scored_long["scenario"] == first_scen) & (scored_long["horizon_years"] == latest_horiz)]
    top_regions = ref.nlargest(int(topn), "score")["region"].unique().tolist()
    line_df = scored_long[scored_long["region"].isin(top_regions)].copy()
    line_df = line_df.pivot_table(index=["region", "scenario", "horizon_years"], values="score").reset_index()
    line_df = line_df.sort_values(["region", "scenario", "horizon_years"])
    chart_df = line_df.pivot_table(index="horizon_years", columns=["region", "scenario"], values="score")
    if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = [f"{r} ({s})" for r, s in chart_df.columns.to_list()]
    st.line_chart(chart_df)

    # Bar chart for top regions at latest horizon
    st.subheader("Top N at latest horizon (bar)")
    bar_df = scored_long[scored_long["horizon_years"] == latest_horiz]
    bar_df = bar_df[bar_df["region"].isin(top_regions)]
    bar_chart_df = bar_df.pivot_table(index="region", columns="scenario", values="score")
    st.bar_chart(bar_chart_df)

    # Contributions chart (single region and scenario)
    st.subheader("Indicator Contributions (stacked for one region)")
    region_pick = st.selectbox("Region", sorted(scored_long["region"].unique().tolist()))
    scen_for_contrib = st.selectbox("Scenario (for contributions)", sorted(scored_long["scenario"].unique().tolist()))
    contrib_df = scored_long[(scored_long["region"] == region_pick) & (scored_long["scenario"] == scen_for_contrib)]
    contrib_df = contrib_df[["horizon_years"] + [c for c in contrib_df.columns if c.startswith("contrib__")]]
    contrib_df = contrib_df.sort_values("horizon_years").set_index("horizon_years")
    contrib_df.columns = [c.replace("contrib__", "") for c in contrib_df.columns]
    st.area_chart(contrib_df)

    # Downloads
    st.subheader("Download results")
    st.download_button(
        "Download scores (tidy long CSV)",
        data=scored_long.to_csv(index=False).encode("utf-8"),
        file_name="scores_long.csv",
        mime="text/csv"
    )
    latest_scope = scored_long[scored_long["horizon_years"] == latest_horiz].copy()
    latest_scope["rank_within_scenario"] = latest_scope.groupby("scenario")["score"].rank(ascending=False, method="min")
    latest_scope["bucket_1_low_5_high"] = (latest_scope["score"] * 5).clip(0, 4).astype(int) + 1
    st.download_button(
        "Download ranked (latest horizon, all regions + buckets)",
        data=(latest_scope
              .sort_values(["scenario", "rank_within_scenario"])
              [["scenario", "region", "nuts3_code", "score", "rank_within_scenario", "bucket_1_low_5_high"]]
              .to_csv(index=False).encode("utf-8")),
        file_name="ranked_latest.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

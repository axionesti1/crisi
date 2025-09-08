import pandas as pd
import numpy as np
import yaml
import streamlit as st
from typing import Dict, List, Tuple, Any

st.set_page_config(page_title="CRISI Scorer", layout="wide")

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_cfg(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def min_max(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)

def normalize_df(df: pd.DataFrame, benefit: Dict[str, bool], indicators: List[str]) -> pd.DataFrame:
    out = df.copy()
    for ind in indicators:
        norm = min_max(out[ind])
        if not benefit.get(ind, True):
            norm = 1 - norm
        out[f"norm__{ind}"] = norm
    return out

def weight_vector(base_w: Dict[str, float], override: Dict[str, float]) -> Dict[str, float]:
    w = base_w.copy()
    w.update(override or {})
    s = sum(w.values())
    if s == 0:
        return w
    return {k: v/s for k, v in w.items()}

def years_to_steps(years: int, step: int = 10) -> float:
    return years / step

def apply_rcp(df: pd.DataFrame, scenario_cfg: Dict[str, Any], years: int) -> pd.DataFrame:
    out = df.copy()
    mult_10y = float(scenario_cfg.get("climate_multiplier_per_10y", 1.0))
    steps = years_to_steps(years)
    factor = mult_10y ** steps
    if "climate_risk" in out.columns:
        out["climate_risk"] = out["climate_risk"] * factor
    return out

def apply_foresight(df: pd.DataFrame, reg: str, tech: str, years: int) -> pd.DataFrame:
    out = df.copy()
    steps = years_to_steps(years)

    REG = {
        "strict": {"climate_risk": 0.92, "income_dependency": 0.94, "unemployment_rate": 0.98},
        "easy":   {"climate_risk": 1.05, "income_dependency": 0.99, "unemployment_rate": 0.96},
    }
    TECH = {
        "strict": {"infra_score": 1.12, "seasonality_index": 0.94, "unemployment_rate": 0.92},
        "easy":   {"infra_score": 1.05, "seasonality_index": 0.98, "unemployment_rate": 0.96},
    }

    def apply_mult(col: str, mult_10y: float):
        if col in out.columns:
            out[col] = out[col] * (mult_10y ** steps)

    for col, m in REG.get(reg, {}).items():
        apply_mult(col, m)
    for col, m in TECH.get(tech, {}).items():
        apply_mult(col, m)

    return out

def score_once(df: pd.DataFrame, weights: Dict[str, float], indicators: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    contrib_cols = []
    s = np.zeros(len(df))
    out = df.copy()
    for ind in indicators:
        cc = f"contrib__{ind}"
        out[cc] = out[f"norm__{ind}"] * weights[ind]
        contrib_cols.append(cc)
        s += out[cc].values
    out["score"] = s
    return out, out[contrib_cols].sum(axis=0)

def run_scenario(df_base: pd.DataFrame, cfg: Dict[str, Any], scenario_name: str, years_list: List[int]) -> pd.DataFrame:
    indicators = list(cfg["weights"].keys())
    benefit = cfg["benefit"]
    scen = cfg["scenarios"][scenario_name]
    base_w = cfg["weights"]
    w = weight_vector(base_w, scen.get("weights_override", {}))

    rows = []
    for y in years_list:
        if scen["type"] == "baseline":
            df_proj = df_base.copy()
        elif scen["type"] == "rcp":
            df_proj = apply_rcp(df_base, scen, y)
        elif scen["type"] == "foresight":
            df_proj = apply_foresight(df_base, scen["regulation"], scen["technology"], y)
        else:
            df_proj = df_base.copy()

        df_norm = normalize_df(df_proj, benefit, indicators)
        scored, _ = score_once(df_norm, w, indicators)
        keep = ["region", "nuts3_code", "lat", "lon", "score"] \
               + [f"norm__{i}" for i in indicators] \
               + [f"contrib__{i}" for i in indicators]
        scored = scored[keep]
        scored["scenario"] = scenario_name
        scored["horizon_years"] = y
        rows.append(scored)

    return pd.concat(rows, ignore_index=True)

st.title("CRISI: Multi-Scenario, Multi-Horizon Scorer")

with st.sidebar:
    st.header("Inputs")
    data_path = st.text_input("Data CSV path", value="data/regions_sample.csv")
    cfg = load_cfg("indicators.yaml")
    horizons_all = cfg.get("horizons", [5,10,15,20,25,30])

    scenarios = list(cfg["scenarios"].keys())
    scen_sel = st.multiselect("Scenarios", options=scenarios, default=["baseline", "rcp45", "rcp85"])
    horizons_sel = st.multiselect("Horizons (years)", options=horizons_all, default=horizons_all)

    topn = st.number_input("Top N regions (charts)", min_value=1, max_value=50, value=10, step=1)

df = load_csv(data_path)

all_scored = []
for sn in scen_sel:
    all_scored.append(run_scenario(df, cfg, sn, horizons_sel))
if len(all_scored) == 0:
    st.stop()
scored_long = pd.concat(all_scored, ignore_index=True)

st.subheader("Scores over Time (line chart)")
ref = scored_long[scored_long["scenario"] == scen_sel[0]]
latest = max(horizons_sel)
top_regions = ref[ref["horizon_years"] == latest].nlargest(topn, "score")["region"].unique().tolist()

line_df = scored_long[scored_long["region"].isin(top_regions)].copy()
line_df = line_df.pivot_table(index=["region","scenario","horizon_years"], values="score").reset_index()
line_df = line_df.sort_values(["region","scenario","horizon_years"])
st.line_chart(
    line_df.pivot_table(index="horizon_years", columns=["region","scenario"], values="score")
)

st.subheader("Top Regions (bar, latest horizon per scenario)")
bar_scope = scored_long[scored_long["horizon_years"] == latest]
bar_df = (bar_scope.sort_values(["scenario","score"], ascending=[True, False])
                  .groupby("scenario")
                  .head(topn))
st.bar_chart(
    bar_df.pivot_table(index="region", columns="scenario", values="score").fillna(0)
)

st.subheader("Indicator Contributions (stacked for one region)")
indicators = list(cfg["weights"].keys())
region_pick = st.selectbox("Region", sorted(scored_long["region"].unique().tolist()))
contrib_df = (scored_long[scored_long["region"] == region_pick]
              [["scenario","horizon_years"] + [f"contrib__{i}" for i in indicators]]
              .sort_values(["scenario","horizon_years"]))
st.area_chart(
    contrib_df.set_index(["horizon_years","scenario"])[[f"contrib__{i}" for i in indicators]]
)

st.subheader("Map")
st.markdown("If you add `data/greece_nuts3.geojson`, we’ll draw a choropleth. Otherwise, you’ll see points.")
geojson_path = "data/greece_nuts3.geojson"
try:
    import geopandas as gpd
    import json
    import pydeck as pdk

    gdf = gpd.read_file(geojson_path)
    nuts_col = None
    for c in ["NUTS_ID", "nuts_id", "nuts3_code", "id", "CODE", "code"]:
        if c in gdf.columns:
            nuts_col = c
            break
    if nuts_col is None:
        st.warning("GeoJSON loaded, but can't find a NUTS-3 code column. Add one (e.g., NUTS_ID) to match `nuts3_code`.")
    else:
        map_df = scored_long[(scored_long["scenario"] == scen_sel[0]) & (scored_long["horizon_years"] == latest)]
        merged = gdf.merge(map_df, left_on=nuts_col, right_on="nuts3_code", how="left")
        merged["bucket"] = (merged["score"] * 5).fillna(0).clip(0,4).astype(int)
        colors = [[240,240,240],[198,219,239],[158,202,225],[107,174,214],[49,130,189]]
        merged["color"] = merged["bucket"].apply(lambda b: colors[int(b)])

        layer = pdk.Layer(
            "GeoJsonLayer",
            data=json.loads(merged.to_json()),
            get_fill_color="properties.color",
            get_line_color=[80,80,80],
            line_width_min_pixels=0.5,
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(latitude=38.5, longitude=24.0, zoom=5)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{region}\nScore: {score}"}))
except Exception as e:
    st.info("Choropleth not available (missing GeoJSON or GeoPandas); showing points.")
    latest_scope = scored_long[(scored_long["scenario"] == scen_sel[0]) & (scored_long["horizon_years"] == latest)]
    if {"lat","lon"}.issubset(latest_scope.columns):
        st.map(latest_scope.rename(columns={"lat":"latitude","lon":"longitude"}))

st.subheader("Download results")
long_cols = ["region","nuts3_code","scenario","horizon_years","score"] \
            + [c for c in scored_long.columns if c.startswith("norm__")] \
            + [c for c in scored_long.columns if c.startswith("contrib__")]
dl_long = scored_long[long_cols].sort_values(["scenario","horizon_years","score"], ascending=[True, True, False])

st.download_button(
    "Download scores (tidy long CSV)",
    data=dl_long.to_csv(index=False).encode("utf-8"),
    file_name="scores_long.csv",
    mime="text/csv"
)

summary = (scored_long[scored_long["horizon_years"] == latest]
           .sort_values(["scenario","score"], ascending=[True, False])
           .groupby("scenario")
           .head(topn)[["scenario","region","nuts3_code","score"]])
st.download_button(
    "Download summary (latest horizon, top N per scenario)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="summary_latest_topN.csv",
    mime="text/csv"
)

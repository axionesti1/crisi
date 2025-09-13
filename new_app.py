import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import yaml
from pathlib import Path
from typing import Dict
import sys

# API and ML libraries
import cdsapi
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Explainability
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# Add src/ to path for scoring module
sys.path.append("src")
from scoring import compute_scores

# Load indicator configuration (Delphi weights and scenarios)
config = yaml.safe_load(open("config/indicators.yaml", "r"))
scenario_configs = config.get("scenarios", {})

# Default per-10-year multipliers for foresight scenarios
REG = {
    "strict": {
        "climate_risk": 0.92,
        "income_dependency": 0.94,
        "unemployment_rate": 0.98,
    },
    "easy": {
        "climate_risk": 1.05,
        "income_dependency": 0.99,
        "unemployment_rate": 0.96,
    },
}

TECH = {
    "strict": {
        "infra_score": 1.12,
        "seasonality_index": 0.94,
        "unemployment_rate": 0.92,
    },
    "easy": {
        "infra_score": 1.05,
        "seasonality_index": 0.98,
        "unemployment_rate": 0.96,
    },
}


def apply_foresight(
    df: pd.DataFrame,
    reg: str,
    tech: str,
    years: int,
    REG: Dict[str, Dict[str, float]] = None,
    TECH: Dict[str, Dict[str, float]] = None,
) -> pd.DataFrame:
    """Project multiple indicators forward for foresight scenarios."""
    out = df.copy()
    steps = years / 10

    def apply_mult(col: str, mult_10y: float) -> None:
        if col in out.columns:
            out[col] = out[col] * (mult_10y ** steps)

    for col, m in (REG or {}).get(reg, {}).items():
        apply_mult(col, m)
    for col, m in (TECH or {}).get(tech, {}).items():
        apply_mult(col, m)
    return out

# Streamlit page config
st.set_page_config(page_title="CRISI Tourism Resilience Dashboard", layout="wide")
st.title("🌍 CRISI – Tourism-Climate Resilience Dashboard")
st.write("This app links climate hazards with tourism economics to assess the resilience of destinations under climate change.")

# Sidebar: scenario and year selection
with st.sidebar:
    st.header("Scenario and Data Options")
    # Prepare scenario display names (capitalize for UI)
    scenario_names = []
    scenario_key_map = {}
    for key, sc in scenario_configs.items():
        stype = sc.get("type", "").lower()
        if stype in ("baseline", "rcp", "foresight"):
            if stype == "rcp":
                name = key.upper()
                if key == "rcp45":
                    name = "RCP 4.5"
                elif key == "rcp85":
                    name = "RCP 8.5"
            elif stype == "baseline":
                name = "Baseline"
            elif stype == "foresight":
                reg = sc.get("regulation", "").capitalize()
                tech = sc.get("technology", "").capitalize()
                name = f"{reg} / {tech}"
            scenario_names.append(name)
            scenario_key_map[name] = key
    if not scenario_names:
        st.error("No valid scenarios found in configuration.")
        st.stop()
    scenario_display = st.selectbox("Climate Scenario", scenario_names, index=0)
    scenario_key = scenario_key_map[scenario_display]
    year = st.slider("Projection Year", 2020, 2100, 2030, step=5)
    st.write(f"Selected scenario: **{scenario_display}**, Year: **{year}**")

@st.cache_data(show_spinner=True)
def load_eurostat_data(year=2019):
    """
    Fetch tourism nights and regional GDP from Eurostat (SDMX-JSON API).
    Returns DataFrame with columns [region_code, tourist_nights, gdp_meur].
    """
    import pandas as pd
    BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
    def fetch_jsonstat(dataset, params):
        url = BASE + dataset
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    def jsonstat_to_df(js):
        dim_ids = js.get("id") or list(js["dimension"].keys())
        cats = {d: js["dimension"][d]["category"] for d in dim_ids}
        codes_by_dim = {d: [code for code, _idx in sorted(cats[d]["index"].items(), key=lambda kv: kv[1])] for d in dim_ids}
        sizes = [len(codes_by_dim[d]) for d in dim_ids]
        vals = js["value"]
        recs = []
        def decode(flat_idx):
            idxs = []
            n = flat_idx
            for size in reversed(sizes):
                idxs.append(n % size)
                n //= size
            idxs.reverse()
            rec = {}
            for d, i in zip(dim_ids, idxs):
                rec[d] = codes_by_dim[d][i]
            return rec
        if isinstance(vals, list):
            for i, v in enumerate(vals):
                if v is None: 
                    continue
                rec = decode(i)
                rec["value"] = v
                recs.append(rec)
        else:
            for k, v in vals.items():
                if v is None: 
                    continue
                rec = decode(int(k))
                rec["value"] = v
                recs.append(rec)
        df = pd.DataFrame.from_records(recs)
        if "geo" in df.columns:
            df = df.rename(columns={"geo": "region_code"})
        if "time" in df.columns:
            df = df.rename(columns={"time": "year"})
            with pd.option_context("mode.chained_assignment", None):
                df["year"] = pd.to_numeric(df["year"], errors="ignore")
        return df

    # Fetch nights at tourist accommodations (NUTS2)
    nights_js = fetch_jsonstat("tour_occ_nin2", {"time": str(year)})
    df_n = jsonstat_to_df(nights_js)
    # Collapse extra dimensions (prefer TOTAL or first)
    if "geo" in df_n.columns or "region_code" in df_n.columns:
        df_n = df_n.rename(columns={"geo": "region_code"})
    extra_cols = [c for c in df_n.columns if c not in {"region_code", "year", "value"}]
    for c in extra_cols:
        if df_n[c].isin(["TOTAL", "TOT", "ALL"]).any():
            df_n = df_n[df_n[c].isin(["TOTAL", "TOT", "ALL"])]
        else:
            df_n = df_n[df_n[c] == df_n[c].iloc[0]]
    df_n = df_n.rename(columns={"value": "tourist_nights"})
    df_n = df_n[["region_code", "tourist_nights"]]
    # Filter only NUTS2 codes (length >= 4)
    df_n = df_n[df_n["region_code"].str.len() >= 4]

    # Fetch regional GDP (NUTS2)
    gdp_js = fetch_jsonstat("nama_10r_2gdp", {"time": str(year)})
    df_g = jsonstat_to_df(gdp_js)
    if "geo" in df_g.columns:
        df_g = df_g.rename(columns={"geo": "region_code"})
    # Filter to main GDP concept & unit if present
    for col, desired in (("na_item", {"B1GQ"}), ("unit", {"MIO_EUR", "CP_MEUR", "MIO_NAC"})):
        if col in df_g.columns:
            if df_g[col].isin(desired).any():
                df_g = df_g[df_g[col].isin(desired)]
            else:
                df_g = df_g[df_g[col] == df_g[col].iloc[0]]
    df_g = df_g.rename(columns={"value": "gdp_meur"})
    df_g = df_g[df_g["region_code"].str.len() >= 4]
    df_g = df_g[["region_code", "gdp_meur"]]

    # Merge nights and GDP
    out = pd.merge(df_n, df_g, on="region_code", how="inner")
    out["tourist_nights"] = pd.to_numeric(out["tourist_nights"], errors="coerce")
    out["gdp_meur"] = pd.to_numeric(out["gdp_meur"], errors="coerce")
    out = out.dropna(subset=["tourist_nights", "gdp_meur"]).reset_index(drop=True)
    return out

@st.cache_data(show_spinner=True)
def load_worldbank_data(year: int, country_codes_iso2: list):
    """
    Fetch World Bank population and GDP per capita for given ISO2 country codes.
    Returns dict: {country_iso2: {"population": ..., "gdp_per_capita": ...}, ...}
    """
    indicators = {"SP.POP.TOTL": "population", "NY.GDP.PCAP.CD": "gdp_per_capita"}
    country_param = ";".join(sorted(set(cc.upper() for cc in country_codes_iso2)))
    base = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
    params = {"date": str(year), "format": "json", "per_page": "20000"}
    store = {}
    for ind_code, out_name in indicators.items():
        url = base.format(countries=country_param, indicator=ind_code)
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            continue
        js = r.json()
        if not isinstance(js, list) or len(js) < 2 or js[1] is None:
            continue
        for entry in js[1]:
            iso2 = entry.get("country", {}).get("id", None)
            if not iso2:
                continue
            val = entry.get("value", None)
            try:
                num = float(val) if val is not None else np.nan
            except:
                num = np.nan
            if iso2 not in store:
                store[iso2] = {}
            store[iso2][out_name] = num
    return store

@st.cache_data(show_spinner=True)
def simulate_open_meteo():
    """
    Fetch a simple weather forecast (dummy example) from Open-Meteo for Berlin.
    """
    url = "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&daily=temperature_2m_max&forecast_days=1"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

# Load and prepare data
with st.spinner("Loading data from Eurostat and World Bank..."):
    try:
        data = load_eurostat_data(year=2019)
    except Exception as e:
        st.error(f"Error fetching Eurostat data: {e}")
        st.stop()
    # Map NUTS2 code to ISO2 country code
    def map_country(nuts):
        cc = nuts[:2].upper()
        cc_map = {'EL': 'GR', 'UK': 'GB'}
        return cc_map.get(cc, cc)
    data["country_code"] = data["region_code"].apply(map_country)
    countries = sorted(data["country_code"].dropna().unique().tolist())
    country_stats = load_worldbank_data(year=2019, country_codes_iso2=countries)
# Merge World Bank data
data['population'] = data['country_code'].apply(lambda cc: country_stats.get(cc, {}).get('population', np.nan))
data['gdp_per_capita'] = data['country_code'].apply(lambda cc: country_stats.get(cc, {}).get('gdp_per_capita', np.nan))
# Compute tourism arrivals per capita (nights per person)
data['arrivals_per_capita'] = data['tourist_nights'] / data['population']
# Clean and drop missing
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=['tourist_nights', 'gdp_meur', 'population', 'arrivals_per_capita'], inplace=True)
data.reset_index(drop=True, inplace=True)

# Baseline climate data simulation (dummy)
regions = data['region_code'].unique()
baseline_temp = {}
for region in regions:
    # simulate base temperature (°C) by region (random-ish)
    baseline_temp[region] = 15 + (hash(region) % 10) * 0.5
data['baseline_temp'] = data['region_code'].map(baseline_temp)
# Normalize baseline temperature to [0,1] (higher = warmer)
if not data.empty:
    data['norm_baseline_temp'] = (data['baseline_temp'] - data['baseline_temp'].min()) / (data['baseline_temp'].max() - data['baseline_temp'].min() + 1e-9)
else:
    data['norm_baseline_temp'] = 0.0

# Compute climate risk and other indicator projections based on scenario
scen_cfg = scenario_configs.get(scenario_key, {})
stype = scen_cfg.get("type", "").lower()
years_forward = year - 2019
data['climate_risk'] = data['norm_baseline_temp']
if stype == "rcp":
    mult10 = float(scen_cfg.get("climate_multiplier_per_10y", 1.0))
    factor = mult10 ** (years_forward / 10.0)
    data['climate_risk'] = data['climate_risk'] * factor
elif stype == "foresight":
    data = apply_foresight(
        data,
        scen_cfg.get("regulation", "easy"),
        scen_cfg.get("technology", "easy"),
        years_forward,
        REG=REG,
        TECH=TECH,
    )
data['climate_risk'] = data['climate_risk'].clip(0.0, 1.0)

# Derive indicators required for compute_scores
data['region_id'] = data['region_code']
data['region_name'] = data.get('region_name', data['region_code'])
data['scenario'] = scenario_key
data['tourism_gdp_share'] = (data['tourist_nights'] / (data['gdp_meur'] * 1e6)).fillna(0.0)
data['seasonality_idx'] = 0.5
data['readiness_proxy'] = (data['gdp_per_capita'] / data['gdp_per_capita'].max()).fillna(0.0)
data['projected_drop'] = 0.1
data['heat_tmax_delta'] = data['climate_risk'] * 5.0
data['heatwave_days'] = data['climate_risk'] * 30.0
data['drought_spei'] = -data['climate_risk']

# Compute resilience scores using the scoring module
score_cols = [
    'region_id', 'region_name', 'scenario',
    'heat_tmax_delta', 'heatwave_days', 'drought_spei',
    'tourism_gdp_share', 'seasonality_idx', 'projected_drop',
    'readiness_proxy'
]
scores = compute_scores(data[score_cols].copy())
data = data.merge(scores[['region_id', 'resilience_score']], on='region_id', how='left')
data.drop(columns=['region_id'], inplace=True)

# --- Machine Learning Model ---
st.subheader("Resilience Score Prediction Model")
feature_cols = ['arrivals_per_capita', 'gdp_per_capita', 'climate_risk']
X = data[feature_cols].fillna(0.0)
y = data['resilience_score']
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model training (Random Forest with simple hyperparam tuning)
rf = RandomForestRegressor(random_state=42)
param_dist = {'n_estimators': [50, 100], 'max_depth': [3, 5, None], 'min_samples_split': [2, 5]}
cv = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=4, cv=3, scoring='neg_mean_squared_error', random_state=42)
cv.fit(X_train, y_train)
best_model = cv.best_estimator_
# Evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Test set performance: MSE = {mse:.2f}, R² = {r2:.2f}")

# Feature importances
if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.write("Feature importances (model):")
    st.dataframe(importance_df)

# --- SHAP Explainability ---
st.subheader("Explainability (SHAP)")
# Use a small background sample for speed
X_bg = X_train.sample(n=min(500, len(X_train)), random_state=42)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_bg, check_additivity=False)
# Mean absolute SHAP values
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
shap_imp_df = pd.DataFrame({'Feature': feature_cols, 'Mean |SHAP|': mean_abs_shap}).sort_values('Mean |SHAP|', ascending=False)
st.write("Mean absolute SHAP values (feature influence):")
st.dataframe(shap_imp_df)
# SHAP summary bar plot
fig_shap = plt.figure()
shap.summary_plot(shap_values, X_bg, feature_names=feature_cols, plot_type="bar", show=False)
st.pyplot(fig_shap)

# --- Map Visualization ---
st.subheader("Resilience Score Map")
geo_url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_BN_03M_2024_4326_LEVL_2.geojson"
try:
    geojson_data = requests.get(geo_url, timeout=30).json()
except Exception as e:
    st.error("Failed to load region boundary data.")
    geojson_data = None

if geojson_data:
    map_df = data[['region_code', 'resilience_score']].copy()
    map_df['resilience_score'] = map_df['resilience_score'].round(1)
    fig_map = px.choropleth(
        map_df,
        geojson=geojson_data,
        locations='region_code',
        featureidkey="properties.NUTS_ID",
        color='resilience_score',
        color_continuous_scale="YlGnBu",
        range_color=(0, 100),
        labels={'resilience_score': 'Resilience Score'},
        title=f"Resilience Score by Region ({scenario_display} {year})"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("Choropleth map of tourism-climate resilience scores for each NUTS2 region.")

# --- Cost-Benefit Analysis Integration ---
st.subheader("Adaptation Cost–Benefit Analysis")
try:
    # Placeholder for CBA: assuming a function in cba module
    from cba.cba import run_cba
    cba_results = run_cba(data, scenario=scenario_key, year=year)
    st.write(cba_results)
except Exception:
    st.info("Cost–Benefit Analysis module not available or failed to run.")

# --- Data Download ---
st.subheader("Download Results")
output_df = data[['region_code', 'resilience_score', 'tourist_nights', 'population', 'gdp_per_capita']].copy()
output_df['scenario'] = f"{scenario_display}_{year}"
csv_data = output_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv_data, file_name=f"resilience_scores_{scenario_display}_{year}.csv", mime="text/csv")

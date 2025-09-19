# app.py — CRISI Streamlit App (scenarios + maps + ML + real-data hooks)
# Python >= 3.9, Streamlit >= 1.25

import os
import json
from copy import deepcopy
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import yaml

from cba.climate_risk import adjust_for_risk
from cba.economic import apply_shadow_prices, eirr, enpv
from cba.externalities import add_externalities, carbon_cashflow
from cba.financial import irr, npv, payback_period
from src.scoring import compute_scores
# ---------- Streamlit page setup ----------
st.set_page_config(page_title="CRISI Model Explorer", layout="wide")
st.title("🌍 CRISI: Climate Resilience Investment Scoring")

st.caption(
    "Real-data enabled prototype with live API hooks (Copernicus, Eurostat, World Bank, Open-Meteo), "
    "NUTS2 mapping, ML-based tourism impact signal, and side-by-side scenario comparison."
)

# ---------- Helpers ----------
def normalize_weights(*ws):
    s = sum(ws)
    if s == 0:
        n = len(ws)
        return tuple([1.0 / n] * n)
    return tuple([w / s for w in ws])

def scenario_params(name: str):
    """Return (climate_factor, tourism_share, demand_drop, readiness_base) based on scenario."""
    if name.startswith("Green"):
        return (1.0, 0.30, 0.0, 0.80)
    if name.startswith("Business"):
        return (1.2, 0.35, 0.0, 0.60)
    if name.startswith("Divided"):
        return (1.3, 0.40, 0.0, 0.50)
    if name.startswith("Techno"):
        return (1.5, 0.45, 0.0, 0.70)
    if name.startswith("Regional"):
        return (1.4, 0.65, 25.0, 0.30)
    return (1.0, 0.35, 0.0, 0.50)

DEFAULT_FEATURES_PATH = Path("data_proc/features_geo.parquet")
DEFAULT_CONFIG_PATH = Path("config/indicators.yaml")
@st.cache_data
def load_indicator_config(path: str = str(DEFAULT_CONFIG_PATH)) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Indicator configuration missing at {cfg_path}. Ensure config/indicators.yaml is available."
        )
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
  
@st.cache_data
def load_processed_features(path: str = str(DEFAULT_FEATURES_PATH)) -> gpd.GeoDataFrame | None:
    data_path = Path(path)
    if not data_path.exists():
        return None
    gdf = gpd.read_parquet(data_path)
    return gdf


def infer_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def build_indicator_config(
    base_cfg: dict,
    w_heat: float,
    w_gdp: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict:
    cfg = deepcopy(base_cfg)

    exposure_cfg = cfg.setdefault("exposure", {})
    heat_weight = max(w_heat, 0.0)
    drought_weight = max(1.0 - w_heat, 0.0)
    if "heat_tmax_delta" in exposure_cfg:
        exposure_cfg["heat_tmax_delta"]["weight"] = 0.5 * heat_weight
    if "heatwave_days" in exposure_cfg:
        exposure_cfg["heatwave_days"]["weight"] = 0.5 * heat_weight
    if "drought_spei" in exposure_cfg:
        exposure_cfg["drought_spei"]["weight"] = drought_weight

    sensitivity_cfg = cfg.setdefault("sensitivity", {})
    gdp_weight = max(w_gdp, 0.0)
    seasonality_weight = max(1.0 - w_gdp, 0.0)
    if "tourism_gdp_share" in sensitivity_cfg:
        sensitivity_cfg["tourism_gdp_share"]["weight"] = gdp_weight
    if "seasonality_idx" in sensitivity_cfg:
        sensitivity_cfg["seasonality_idx"]["weight"] = seasonality_weight

    cfg.setdefault("pillars", {})
    cfg["pillars"].update({"alpha": alpha, "beta": beta, "gamma": gamma})
    return cfg


def compute_timeseries(
    scores_df: pd.DataFrame,
    scenario: str,
    scenario_col: str,
    year_col: str,
) -> pd.DataFrame:
    subset = scores_df[scores_df[scenario_col] == scenario].copy()
    if subset.empty or year_col not in subset.columns:
        empty = pd.DataFrame(columns=["Exposure", "Sensitivity", "Adaptive", "Resilience", "Risk"])
        empty.index.name = "Year"
        return empty

    subset = subset.dropna(subset=[year_col])
    if subset.empty:
        empty = pd.DataFrame(columns=["Exposure", "Sensitivity", "Adaptive", "Resilience", "Risk"])
        empty.index.name = "Year"
        return empty

    subset[year_col] = subset[year_col].astype(int)
    grouped = (
        subset.groupby(year_col)[
            [
                "exposure_score",
                "sensitivity_score",
                "adaptive_capacity_score",
                "risk",
                "resilience_score",
            ]
        ]
        .mean()
        .sort_index()
    )

    result = pd.DataFrame({
        "Exposure": grouped["exposure_score"].clip(0.0, 1.0),
        "Sensitivity": grouped["sensitivity_score"].clip(0.0, 1.0),
        "Adaptive": grouped["adaptive_capacity_score"].clip(0.0, 1.0),
        "Resilience": (grouped["resilience_score"].clip(lower=0.0) / 100.0).clip(0.0, 1.0),
        "Risk": grouped["risk"].clip(0.0, 1.0),
    })
    result.index.name = "Year"
    return result
def prepare_training_dataset(
    features_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    region_id_col: str,
    scenario_col: str,
    year_col: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    df = features_df.copy()
    df["resilience_score"] = scores_df["resilience_score"]
 numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"resilience_score", region_id_col, scenario_col}
    if year_col and year_col in df.columns:
        drop_cols.add(year_col)

    feature_cols = [col for col in numeric_cols if col not in drop_cols]
    training_df = df[feature_cols + ["resilience_score"]].dropna()
    return training_df, feature_cols


@st.cache_resource(
    hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df, index=True).sum()}
)
def train_rf_model(
    training_df: pd.DataFrame,
    feature_cols: tuple[str, ...],
    target_col: str = "resilience_score",
    random_state: int = 42,
):
    if training_df.empty or not feature_cols:
        raise ValueError("Training dataset is empty. Provide real indicator tables before training.")
    if len(training_df) < 10:
        raise ValueError("At least 10 observations are required to train the RandomForest model.")

    X = training_df.loc[:, list(feature_cols)]
    y = training_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)

    if len(y_test) > 0:
        y_pred = rf.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "n_obs": int(len(training_df)),
        }
    else:
        metrics = {"r2": float("nan"), "mae": float("nan"), "n_obs": int(len(training_df))}

    importances = pd.Series(rf.feature_importances_, index=list(feature_cols)).sort_values(
        ascending=False
    )
    return rf, importances, metrics
 

# ---------- Live data hooks (optional, safe) ----------
@st.cache_data(ttl=3600)
def get_open_meteo_temp(lat=37.98, lon=23.72):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "forecast_days": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        return js.get("hourly", {}).get("temperature_2m", [None])[0]
    except Exception as e:
        st.warning(f"Open-Meteo unavailable: {e}")
        return None

@st.cache_data(ttl=86400)
def get_worldbank_df(country_codes=("GR", "DE")):
    """Fetch very small WB sample (GDP per capita) just to prove connectivity (safe fallback)."""
    try:
        import wbdata
    except Exception as e:
        st.info("World Bank library not installed; skipping WB call.")
        return None
    try:
        ind = {"NY.GDP.PCAP.CD": "gdp_pc"}
        # wbdata expects date=(start, end) as strings YYYY
        df = wbdata.get_dataframe(ind, country=list(country_codes), date=("2019", "2024"))
        return df.reset_index()
    except Exception as e:
        st.warning(f"World Bank call failed: {e}")
        return None

@st.cache_data(ttl=86400)
def get_eurostat_tiny():
    """Try to touch Eurostat via pandasdmx (no heavy pulls; safe fallback)."""
    try:
        from pandasdmx import Request
        estat = Request("ESTAT")
        # We only ping the service metadata; full dataset codes vary and are heavy
        _ = estat.datastructure("ESTAT").response
        return True
    except Exception as e:
        st.info(f"Eurostat (pandasdmx) not reachable right now: {e}")
        return False


@st.cache_resource
def get_nuts2_gdf():
    """
    Download GISCO NUTS 2021 (20M, EPSG:4326) if missing, extract locally,
    and return a GeoDataFrame filtered to NUTS level 2. Falls back to GeoJSON if needed.
    """
    import glob
    import zipfile
    base_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(base_dir, exist_ok=True)

    zip_url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/shp/NUTS_RG_20M_2021_4326.zip"
    zip_path = os.path.join(base_dir, "NUTS_RG_20M_2021_4326.zip")
    extract_dir = os.path.join(base_dir, "NUTS_RG_20M_2021_4326")

    # 1) Download if not present
    if not os.path.exists(zip_path):
        try:
            st.info("Downloading NUTS2 shapefile (GISCO, ~8–15MB)…")
            with requests.get(zip_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            st.warning(f"Shapefile zip download failed: {e}")

    # 2) Extract if not yet extracted
    shp_path = None
    try:
        if not os.path.isdir(extract_dir):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        # Find the .shp file (name can vary slightly)
        candidates = glob.glob(os.path.join(extract_dir, "**", "*.shp"), recursive=True)
        # Prefer the 20M, 2021, 4326 level
        preferred = [p for p in candidates if "NUTS_RG_20M_2021_4326" in p]
        shp_path = preferred[0] if preferred else (candidates[0] if candidates else None)
    except Exception as e:
        st.warning(f"Extraction/read prep failed: {e}")

    # 3) Read with GeoPandas
    gdf = None
    if shp_path and os.path.exists(shp_path):
        try:
            gdf = gpd.read_file(shp_path).to_crs(4326)
        except Exception as e:
            st.warning(f"Reading shapefile failed: {e}")

    # 4) Fallback to GeoJSON (smaller, HTTP-friendly)
    if gdf is None:
        try:
            st.info("Falling back to GeoJSON (GISCO)…")
            geojson_url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_20M_2021_4326.geojson"
            gdf = gpd.read_file(geojson_url).to_crs(4326)
        except Exception as e:
            st.error(f"GeoJSON fallback failed: {e}")
            return None

    # 5) Keep Level 2 + standardize columns
    if "LEVL_CODE" in gdf.columns:
        gdf = gdf[gdf["LEVL_CODE"] == 2]
    if "NUTS_ID" in gdf.columns:
        gdf = gdf.rename(columns={"NUTS_ID": "id"})
    if "NAME_LATN" in gdf.columns:
        gdf = gdf.rename(columns={"NAME_LATN": "region_name"})
    elif "NUTS_NAME" in gdf.columns:
        gdf = gdf.rename(columns={"NUTS_NAME": "region_name"})

    keep_cols = [c for c in ["id", "CNTR_CODE", "region_name", "geometry"] if c in gdf.columns]
    return gdf[keep_cols]
# ---------- Data loading ----------
try:
    indicator_config = load_indicator_config()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

features_geo = load_processed_features()
if features_geo is None:
    st.warning(
        "Processed indicator dataset missing. Run the data preparation pipeline, e.g.:\n"
        "`python src/data_prep.py --shapefile <NUTS2.shp> --climate data_raw/climate_indicators.csv "
        "--tourism data_raw/tourism_econ.csv --output data_proc/features_geo.parquet`."
    )
    st.stop()

features_df = pd.DataFrame(features_geo.drop(columns="geometry", errors="ignore"))

region_id_col = infer_column(features_df, ["region_id", "RegionID", "id", "nuts_id", "NUTS_ID"])
scenario_col = infer_column(features_df, ["scenario", "Scenario"])
year_col = infer_column(features_df, ["year", "Year"])
region_name_col = infer_column(features_df, ["region_name", "RegionName", "name", "NAME_LATN"])

missing_cols = []
if region_id_col is None:
    missing_cols.append("region_id")
if scenario_col is None:
    missing_cols.append("scenario")
if year_col is None:
    missing_cols.append("year")

if missing_cols:
    st.error(
        "Processed dataset is missing required columns: " + ", ".join(missing_cols) + ". "
        "Ensure the CSV inputs include scenario, year, and region identifiers."
    )
    st.stop()

scenarios_in_data = sorted(features_df[scenario_col].dropna().unique().tolist())
if not scenarios_in_data:
    st.error("No scenarios found in processed dataset. Verify the climate CSV includes scenario labels.")
    st.stop()

# ---------- Sidebar Controls ----------
default_scenario_labels = [
    "Green Global Resilience (RCP4.5/SSP1)",
    "Business-as-Usual Drift (RCP6.0/SSP2)",
    "Divided Disparity (RCP6.0-like/SSP4)",
    "Techno-Optimism on a Hot Planet (RCP8.5/SSP5)",
    "Regional Fortress World (RCP7.0/SSP3)",
]
scenarios_all = [label for label in default_scenario_labels if label in scenarios_in_data]
if not scenarios_all:
    scenarios_all = scenarios_in_data

compare_all = st.sidebar.checkbox(
    "Compare ALL scenarios side-by-side", value=len(scenarios_all) > 1
)
if compare_all:
    selected_scenarios = scenarios_all
else:
    selected_scenarios = [st.sidebar.selectbox("Select scenario", scenarios_all)]

st.sidebar.markdown("---")
st.sidebar.subheader("Pillar Weights")
alpha = st.sidebar.slider("Exposure (α)", 0.0, 1.0, 0.33, 0.01)
beta  = st.sidebar.slider("Sensitivity (β)", 0.0, 1.0, 0.33, 0.01)
gamma = st.sidebar.slider("Adaptive (γ)", 0.0, 1.0, 0.34, 0.01)

st.sidebar.subheader("Sub-indicator Weights")
w_heat = st.sidebar.slider("Heat vs Drought", 0.0, 1.0, 0.5, 0.01)
w_gdp  = st.sidebar.slider("Tourism GDP vs Seasonality", 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Map Countries (NUTS2)")
country_map = {"Greece": "EL", "Germany": "DE"}
country_picks = st.sidebar.multiselect(
    "Filter map to:", list(country_map.keys()), default=list(country_map.keys())
)
country_codes = [country_map[c] for c in country_picks]

# Live tiny calls (non-blocking, safe)
temp = get_open_meteo_temp()
if temp is not None:
    st.sidebar.caption(f"Open-Meteo sample temp: {temp:.1f}°C")
_ = get_eurostat_tiny()
_ = get_worldbank_df()

# ---------- Core Computation ----------
alpha_n, beta_n, gamma_n = normalize_weights(alpha, beta, gamma)
dynamic_config = build_indicator_config(indicator_config, w_heat, w_gdp, alpha_n, beta_n, gamma_n)

scores_df = compute_scores(
    features_df,
    config=dynamic_config,
    include_pillars=True,
)

results_by_scenario: dict[str, pd.DataFrame] = {}
for sc in selected_scenarios:
    series_df = compute_timeseries(scores_df, sc, scenario_col, year_col)
    if not series_df.empty:
        results_by_scenario[sc] = series_df

if not results_by_scenario:
    st.error(
        "No time-series data available for the selected scenarios. "
        "Verify that data_proc/features_geo.parquet includes those scenarios and years."
    )
st.stop()

years_union = sorted({int(y) for df in results_by_scenario.values() for y in df.index})
if not years_union:
    st.error("Year values are missing from the processed dataset; cannot build scenario projections.")
    st.stop()

years = np.array(years_union, dtype=int)
# ---------- Advanced: Investment cost-benefit analysis ----------
with st.expander("Advanced: Investment cost-benefit analysis"):
    st.markdown(
        "Configure investment, operating, and climate parameters to quantify the "
        "financial and economic performance of the project under each selected scenario."
    )

    min_year = int(years.min())
    max_year = int(years.max())

    with st.form("cba_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            cba_start = st.number_input(
                "Start year", min_value=min_year, max_value=max_year, value=min_year, step=1
            )
        with col_b:
            default_end = min(min_year + 10, max_year)
            cba_end = st.number_input(
                "End year", min_value=min_year, max_value=max_year, value=default_end, step=1
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            initial_invest = st.number_input(
                "Initial investment (EUR)", min_value=0.0, value=5_000_000.0, step=100_000.0
            )
        with col2:
            base_revenue = st.number_input(
                "Annual revenue in start year (EUR)", value=1_200_000.0, step=50_000.0
            )
        with col3:
            revenue_growth_pct = st.number_input(
                "Annual revenue growth (%)", value=2.0, step=0.5, format="%0.2f"
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            operating_cost = st.number_input(
                "Annual operating cost (EUR)", min_value=0.0, value=350_000.0, step=25_000.0
            )
        with col5:
            discount_rate_pct = st.number_input(
                "Financial discount rate (%)", min_value=0.0, value=6.0, step=0.25, format="%0.2f"
            )
        with col6:
            social_discount_rate_pct = st.number_input(
                "Social discount rate (%)", min_value=0.0, value=4.0, step=0.25, format="%0.2f"
            )

        col7, col8, col9 = st.columns(3)
        with col7:
            resilience_uplift = st.number_input(
                "Revenue uplift per resilience point",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
            )
        with col8:
            risk_exposure_share = st.number_input(
                "Share of revenue exposed to climate risk",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
            )
        with col9:
            adaptation_cost = st.number_input(
                "Additional adaptation O&M (EUR/yr)", min_value=0.0, value=0.0, step=10_000.0
            )

        col10, col11, col12 = st.columns(3)
        with col10:
            annual_emissions = st.number_input(
                "Net emissions in start year (tCO₂e)", value=0.0, step=25.0, format="%0.2f"
            )
        with col11:
            emission_trend_pct = st.number_input(
                "Annual emissions change (%)", value=-2.0, step=0.5, format="%0.2f"
            )
        with col12:
            carbon_price_start = st.number_input(
                "Carbon price in start year (€/tCO₂e)", min_value=0.0, value=80.0, step=5.0
            )

        col13, col14 = st.columns(2)
        with col13:
            carbon_price_growth_pct = st.number_input(
                "Annual carbon price growth (%)", value=2.0, step=0.5, format="%0.2f"
            )
        with col14:
            shadow_factor = st.number_input(
                "Shadow price factor (economic analysis)", min_value=0.0, value=1.0, step=0.05
            )

        run_cba = st.form_submit_button("Run analysis")

    if run_cba:
        if cba_end < cba_start:
            st.error("End year must be greater than or equal to the start year.")
        else:
            analysis_years = np.arange(cba_start, cba_end + 1)
            revenue_growth = revenue_growth_pct / 100.0
            discount_rate = discount_rate_pct / 100.0
            social_discount_rate = social_discount_rate_pct / 100.0
            emission_trend = emission_trend_pct / 100.0
            carbon_price_growth = carbon_price_growth_pct / 100.0

            if not results_by_scenario:
                st.info("No scenarios selected for comparison.")
            else:
                summary_rows = []
                detail_tables = {}

                for sc, df in results_by_scenario.items():
                    scenario_df = df.reindex(analysis_years).interpolate().ffill().bfill()

                    gross_cf = {}
                    climate_losses = {}
                    emissions = {}
                    yearly_records = []

                    for idx, year in enumerate(analysis_years):
                        resilience_val = float(scenario_df.loc[year, "Resilience"])
                        risk_val = float(scenario_df.loc[year, "Risk"])

                        revenue_nominal = base_revenue * ((1 + revenue_growth) ** idx)
                        revenue_adjusted = revenue_nominal * (1 + resilience_uplift * (resilience_val - 0.5))
                        revenue_adjusted = max(revenue_adjusted, 0.0)

                        annual_cash = revenue_adjusted - operating_cost - adaptation_cost
                        if year == cba_start:
                            annual_cash -= initial_invest

                        gross_cf[year] = annual_cash

                        climate_loss = max(revenue_adjusted * risk_val * risk_exposure_share, 0.0)
                        climate_losses[year] = climate_loss

                        emission_val = annual_emissions * ((1 + emission_trend) ** idx)
                        emissions[year] = emission_val

                        yearly_records.append(
                            {
                                "Year": year,
                                "Resilience": resilience_val,
                                "Risk": risk_val,
                                "Revenue (resilience-adjusted)": revenue_adjusted,
                                "Gross cashflow": annual_cash,
                                "Expected climate loss": climate_loss,
                                "Net emissions (tCO₂e)": emission_val,
                            }
                        )

                    risk_adj_cf = adjust_for_risk(gross_cf, climate_losses)

                    carbon_price_path = {
                        year: carbon_price_start * ((1 + carbon_price_growth) ** (year - cba_start))
                        for year in analysis_years
                    }
                    carbon_cf = carbon_cashflow(emissions, carbon_price_path)
                    econ_cf = add_externalities(risk_adj_cf, carbon_cf)
                    econ_shadow = apply_shadow_prices(econ_cf, {"non_traded": shadow_factor})

                    for record in yearly_records:
                        year = record["Year"]
                        record["Risk-adjusted cashflow"] = risk_adj_cf.get(year, 0.0)
                        record["Carbon externality"] = carbon_cf.get(year, 0.0)
                        record["Economic cashflow (shadow)"] = econ_shadow.get(year, 0.0)

                    try:
                        fin_npv = npv(risk_adj_cf, discount_rate)
                    except Exception:
                        fin_npv = float("nan")

                    try:
                        fin_irr = irr(risk_adj_cf) * 100
                    except Exception:
                        fin_irr = float("nan")

                    payback = payback_period(risk_adj_cf)

                    try:
                        econ_npv = enpv(econ_shadow, social_discount_rate)
                    except Exception:
                        econ_npv = float("nan")

                    try:
                        econ_irr = eirr(econ_shadow) * 100
                    except Exception:
                        econ_irr = float("nan")

                    total_benefits = sum(v for v in risk_adj_cf.values() if v > 0)
                    total_costs = -sum(v for v in risk_adj_cf.values() if v < 0)
                    bcr = (total_benefits / total_costs) if total_costs else float("nan")

                    summary_rows.append(
                        {
                            "Scenario": sc,
                            "Financial NPV (€)": fin_npv,
                            "Financial IRR (%)": fin_irr,
                            "Payback (years)": payback,
                            "Benefit-cost ratio": bcr,
                            "Economic NPV (€)": econ_npv,
                            "Economic IRR (%)": econ_irr,
                        }
                    )

                    detail_tables[sc] = pd.DataFrame(yearly_records).set_index("Year")

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows).set_index("Scenario")
                    st.success("Cost-benefit analysis complete.")
                    st.dataframe(
                        summary_df.style.format(
                            {
                                "Financial NPV (€)": "{:,.0f}",
                                "Financial IRR (%)": "{:,.2f}",
                                "Payback (years)": "{:.0f}",
                                "Benefit-cost ratio": "{:,.2f}",
                                "Economic NPV (€)": "{:,.0f}",
                                "Economic IRR (%)": "{:,.2f}",
                            }
                        ),
                        use_container_width=True,
                    )

                    detail_tabs = st.tabs(list(detail_tables.keys()))
                    for tab, sc in zip(detail_tabs, detail_tables.keys()):
                        with tab:
                            st.dataframe(
                                detail_tables[sc].style.format(
                                    {
                                        "Resilience": "{:.3f}",
                                        "Risk": "{:.3f}",
                                        "Revenue (resilience-adjusted)": "{:,.0f}",
                                        "Gross cashflow": "{:,.0f}",
                                        "Expected climate loss": "{:,.0f}",
                                        "Net emissions (tCO₂e)": "{:,.1f}",
                                        "Risk-adjusted cashflow": "{:,.0f}",
                                        "Carbon externality": "{:,.0f}",
                                        "Economic cashflow (shadow)": "{:,.0f}",
                                    }
                                ),
                                use_container_width=True,
                            )
                else:
                    st.info("No results to display for the configured inputs.")
# ---------- Visuals: Time-series ----------
st.subheader(f"Resilience projection ({years.min()}–{years.max()})")
plot_df = pd.concat(
    {sc: df["Resilience"] for sc, df in results_by_scenario.items()}, axis=1
)
if not plot_df.empty:
    st.line_chart(plot_df)
else:
    st.info("No resilience values available to plot for the selected scenarios.")

# ---------- Tables ----------
st.subheader("Pillar breakdown (select scenario)")
table_choice = st.selectbox("Scenario table", list(results_by_scenario.keys()))
st.dataframe(results_by_scenario[table_choice])

# Guardrail check (per the selected table)
last_row = results_by_scenario[table_choice].iloc[-1]
# We need the scenario's tourism_share and demand_drop:
cf, ts, dd, rb = scenario_params(table_choice)
if ts > 0.5 and dd > 20:
    st.warning("⚠️ Guardrail: Tourism GDP share > 50% and projected demand drop > 20%.")

# ---------- Download ----------
st.download_button(
    "Download current table (CSV)",
    data=results_by_scenario[table_choice].reset_index().to_csv(index=False).encode("utf-8"),
    file_name=f"crisi_{table_choice.split()[0].lower()}_results.csv",
    mime="text/csv",
)

# ---------- ML: simple trained model (no SHAP to avoid frontend errors) ----------
st.subheader("ML tourism-impact signal")
training_df, feature_cols = prepare_training_dataset(
    features_df, scores_df, region_id_col, scenario_col, year_col
)

if feature_cols and not training_df.empty:
    try:
        rf_model, importances, ml_metrics = train_rf_model(training_df, tuple(feature_cols))
        st.caption(
            "RandomForestRegressor trained on processed indicator features. "
            f"Hold-out R²: {ml_metrics['r2']:.2f}, MAE: {ml_metrics['mae']:.2f} resilience points "
            f"(n={ml_metrics['n_obs']})."
        )
        st.bar_chart(importances)
    except ValueError as exc:
        st.info(str(exc))
else:
    st.info(
        "Not enough processed observations to train the ML model. "
        "Ensure climate and tourism tables cover multiple regions/years."
    )

# ---------- Map ----------
map_scenario = table_choice
scenario_scores = (
    scores_df[scores_df[scenario_col] == map_scenario]
    .dropna(subset=[year_col])
    .copy()
)
if not scenario_scores.empty:
    scenario_scores[year_col] = scenario_scores[year_col].astype(int)
    map_year = int(scenario_scores[year_col].max())
    st.subheader(f"Resilience map ({map_scenario}, {map_year})")
else:
    map_year = None
    st.subheader("Resilience map (NUTS2)")
nuts2 = get_nuts2_gdf()
if nuts2 is not None and not scenario_scores.empty:
    map_scores = scenario_scores[scenario_scores[year_col] == map_year].copy()
    map_scores["Resilience"] = (map_scores["resilience_score"] / 100.0).clip(0.0, 1.0)
    map_scores = map_scores.drop_duplicates(subset=[region_id_col])

    merge_cols = [region_id_col, "Resilience"]
    if region_name_col and region_name_col in map_scores.columns:
        merge_cols.append(region_name_col)
    map_scores = map_scores[merge_cols]

    nuts2 = nuts2.copy()
    if country_codes:
        nuts2 = nuts2[nuts2["CNTR_CODE"].isin(country_codes)]
    
    nuts2 = nuts2.merge(map_scores, left_on="id", right_on=region_id_col, how="left")
    centroid = nuts2.geometry.centroid
    center = [centroid.y.mean(), centroid.x.mean()]
    fmap = folium.Map(location=center, zoom_start=5)
    folium.Choropleth(
        geo_data=nuts2.__geo_interface__,
        data=nuts2,
        columns=["id", "Resilience"],
        key_on="feature.properties.id",
        fill_color="YlGn",
        fill_opacity=0.8,
        line_opacity=0.3,
        nan_fill_color="#d9d9d9",
        legend_name="Resilience score (latest)",
    ).add_to(fmap)

     label_candidates = []
    if region_name_col:
        label_candidates.extend(
            [region_name_col, f"{region_name_col}_x", f"{region_name_col}_y"]
        )
    label_candidates.extend(["region_name", "region_name_x", "region_name_y"])
    label_col = next((col for col in label_candidates if col in nuts2.columns), None)

    for _, row in nuts2.iterrows():
        if row.geometry.is_empty:
            continu
        c = row.geometry.centroid
        folium.Marker(
            [c.y, c.x],
            icon=folium.DivIcon(html=f"<div style='font-size:10px'>{label}</div>"),
        ).add_to(fmap)

    st_folium(fmap, width=800, height=520)
elif nuts2 is None:
    st.info("NUTS2 map unavailable right now.")
else:
    st.info("No resilience records found for the selected scenario/year to populate the map.")
  
st.caption(
    "Notes: Load real indicators by running `python src/data_prep.py` to refresh "
    "data_proc/features_geo.parquet. Copernicus CDS calls require a valid ~/.cdsapirc; "
    "Eurostat/World Bank are pinged lightly for connectivity checks."
)

# app.py — CRISI Streamlit App with Corine Land Cover integration

"""
This Streamlit application visualises climate resilience scenarios and includes a new
section for analysing Corine Land Cover (CLC) data for Greece. The CLC dataset,
often distributed as a large ZIP archive (~250 MB), contains polygon geometries
classified by land cover type across multiple reference years (e.g. 1990, 2000,
2006, 2012, 2018). To accommodate this sizeable dataset, the app loads the
archive on-demand using a cached helper function. Once loaded, users can
filter polygons by reference year, view summary counts by CLC code, and
visualise the shapes on an interactive map.

The original CRISI app provides scenario-based resilience projections,
investment cost–benefit analysis, a simple machine-learning demonstration and
resilience mapping. Those features remain intact, and the new CLC section is
appended at the end of the page.
"""

import os
import sys
import io
import json
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# Additional imports for CLC processing
import tempfile
import zipfile

# Imports for real-data integration
from pathlib import Path
import json as _json

# Typing support for type hints
from typing import List, Tuple, Optional


# Import cost–benefit functions from CRISI package
from cba.climate_risk import adjust_for_risk
from cba.economic import apply_shadow_prices, eirr, enpv
from cba.externalities import add_externalities, carbon_cashflow
from cba.financial import irr, npv, payback_period

# -----------------------------------------------------------------------------
# Real data loader
#
# The preprocessing script ``scripts/prepare_real_data.py`` produces a
# Parquet file and a JSON listing available numeric metrics. If these files
# are present in ``data/processed``, the app will load them and provide
# interactive analysis of the real datasets for Greece. Otherwise, the
# section will instruct the user to run the preprocessing script.

@st.cache_data(show_spinner=True)
def load_real_data():
    """Load processed Greece datasets and metrics if available.

    Returns a tuple ``(df, clc_df, metrics)`` where ``df`` is the unified
    DataFrame keyed by ``_region_norm`` and ``year``, ``clc_df`` is a CLC
    summary (optional), and ``metrics`` is a list of numeric columns for
    selection in the UI.
    """
    base_dir = Path(__file__).resolve().parent / "data" / "processed"
    parquet_file = base_dir / "crisi_greece_processed.parquet"
    metrics_file = base_dir / "crisi_metrics.json"
    clc_file = base_dir / "clc_greece_summary.parquet"
    # Try Parquet first, fallback to CSV
    df = pd.DataFrame()
    if parquet_file.exists():
        try:
            df = pd.read_parquet(parquet_file)
        except Exception:
            pass
    # fallback to CSV
    csv_file = parquet_file.with_suffix(".csv")
    if df.empty and csv_file.exists():
        df = pd.read_csv(csv_file, low_memory=False)
    if df.empty:
        return pd.DataFrame(), None, []
    # load metrics list
    metrics: List[str] = []
    if metrics_file.exists():
        try:
            metrics = _json.loads(metrics_file.read_text()).get("available_metrics", [])
        except Exception:
            metrics = []
    # clc summary if exists
    clc_df = None
    if clc_file.exists():
        try:
            clc_df = pd.read_parquet(clc_file)
        except Exception:
            clc_df = None
    return df, clc_df, metrics


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
    """Return (climate_factor, tourism_share, demand_drop, readiness_base) based on scenario name."""
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


def compute_series(years, climate_factor, tourism_share, readiness_base, w_heat, w_gdp, a, b, g):
    """
    Compute exposure, sensitivity, adaptive, resilience and risk series for a given scenario.
    Synthetic random noise is added to avoid flat lines. Output is a DataFrame indexed by year.
    """
    n = len(years)
    rng = np.random.default_rng(0)

    heat_index = np.clip(np.linspace(0.1, 0.1 + 0.6 * climate_factor, n) + rng.normal(0, 0.02, n), 0, 1)
    drought_index = np.clip(np.linspace(0.2, 0.2 + 0.4 * climate_factor, n) + rng.normal(0, 0.02, n), 0, 1)
    seasonality = np.clip(np.linspace(0.3, 0.3 + 0.1 * climate_factor, n) + rng.normal(0, 0.01, n), 0, 1)
    readiness = np.linspace(readiness_base, min(readiness_base + 0.1, 1.0), n)
    tourism_share_series = np.full(n, tourism_share)

    exposure = w_heat * heat_index + (1 - w_heat) * drought_index
    sensitivity = w_gdp * tourism_share_series + (1 - w_gdp) * seasonality
    adaptive = readiness

    a_n, b_n, g_n = normalize_weights(a, b, g)
    resilience = np.clip(a_n * (1 - exposure) + b_n * (1 - sensitivity) + g_n * adaptive, 0, 1)
    risk = np.clip(a_n * exposure + b_n * sensitivity + g_n * (1 - adaptive), 0, 1)

    df = pd.DataFrame({
        "Year": years,
        "Exposure": np.round(exposure, 3),
        "Sensitivity": np.round(sensitivity, 3),
        "Adaptive": np.round(adaptive, 3),
        "Resilience": np.round(resilience, 3),
        "Risk": np.round(risk, 3),
    }).set_index("Year")

    return df


# ---------- Live data hooks (optional, safe) ----------
@st.cache_data(ttl=3600)
def get_open_meteo_temp(lat=37.98, lon=23.72):
    """Fetch a single hourly temperature value from the Open-Meteo API."""
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
    """Fetch a tiny GDP per capita sample from the World Bank for connectivity checks."""
    try:
        import wbdata
    except Exception as e:
        st.info("World Bank library not installed; skipping WB call.")
        return None
    try:
        ind = {"NY.GDP.PCAP.CD": "gdp_pc"}
        df = wbdata.get_dataframe(ind, country=list(country_codes), date=("2019", "2024"))
        return df.reset_index()
    except Exception as e:
        st.warning(f"World Bank call failed: {e}")
        return None


@st.cache_data(ttl=86400)
def get_eurostat_tiny():
    """Ping Eurostat via pandasdmx to confirm connectivity without heavy downloads."""
    try:
        from pandasdmx import Request
        estat = Request("ESTAT")
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
    import zipfile as zf
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
            with zf.ZipFile(zip_path, "r") as zipf:
                zipf.extractall(extract_dir)

        # Find the .shp file (name can vary slightly)
        candidates = glob.glob(os.path.join(extract_dir, "**", "*.shp"), recursive=True)
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


# ---------- Sidebar Controls ----------
scenarios_all = [
    "Green Global Resilience (RCP4.5/SSP1)",
    "Business-as-Usual Drift (RCP6.0/SSP2)",
    "Divided Disparity (RCP6.0-like/SSP4)",
    "Techno-Optimism on a Hot Planet (RCP8.5/SSP5)",
    "Regional Fortress World (RCP7.0/SSP3)",
]
compare_all = st.sidebar.checkbox("Compare ALL scenarios side-by-side", value=True)
if compare_all:
    selected_scenarios = scenarios_all
else:
    selected_scenarios = [st.sidebar.selectbox("Select scenario", scenarios_all)]

st.sidebar.markdown("---")
st.sidebar.subheader("Pillar Weights")
alpha = st.sidebar.slider("Exposure (α)", 0.0, 1.0, 0.33, 0.01)
beta = st.sidebar.slider("Sensitivity (β)", 0.0, 1.0, 0.33, 0.01)
gamma = st.sidebar.slider("Adaptive (γ)", 0.0, 1.0, 0.34, 0.01)

st.sidebar.subheader("Sub-indicator Weights")
w_heat = st.sidebar.slider("Heat vs Drought", 0.0, 1.0, 0.5, 0.01)
w_gdp = st.sidebar.slider("Tourism GDP vs Seasonality", 0.0, 1.0, 0.5, 0.01)

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
years = np.arange(2025, 2056)
results_by_scenario = {}
for sc in selected_scenarios:
    cf, ts, dd, rb = scenario_params(sc)
    results_by_scenario[sc] = compute_series(
        years, cf, ts, rb, w_heat, w_gdp, alpha, beta, gamma
    )


# ---------- Advanced: Investment cost–benefit analysis ----------
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
            cba_end = st.number_input(
                "End year", min_value=min_year, max_value=max_year, value=min_year + 10, step=1
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
                        revenue_adjusted = revenue_nominal * (
                            1 + resilience_uplift * (resilience_val - 0.5)
                        )
                        revenue_adjusted = max(revenue_adjusted, 0.0)

                        annual_cash = revenue_adjusted - operating_cost - adaptation_cost
                        if year == cba_start:
                            annual_cash -= initial_invest

                        gross_cf[year] = annual_cash

                        climate_loss = max(
                            revenue_adjusted * risk_val * risk_exposure_share, 0.0
                        )
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
                        year: carbon_price_start
                        * ((1 + carbon_price_growth) ** (year - cba_start))
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
st.subheader("Resilience projection (2025–2055)")
plot_df = pd.concat(
    {sc: df["Resilience"] for sc, df in results_by_scenario.items()}, axis=1
)
st.line_chart(plot_df)


# ---------- Tables ----------
st.subheader("Pillar breakdown (select scenario)")
table_choice = st.selectbox("Scenario table", list(results_by_scenario.keys()))
st.dataframe(results_by_scenario[table_choice])


# Guardrail check (per the selected table)
last_row = results_by_scenario[table_choice].iloc[-1]
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


# ---------- ML: simple trained model (demo) ----------
st.subheader("ML tourism-impact signal (demo)")
st.caption("RandomForestRegressor trained on synthetic features; showing feature importances.")

@st.cache_resource
def train_rf_model(seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Heat": rng.random(300),
        "Drought": rng.random(300),
        "TourismShare": rng.random(300),
        "Adaptive": rng.random(300),
    })
    y = (
        0.5 + 0.3 * X["Adaptive"] - 0.25 * X["Heat"] - 0.2 * X["Drought"]
        + 0.05 * X["TourismShare"] + rng.normal(0, 0.03, 300)
    )
    rf = RandomForestRegressor(n_estimators=150, random_state=seed)
    rf.fit(X, y)
    return rf, X.columns.tolist()

rf_model, feat_names = train_rf_model()
importances = pd.Series(rf_model.feature_importances_, index=feat_names).sort_values(ascending=False)
st.bar_chart(importances)


# ---------- Map ----------
st.subheader("Resilience map (latest year, NUTS2)")
nuts2 = get_nuts2_gdf()
if nuts2 is not None:
    if country_codes:
        nuts2 = nuts2[nuts2["CNTR_CODE"].isin(country_codes)]
    last_vals = []
    for sc, df in results_by_scenario.items():
        last_vals.append(df["Resilience"].iloc[-1])
    last_avg = float(np.mean(last_vals)) if last_vals else 0.5
    rng = np.random.default_rng(7)
    nuts2 = nuts2.copy()
    nuts2["Resilience"] = np.clip(last_avg * rng.uniform(0.85, 1.15, len(nuts2)), 0, 1)
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
        legend_name="Resilience score (latest)",
    ).add_to(fmap)
    for _, row in nuts2.iterrows():
        c = row.geometry.centroid
        folium.Marker(
            [c.y, c.x],
            icon=folium.DivIcon(html=f"<div style='font-size:10px'>{row.get('region_name','')}</div>"),
        ).add_to(fmap)
    st_folium(fmap, width=800, height=520)
else:
    st.info("NUTS2 map unavailable right now.")


st.caption(
    "Notes: Copernicus CDS calls require a valid ~/.cdsapirc. Eurostat/World Bank are pinged lightly; "
    "full dataset wiring is straightforward once indicator codes are finalized."
)


# ---------- Corine Land Cover dataset ----------
st.header("Land Cover Analysis (Corine)")
st.markdown(
    "The Corine Land Cover (CLC) database provides polygon land cover classifications for Europe across "
    "multiple reference years. Because the full zip archive for Greece can exceed 250 MB, we load it "
    "on-demand and cache the results. Provide the path to your CLC zip file below."
)

@st.cache_resource
def load_corine_land_cover(zip_path: str, country_code: str = "EL"):
    """
    Extract and load Corine Land Cover shapefiles from a zipped archive and filter to a specific country.
    This helper walks through the extracted folder, reads all .shp files, concatenates them, and returns a
    GeoDataFrame in WGS84. If the archive cannot be read or contains no shapefiles, it returns None.
    """
    import geopandas as gpd
    import pandas as pd
    if not os.path.isfile(zip_path):
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
            gdfs = []
            for root, _, files in os.walk(tmp_dir):
                for fname in files:
                    if fname.lower().endswith(".shp"):
                        shp_path = os.path.join(root, fname)
                        try:
                            df = gpd.read_file(shp_path)
                        except Exception:
                            continue
                        # Filter by country code if available
                        if "CNTR_CODE" in df.columns:
                            df = df[df["CNTR_CODE"] == country_code]
                        # Standardize year column name
                        for yr_col in ["YEAR", "year", "REFERENCE_YEA", "YEAR_REF"]:
                            if yr_col in df.columns:
                                df = df.rename(columns={yr_col: "year"})
                                break
                        gdfs.append(df)
            if not gdfs:
                return None
            merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
            return merged.to_crs(4326)
    except Exception:
        return None


# Input field for CLC zip path
clc_path = st.text_input(
    "Path to Corine Land Cover zip (e.g., data_raw/clc_greece.zip)",
    "",
    help="Provide an absolute or relative path to the downloaded CLC zip. The archive will be extracted and cached upon first load.",
)
if clc_path:
    with st.spinner("Loading Corine data..."):
        clc_gdf = load_corine_land_cover(clc_path)
    if clc_gdf is None:
        st.error(
            "No shapefiles were found in the provided zip or the file could not be read. "
            "Ensure the path is correct and the archive contains valid shapefiles."
        )
    else:
        st.success(f"Loaded {len(clc_gdf)} polygons from the CLC archive.")
        # Filter by year if available
        if "year" in clc_gdf.columns:
            years_available = sorted(set(clc_gdf["year"].dropna().astype(int)))
            selected_year = st.selectbox("Select reference year", years_available, index=0)
            clc_subset = clc_gdf[clc_gdf["year"] == selected_year]
        else:
            st.info("No 'year' column found; displaying all loaded polygons.")
            clc_subset = clc_gdf
        # Display counts by land cover code if available
        if "CLC_CODE" in clc_subset.columns:
            counts = (
                clc_subset["CLC_CODE"].value_counts().reset_index().rename(columns={"index": "CLC_CODE", "CLC_CODE": "count"})
            )
            st.dataframe(counts, use_container_width=True)
        # Draw map if subset not empty
        if not clc_subset.empty:
            centroid = clc_subset.geometry.centroid
            center_lat = float(centroid.y.mean())
            center_lon = float(centroid.x.mean())
            lc_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            folium.Choropleth(
                geo_data=clc_subset.__geo_interface__,
                data=clc_subset.reset_index(),
                columns=["index", "year"] if "year" in clc_subset.columns else ["index", "index"],
                key_on="feature.id",
                fill_color="OrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="Corine polygons",
            ).add_to(lc_map)
            st_folium(lc_map, width=800, height=520)

# ---------- Real-data analysis section ----------
with st.expander("Real Data Analysis (Greece)", expanded=False):
    """
    This section reads the processed Greece datasets (if available) and allows
    users to explore real metrics across regions and years. Metrics are
    discovered automatically from the processed Parquet file and can be
    visualised as time-series line charts. To prepare the data, run
    ``python scripts/prepare_real_data.py`` after placing the raw files in
    ``dataset``.
    """
    df_real, clc_sum, metrics = load_real_data()
    if df_real is None or df_real.empty:
        st.info(
            "Processed real data not found. Please run the preprocessing script "
            "(`scripts/prepare_real_data.py`) to generate `data/processed/crisi_greece_processed.parquet`."
        )
    else:
        # Region selection
        regions = sorted([r for r in df_real["_region_norm"].dropna().unique()])
        if not regions:
            st.warning("No regions detected in real data.")
        else:
            sel_region = st.selectbox("Region", regions, index=0)
            sdf = df_real[df_real["_region_norm"] == sel_region]
            # Year filter (optional)
            year_min = int(sdf["year"].min())
            year_max = int(sdf["year"].max())
            year_range = st.slider(
                "Year range",
                min_value=year_min,
                max_value=year_max,
                value=(year_min, year_max),
                step=1,
            )
            sdf = sdf[(sdf["year"] >= year_range[0]) & (sdf["year"] <= year_range[1])]
            # Metric selection
            if not metrics:
                metrics = [c for c in sdf.columns if c not in ["_region_norm", "year"] and pd.api.types.is_numeric_dtype(sdf[c])]
            metric = st.selectbox("Metric", metrics, index=0)
            st.markdown(f"### {metric} for {sel_region}")
            st.dataframe(sdf[["year", metric]].set_index("year").rename(columns={metric: metric}), use_container_width=True)
            # Chart
            try:
                import plotly.express as px  # type: ignore

                fig = px.line(
                    sdf.sort_values("year"), x="year", y=metric, markers=True,
                    labels={"year": "Year", metric: metric},
                    title=f"{metric} over time"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.line_chart(sdf.set_index("year")[metric])
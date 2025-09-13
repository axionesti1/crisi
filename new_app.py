import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

# For data APIs
import cdsapi                # Copernicus Climate Data Store API

# For ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
# Optionally, include XGBoost (ensure it's installed in requirements)
# from xgboost import XGBRegressor

# For explainability
import shap
import matplotlib.pyplot as plt

# Config streamlit page
st.set_page_config(page_title="CRISI Tourism-Climate Resilience Dashboard", layout="wide")

st.title("🌍 CRISI – Tourism Climate Resilience Dashboard")
st.write("This app links climate hazards with tourism economics to assess the resilience of destinations under climate change.")

# --- Data Loading and Caching ---

@st.cache_data(show_spinner=True)
def load_eurostat_data(year=2019):
    """
    Fetch tourism & GDP data from Eurostat REST API (no pandasdmx).
    - tour_occ_nin2: nights spent at tourist accommodations (NUTS2)
    - nama_10r_2gdp: regional GDP (NUTS2)
    Returns: DataFrame [region_code, tourist_nights, gdp_meur]
    """
    import requests
    import pandas as pd

    BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

    def fetch_jsonstat(dataset, params):
        # New Eurostat endpoint – no language segment; returns SDMX-JSON
        url = BASE + dataset
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def jsonstat_to_df(js):
        # Generic SDMX-JSON → DataFrame
        dim_ids = js.get("id") or list(js["dimension"].keys())
        cats = {d: js["dimension"][d]["category"] for d in dim_ids}
        codes_by_dim = {
            d: [code for code, _idx in sorted(cats[d]["index"].items(), key=lambda kv: kv[1])]
            for d in dim_ids
        }
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
        # standardize common cols
        if "geo" in df.columns:
            df = df.rename(columns={"geo": "region_code"})
        if "time" in df.columns:
            df = df.rename(columns={"time": "year"})
            # make year int if it looks like one
            with pd.option_context("mode.chained_assignment", None):
                df["year"] = pd.to_numeric(df["year"], errors="ignore")
        return df

    # ---- Nights spent (tour_occ_nin2) ----
    # Keep params minimal; Eurostat 404s if you pass invalid filters.
    nights_js = fetch_jsonstat("tour_occ_nin2", {"time": str(year)})
    df_n = jsonstat_to_df(nights_js)

    # Filter/aggregate to a single 'total' slice if extra dims exist
    keep_cols = {"region_code", "year", "value"}
    extra_dims = [c for c in df_n.columns if c not in keep_cols]
    # Prefer any 'TOTAL'-like codes if present; otherwise pick the first category to avoid double counting
    pref_totals = {"TOTAL", "TOT", "ALL"}
    for c in extra_dims:
        if df_n[c].isin(pref_totals).any():
            df_n = df_n[df_n[c].isin(pref_totals)]
        else:
            first_code = df_n[c].iloc[0]
            df_n = df_n[df_n[c] == first_code]
    df_n = df_n.rename(columns={"value": "tourist_nights"})
    df_n = df_n[["region_code", "tourist_nights"]]
    # NUTS2 codes are typically ≥4 chars (e.g., EL30, DE21). Keep those.
    df_n = df_n[df_n["region_code"].str.len().ge(4)]

    # ---- Regional GDP (nama_10r_2gdp) ----
    # Again, minimal params (just time). We'll filter to the main GDP concept afterward.
    gdp_js = fetch_jsonstat("nama_10r_2gdp", {"time": str(year)})
    df_g = jsonstat_to_df(gdp_js)

    # Try to prefer main GDP concept & unit if present
    # Common codes: na_item='B1GQ' (GDP) and unit like 'MIO_EUR' or 'CP_MEUR' depending on dataset.
    for col, desired in (("na_item", {"B1GQ"}), ("unit", {"MIO_EUR", "CP_MEUR", "MIO_NAC"})):
        if col in df_g.columns:
            if df_g[col].isin(desired).any():
                df_g = df_g[df_g[col].isin(desired)]
            else:
                # fallback: first category
                df_g = df_g[df_g[col] == df_g[col].iloc[0]]

    if "geo" in df_g.columns:
        df_g = df_g.rename(columns={"geo": "region_code"})
    df_g = df_g.rename(columns={"value": "gdp_meur"})
    df_g = df_g[df_g["region_code"].str.len().ge(4)]
    df_g = df_g[["region_code", "gdp_meur"]]

    # Merge
    out = pd.merge(df_n, df_g, on="region_code", how="inner")
    out["tourist_nights"] = pd.to_numeric(out["tourist_nights"], errors="coerce")
    out["gdp_meur"] = pd.to_numeric(out["gdp_meur"], errors="coerce")
    out = out.dropna(subset=["tourist_nights", "gdp_meur"]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=True)
def load_worldbank_data(year: int, country_codes_iso2: list[str]) -> dict:
    """
    Fetch World Bank indicators for specified ISO2 countries via REST API.
    Returns: dict like {"GR": {"population": 10_482_487, "gdp_per_capita": 19876.5}, ...}
    """
    import math

    indicators = {
        "SP.POP.TOTL": "population",
        "NY.GDP.PCAP.CD": "gdp_per_capita",
    }

    # WB API allows multiple countries separated by ';' (ISO2 code expected in 'country.id')
    country_param = ";".join(sorted(set(cc.upper() for cc in country_codes_iso2)))

    base = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
    params = {"date": str(year), "format": "json", "per_page": "20000"}

    store: dict[str, dict] = {}

    for ind_code, out_name in indicators.items():
        url = base.format(countries=country_param, indicator=ind_code)
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        js = resp.json()

        # Response format: [metadata, data_list]
        if not isinstance(js, list) or len(js) < 2 or js[1] is None:
            # No data available; continue with NAs
            continue

        for row in js[1]:
            # row['country']['id'] is ISO2
            try:
                iso2 = row["country"]["id"]
            except Exception:
                continue
            val = row.get("value", None)
            # Coerce to float where possible
            if val is None:
                num = math.nan
            else:
                try:
                    num = float(val)
                except Exception:
                    num = math.nan

            if iso2 not in store:
                store[iso2] = {}
            store[iso2][out_name] = num

    return store


@st.cache_data(show_spinner=True)
def load_climate_data():
    """Fetch or simulate climate hazard indicators (baseline + scenario deltas)."""
    # In a real app, we might retrieve data via cdsapi. Here, we'll simulate a climate metric.
    # For example, baseline average summer temperature by region (simulated) and an increase under scenarios.
    # We'll create a dummy dictionary of region_code -> baseline_temp.
    baseline_temp = {}
    for region in data['region_code'].unique():
        # simulate baseline temp by latitude: just random or based on region code
        baseline_temp[region] = 25 + (hash(region) % 10) * 0.1  # random-ish base temp
    return baseline_temp

# --- Load and combine data ---
with st.spinner("Loading data from Eurostat and World Bank..."):
    try:
        data = load_eurostat_data(year=2019)
    except Exception as e:
        st.error(f"Error fetching Eurostat data: {e}")
        st.stop()

    # derive ISO2 from NUTS2 (first 2 chars; map exceptions)
    def map_country_code(nuts_code):
        cc = nuts_code[:2].upper()
        cc_map = {'EL': 'GR', 'UK': 'GB'}  # NUTS->ISO2 fixes
        return cc_map.get(cc, cc)

    data["country_code"] = data["region_code"].apply(map_country_code)

    unique_countries = sorted(data["country_code"].dropna().unique().tolist())
    country_stats = load_worldbank_data(year=2019, country_codes_iso2=unique_countries)

# Merge country-level data into regional dataframe
# Assume region_code first 2 letters correspond to country (NUTS2).
def map_country_code(nuts_code):
    cc = nuts_code[:2].upper()
    # Map exceptions: NUTS uses 'EL' for Greece, 'UK' for United Kingdom, etc.
    cc_map = {'EL': 'GR', 'UK': 'GB'}  # ISO2: Greece->GR, UK->GB
    return cc_map.get(cc, cc)

data['country_code'] = data['region_code'].apply(map_country_code)
# Add WB indicators to each region row
data['population'] = data['country_code'].apply(lambda cc: country_stats.get(cc, {}).get('population', np.nan))
data['gdp_per_capita'] = data['country_code'].apply(lambda cc: country_stats.get(cc, {}).get('gdp_per_capita', np.nan))

# Compute additional features
data['tourism_intensity'] = data['tourist_nights'] / data['population']  # nights per person, as a tourism dependence metric
# Handle any infinities or missing:
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with missing key data if any (simplify)
data.dropna(subset=['tourist_nights', 'gdp_meur', 'population'], inplace=True)

# --- Define target (Resilience Score) ---
# Here we create a simple resilience score for demonstration.
# We'll say higher tourism intensity and higher temperature increase => higher risk => lower resilience.
# Higher GDP per cap => more adaptive capacity => higher resilience.
# This is an illustrative formula.
data['resilience_score'] = 0.0
# Normalize indicators for combination
# (In practice, use proper normalization as per config weights:contentReference[oaicite:19]{index=19}:contentReference[oaicite:20]{index=20})
tn_max = data['tourism_intensity'].max()
gdp_max = data['gdp_per_capita'].max()
data['norm_tourism_intensity'] = data['tourism_intensity'] / tn_max
data['norm_gdp_per_capita'] = data['gdp_per_capita'] / gdp_max
# Assign a dummy climate risk indicator (this will be updated per scenario)
data['climate_risk'] = 0.0  # placeholder, e.g. projected temp increase (will set later)

# Initial climate data (baseline)
baseline_temp = load_climate_data()
# We use baseline_temp to fill climate_risk initially as baseline temperature (normalized)
temps = []
for region in data['region_code']:
    temps.append(baseline_temp.get(region, 0.0))
data['baseline_temp'] = temps
# Normalize baseline_temp for use
if len(data) > 0:
    data['norm_baseline_temp'] = (data['baseline_temp'] - data['baseline_temp'].min()) / (data['baseline_temp'].max() - data['baseline_temp'].min())
else:
    data['norm_baseline_temp'] = 0.0
# Set initial climate risk indicator as normalized baseline temp (as a proxy: higher temp => more climate risk for tourism maybe)
data['climate_risk'] = data['norm_baseline_temp']

# Now define resilience_score (simple formula: higher GDP => higher resilience, higher tourism intensity or climate risk => lower resilience)
data['resilience_score'] = (data['norm_gdp_per_capita'] * 0.4) - (data['norm_tourism_intensity'] * 0.3) - (data['climate_risk'] * 0.3)
# Scale to 0-100
min_res, max_res = data['resilience_score'].min(), data['resilience_score'].max()
data['resilience_score'] = 100 * (data['resilience_score'] - min_res) / (max_res - min_res + 1e-9)

# --- Sidebar for scenario selection ---
st.sidebar.header("Scenario Selection")
scenario = st.sidebar.selectbox("Climate Scenario", ["RCP4.5", "RCP8.5"])
year = st.sidebar.selectbox("Year", [2030, 2050])

st.sidebar.write("Selected scenario:", scenario, year)

# Adjust climate risk based on scenario (simple multiplier for demo)
# In a real case, we'd retrieve actual projections for the selected year & scenario.
if scenario == "RCP4.5":
    # Moderate increase by selected year
    # e.g. +1°C by 2050 (relative to baseline), scaled by year
    temp_increase = 1.0 * ((year - 2019) / (2050 - 2019))  # proportion of 1°C
elif scenario == "RCP8.5":
    # Larger increase, e.g. +3°C by 2050
    temp_increase = 3.0 * ((year - 2019) / (2050 - 2019))
else:
    temp_increase = 0.0

# Update climate_risk feature: assume climate_risk = normalized (baseline_temp + temp_increase)
data['proj_temp'] = data['baseline_temp'] + temp_increase
# Normalize projected temperature similarly
data['climate_risk'] = (data['proj_temp'] - data['proj_temp'].min()) / (data['proj_temp'].max() - data['proj_temp'].min() + 1e-9)

# Recalculate resilience_score for the scenario (keeping same formula structure)
data['resilience_score'] = (data['norm_gdp_per_capita'] * 0.4) - (data['norm_tourism_intensity'] * 0.3) - (data['climate_risk'] * 0.3)
# Rescale to 0-100
min_res, max_res = data['resilience_score'].min(), data['resilience_score'].max()
data['resilience_score'] = 100 * (data['resilience_score'] - min_res) / (max_res - min_res + 1e-9)

# --- Machine Learning: Train model on current data (baseline) to predict resilience_score ---
# Features for model (we exclude region identifiers and direct target)
feature_cols = ['tourism_intensity', 'gdp_per_capita', 'climate_risk']
X = data[feature_cols].fillna(0.0)
y = data['resilience_score']

# Split data for training (though we might use all data if it's small)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with hyperparameter tuning (Random Forest)
rf = RandomForestRegressor(random_state=42)
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5],
}
cv = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=3, scoring='neg_mean_squared_error', random_state=42)
cv.fit(X_train, y_train)
best_model = cv.best_estimator_

# Evaluate model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Test MSE: {mse:.2f}, R²: {r2:.2f}")
# (In a real scenario with more data, we’d also display cross-val scores, etc.)

# Feature importances from model
importances = None
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    importance_df.sort_values('importance', ascending=False, inplace=True)
    st.write("Feature importances (model):")
    st.dataframe(importance_df.reset_index(drop=True))

# SHAP explainability on a sample of training data
st.subheader("Feature Importance and Explainability")
# Using SHAP to explain the model predictions
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_train)

# Bar chart of mean absolute SHAP values for each feature
shap_avg = np.mean(np.abs(shap_values.values), axis=0)
shap_importance_df = pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': shap_avg})
shap_importance_df.sort_values('mean_abs_shap', ascending=False, inplace=True)
st.write("Mean absolute SHAP value (feature importance):")
st.dataframe(shap_importance_df.reset_index(drop=True))

# We can also plot SHAP summary plot (as a static image)
fig, ax = plt.subplots()
shap.summary_plot(shap_values.values, X_train, feature_names=feature_cols, plot_type="bar", show=False)
plt.tight_layout()
st.pyplot(fig)

# --- Map Visualization ---
st.subheader("Resilience Scores by Region")
# Load geojson for NUTS2 regions (from GISCO)
geo_url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_BN_03M_2024_4326_LEVL_2.geojson"
try:
    geojson_data = requests.get(geo_url).json()
except Exception as e:
    st.error("Could not load region boundaries GeoJSON.")
    geojson_data = None

if geojson_data:
    import plotly.express as px
    # Prepare data for mapping
    map_df = data[['region_code', 'resilience_score']].copy()
    map_df['resilience_score'] = map_df['resilience_score'].round(1)
    fig_map = px.choropleth(
        map_df, geojson=geojson_data, locations='region_code', featureidkey="properties.NUTS_ID",
        color='resilience_score', color_continuous_scale="YlGnBu", range_color=(0,100),
        labels={'resilience_score': 'Resilience Score'}, title=f"Resilience Score ({scenario} {year})"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # Add an explanation or caption for the map
    st.caption("Choropleth map of tourism-climate resilience scores for each NUTS2 region. Use the sidebar to change scenario assumptions.")

# --- Data Download ---
st.subheader("Download Results")
csv_data = data[['region_code','resilience_score', 'tourist_nights','population','gdp_meur','gdp_per_capita']].copy()
csv_data['scenario'] = f"{scenario}_{year}"
csv = csv_data.to_csv(index=False)
st.download_button("Download CSV of results", data=csv, file_name=f"resilience_scores_{scenario}_{year}.csv", mime="text/csv")

# Optionally, allow downloading the map as an image (using plotly static image if kaleido is installed)
try:
    img_bytes = fig_map.to_image(format="png")
    st.download_button("Download map as PNG", data=img_bytes, file_name=f"resilience_map_{scenario}_{year}.png", mime="image/png")
except Exception as e:
    st.write("Use the plotly toolbar to save the map as an image.")

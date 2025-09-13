import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

# For data APIs

import wbdata                # World Bank data
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
    Fetch tourism & economic data from Eurostat REST API (no pandasdmx).
    - tour_occ_nin2: nights spent at tourist accommodation (NUTS2)
    - nama_10r_2gdp: regional GDP (Mio EUR, NUTS2)
    Returns: DataFrame with columns [region_code, tourist_nights, gdp_meur]
    """
    import itertools

    def fetch_jsonstat(dataset_code, params):
        base = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/JSON/en/"
        url = base + dataset_code
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def jsonstat_to_df(js):
        # Parse JSON-stat (Eurostat v1.0) into a flat DataFrame of dimension codes + value
        dim = js["dimension"]
        dim_ids = js["id"] if "id" in js else list(dim.keys())
        categories = {d: dim[d]["category"] for d in dim_ids}
        # ordered codes per dimension
        code_lists = {
            d: [c for c, _ in sorted(categories[d]["index"].items(), key=lambda kv: kv[1])]
            for d in dim_ids
        }
        # total size (cartesian product)
        sizes = [len(code_lists[d]) for d in dim_ids]
        # values can be dense list or sparse dict
        values = js["value"]
        records = []

        if isinstance(values, list):
            # dense array (length == product of sizes)
            for flat_idx, v in enumerate(values):
                if v is None:
                    continue
                idxs = []
                n = flat_idx
                # decode flat index into multi-index
                for size in reversed(sizes):
                    idxs.append(n % size)
                    n //= size
                idxs = list(reversed(idxs))
                rec = {}
                for d, i in zip(dim_ids, idxs):
                    rec[d] = code_lists[d][i]
                rec["value"] = v
                records.append(rec)
        else:
            # sparse dict {flat_idx_str: value}
            for flat_idx_str, v in values.items():
                if v is None:
                    continue
                flat_idx = int(flat_idx_str)
                idxs = []
                n = flat_idx
                for size in reversed(sizes):
                    idxs.append(n % size)
                    n //= size
                idxs = list(reversed(idxs))
                rec = {}
                for d, i in zip(dim_ids, idxs):
                    rec[d] = code_lists[d][i]
                rec["value"] = v
                records.append(rec)

        return pd.DataFrame.from_records(records)

    # --- Nights spent at tourist accommodation (NUTS2) ---
    # dataset: tour_occ_nin2
    # filters: unit=NR (number), and time=year
    nights_js = fetch_jsonstat(
        "tour_occ_nin2",
        {"time": str(year), "unit": "NR"}
    )
    df_n = jsonstat_to_df(nights_js)
    # standardize column names
    if "geo" in df_n.columns:
        df_n.rename(columns={"geo": "region_code"}, inplace=True)
    if "time" in df_n.columns:
        df_n.rename(columns={"time": "year"}, inplace=True)
    df_n = df_n[["region_code", "year", "value"]]
    df_n = df_n.rename(columns={"value": "tourist_nights"})
    # keep only NUTS2 (Eurostat codes are like EL30, DE60 etc.; ensure length >=4 and no “TOTAL”)
    df_n = df_n[df_n["region_code"].str.len().ge(4)].copy()

    # --- Regional GDP at current market prices (NUTS2) ---
    # dataset: nama_10r_2gdp (value in million EUR)
    gdp_js = fetch_jsonstat(
        "nama_10r_2gdp",
        {"time": str(year)}
    )
    df_g = jsonstat_to_df(gdp_js)
    if "geo" in df_g.columns:
        df_g.rename(columns={"geo": "region_code"}, inplace=True)
    if "time" in df_g.columns:
        df_g.rename(columns={"time": "year"}, inplace=True)
    df_g = df_g[["region_code", "year", "value"]]
    df_g = df_g.rename(columns={"value": "gdp_meur"})
    df_g = df_g[df_g["region_code"].str.len().ge(4)].copy()

    # Merge on region_code
    df = pd.merge(
        df_n[["region_code", "tourist_nights"]],
        df_g[["region_code", "gdp_meur"]],
        on="region_code",
        how="inner"
    )

    # basic sanity cast
    df["tourist_nights"] = pd.to_numeric(df["tourist_nights"], errors="coerce")
    df["gdp_meur"] = pd.to_numeric(df["gdp_meur"], errors="coerce")

    # Drop NAs
    df = df.dropna(subset=["tourist_nights", "gdp_meur"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=True)
def load_worldbank_data(year=2019):
    """Fetch country-level socio-economic indicators from World Bank for given year."""
    # Define indicators to fetch
    indicators = {
        'SP.POP.TOTL': 'population',
        'NY.GDP.PCAP.CD': 'gdp_per_capita'
        # (Add more indicators as needed, e.g. tourism % GDP if available)
    }
    # Fetch data for all countries for the specified year
    df_wb = wbdata.get_dataframe(indicators, country='all', data_date=str(year))
    df_wb = df_wb.reset_index()
    # The DataFrame typically has columns: country, date, indicator values
    df_wb = df_wb[df_wb['date'] == year]  # filter the year if needed
    df_wb = df_wb[['country', 'population', 'gdp_per_capita']]
    # Country codes in wbdata might be ISO3 or ISO2 country codes. We'll assume ISO2 for mapping.
    # Create a lookup from country code to indicators
    country_data = {}
    for _, row in df_wb.iterrows():
        country_code = str(row['country']).upper()
        country_data[country_code] = {
            'population': row['population'],
            'gdp_per_capita': row['gdp_per_capita']
        }
    return country_data

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
    country_stats = load_worldbank_data(year=2019)

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

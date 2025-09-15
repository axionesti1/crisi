import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="CRISI Model Explorer", layout="wide")
st.title("🌍 CRISI: Climate Resilience Investment Scoring (Mock Implementation)")

# Scenario options (user-selectable RCP/SSP narratives)
scenario = st.selectbox(
    "Select Scenario (RCP/SSP pathway):",
    [
        "Green Global Resilience (RCP4.5/SSP1)",
        "Business-as-Usual Drift (RCP6.0/SSP2)",
        "Divided Disparity (RCP6.0-like/SSP4)",
        "Techno-Optimism on a Hot Planet (RCP8.5/SSP5)",
        "Regional Fortress World (RCP7.0/SSP3)",
    ]
)
st.markdown(f"**Selected scenario:** {scenario}")

# Sidebar: Pillar weights (alpha, beta, gamma)
st.sidebar.header("Pillar Weights")
alpha = st.sidebar.slider("Exposure weight (α)", 0.0, 1.0, 0.33, step=0.01)
beta  = st.sidebar.slider("Sensitivity weight (β)", 0.0, 1.0, 0.33, step=0.01)
gamma = st.sidebar.slider("Adaptive capacity weight (γ)", 0.0, 1.0, 0.33, step=0.01)

# Sidebar: Indicator weights
st.sidebar.header("Indicator Weights")
w_heat = st.sidebar.slider("Heat index weight (vs Drought)", 0.0, 1.0, 0.5, step=0.01)
w_gdp  = st.sidebar.slider("Tourism GDP Share weight (vs Seasonality)", 0.0, 1.0, 0.5, step=0.01)

# Generate mock time series (years 2025–2055)
years = np.arange(2025, 2056)
n_years = len(years)

# Define scenario-based factors for mock data
if scenario.startswith("Green Global"):
    climate_factor = 1.0
    tourism_share = 0.30
    demand_drop = 0.0
elif scenario.startswith("Business-as-Usual"):
    climate_factor = 1.2
    tourism_share = 0.35
    demand_drop = 0.0
elif scenario.startswith("Divided Disparity"):
    climate_factor = 1.3
    tourism_share = 0.40
    demand_drop = 0.0
elif scenario.startswith("Techno-Optimism"):
    climate_factor = 1.5
    tourism_share = 0.45
    demand_drop = 0.0
elif scenario.startswith("Regional Fortress"):
    climate_factor = 1.4
    tourism_share = 0.65
    demand_drop = 25.0
else:
    climate_factor = 1.0
    tourism_share = 0.35
    demand_drop = 0.0

# 1) Exposure indices (scaled 0–1): heat and drought increase with time/climate factor
np.random.seed(0)
heat_index = np.linspace(0.1, 0.1 + 0.6 * climate_factor, n_years) \
             + np.random.normal(0, 0.02, n_years)
drought_index = np.linspace(0.2, 0.2 + 0.4 * climate_factor, n_years) \
                + np.random.normal(0, 0.02, n_years)
heat_index = np.clip(heat_index, 0, 1)
drought_index = np.clip(drought_index, 0, 1)

# 2) Sensitivity indices (0–1): tourism share (constant) and seasonality trend
tourism_share_series = np.full(n_years, tourism_share)
seasonality = np.linspace(0.3, 0.3 + 0.1 * climate_factor, n_years) \
              + np.random.normal(0, 0.01, n_years)
seasonality = np.clip(seasonality, 0, 1)

# 3) Adaptive capacity index (0–1): baseline + small improvement
# (e.g. higher in optimistic scenarios)
if scenario.startswith("Green Global"):
    readiness_base = 0.8
elif scenario.startswith("Techno-Optimism"):
    readiness_base = 0.7
elif scenario.startswith("Business-as-Usual"):
    readiness_base = 0.6
elif scenario.startswith("Regional Fortress"):
    readiness_base = 0.3
elif scenario.startswith("Divided Disparity"):
    readiness_base = 0.5
else:
    readiness_base = 0.5
readiness_index = np.linspace(readiness_base, min(readiness_base + 0.1, 1.0), n_years)

# Compute pillar scores by combining indicators with user weights
exposure_score = w_heat * heat_index + (1 - w_heat) * drought_index
sensitivity_score = w_gdp * tourism_share_series + (1 - w_gdp) * seasonality
adaptive_score = readiness_index

# Normalize pillar weights (to sum=1)
weight_sum = alpha + beta + gamma
if weight_sum == 0:
    alpha_norm = beta_norm = gamma_norm = 1/3
else:
    alpha_norm = alpha / weight_sum
    beta_norm  = beta  / weight_sum
    gamma_norm = gamma / weight_sum

# Resilience and risk indices (0–1)
resilience = (alpha_norm * (1 - exposure_score) +
              beta_norm  * (1 - sensitivity_score) +
              gamma_norm * adaptive_score)
resilience = np.clip(resilience, 0, 1)
risk = (alpha_norm * exposure_score +
        beta_norm  * sensitivity_score +
        gamma_norm * (1 - adaptive_score))
risk = np.clip(risk, 0, 1)

# ----- Outputs -----
st.subheader("Resilience Score Projection (2025–2055)")
res_df = pd.DataFrame({"Year": years, "Resilience Score": resilience})
res_df = res_df.set_index("Year")
st.line_chart(res_df)  # time-series chart

st.subheader("Yearly Pillar Scores and Risk")
table_df = pd.DataFrame({
    "Year": years,
    "Exposure": np.round(exposure_score, 3),
    "Sensitivity": np.round(sensitivity_score, 3),
    "Adaptive": np.round(adaptive_score, 3),
    "Resilience": np.round(resilience, 3),
    "Risk": np.round(risk, 3)
})
table_df = table_df.set_index("Year")
st.dataframe(table_df)  # interactive table

# Guardrail flag
if (tourism_share > 0.5) and (demand_drop > 20):
    st.warning(
        "⚠️ **Guardrail Alert:** Tourism GDP share > 50% *and* demand drop > 20%. "
        "This scenario breaches the resilience guardrail condition."
    )

# CSV download
csv = table_df.reset_index().to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Results as CSV",
    data=csv,
    file_name="crisi_resilience_projections.csv",
    mime="text/csv"
)

st.subheader("Resilience Scores by Region (Choropleth)")

# Mock GeoJSON for three example regions
regions_geo = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"id": "A", "name": "Region A"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[10, 10], [10, 11], [11, 11], [11, 10], [10, 10]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"id": "B", "name": "Region B"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[20, 10], [20, 11], [21, 11], [21, 10], [20, 10]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"id": "C", "name": "Region C"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[10, 20], [10, 21], [11, 21], [11, 20], [10, 20]]]
            }
        }
    ]
}

# Assign mock final-year resilience scores to each region
final_res = resilience[-1]
regions_data = pd.DataFrame({
    "id": ["A", "B", "C"],
    "Resilience": [
        np.clip(final_res * 1.0, 0, 1),
        np.clip(final_res * 0.8, 0, 1),
        np.clip(final_res * 1.2, 0, 1)
    ]
})

# Create Folium map
m = folium.Map(location=[15, 15], zoom_start=4)
folium.Choropleth(
    geo_data=regions_geo,
    data=regions_data,
    columns=["id", "Resilience"],
    key_on="feature.properties.id",
    fill_color="YlGn", fill_opacity=0.7, line_opacity=0.5,
    legend_name="Resilience Score"
).add_to(m)

# Label regions by centroid
for feature in regions_geo["features"]:
    coords = np.array(feature["geometry"]["coordinates"][0])
    centroid = [coords[:,1].mean(), coords[:,0].mean()]
    folium.map.Marker(
        location=[centroid[0], centroid[1]],
        icon=folium.DivIcon(html=f"<div style='font-size:12pt'>{feature['properties']['name']}</div>")
    ).add_to(m)

# Display the interactive map
st_data = st_folium(m, width=700, height=500)

# Note: All data here are synthetic.  For a real application, replace the mock series
# above with actual climate projections, tourism/economic data, and/or an ML scoring model:contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}.

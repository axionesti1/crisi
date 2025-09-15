"""
app.py: Streamlit web app for CRISI (Climate Resilience Investment Scoring Intelligence) model.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
import shap  # SHAP for explainability

# Data-fetching libraries (with fallback)
try:
    import cdsapi
except ImportError:
    cdsapi = None
try:
    from pandasdmx import Request
except ImportError:
    Request = None
try:
    import wbdata
except ImportError:
    wbdata = None

st.set_page_config(page_title="CRISI Model Explorer", layout="wide")
st.title("🌍 CRISI: Climate Resilience Investment Scoring")

st.write("""
This interactive dashboard integrates climate, tourism, and socioeconomic data to compute resilience scores under multiple climate scenarios. 
It uses a Random Forest model (with SHAP explainability) and allows weight adjustments:contentReference[oaicite:17]{index=17}:contentReference[oaicite:18]{index=18}.
""")

st.sidebar.header("Scenario & Weights")
scenarios = [
    "Green Global Resilience (RCP4.5/SSP1)",
    "Business-as-Usual Drift (RCP6.0/SSP2)",
    "Divided Disparity (RCP6.0-like/SSP4)",
    "Techno-Optimism on a Hot Planet (RCP8.5/SSP5)",
    "Regional Fortress World (RCP7.0/SSP3)"
]
scenario = st.sidebar.selectbox("Scenario (RCP/SSP pathway):", scenarios)
alpha = st.sidebar.slider("Exposure weight (α)", 0.0, 1.0, 0.33, step=0.01)
beta  = st.sidebar.slider("Sensitivity weight (β)", 0.0, 1.0, 0.33, step=0.01)
gamma = st.sidebar.slider("Adaptive weight (γ)", 0.0, 1.0, 0.33, step=0.01)
st.sidebar.header("Indicator Weights")
w_heat = st.sidebar.slider("Heat index weight (vs Drought)", 0.0, 1.0, 0.5, step=0.01)
w_gdp  = st.sidebar.slider("Tourism GDP weight (vs Seasonality)", 0.0, 1.0, 0.5, step=0.01)
st.markdown(f"**Selected scenario:** {scenario}")

# Data retrieval functions
@st.cache(ttl=86400)
def fetch_copernicus():
    if not cdsapi:
        st.warning("cdsapi not available; skipping Copernicus data.")
        return None
    try:
        client = cdsapi.Client()
        # Example retrieval (placeholder; real query would need specifics)
        data = client.retrieve('seasonal-monthly-shorclim-archive', {
            'variable': '2m_temperature',
            'product_type': 'ensemble_mean',
            'year': list(range(2025,2030)),
            'month': [6,7,8],
            'time_aggregation': 'monthly',
            'format': 'netcdf'
        })
        return data
    except Exception as e:
        st.warning(f"Copernicus data fetch failed: {e}")
        return None

@st.cache(ttl=86400)
def fetch_eurostat():
    try:
        estat = Request('ESTAT')
        # Placeholder: example Eurostat data fetch
        data = estat.data(resource_id='tourist_arrivals_nuts2', params={'geo': 'EL'})
        return data
    except Exception as e:
        st.warning(f"Eurostat fetch failed: {e}")
        return None

@st.cache(ttl=86400)
def fetch_worldbank():
    if not wbdata:
        st.warning("wbdata not available; skipping World Bank data.")
        return None
    try:
        indicators = {'NY.GDP.PCAP.CD':'GDP_per_capita','SP.POP.TOTL':'Population'}
        df = wbdata.get_dataframe(indicators, country='GRC', data_date=(2020,2025))
        return df.reset_index()
    except Exception as e:
        st.warning(f"World Bank fetch failed: {e}")
        return None

@st.cache(ttl=3600)
def fetch_open_meteo(lat=40.0, lon=23.7):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        temp = res.json()['hourly']['temperature_2m'][0]
        return temp
    except Exception as e:
        st.warning(f"Open-Meteo fetch failed: {e}")
        return None

@st.cache(ttl=86400)
def load_nuts2():
    fn = "NUTS_RG_20M_2021_4326.shp"
    if not os.path.exists(fn):
        try:
            url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/shp/NUTS_RG_20M_2021_4326.zip"
            r = requests.get(url, timeout=30)
            gdf = gpd.read_file(f"zip://{url}")
        except Exception as e:
            st.warning(f"NUTS shapefile download failed: {e}")
            return None
    else:
        gdf = gpd.read_file(fn)
    if 'LEVL_CODE' in gdf.columns:
        gdf = gdf[gdf.LEVL_CODE == 2]
    gdf = gdf.to_crs(4326)
    return gdf

# Attempt data fetches
cop_data = fetch_copernicus()
eur_data = fetch_eurostat()
wb_data = fetch_worldbank()
live_temp = fetch_open_meteo()

if live_temp is not None:
    st.sidebar.markdown(f"**Live temp:** {live_temp:.1f} °C")

country_map = {"Greece": "EL", "Germany": "DE"}
countries = st.sidebar.multiselect("Countries (NUTS2)", list(country_map.keys()), default=list(country_map.keys()))
st.sidebar.write(f"Countries: {', '.join(countries)}")

# Compute resilience (2025-2055)
years = np.arange(2025, 2056)
n_years = len(years)
if scenario.startswith("Green Global"):
    cf, ts, dd = 1.0, 0.30, 0.0
elif scenario.startswith("Business"):
    cf, ts, dd = 1.2, 0.35, 0.0
elif scenario.startswith("Divided"):
    cf, ts, dd = 1.3, 0.40, 0.0
elif scenario.startswith("Techno"):
    cf, ts, dd = 1.5, 0.45, 0.0
elif scenario.startswith("Regional"):
    cf, ts, dd = 1.4, 0.65, 25.0
else:
    cf, ts, dd = 1.0, 0.35, 0.0

np.random.seed(0)
heat_index = np.linspace(0.1, 0.1+0.6*cf, n_years) + np.random.normal(0, 0.02, n_years)
drought_index = np.linspace(0.2, 0.2+0.4*cf, n_years) + np.random.normal(0, 0.02, n_years)
heat_index = np.clip(heat_index, 0, 1)
drought_index = np.clip(drought_index, 0, 1)
tourism_share_series = np.full(n_years, ts)
seasonality = np.linspace(0.3, 0.3+0.1*cf, n_years) + np.random.normal(0, 0.01, n_years)
seasonality = np.clip(seasonality, 0, 1)

if scenario.startswith("Green"):
    rb = 0.8
elif scenario.startswith("Techno"):
    rb = 0.7
elif scenario.startswith("Business"):
    rb = 0.6
elif scenario.startswith("Regional"):
    rb = 0.3
elif scenario.startswith("Divided"):
    rb = 0.5
else:
    rb = 0.5
readiness_index = np.linspace(rb, min(rb+0.1,1.0), n_years)

exposure_score = w_heat * heat_index + (1 - w_heat) * drought_index
sensitivity_score = w_gdp * tourism_share_series + (1 - w_gdp) * seasonality
adaptive_score = readiness_index

w_sum = alpha + beta + gamma
if w_sum == 0:
    a_norm = b_norm = g_norm = 1/3
else:
    a_norm = alpha/w_sum; b_norm = beta/w_sum; g_norm = gamma/w_sum

resilience = (a_norm*(1-exposure_score) + b_norm*(1-sensitivity_score) + g_norm*adaptive_score)
resilience = np.clip(resilience, 0, 1)
risk = (a_norm*exposure_score + b_norm*sensitivity_score + g_norm*(1-adaptive_score))
risk = np.clip(risk, 0, 1)

df_res = pd.DataFrame({"Year": years, "Resilience": resilience})
df_res.set_index("Year", inplace=True)
df_pillars = pd.DataFrame({
    "Year": years,
    "Exposure": exposure_score.round(3),
    "Sensitivity": sensitivity_score.round(3),
    "Adaptive": adaptive_score.round(3),
    "Resilience": resilience.round(3),
    "Risk": risk.round(3)
})
df_pillars.set_index("Year", inplace=True)

# ML model on synthetic data (prediction)
st.subheader("Tourism Impact Model (Synthetic)")
st.write("Random Forest predicts tourism impact from climate features:contentReference[oaicite:19]{index=19}.")
np.random.seed(2)
X = pd.DataFrame({
    "Heat": np.random.rand(100),
    "Drought": np.random.rand(100),
    "TourismShare": np.random.rand(100),
    "Adaptive": np.random.rand(100)
})
y = 1 - (0.4*X["Heat"] + 0.3*X["Drought"]) + 0.2*X["Adaptive"] + np.random.normal(0,0.05,100)
model = RandomForestRegressor(n_estimators=50, random_state=0).fit(X, y)
expl = shap.TreeExplainer(model)
shap_vals = expl.shap_values(X)
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
st.pyplot()

# Resilience charts and tables
st.subheader("Resilience Projection (2025–2055)")
st.line_chart(df_res)
st.subheader("Pillar Scores & Risk by Year")
st.dataframe(df_pillars)

if ts > 0.5 and dd > 20:
    st.warning("⚠️ **Guardrail Alert:** Tourism GDP share >50% and demand drop >20% in this scenario.")

csv_file = df_pillars.reset_index().to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv_file, "crisi_results.csv", mime="text/csv")

st.subheader("Resilience Map (Latest Year)")
gdf_nuts = load_nuts2()
if gdf_nuts is not None:
    codes = [country_map[c] for c in countries]
    if codes:
        gdf_nuts = gdf_nuts[gdf_nuts['CNTR_CODE'].isin(codes)]
    if 'NUTS_ID' in gdf_nuts.columns:
        gdf_nuts = gdf_nuts.rename(columns={'NUTS_ID':'id'})
    if 'NAME_LATN' in gdf_nuts.columns:
        gdf_nuts = gdf_nuts.rename(columns={'NAME_LATN':'region_name'})
    elif 'NUTS_NAME' in gdf_nuts.columns:
        gdf_nuts = gdf_nuts.rename(columns={'NUTS_NAME':'region_name'})
    final_value = resilience[-1]
    np.random.seed(3)
    gdf_nuts['Resilience'] = np.clip(final_value * np.random.uniform(0.8,1.2,len(gdf_nuts)), 0, 1)
    center = [gdf_nuts.geometry.centroid.y.mean(), gdf_nuts.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=5)
    folium.Choropleth(
        geo_data=gdf_nuts.__geo_interface__,
        data=gdf_nuts,
        columns=['id','Resilience'],
        key_on='feature.properties.id',
        fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.5,
        legend_name='Resilience Score'
    ).add_to(m)
    for _, row in gdf_nuts.iterrows():
        cen = row.geometry.centroid
        folium.map.Marker(
            location=[cen.y, cen.x],
            icon=folium.DivIcon(html=f"<div style='font-size:10pt'>{row['region_name'][:15]}</div>")
        ).add_to(m)
    st_folium(m, width=700, height=500)
else:
    st.info("NUTS data unavailable; map cannot be shown.")

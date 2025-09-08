import streamlit as st
import pandas as pd
import yaml


def load_data(csv_path):
    return pd.read_csv(csv_path)


def load_scenarios(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)['scenarios']


def normalize_series(series, benefit):
    if series.nunique() == 1:
        return pd.Series([0.5] * len(series), index=series.index)
    min_val = series.min()
    max_val = series.max()
    norm = (series - min_val) / (max_val - min_val)
    return norm if benefit else 1 - norm


def compute_scores(df, indicators):
    scores = pd.Series(0.0, index=df.index)
    details = {}
    for name, config in indicators.items():
        weight = config['weight']
        benefit = config['benefit']
        norm = normalize_series(df[name], benefit)
        scores += weight * norm
        details[f'{name}_norm'] = norm
        details[f'{name}_contrib'] = weight * norm
    result = df.copy()
    for col, val in details.items():
        result[col] = val
    result['score'] = scores
    result = result.sort_values('score', ascending=False)
    return result


st.title('Tourism Resilience Scoring')
st.sidebar.header('Settings')
data_path = st.sidebar.text_input('Data CSV path', 'data/regions_sample.csv')
scenarios = load_scenarios('indicators.yaml')
scenario = st.sidebar.selectbox('Scenario', list(scenarios.keys()))
if st.sidebar.button('Run'):
    df = load_data(data_path)
    indicators = scenarios[scenario]
    result = compute_scores(df, indicators)
    st.subheader('Ranking')
    st.dataframe(result)
    st.subheader('Map')
    st.map(result[['lat', 'lon']])
    csv_data = result.to_csv(index=False).encode('utf-8')
    st.download_button('Download scored CSV', csv_data, file_name='scored_regions.csv', mime='text/csv')

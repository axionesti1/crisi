🌍 CRISI – Climate Resilience Investment Scoring Intelligence

CRISI (Climate Resilience Investment Scoring Intelligence) is an AI‑powered, open‑source tool that links climate hazards with tourism economics to assess the resilience of destinations and projects under climate change.
It produces spatially explicit resilience scores and supports policy foresight by integrating climate scenarios, economic impacts and adaptation pathways.

✨ Key Features

Tourism‑specific climate risk scoring – connects hazards (heatwaves, snow loss, floods, sea‑level rise) with regional tourism demand and GDP.

Foresight scenarios – explore resilience under multiple climate pathways (e.g. RCP 4.5 / 8.5, SSPs) and adaptation options.

Multi‑source data pipeline – automated integration of open datasets:

Eurostat
 – tourism & economics.

Copernicus CDS
 – climate projections.

EURO‑CORDEX
 – regional climate models.

World Bank
 – socioeconomic indicators.

GISCO
 – spatial data (NUTS regions).

Machine learning & explainability – predictive modelling (Random Forest, XGBoost) with interpretable outputs (SHAP values, feature importance).

Geospatial analysis – GIS joins, risk mapping and interactive dashboards.

Policy decision support – outputs include climate‑adjusted ROI, resilience maps and regional rankings for adaptation funding.

📂 Repository Structure
crisi/
│── README.md
│── LICENSE
│── requirements.txt
│── data/            # raw & processed data
│── docs/            # methodology, PhD materials, benchmarking
│── src/             # core code
│   ├── pipeline/    # data collection & preprocessing
│   ├── models/      # ML models & scoring engine
│   ├── gis/         # spatial joins & mapping
│   ├── ui/          # dashboard (Streamlit/Dash)
│   └── utils/       # helpers & config
│── notebooks/       # Jupyter experiments
└── tests/           # unit tests

⚙️ Installation

Clone the repository and set up a Python virtual environment:

git clone https://github.com/YOUR_USERNAME/crisi.git
cd crisi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

🚀 Usage

Fetch data

python src/pipeline/eurostat_api.py
python src/pipeline/copernicus_api.py


Train model

python src/models/training.py


Run dashboard

streamlit run src/ui/app_streamlit.py

📊 Example Outputs

Resilience map – Europe’s NUTS2 regions scored on climate–tourism resilience.

Scenario explorer – compare 2030 vs 2050 under RCP 4.5 vs 8.5.

Policy insights – identify tourism hotspots needing adaptation funding.

📘 References

CRISI is grounded in cutting‑edge research on climate, tourism and investment policy.
The methodology integrates:

Systematic literature review,

Multi‑source spatial–economic datasets,

Expert elicitation (Delphi method),

Machine learning and scenario planning,

Case‑study validation across Europe.

📜 License

Released under the MIT License for open science and collaboration.

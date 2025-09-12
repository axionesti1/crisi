import streamlit as st

def run_scenarios():
    st.header("Climate & Foresight Scenarios")
    scenarios = {
        "Green Resilient Europe": "EU Foresight + IPCC RCP4.5",
        "High-Tech Sustainability": "Singapore Foresight + OECD + RCP2.6",
        "Fragmented Protectionism": "OECD + World Bank stress scenarios + RCP6.0",
        "Tourism at the Crossroads": "UNWTO + EIB foresight + RCP8.5",
        "Resilient Renaissance": "EU & UN foresight + World Bank + RCP4.5",
        "Business-as-Usual Decline": "IPCC RCP8.5 baseline"
    }
    choice = st.selectbox("Select a scenario", list(scenarios.keys()))
    st.write("**Description:**", scenarios[choice])

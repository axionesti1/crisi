import streamlit as st
from modules.scenarios import run_scenarios
from modules.cba import run_cba

st.title("CRISI: Climate Resilience Investment Scoring Intelligence")

menu = ["Home", "Scenarios", "Cost-Benefit Analysis"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.write("Welcome to CRISI – AI-powered foresight and CBA tool for tourism resilience.")
elif choice == "Scenarios":
    run_scenarios()
elif choice == "Cost-Benefit Analysis":
    run_cba()

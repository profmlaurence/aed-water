import streamlit as st
from shared import load_dataset_sidebar, render_footer
from analysis import AnalysisPlots

st.set_page_config(page_title="MonitorWater - Análise Avançada", page_icon="🔬", layout="wide")

# ── Sidebar: Seleção do Dataset ──────────────────────────────────────────────
df = load_dataset_sidebar()
campanha = st.session_state.get("current_campanha", "")

st.badge(f"Campanha: {campanha}")

analysis = AnalysisPlots(df)
analysis.run()

render_footer()

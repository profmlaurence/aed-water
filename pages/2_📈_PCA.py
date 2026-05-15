import streamlit as st
from shared import load_dataset_sidebar
from pca import PlotsData

st.set_page_config(page_title="MonitorWater - Análise Exploratória", page_icon="📈", layout="wide")

# ── Sidebar: Seleção do Dataset ──────────────────────────────────────────────
df = load_dataset_sidebar()
campanha = st.session_state.get("current_campanha", "")

st.caption(f"Campanha: **{campanha}**")

plots = PlotsData(df)
plots.run()

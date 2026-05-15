import streamlit as st
from shared import load_dataset_sidebar, render_footer, setup_page
from pca import PlotsData

setup_page()

# ── Sidebar: Seleção do Dataset ──────────────────────────────────────────────
df = load_dataset_sidebar()
campanha = st.session_state.get("current_campanha", "")

st.badge(f"Campanha: {campanha}")

plots = PlotsData(df)
plots.run()

render_footer()

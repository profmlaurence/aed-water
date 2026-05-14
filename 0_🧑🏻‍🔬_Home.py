import streamlit as st
from shared import load_dataset_sidebar

st.set_page_config(page_title="Análise Exploratória de Água", page_icon="📊", layout="wide")

# ── Cabeçalho ────────────────────────────────────────────────────────────────
st.title("Análise Exploratória de Água")
st.caption("Explore as propriedades físico-químicas do dataset de qualidade da água.")
st.subheader("Mikaela")

# ── Sidebar: Seleção do Dataset ──────────────────────────────────────────────
df = load_dataset_sidebar()
campanha = st.session_state.get("current_campanha", "")

# ── Conteúdo principal ───────────────────────────────────────────────────────
st.markdown("### Dados da Campanha")
st.info(f"**Campanha selecionada:** {campanha}")

col1, col2, col3 = st.columns(3)
col1.metric("Linhas", df.shape[0])
col2.metric("Colunas", df.shape[1])
col3.metric("Valores Nulos", int(df.isnull().sum().sum()))

with st.expander("Amostra dos dados brutos", expanded=True):
    st.dataframe(df)

st.divider()

st.markdown(
    "Selecione uma das páginas no menu lateral para iniciar a análise."
)

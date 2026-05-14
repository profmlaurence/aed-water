import os
import streamlit as st
import pandas as pd


def load_dataset_sidebar() -> pd.DataFrame:
    """
    Exibe o seletor de campanha no sidebar e retorna o DataFrame carregado.
    Salva o dataset e o nome da campanha no session_state.
    """
    st.sidebar.header("Dataset")
    datasets = sorted([f for f in os.listdir("dataset") if f.endswith(".csv")])
    option_campanha = st.sidebar.radio("Campanha", datasets)

    file = pd.read_csv(f"dataset/{option_campanha}", sep=";")
    df = pd.DataFrame(file)

    # Salvar no session_state para acesso compartilhado
    st.session_state["current_dataset"] = df
    st.session_state["current_campanha"] = option_campanha

    return df


def render_footer():
    """
    Renderiza o rodapé padrão da aplicação com os logotipos do LAPEQ e UFT.
    """
    st.divider()
    
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.9em;'>"
        "Laboratório de Pesquisa em Química Ambiental e de Biocombustíveis - LAPEQ"
        "</p>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 0.5, 1, 1])
    with col2:
        st.image("assets/logo-uft.png", width=80)
    with col4:
        st.image("assets/logo-lapeq.jpg", width=80)

import os
import streamlit as st
import pandas as pd
import io
from PIL import Image


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
    st.markdown("---")

    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        st.image("assets/lapeq.avif")

    st.markdown(
        "<div style='text-align: center'><strong>Laboratório de Pesquisa em Química Ambiental e de Biocombustíveis (LAPEQ)</strong></div>", 
        unsafe_allow_html=True
    )

def conn_genai():
    """Configura e retorna o cliente do Gemini."""
    try:
        from google import genai
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except (FileNotFoundError, KeyError):
            api_key = os.environ.get("GEMINI_API_KEY")
            
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Erro ao conectar com a IA: {e}")
        return None

def analisar_dados_com_ia(dados_txt, prompt=None):
    """
    Recebe um texto com dados numéricos e solicita análise ao Gemini.
    """
    if prompt is None:
        prompt = "Analise os seguintes dados técnicos e descreva os principais padrões, correlações ou anomalias de forma clara e objetiva."

    client = conn_genai()
    if not client:
        return "⚠️ Erro de conexão com a IA."

    try:
        with st.spinner("IA analisando os dados..."):
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{prompt}\n\nDados:\n{dados_txt}"
            )
            return response.text
    except Exception as e:
        return f"❌ Erro na análise: {e}"

def analyse_graph(image_bytes, prompt="Analise este gráfico de qualidade da água e descreva os principais padrões, correlações ou anomalias de forma clara e objetiva. RESUMA EM 150 palavras no máximo"):
    """
    Recebe a imagem de um gráfico em bytes, envia para o Gemini e retorna a análise.
    """
    client = conn_genai()
    if not client:
        return "⚠️ Não foi possível conectar com a IA para analisar o gráfico."
        
    try:
        # Converte os bytes em uma imagem PIL, que é o formato esperado pelo modelo multimodal
        image = Image.open(io.BytesIO(image_bytes))
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image]
        )
        return response.text
    except Exception as e:
        st.error(f"Erro ao analisar o gráfico com a IA: {e}")
        return None

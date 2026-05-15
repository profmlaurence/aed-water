import os
import streamlit as st
import pandas as pd
import io
from PIL import Image


def setup_page():
    st.set_page_config(
        page_title="Análise Exploratória de Água",
        page_icon="📊",
        layout="wide"
    )


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
        if os.path.exists("assets/lapeq.avif"):
            st.image("assets/lapeq.avif")
        elif os.path.exists("assets/logo-lapeq.jpg"):
            st.image("assets/logo-lapeq.jpg", width=80)

    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
        "<strong>Laboratório de Pesquisa em Química Ambiental e de Biocombustíveis (LAPEQ)</strong><br>"
        "Universidade Federal do Tocantins - UFT"
        "</div>", 
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
            
        if not api_key:
            return None
            
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
        return "⚠️ Chave de API não configurada ou erro de conexão."

    try:
        with st.spinner("IA analisando os dados..."):
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{prompt}\n\nDados:\n{dados_txt}"
            )
            return response.text
    except Exception as e:
        return f"❌ Erro na análise: {e}"


def analyse_graph(image_bytes, prompt="Você é um especialista em qualidade da água auxiliando no Laboratório de Pesquisa em Química Ambiental (LAPEQ). Analise este gráfico e descreva os principais padrões, correlações ou anomalias de forma clara e objetiva. RESUMA EM 150 palavras no máximo"):
    """
    Recebe a imagem de um gráfico em bytes, envia para o Gemini e retorna a análise.
    """
    client = conn_genai()
    if not client:
        return "⚠️ Não foi possível conectar com a IA para analisar o gráfico."
        
    try:
        from google.genai import types
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


def explicar_grafico(fig, user_prompt: str = "Extraia os principais insights deste gráfico de qualidade da água.") -> str:
    """
    Consome a API do Gemini enviando a imagem do gráfico (Plotly ou Matplotlib) para análise.
    """
    client = conn_genai()
    if not client:
        return "⚠️ **Chave de API não configurada.** Defina `GEMINI_API_KEY` nos secrets do Streamlit."

    # Converte o gráfico para bytes de imagem
    try:
        if hasattr(fig, "to_image"): # Provavelmente Plotly
            import plotly.io as pio
            # Configuração para ambientes Linux Headless (Streamlit Cloud)
            try:
                if hasattr(pio.kaleido, "scope"):
                    pio.kaleido.scope.chromium_args = ("--headless", "--no-sandbox", "--single-process", "--disable-gpu")
            except Exception:
                pass
            img_bytes = fig.to_image(format="png")
        elif hasattr(fig, "savefig"): # Provavelmente Matplotlib
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            img_bytes = buf.getvalue()
            buf.close()
        else:
            return "❌ Erro: O objeto fornecido não é um gráfico suportado (Plotly ou Matplotlib)."
    except Exception as e:
        return f"❌ Erro ao gerar a imagem do gráfico: {type(e).__name__}: {e}. Certifique-se de que o pacote `kaleido==0.2.1` está no `requirements.txt` e o `packages.txt` contém as dependências do sistema."

    return analyse_graph(img_bytes, user_prompt)

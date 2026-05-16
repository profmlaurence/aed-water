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
            return "⚠️ **Chave de API não configurada.** Defina `GEMINI_API_KEY` nos secrets do Streamlit ou variáveis de ambiente."
            
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Erro ao conectar com a IA: {e}")
        return None

def analyse_graph(fig, user_prompt: str = "Extraia os principais insights deste gráfico de qualidade da água. Responda no máximo em 150 palavras"):
    """
    Analisa um gráfico usando a IA do Gemini.
    Gráficos Matplotlib são enviados como imagem. Gráficos Plotly são enviados como dados estruturados (JSON)
    para evitar o uso do Kaleido/Chromium no Cloud Run.
    """
    from google.genai import types
    # import traceback
    
    try:
        # 1. Inicializa o cliente
        client = conn_genai()
        if isinstance(client, str): # Tratamento se conn_genai retornar string de erro
            return client
        if not client:
            return "⚠️ Não foi possível conectar com a IA. Verifique sua chave API."

        # 2. Monta o conteúdo dinamicamente
        contents = [user_prompt]

        if isinstance(fig, bytes):
            contents.append(types.Part.from_bytes(data=fig, mime_type="image/png"))
            
        elif hasattr(fig, "savefig"): # Matplotlib (gera imagem nativamente em Python puro)
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            img_bytes = buf.getvalue()
            buf.close()
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            
        elif hasattr(fig, "to_json"): # Plotly (Envio de dados estruturados sem Kaleido)
            # Como o dataset é leve, enviamos o JSON com as coordenadas e layouts do gráfico
            # O Gemini 2.5 Flash entende a estrutura do Plotly nativamente e extrai os insights!
            graph_json = fig.to_json()
            contents.append(f"\n\n--- DADOS E ESTRUTURA DO GRÁFICO (PLOTLY JSON) ---\n{graph_json}")
            
        else:
            return "❌ Erro: O objeto fornecido não é um gráfico suportado."

        # 3. Envia para o Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents
        )
        
        return response.text
        
    except Exception as e:
        error_details = traceback.format_exc()
        st.error(f"Erro na comunicação com a IA: {e}")
        with st.expander("Detalhes do erro para debug"):
            st.code(error_details)
        return "❌ Ocorreu um erro ao processar o gráfico."


def render_ia_analyze_button(fig, key: str, prompt: str = None):
    """
    Adiciona um botão para analisar o gráfico com IA.
    Esta função é compartilhada entre todos os módulos do sistema.
    """
    if st.button("✨ Analisar com IA", key=key, type="secondary"):
        with st.spinner("Processando o gráfico..."):
            if prompt:
                analise = analyse_graph(fig, user_prompt=prompt)
            else:
                analise = analyse_graph(fig)
            st.caption("🤖 **Modelo:** gemini-2.5-flash-lite")
            st.info(analise)


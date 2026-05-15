import os
import streamlit as st
import pandas as pd


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


def explicar_grafico(fig, user_prompt: str = "Extraia os principais insights deste gráfico de qualidade da água.") -> str:
    """
    Consome a API do Gemini enviando a imagem do gráfico (Plotly ou Matplotlib) para análise.
    """
    try:
        from google import genai
        from google.genai import types
        import os
        import io

        # Tenta obter a chave da API
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except (FileNotFoundError, KeyError):
            api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            return "⚠️ **Chave de API não configurada.** Defina `GEMINI_API_KEY` nos secrets do Streamlit."

        client = genai.Client(api_key=api_key)

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

        prompt = f"""
        Você é um especialista em qualidade da água auxiliando no Laboratório de Pesquisa em Química Ambiental (LAPEQ).
        Analise detalhadamente o gráfico em anexo e responda.
        
        Contexto ou Pergunta:
        "{user_prompt}"
        
        Responda de forma resumida, máximo 150 palavras.
        """

        contents = [
            types.Part.from_bytes(data=img_bytes, mime_type='image/png'),
            prompt
        ]

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )

        return response.text

    except ImportError:
        return "❌ Pacote `google-genai` não instalado."
    except Exception as e:
        return f"❌ Erro ao conectar com a IA: {e}"

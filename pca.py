import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from shared import render_ia_analyze_button


class PlotsData:
    """
    Classe para encapsular a lógica de análise exploratória dos dados.
    """

    def __init__(self, dataset: pd.DataFrame):
        """Inicializa com o dataset a ser analisado."""
        self.dataset = dataset
        self.model = st.session_state.get("model")

    def _get_numeric_data(self) -> pd.DataFrame:
        """Retorna apenas as colunas numéricas do dataset."""
        return self.dataset.select_dtypes(include=[np.number])

    def plot_pca(self):
        """Exibe os gráficos de PCA (Dispersão e Biplot)."""
        st.markdown("### PCA")

        numeric_data = self._get_numeric_data()
        original_cols = numeric_data.columns
        # 1. Remover colunas com variância zero (constantes)
        numeric_data = numeric_data.loc[:, numeric_data.std() > 1e-8]
        
        removed_cols = [col for col in original_cols if col not in numeric_data.columns]
        if removed_cols:
            st.info(f"**Colunas ignoradas (variância zero/constantes):** {', '.join(removed_cols)}")

        if numeric_data.shape[1] < 2:
            st.warning(
                "O dataset não possui colunas numéricas suficientes "
                "para realizar a análise PCA."
            )
            return
        
        # 2. Imputar valores NaN com a média (para não perder as linhas das coletas)
        imputer = SimpleImputer(strategy='mean')
        imputed_data = imputer.fit_transform(numeric_data)

        # 3. Padronizar os dados (Média 0, Desvio Padrão 1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(imputed_data)

        # Convertendo de volta para DataFrame apenas para manter os nomes das colunas
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_data.columns, index=numeric_data.index)


        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_df)
        pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'], index=scaled_df.index)

        tab_scatter, tab_biplot = st.tabs(
            ["Gráfico de Dispersão PCA", "Biplot PCA"]
        )

        with tab_scatter:
            fig_scatter = self._plot_pca_scatter(pca_df)
            render_ia_analyze_button(fig_scatter, key='key_pca_scatter',
            prompt="""Analise a projeção das amostras neste gráfico de dispersão PCA (scores plot).
                        Focando na estrutura espacial desses dados tabulares, resuma em até 200 palavras:
                        A dispersão e separabilidade dos pontos ao longo dos eixos PC1 e PC2.
                        A presença de clusters naturais (agrupamentos de instâncias similares) e se há sobreposição entre eles.
                        A existência de instâncias isoladas (outliers).
                        Seja técnico e estritamente objetivo.
                        Formate em tópicos.
                    """)

        with tab_biplot:
            fig_biplot = self._plot_pca_biplot(pca, pca_result, scaled_df)
            render_ia_analyze_button(fig_biplot, key='key_pca_biplot',
            prompt="""Analise este biplot PCA e extraia os insights mais críticos em no máximo 200 palavras. 
                        Estruture sua resposta abordando obrigatoriamente:
                        1 - A proporção de variância explicada pelos eixos PC1 e PC2.
                        2 - Variáveis altamente correlacionadas (setas apontando na mesma direção) e inversamente correlacionadas (direções opostas).
                        3 - As variáveis com maior peso/importância (setas mais longas).
                        4 - Existência de clusters evidentes ou amostras atípicas (outliers).
                        Faça uma formatação em tópicos.
                    """)
        

        # with tab_explicacao:
        #     self.explicar_pca(pca, scaled_df.columns)

    def _plot_pca_scatter(self, pca_df: pd.DataFrame):
        """Gráfico de dispersão dos componentes principais."""
        st.markdown("##### Gráfico de Dispersão PCA")
        
        fig = px.scatter(
            pca_df, x='PCA1', y='PCA2',
            title='Dispersão PCA',
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            xaxis_title='PCA1',
            yaxis_title='PCA2',
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        # Adicionar eixos x e y em 0
        fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.3)
        fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.3)
        
        st.plotly_chart(fig, width='content')

        return fig

    def _plot_pca_biplot(self, pca: PCA, pca_result: np.ndarray,
                         scaled_df: pd.DataFrame):
        """Biplot PCA com scores e loadings."""
        st.markdown("##### Biplot PCA")

        fig = go.Figure()

        # Plotar os pontos dos dados (scores)
        fig.add_trace(go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers',
            name='Amostras',
            marker=dict(color='rgba(102, 194, 165, 0.7)', size=8, line=dict(color='DarkSlateGrey', width=1)),
            hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ))

        # Plotar as setas das variáveis (loadings)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        annotations = []
        for i, var in enumerate(scaled_df.columns):
            # Adicionar a linha da seta
            fig.add_trace(go.Scatter(
                x=[0, loadings[i, 0]],
                y=[0, loadings[i, 1]],
                mode='lines',
                line=dict(color='red', width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Adicionar a anotação do texto (número da variável) na ponta da seta
            annotations.append(
                dict(
                    x=loadings[i, 0] * 1.15,
                    y=loadings[i, 1] * 1.15,
                    xref="x", yref="y",
                    text=str(i + 1),
                    showarrow=False,
                    font=dict(color="red", size=12, family="Arial Black"),
                    hovertext=var
                )
            )

        fig.update_layout(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
            title='Biplot PCA',
            annotations=annotations,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        # Adicionar eixos x e y em 0
        fig.add_hline(y=0, line_width=1, line_color="black", opacity=0.3)
        fig.add_vline(x=0, line_width=1, line_color="black", opacity=0.3)

        st.plotly_chart(fig, width='content')

        with st.expander("**Legenda das Variáveis (Setas Vermelhas)**",icon="🔢"):
            st.dataframe({"Variável": scaled_df.columns})
            
        return fig

    def run(self):
        """Executa a análise exploratória completa."""
        st.title("Análises Exploratória dos Dados")

        try:
            # self.plot_correlation_matrix()
            self.plot_pca()
        except Exception as e:
            st.error(f"Erro na análise exploratória: {str(e)}")


if __name__ == "__main__":
    dataset = st.session_state.get("current_dataset")

    if dataset is not None:
        page = PlotsData(dataset)
        page.run()
    else:
        st.warning("Nenhum dataset carregado. Selecione um dataset na página de dados.")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import io


class PlotsData:
    """
    Classe para encapsular a lógica de análise exploratória dos dados.
    """

    def __init__(self, dataset: pd.DataFrame):
        """Inicializa com o dataset a ser analisado."""
        self.dataset = dataset

    def _get_numeric_data(self) -> pd.DataFrame:
        """Retorna apenas as colunas numéricas do dataset."""
        return self.dataset.select_dtypes(include=[np.number])

    def _add_download_button(self, fig: plt.Figure, label: str, file_name: str):
        """Salva a figura matplotlib em buffer e exibe o botão de download."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button(
            label=label,
            data=buf.getvalue(),
            file_name=file_name,
            mime="image/png"
        )

    def plot_correlation_matrix(self):
        """Exibe a matriz de correlação das variáveis numéricas."""
        st.markdown("### Matriz de Correlação")

        numeric_data = self._get_numeric_data()
        original_cols = numeric_data.columns
        
        # Remover colunas com variância zero ou com todos os valores NaN
        numeric_data = numeric_data.loc[:, numeric_data.std() > 1e-8]

        removed_cols = [col for col in original_cols if col not in numeric_data.columns]
        # if removed_cols:
        #     st.info(f"**Colunas ignoradas (variância zero/constantes):** {', '.join(removed_cols)}")

        if numeric_data.shape[1] < 2:
            st.warning(
                "O dataset não possui colunas numéricas suficientes "
                "para calcular a matriz de correlação."
                f"Linhas: {numeric_data.shape[0]}, Colunas: {numeric_data.shape[1]}"
            )
            return

        corr = numeric_data.corr()
        st.dataframe(
            corr.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
            .format("{:.2f}", na_rep="NaN")
        )

        # Renderizar Matplotlib em background para permitir o download em imagem
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        
        if len(corr.columns) <= 15:
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    if not np.isnan(corr.iloc[i, j]):
                        val = corr.iloc[i, j]
                        color = 'white' if abs(val) > 0.5 else 'black'
                        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8)
                        
        ax.set_title("Matriz de Correlação")
        plt.tight_layout()

        self._add_download_button(fig, "⬇️ Baixar Matriz de Correlação", "matriz_correlacao.png")
        plt.close(fig)

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
            self._plot_pca_scatter(pca_df)

        with tab_biplot:
            self._plot_pca_biplot(pca, pca_result, scaled_df)

    def _plot_pca_scatter(self, pca_df: pd.DataFrame):
        """Gráfico de dispersão dos componentes principais."""
        st.markdown("##### Gráfico de Dispersão PCA")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(pca_df['PCA1'], pca_df['PCA2'], alpha=0.6, edgecolors='k')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.set_title('Dispersão PCA')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        st.pyplot(fig)

        self._add_download_button(fig, "⬇️ Baixar Imagem da Dispersão", "dispersao_pca.png")

    def _plot_pca_biplot(self, pca: PCA, pca_result: np.ndarray,
                         scaled_df: pd.DataFrame):
        """Biplot PCA com scores e loadings."""
        st.markdown("##### Biplot PCA")

        # with col1:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plotar os pontos dos dados (scores)
        ax.scatter(pca_result[:, 0], pca_result[:, 1],
                    alpha=0.6, edgecolors='k')

        # Plotar as setas das variáveis (loadings)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        for i, var in enumerate(scaled_df.columns):
            ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                        head_width=0.1, head_length=0.1,
                        fc='red', ec='red', alpha=0.7)
            ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, str(i + 1),
                    fontsize=10, fontweight='bold', color='red')

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Biplot PCA')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        st.pyplot(fig)

        self._add_download_button(fig, "⬇️ Baixar Imagem do Biplot", "biplot_pca.png")

        # st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Legenda**")
            st.dataframe({"Variáveis": scaled_df.columns})
            # for i, var in enumerate(scaled_df.columns, 1):
            #     st.badge(f"{i} - {var}", color="red")
            

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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io


class AnalysisPlots:
    """
    Classe para encapsular a lógica de análise avançada dos dados
    de qualidade da água.
    """

    def __init__(self, dataset: pd.DataFrame):
        """Inicializa com o dataset a ser analisado."""
        self.dataset = dataset
        self.features = list(
            dataset.select_dtypes(include=[np.number]).columns
        )

    def _get_numeric_data(self) -> pd.DataFrame:
        """Retorna apenas as colunas numéricas do dataset."""
        return self.dataset.select_dtypes(include=[np.number])

    def _add_download_button(self, fig, label: str, file_name: str):
        """Salva a figura matplotlib em buffer e exibe o botão de download."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button(
            label=label,
            data=buf.getvalue(),
            file_name=file_name,
            mime="image/png"
        )

    # ── Tab 1: Visão Geral ────────────────────────────────────────────────────

    def _plot_correlation_heatmap(self):
        """Mapa de calor de correlação."""
        numeric_data = self._get_numeric_data()
        numeric_data = numeric_data.loc[:, numeric_data.std() > 1e-8]

        if numeric_data.shape[1] < 2:
            st.warning("Colunas numéricas insuficientes para a correlação.")
            return

        corr = numeric_data.corr()

        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1, aspect='auto'
        )
        fig.update_layout(margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    def _tab_visao_geral(self):
        """Conteúdo da tab Visão Geral."""
        st.header("📋 Visão Geral")

        st.subheader("Correlação entre Variáveis")
        st.markdown(
            "O mapa de calor abaixo mostra a correlação de Pearson entre as "
            "diversas propriedades da água. Valores próximos de **+1** ou **-1** "
            "indicam relações lineares fortes."
        )
        self._plot_correlation_heatmap()

    # ── Tab 2: Distribuições ──────────────────────────────────────────────────

    def _plot_feature_distribution(self, feature: str):
        """Histograma + Box Plot de uma variável."""
        fig = px.histogram(
            self.dataset, x=feature, marginal='box',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    def _plot_violin(self, feature: str):
        """Gráfico violino de uma variável."""
        fig = px.violin(
            self.dataset, y=feature, box=True, points='outliers',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    def _tab_distribuicoes(self):
        """Conteúdo da tab Distribuições."""
        st.header("📈 Distribuições")

        if not self.features:
            st.warning("Nenhuma variável numérica disponível.")
            return

        st.subheader(
            "Histograma + Box Plot",
            help='O histograma mostra a distribuição de frequência dos dados, '
                 'enquanto o box plot mostra a mediana, os quartis e os outliers.'
        )
        feat_hist = st.selectbox(
            "Selecione a propriedade:", self.features, key="dist_selectbox"
        )
        self._plot_feature_distribution(feat_hist)

        st.divider()

        st.subheader(
            "Gráfico Violino",
            help='O formato do "violino" evidencia distribuições bimodais '
                 'e outliers com mais riqueza que um boxplot comum.'
        )
        feat_violin = st.selectbox(
            "Selecione a propriedade:", self.features, key="violin_selectbox"
        )
        self._plot_violin(feat_violin)

    # ── Tab 3: Análise Multivariada ───────────────────────────────────────────

    def _plot_parallel_coordinates(self):
        """Gráfico de coordenadas paralelas."""
        numeric_data = self._get_numeric_data()
        numeric_data = numeric_data.loc[:, numeric_data.std() > 1e-8]

        if numeric_data.shape[1] < 2:
            st.warning("Colunas numéricas insuficientes para coordenadas paralelas.")
            return

        fig = px.parallel_coordinates(
            numeric_data,
            color_continuous_scale=px.colors.diverging.Tealrose
        )
        fig.update_layout(margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    def _plot_spider_chart(self):
        """Gráfico de radar (Spider Chart) com a média de cada variável."""
        numeric_data = self._get_numeric_data()
        numeric_data = numeric_data.loc[:, numeric_data.std() > 1e-8]

        if numeric_data.shape[1] < 3:
            st.warning("Colunas numéricas insuficientes para o gráfico de radar.")
            return

        # Normalizar os dados (min-max)
        normalized = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
        means = normalized.mean()
        categories = means.index.tolist()

        values = means.tolist()
        values.append(values[0])  # Fechar o polígono
        cats = categories + [categories[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=cats, fill='toself',
            name='Média Normalizada',
            line=dict(color='#66cdaa')
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            margin=dict(t=30, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    def _plot_scatter_matrix(self):
        """Matriz de dispersão."""
        numeric_data = self._get_numeric_data()
        numeric_data = numeric_data.loc[:, numeric_data.std() > 1e-8]

        dims = numeric_data.columns[:6].tolist()  # Limitar para performance

        fig = px.scatter_matrix(
            numeric_data,
            dimensions=dims,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=700, margin=dict(t=30, b=30))
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    def _tab_multivariada(self):
        """Conteúdo da tab Análise Multivariada."""
        st.header("🔗 Análise Multivariada")

        st.subheader("Coordenadas Paralelas")
        st.caption(
            "Cada linha representa uma amostra passando por todos os eixos. "
            "Ajuda a encontrar padrões e faixas de valores entre as variáveis."
        )
        self._plot_parallel_coordinates()

        st.divider()

        st.subheader("Gráfico de Radar (Spider Chart)")
        st.caption(
            "Assinatura visual usando a média normalizada de cada propriedade. "
            "Permite uma visão holística rápida das variáveis."
        )
        self._plot_spider_chart()

        st.divider()

        st.subheader("Matriz de Dispersão")
        st.caption(
            "Visão cruzada de todas as propriedades entre si."
        )
        self._plot_scatter_matrix()

    # ── Tab 4: Avançado ───────────────────────────────────────────────────────

    def _plot_3d_scatter(self, x: str, y: str, z: str):
        """Gráfico de dispersão 3D."""
        fig = px.scatter_3d(
            self.dataset, x=x, y=y, z=z,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=600, margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    def _tab_avancado(self):
        """Conteúdo da tab Avançado."""
        st.header("🧪 Avançado")

        if len(self.features) < 3:
            st.warning("O dataset precisa de pelo menos 3 variáveis numéricas.")
            return

        st.subheader(
            "Dispersão 3D",
            help="Explore a relação entre três propriedades da água "
                 "em um espaço tridimensional."
        )
        st.caption(
            "Selecione 3 propriedades para explorar clusters espaciais "
            "interativamente."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            x_3d = st.selectbox(
                "Eixo X:", self.features,
                index=0,
                key='3d_x',
            )
        with col2:
            y_3d = st.selectbox(
                "Eixo Y:", self.features,
                index=min(1, len(self.features) - 1),
                key='3d_y',
            )
        with col3:
            z_3d = st.selectbox(
                "Eixo Z:", self.features,
                index=min(2, len(self.features) - 1),
                key='3d_z',
            )
        self._plot_3d_scatter(x=x_3d, y=y_3d, z=z_3d)

    # ── Entry Point ───────────────────────────────────────────────────────────

    def run(self):
        """Executa a análise completa com tabs."""
        st.title("Análise da Qualidade da Água")

        # Métricas rápidas
        total = len(self.dataset)
        num_features = len(self.features)

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total de Amostras", f"{total:,}")
        col_m2.metric("Nº de Propriedades", num_features)

        try:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Visão Geral",
                "📈 Distribuições",
                "🔗 Análise Multivariada",
                "🧪 Avançado"
            ])

            with tab1:
                self._tab_visao_geral()
            with tab2:
                self._tab_distribuicoes()
            with tab3:
                self._tab_multivariada()
            with tab4:
                self._tab_avancado()

        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")
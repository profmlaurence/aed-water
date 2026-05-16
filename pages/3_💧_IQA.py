import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shared import load_dataset_sidebar, render_footer, setup_page
from iqa import IQACalculator

setup_page()

# ── Sidebar: Seleção do Dataset ──────────────────────────────────────────────
df = load_dataset_sidebar()
campanha = st.session_state.get("current_campanha", "")

st.title("💧 Índice de Qualidade da Água (IQA)")

st.caption(f"Campanha: **{campanha}** — Modelo CETESB/NSF")

st.markdown(
    """
    O IQA é calculado pelo **produtório ponderado** de 9 parâmetros 
    de qualidade da água, resultando em um valor de **0 a 100**.
    """
)

# ── Tabela de Classificação ──────────────────────────────────────────────────
with st.expander("📋 Tabela de Classificação do IQA", expanded=False):
    class_table = IQACalculator.get_classificacao_table()
    st.dataframe(class_table, hide_index=True)

st.divider()

# ── Calcular IQA ─────────────────────────────────────────────────────────────
calc = IQACalculator()

# Verificar quais colunas estão disponíveis
st.markdown("### Mapeamento de Parâmetros")
col_status = []
for param, col in calc.COLUMN_MAP.items():
    if param == 'nitrogenio':
        found = all(
            calc._find_column(df, c.strip()) is not None for c in col
        )
        col_name = ' + '.join(col)
    else:
        found = calc._find_column(df, col.strip()) is not None
        col_name = col
    col_status.append({
        'Parâmetro IQA': param.upper(),
        'Coluna no Dataset': col_name,
        'Peso (wi)': calc.PESOS[param],
        'Status': '✅ Encontrado' if found else '❌ Ausente',
    })

status_df = pd.DataFrame(col_status)
st.dataframe(status_df, hide_index=True)

available = sum(1 for s in col_status if '✅' in s['Status'])
st.info(f"**{available}/9** parâmetros disponíveis no dataset.")

if available < 3:
    st.error("Parâmetros insuficientes para calcular o IQA.")
    st.stop()

st.divider()

# ── Resultados ────────────────────────────────────────────────────────────────
st.markdown("### Resultados do IQA por Ponto de Coleta")

results_df = calc.calculate_from_dataframe(df)
st.dataframe(
    results_df.style.format(
        {col: '{:.1f}' for col in results_df.columns if col.startswith('q(') or col == 'IQA'},
        na_rep='—'
    ),
    hide_index=True,
)

# ── Gráfico de barras ────────────────────────────────────────────────────────
st.markdown("### Visualização")

tab_bar, tab_radar, tab_detail = st.tabs([
    "📊 IQA por Ponto", "🕸️ Radar de Qualidade", "📋 Detalhes por Ponto"
])

with tab_bar:
    iqa_values = results_df['IQA'].tolist()
    pontos = results_df['Ponto de Coleta'].tolist()

    colors = []
    for v in iqa_values:
        if pd.isna(v):
            colors.append('#cccccc')
        elif v >= 79:
            colors.append('#2ecc71')
        elif v >= 51:
            colors.append('#3498db')
        elif v >= 36:
            colors.append('#f1c40f')
        elif v >= 19:
            colors.append('#e67e22')
        else:
            colors.append('#e74c3c')

    fig = go.Figure(data=[
        go.Bar(
            x=[f'Ponto {p}' for p in pontos],
            y=iqa_values,
            marker_color=colors,
            text=[f'{v:.1f}' if pd.notna(v) else '—' for v in iqa_values],
            textposition='outside',
        )
    ])

    # Linhas de referência das faixas
    for limit, color, label in [
        (79, '#2ecc71', 'Ótima'), (51, '#3498db', 'Boa'),
        (36, '#f1c40f', 'Regular'), (19, '#e67e22', 'Ruim')
    ]:
        fig.add_hline(
            y=limit, line_dash='dot', line_color=color,
            annotation_text=label, annotation_position='right'
        )

    fig.update_layout(
        yaxis_title='IQA', xaxis_title='Ponto de Coleta',
        yaxis_range=[0, 105], height=500,
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, width='stretch')

with tab_radar:
    qi_cols = [c for c in results_df.columns if c.startswith('q(')]
    selected_point = st.selectbox(
        "Selecione o ponto de coleta:",
        results_df['Ponto de Coleta'].tolist(),
        key='radar_point'
    )

    row = results_df[results_df['Ponto de Coleta'] == selected_point].iloc[0]
    categories = [c.replace('q(', '').replace(')', '') for c in qi_cols]
    values = [row[c] if pd.notna(row[c]) else 0 for c in qi_cols]
    values.append(values[0])  # fechar o polígono
    cats = categories + [categories[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values, theta=cats, fill='toself',
        name=f'Ponto {selected_point}',
        line=dict(color='#3498db')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500, margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig_radar, width='stretch')

    iqa_val = row['IQA']
    classif = row['Classificação']
    if pd.notna(iqa_val):
        st.metric(
            label=f"IQA — Ponto {selected_point}",
            value=f"{iqa_val:.1f}",
            delta=classif
        )

with tab_detail:
    selected_detail = st.selectbox(
        "Selecione o ponto de coleta:",
        results_df['Ponto de Coleta'].tolist(),
        key='detail_point'
    )
    detail_row = results_df[
        results_df['Ponto de Coleta'] == selected_detail
    ].iloc[0]

    qi_data = []
    param_labels = {
        'od': 'Oxigênio Dissolvido', 'coliformes': 'Coliformes',
        'ph': 'pH', 'dbo': 'DBO',
        'temperatura': 'Temperatura', 'nitrogenio': 'Nitrogênio Total',
        'fosforo': 'Fósforo Total', 'turbidez': 'Turbidez',
        'residuos': 'Resíduos Totais',
    }
    qi_col_map = {
        'od': 'q(OD)', 'coliformes': 'q(Coliformes)',
        'ph': 'q(pH)', 'dbo': 'q(DBO)',
        'temperatura': 'q(Temp)', 'nitrogenio': 'q(N Total)',
        'fosforo': 'q(P Total)', 'turbidez': 'q(Turbidez)',
        'residuos': 'q(Resíduos)',
    }

    for param, label in param_labels.items():
        qi_col = qi_col_map[param]
        qi_val = detail_row[qi_col] if pd.notna(detail_row.get(qi_col)) else None
        peso = calc.PESOS[param]

        # Classificar a qualidade individual
        if qi_val is not None:
            if qi_val >= 80:
                status = '🟢 Ótimo'
            elif qi_val >= 50:
                status = '🔵 Bom'
            elif qi_val >= 30:
                status = '🟡 Regular'
            elif qi_val >= 15:
                status = '🟠 Ruim'
            else:
                status = '🔴 Péssimo'
        else:
            status = '⚪ N/D'

        qi_data.append({
            'Parâmetro': label,
            'Peso (wi)': peso,
            'qi (0-100)': f'{qi_val:.1f}' if qi_val else '—',
            'Qualidade': status,
        })

    st.dataframe(pd.DataFrame(qi_data), hide_index=True)

st.divider()

# ── Referências ───────────────────────────────────────────────────────────────
with st.expander("📚 Referências e Metodologia"):
    st.markdown("""
    **Fórmula:** `IQA = ∏(qi^wi)` — Produtório ponderado

    **Fontes:**
    - CETESB — Companhia Ambiental do Estado de São Paulo
    - Brown, R.M. et al. (1970) — *A Water Quality Index: Do We Dare?*
    - Von Sperling, M. — *Introdução à qualidade das águas e ao tratamento de esgotos*
    - ANA — Agência Nacional de Águas

    **Observações:**
    - O OD é convertido para **% de saturação** antes do cálculo.
    - *Escherichia coli* é utilizada como proxy para coliformes termotolerantes.
    - Nitrogênio Total = Amônia + Nitrito + Nitrato + Orgânico.
    - Quando um parâmetro está ausente, os pesos são **renormalizados**
      para manter a escala de 0–100.
    """)

st.divider()

if st.button("Gerar Explicação Geral com IA", icon="🧠"):
    with st.spinner("Analisando os resultados..."):
        explicacao = calc.explicar_iqa(
            results_df, 
            user_prompt="Faça um resumo geral da qualidade da água dos pontos coletados, destacando os pontos críticos e os melhores pontos."
        )
        st.caption("🤖 **Modelo:** gemini-2.5-flash-lite")
        st.markdown(explicacao)

render_footer()

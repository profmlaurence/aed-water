# 📊 MonitorWater — Análise Exploratória de Qualidade da Água

Aplicação web interativa desenvolvida com [Streamlit](https://streamlit.io/) para análise exploratória de dados (EDA) de campanhas de monitoramento de qualidade da água.

O sistema permite carregar dados de diferentes campanhas de coleta, visualizar propriedades físico-químicas e realizar análises estatísticas multivariadas, incluindo PCA (Análise de Componentes Principais).

---

## 🗂️ Estrutura do Projeto

```
aed-water/
├── 0_🧑🏻‍🔬_Home.py              # Página inicial — visão geral do dataset
├── shared.py                    # Funções compartilhadas (seletor de campanha)
├── analysis.py                  # Classe AnalysisPlots — análises avançadas
├── plots_data.py                # Classe PlotsData — matriz de correlação e PCA
├── dataset/
│   ├── Campanha 1.csv           # Dados da 1ª campanha de coleta
│   ├── Campanha 2.csv           # Dados da 2ª campanha de coleta
│   └── all.csv                  # Dados consolidados
├── pages/
│   ├── 1_🔬_Análise Exploratória.py   # Página de análise exploratória
│   └── 2_📈_PCA.py                    # Página de PCA e correlação
└── .gitignore
```

## ✨ Funcionalidades

### 🏠 Home
- Seleção interativa de campanha via sidebar (disponível em todas as páginas)
- Métricas rápidas: linhas, colunas e valores nulos
- Visualização da tabela de dados brutos

### 🔬 Análise Exploratória (`AnalysisPlots`)
- **Correlação** — Mapa de calor interativo (Plotly)
- **Distribuições** — Histograma + Box Plot e Gráfico Violino por variável
- **Análise Multivariada** — Coordenadas Paralelas, Gráfico de Radar (Spider Chart) e Matriz de Dispersão
- **Avançado** — Dispersão 3D interativa com seleção de eixos

### 📈 PCA (`PlotsData`)
- **Matriz de Correlação** — Tabela estilizada com gradiente de cores e botão de download
- **PCA (Análise de Componentes Principais)**:
  - Filtragem automática de colunas constantes (variância ≈ 0)
  - Imputação de valores faltantes com a média (`SimpleImputer`)
  - Padronização dos dados (`StandardScaler`)
  - Gráfico de Dispersão PCA com botão de download
  - Biplot PCA com setas numeradas e quadro de legenda
  - Botões de download para todas as imagens geradas

## 🧪 Dados

Os arquivos CSV utilizam `;` como separador e contêm **39 variáveis físico-químicas**, incluindo:

| Categoria | Exemplos |
|---|---|
| Físicas | Temperatura, Turbidez, Cor Verdadeira, Condutividade Elétrica |
| Químicas | pH, Oxigênio Dissolvido, DBO, Nitrogênio, Fósforo Total |
| Metais | Ferro, Manganês, Zinco, Cromo, Alumínio, Cobre |
| Alcalinidade | Carbonato, Bicarbonato, Total |
| Dureza | Cálcio, Magnésio, Total |
| Microbiologia | Coliformes Totais, *Escherichia coli* |
| Outros | Glifosato, Sulfato, Sódio, Potássio, Clorofila, Salinidade |

## 🚀 Como Executar

### Pré-requisitos

- Python 3.12+
- pip

### Instalação

```bash
# Clonar o repositório
git clone <url-do-repositório>
cd aed-water

# Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Instalar as dependências
pip install streamlit pandas numpy matplotlib plotly scikit-learn
```

### Executar

```bash
streamlit run "0_🧑🏻‍🔬_Home.py"
```

A aplicação abrirá automaticamente em `http://localhost:8501`.

## 📦 Dependências

| Pacote | Versão | Uso |
|---|---|---|
| `streamlit` | 1.57+ | Framework da aplicação web |
| `pandas` | 3.0+ | Manipulação de dados |
| `numpy` | 2.4+ | Operações numéricas |
| `matplotlib` | 3.10+ | Gráficos estáticos (PCA, Correlação) |
| `plotly` | 6.7+ | Gráficos interativos (Distribuições, 3D) |
| `scikit-learn` | 1.8+ | PCA, Imputação, Padronização |

## 📐 Arquitetura

```
┌──────────────────────────────────────────────────────┐
│                    shared.py                         │
│              load_dataset_sidebar()                  │
│         (seletor de campanha no sidebar)             │
└──────────────┬───────────────┬───────────────────────┘
               │               │
    ┌──────────▼──┐     ┌──────▼──────┐
    │ plots_data  │     │  analysis   │
    │  .py        │     │  .py        │
    │             │     │             │
    │ PlotsData   │     │AnalysisPlots│
    │ - Correlação│     │- Histograma │
    │ - PCA       │     │- Violino    │
    │ - Biplot    │     │- Radar      │
    │ - Download  │     │- 3D Scatter │
    └──────┬──────┘     └──────┬──────┘
           │                   │
    ┌──────▼──────┐     ┌──────▼──────────────┐
    │ pages/      │     │ pages/              │
    │ 2_📈_PCA.py │     │ 1_🔬_Análise        │
    │             │     │   Exploratória.py   │
    └─────────────┘     └─────────────────────┘
```

## 👩‍🔬 Autora

**Mikaela**

---

> Desenvolvido como ferramenta de apoio à análise de dados ambientais de monitoramento hídrico.

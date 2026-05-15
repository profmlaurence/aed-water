import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class IQACalculator:
    """
    Calculadora do Índice de Qualidade da Água (IQA) — Modelo CETESB/NSF.

    Utiliza o produtório ponderado dos 9 parâmetros:
        IQA = ∏(qi^wi)

    As curvas de qualidade (qi) são aproximadas por interpolação linear
    a partir dos pontos digitalizados das curvas oficiais da NSF/CETESB.

    Referências:
        - CETESB (Companhia Ambiental do Estado de São Paulo)
        - Brown, R.M. et al. (1970) - A Water Quality Index: Do We Dare?
        - Von Sperling, M. - Introdução à qualidade das águas
    """

    # ── Pesos (wi) — Modelo CETESB ────────────────────────────────────────────
    PESOS = {
        'od':           0.17,
        'coliformes':   0.15,
        'ph':           0.12,
        'dbo':          0.10,
        'temperatura':  0.10,
        'nitrogenio':   0.10,
        'fosforo':      0.10,
        'turbidez':     0.08,
        'residuos':     0.08,
    }

    # ── Classificação ─────────────────────────────────────────────────────────
    CLASSIFICACAO = [
        (79, 100, 'Ótima',    '🟢'),
        (51,  79, 'Boa',      '🔵'),
        (36,  51, 'Regular',  '🟡'),
        (19,  36, 'Ruim',     '🟠'),
        ( 0,  19, 'Péssima',  '🔴'),
    ]

    # ── Mapeamento de colunas do dataset → parâmetros IQA ─────────────────────
    COLUMN_MAP = {
        'od':           'Oxigenio Dissolvido',
        'coliformes':   'Escherichia coli',
        'ph':           'pH',
        'dbo':          'Demanda Bioquímica de Oxigênio',
        'temperatura':  'Temperatura',
        'turbidez':     'Turbidez',
        'fosforo':      'Fósforo Total',
        'residuos':     'Residuos Totais',
        # Nitrogênio Total = soma das frações
        'nitrogenio':   ['Nitrogênio - Amônia', 'Nitrogênio - Nitrito',
                         'Nitrogênio - Nitrato', 'Orgânico'],
    }

    # ── Curvas de qualidade (qi) ──────────────────────────────────────────────
    # Pontos digitalizados das curvas oficiais NSF/CETESB.
    # Cada curva: (valores_medidos, qi_correspondente)
    # Valores fora do intervalo são clamped ao mín/máx de qi.

    # OD: porcentagem de saturação (%) → qi
    _CURVE_OD = (
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [3,  8, 15, 25, 38, 52, 65, 78, 88, 95, 100,  95,  85,  75,  65],
    )

    # Coliformes termotolerantes (NMP/100mL) → qi (escala log)
    _CURVE_COLIFORMES = (
        [1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000],
        [98, 95, 90, 80, 65, 55, 35, 25, 12, 8, 4, 2],
    )

    # pH → qi
    _CURVE_PH = (
        [2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.4, 7.8, 8.0,
         8.5, 9.0, 10.0, 11.0, 12.0],
        [2, 5, 10, 20, 42, 60, 82, 93, 85, 78,
         60, 42, 18, 8, 2],
    )

    # DBO (mg/L) → qi
    _CURVE_DBO = (
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30],
        [100, 92, 82, 72, 62, 52, 42, 28, 18, 12, 7, 3, 2],
    )

    # Variação de temperatura (°C) → qi
    # Como não temos ponto de referência upstream, usamos a temperatura
    # absoluta com uma curva que penaliza extremos.
    # Para o modelo CETESB em regiões tropicais, temperaturas até ~25°C
    # são consideradas ideais; acima disso há perda de qualidade.
    _CURVE_TEMPERATURA = (
        [0, 5, 10, 15, 20, 25, 28, 30, 32, 35, 40],
        [45, 55, 65, 80, 90, 93, 88, 78, 65, 45, 25],
    )

    # Nitrogênio Total (mg/L) → qi
    _CURVE_NITROGENIO = (
        [0, 0.5, 1, 2, 3, 5, 10, 20, 50, 100],
        [98, 90, 80, 65, 55, 38, 20, 10, 4, 1],
    )

    # Fósforo Total (mg/L) → qi
    _CURVE_FOSFORO = (
        [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        [98, 92, 85, 72, 55, 38, 20, 10, 5, 2, 1],
    )

    # Turbidez (NTU) → qi
    _CURVE_TURBIDEZ = (
        [0, 1, 5, 10, 20, 30, 40, 60, 80, 100],
        [100, 97, 85, 72, 52, 38, 28, 15, 8, 5],
    )

    # Resíduos Totais (mg/L) → qi
    _CURVE_RESIDUOS = (
        [0, 50, 100, 150, 200, 250, 300, 350, 400, 500],
        [85, 80, 72, 62, 52, 42, 32, 22, 15, 5],
    )

    def __init__(self):
        """Inicializa as funções de interpolação das curvas de qualidade."""
        self._interpolators = {
            'od':          self._build_interpolator(*self._CURVE_OD),
            'coliformes':  self._build_interpolator_log(*self._CURVE_COLIFORMES),
            'ph':          self._build_interpolator(*self._CURVE_PH),
            'dbo':         self._build_interpolator(*self._CURVE_DBO),
            'temperatura': self._build_interpolator(*self._CURVE_TEMPERATURA),
            'nitrogenio':  self._build_interpolator(*self._CURVE_NITROGENIO),
            'fosforo':     self._build_interpolator(*self._CURVE_FOSFORO),
            'turbidez':    self._build_interpolator(*self._CURVE_TURBIDEZ),
            'residuos':    self._build_interpolator(*self._CURVE_RESIDUOS),
        }

    @staticmethod
    def _build_interpolator(x_points, y_points):
        """Cria interpolador linear com clamping nos limites."""
        return interp1d(
            x_points, y_points,
            kind='linear', bounds_error=False,
            fill_value=(y_points[0], y_points[-1])
        )

    @staticmethod
    def _build_interpolator_log(x_points, y_points):
        """Cria interpolador em escala logarítmica (para coliformes)."""
        log_x = np.log10(np.maximum(x_points, 1e-10))
        return interp1d(
            log_x, y_points,
            kind='linear', bounds_error=False,
            fill_value=(y_points[0], y_points[-1])
        )

    def _get_do_saturation(self, do_mg_l: float, temp_c: float) -> float:
        """
        Converte OD (mg/L) para porcentagem de saturação.
        Usa a fórmula de Benson & Krause (1984) para OD de saturação.
        """
        # OD de saturação em mg/L em função da temperatura (°C) ao nível do mar
        t = temp_c
        od_sat = (468.0 / (31.6 + t))
        return min((do_mg_l / od_sat) * 100, 140) if od_sat > 0 else 0

    def calculate_qi(self, parameter: str, value: float,
                     temp_c: float = None) -> float:
        """
        Calcula o qi (nota de qualidade) de um parâmetro.

        Args:
            parameter: nome do parâmetro ('od', 'ph', etc.)
            value: valor medido
            temp_c: temperatura (necessária apenas para OD)

        Returns:
            qi: valor entre 0 e 100
        """
        if np.isnan(value):
            return np.nan

        if parameter == 'od' and temp_c is not None:
            value = self._get_do_saturation(value, temp_c)

        if parameter == 'coliformes':
            value = max(value, 1)
            value = np.log10(value)
            qi = float(self._interpolators[parameter](value))
        else:
            qi = float(self._interpolators[parameter](value))

        return np.clip(qi, 0, 100)

    def calculate_iqa(self, row: dict, temp_c: float = None) -> dict:
        """
        Calcula o IQA para uma amostra (linha do dataset).

        Args:
            row: dicionário {parâmetro: valor} com os 9 parâmetros.
            temp_c: temperatura da amostra (para cálculo do OD).

        Returns:
            dict com 'iqa', 'classificacao', 'cor', 'qi_values'
        """
        qi_values = {}
        for param in self.PESOS:
            value = row.get(param, np.nan)
            if param == 'od':
                qi = self.calculate_qi(param, value, temp_c=temp_c)
            else:
                qi = self.calculate_qi(param, value)
            qi_values[param] = qi

        # Verificar parâmetros válidos (não NaN)
        valid = {k: v for k, v in qi_values.items() if not np.isnan(v)}

        if not valid:
            return {
                'iqa': np.nan,
                'classificacao': 'N/D',
                'cor': '⚪',
                'qi_values': qi_values,
            }

        # Produtório ponderado (modelo CETESB)
        # Normalizar pesos para os parâmetros disponíveis
        total_weight = sum(self.PESOS[k] for k in valid)
        iqa = 1.0
        for param, qi in valid.items():
            w = self.PESOS[param] / total_weight  # peso normalizado
            qi = max(qi, 0.01)  # evitar log(0)
            iqa *= qi ** w

        iqa = np.clip(iqa, 0, 100)

        # Classificação
        classificacao = 'N/D'
        cor = '⚪'
        for low, high, label, emoji in self.CLASSIFICACAO:
            if low <= iqa < high:
                classificacao = label
                cor = emoji
                break

        return {
            'iqa': round(iqa, 1),
            'classificacao': classificacao,
            'cor': cor,
            'qi_values': qi_values,
        }

    @staticmethod
    def _to_float(val):
        """Converte de forma segura para float, tratando strings com vírgula."""
        if pd.isna(val):
            return np.nan
        try:
            if isinstance(val, str):
                return float(val.replace(',', '.'))
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    def calculate_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o IQA para cada linha do DataFrame.

        Mapeia automaticamente as colunas do dataset para os parâmetros IQA.

        Returns:
            DataFrame com colunas: Ponto, IQA, Classificação, + qi de cada parâmetro
        """
        results = []

        for idx, row in df.iterrows():
            # Extrair valores usando o mapeamento de colunas
            params = {}
            temp_c = None

            for param, col in self.COLUMN_MAP.items():
                if param == 'nitrogenio':
                    # Somar as frações de nitrogênio
                    total = 0
                    has_value = False
                    for col_name in col:
                        col_name = col_name.strip()
                        matched = self._find_column(df, col_name)
                        if matched is not None:
                            val = self._to_float(row[matched])
                            if not np.isnan(val):
                                total += val
                                has_value = True
                    params[param] = total if has_value else np.nan
                else:
                    col_name = col.strip()
                    matched = self._find_column(df, col_name)
                    if matched is not None:
                        params[param] = self._to_float(row[matched])
                    else:
                        params[param] = np.nan

            # Temperatura para calcular saturação de OD
            temp_col = self._find_column(df, 'Temperatura')
            if temp_col is not None:
                temp_c = self._to_float(row[temp_col])

            result = self.calculate_iqa(params, temp_c=temp_c)
            result['ponto'] = idx + 1
            results.append(result)

        # Montar DataFrame de resultados
        output_rows = []
        for r in results:
            row_data = {
                'Ponto de Coleta': r['ponto'],
                'IQA': r['iqa'],
                'Classificação': f"{r['cor']} {r['classificacao']}",
            }
            # Adicionar qi de cada parâmetro
            param_labels = {
                'od': 'q(OD)', 'coliformes': 'q(Coliformes)',
                'ph': 'q(pH)', 'dbo': 'q(DBO)',
                'temperatura': 'q(Temp)', 'nitrogenio': 'q(N Total)',
                'fosforo': 'q(P Total)', 'turbidez': 'q(Turbidez)',
                'residuos': 'q(Resíduos)',
            }
            for param, label in param_labels.items():
                qi = r['qi_values'].get(param, np.nan)
                row_data[label] = round(qi, 1) if not np.isnan(qi) else None

            output_rows.append(row_data)

        return pd.DataFrame(output_rows)

    def explicar_iqa(self, results_df: pd.DataFrame, user_prompt: str) -> str:
        """
        Consome a API do Gemini para explicar os resultados do IQA baseando-se na pergunta do usuário.
        """
        try:
            from google import genai
            import streamlit as st
            import os
            
            # Tenta obter a chave da API
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except (FileNotFoundError, KeyError):
                api_key = os.environ.get("GEMINI_API_KEY")

            if not api_key:
                return "⚠️ **Chave de API não configurada.** Defina `GEMINI_API_KEY` nos secrets do Streamlit ou variáveis de ambiente."

            client = genai.Client(api_key=api_key)
            
            df_str = results_df.copy()
            df_str["Ponto de Coleta"] = df_str["Ponto de Coleta"].astype(str)
            data_csv = df_str.to_csv(index=False)
            
            prompt = f"""
            Você é um especialista em qualidade da água auxiliando no Laboratório de Pesquisa em Química Ambiental (LAPEQ).
            Analise os seguintes dados do Índice de Qualidade da Água (IQA) e responda à pergunta do usuário de forma didática.
            
            DADOS DE IQA CALCULADOS:
            {data_csv}
            
            A classificação do IQA segue a tabela:
            - 80 a 100: Ótima
            - 51 a 79: Boa
            - 36 a 50: Regular
            - 19 a 35: Ruim
            - 0 a 18: Péssima
            
            PERGUNTA DO USUÁRIO:
            "{user_prompt}"
            """

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            return response.text
            
        except ImportError:
            return "❌ O pacote `google-genai` não está instalado. Execute `pip install google-genai`."
        except Exception as e:
            return f"❌ Erro ao conectar com IA: {e}"
    
    @staticmethod
    def _find_column(df: pd.DataFrame, target: str):
        """Busca uma coluna no DataFrame ignorando espaços extras."""
        target_clean = target.strip().lower()
        for col in df.columns:
            if col.strip().lower() == target_clean:
                return col
        return None

    @classmethod
    def get_classificacao_table(cls) -> pd.DataFrame:
        """Retorna a tabela de classificação como DataFrame."""
        return pd.DataFrame([
            {'Faixa': f'{low} – {high}', 'Classificação': f'{emoji} {label}'}
            for low, high, label, emoji in cls.CLASSIFICACAO
        ])

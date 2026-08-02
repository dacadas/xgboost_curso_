import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from scipy import stats


# Configuração para exibição completa de colunas
pd.set_option('display.max_columns', None)

# Fixar Semente para Reprodutibilidade
semilla = 13
np.random.seed(semilla)
random.seed(semilla)



# 1. Divisão Treino / Teste (Prevenção de Data Leakage)
# 2. Tratamento de NaNs (Imputação)
# 3. Capping de Outliers (Percentis / Winsorizer)
# 4. Correção de Assimetria / Skewness (Log / Yeo-Johnson)  
# 5. Análise de Correlação e Eliminação de Multicolinearidade
# 6. Encoding de Categóricas e Scaler


# =============================================================================
# CONFIGURAÇÃO INICIAL E CARREGAMENTO
# =============================================================================
path = 'D:/Diego/Curso Udemy/xgboost_curso_/Data_Files/'
data = pd.read_csv( path + 'House_Price.csv')
target_col = 'price'

# data = pd.read_csv( path + 'train.csv')
# target_col = 'SalePrice'


X = data.drop(columns=[target_col])
y = data[target_col]

# =============================================================================
# 1 - Separar em Treino e Teste (Prevenção de Data Leakage)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=semilla
)

# =============================================================================
#  2 - Identificação inicial  tipo de dado 
# =============================================================================
# A) Detectar colunas de data automaticamente (por dtype ou tentando converter texto)
vars_date = []
vars_year = []
current_year = 2026 # Ano limite para identificação de colunas de ano

for col in X_train.columns:
    s = X_train[col]
    
    # Caso 1: Já está formatado como datetime nativo
    if pd.api.types.is_datetime64_any_dtype(s):
        vars_date.append(col)
        continue
        
    # Caso 2: Está em formato texto (object), mas possui conteúdo de data
    if s.dtype == 'object':
        try:
            parsed = pd.to_datetime(s, errors='coerce', format='mixed')
            if parsed.notnull().mean() > 0.8:
                vars_date.append(col)
                continue
        except Exception:
            pass
            
    # Caso 3: É numérico (int/float), mas representa um ANO (ex: 1800 até o ano atual)
    if pd.api.types.is_numeric_dtype(s):
        clean_s = s.dropna()
        if len(clean_s) > 0:
            # Verifica se os valores são inteiros e se estão dentro da faixa histórica plausível de anos
            is_integer_like = (clean_s % 1 == 0).all()
            min_val, max_val = clean_s.min(), clean_s.max()
            
            if is_integer_like and (1800 <= min_val <= current_year) and (1800 <= max_val <= current_year):
                vars_year.append(col)

# Detecção de pares de Ano e Mês para diagnóstico posterior de sazonalidade
col_ano_venda = [c for c in vars_year if 'sold' in c.lower() or 'sale' in c.lower()]
col_mes_venda = [c for c in X_train.columns if ('mo' in c.lower() or 'month' in c.lower()) and 'sold' in c.lower()]

# Categorias (exclui colunas de data)
vars_cat = [
    col for col in X_train.select_dtypes(include=['object', 'category']).columns.tolist() 
    if col not in vars_date
]

# Numéricas puras (EXCLUI colunas de data E colunas de ano!)
vars_num = [
    col for col in X_train.select_dtypes(include=['int64', 'float64']).columns.tolist() 
    if col not in vars_date and col not in vars_year
]

# =============================================================================
# 3. IDENTIFICAÇÃO DE VARIÁVEIS CONSTANTES / SEM VARIÂNCIA (Passo 8a)
# =============================================================================
print("="*60)
print(" 2. DIAGNÓSTICO DE VARIÂNCIA (FILTRO DE CONSTANTES)")
print("="*60)
limite_constante = 0.995
vars_constantes = []

for col in X_train.columns:
    top_freq_pct = X_train[col].value_counts(normalize=True, dropna=False).values[0]
    if top_freq_pct >= limite_constante:
        vars_constantes.append(col)
        print(f" -> REMOVER '{col}': {top_freq_pct*100:.1f}% dos dados são idênticos.")

# Variáveis válidas sem constantes
vars_num_validas = [col for col in vars_num if col not in vars_constantes]
vars_cat_validas = [col for col in vars_cat if col not in vars_constantes]
print(f"Variáveis constantes identificadas: {vars_constantes}\n")


# =============================================================================
# 4. DIAGNÓSTICO DE VALORES FALTANTES - NaNs (Passo 2)
# =============================================================================
print("="*60)
print(" 3. DIAGNÓSTICO DE DADOS AUSENTES (NaNs)")
print("="*60)
nan_pct = X_train.isnull().mean()
vars_com_nan = nan_pct[nan_pct > 0]

vars_cat_missing_label = []
vars_cat_frequent = []
vars_num_median = []

for col, pct in vars_com_nan.items():
    if col in vars_constantes:
        continue
    print(f" -> Coluna '{col}': {pct*100:.2f}% de ausência")
    
    # Critérios para Categóricas
    if col in vars_cat_validas:
        if pct > 0.10: # > 10%: Criar nova classe 'Missing'
            vars_cat_missing_label.append(col)
        else:          # < 10%: Imputação pela Moda (frequente)
            vars_cat_frequent.append(col)
            
    # Critérios para Numéricas
    elif col in vars_num_validas:
        vars_num_median.append(col) # Mediana (mais robusta contra assimetria)

print(f"\nDecisão -> Categóricas Imputadas com 'Missing': {vars_cat_missing_label}")
print(f"Decisão -> Categóricas Imputadas com Moda: {vars_cat_frequent}")
print(f"Decisão -> Numéricas Imputadas com Mediana: {vars_num_median}\n")

# =============================================================================
# 5. DIAGNÓSTICO DE ASSIMETRIA E OUTLIERS (Passos 3 e 4)
# =============================================================================
print("="*60)
print(" 4. DIAGNÓSTICO DE ASSIMETRIA (SKEWNESS) E OUTLIERS")
print("="*60)

# A) Teste de Assimetria (|Skewness| > 1.0 = Alta Assimetria / Cauda Longa)
skewness = X_train[vars_num_validas].skew()
vars_alta_assimetria = skewness[abs(skewness) > 1.0].index.tolist()

print("Variáveis com Cauda Longa (|Skewness| > 1.0):")
for col in vars_alta_assimetria:
    print(f" -> '{col}': Skewness = {skewness[col]:.2f} (Aplicar Yeo-Johnson)")

# B) Teste de Outliers (Regra do Intervalo Interquartil - IQR 1.5x)
vars_outliers_capping = []
for col in vars_num_validas:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    outliers = X_train[(X_train[col] < limite_inferior) | (X_train[col] > limite_superior)]
    if len(outliers) > 0:
        vars_outliers_capping.append(col)

print(f"\nDecisão -> Variáveis numéricas para Capping (IQR 1.5x): {vars_outliers_capping}\n")


# =============================================================================
# 6. DIAGNÓSTICO DE CORRELAÇÃO E MULTICOLINEARIDADE (Passo 5)
# =============================================================================
print("="*60)
print(" 5. ANÁLISE DE CORRELAÇÃO (APÓS TRANSFORMAÇÃO YEO-JOHNSON)")
print("="*60)

# Cópia temporária do Treino para simular a saída das etapas anteriores
X_train_eda = X_train.copy()

# A) Imputação temporária dos NaNs numéricos para permitir o cálculo de correlação
for col in vars_num_median:
    X_train_eda[col] = X_train_eda[col].fillna(X_train_eda[col].median())

# B) Aplicação temporária do Yeo-Johnson nas variáveis de alta assimetria
if len(vars_alta_assimetria) > 0:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    X_train_eda[vars_alta_assimetria] = pt.fit_transform(X_train_eda[vars_alta_assimetria])

# Matriz de Correlação de Pearson com todas as variáveis numéricas válidas já transformadas
corr_matrix = X_train_eda[vars_num_validas].corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identificar pares com Correlação > 0.85
alta_corr = [
    (col1, col2, upper_tri.loc[col1, col2])
    for col1 in upper_tri.columns
    for col2 in upper_tri.index
    if upper_tri.loc[col1, col2] > 0.85
]

print("Pares com alta correlação (|r| > 0.85) identificados:")
if len(alta_corr) > 0:
    for col1, col2, r in alta_corr:
        print(f" -> {col1} x {col2}: r = {r:.4f}")
else:
    print(" -> Nenhum par com correlação > 0.85 foi encontrado.\n")

# print("\nAnálise de Engenharia:")
# print(" -> 'f1' x 'f2' têm r > 0.99. Substituir pela média 'dist_mean'(MathematicalCombination). e descartar originais")
# print(" -> 'f3' x 'f4' têm r = 0.85. Uma delas será eliminada por DropCorrelatedFeatures.\n")

# Decisão de Engenharia x Exclusão:
# 1 - criar 'dist_mean' (MathematicalCombination) e descartar as 4 originais.
# 2 - DropCorrelatedFeatures(threshold=0.85) para remover colunas remanescentes duplicadas.

# =============================================================================
# 7. CARDINALIDADE E ORDINALIDADE CATEGÓRICA (Passo 6)
# =============================================================================
print("="*60)
print(" 6. DIAGNÓSTICO DE CATEGÓRICAS (ENCODING)")
print("="*60)
vars_cat_nominal_baixa = []  # Para OneHotEncoder
vars_cat_nominal_alta = []   # Para TargetEncoder / RareLabelEncoder

for col in vars_cat_validas:
    cardinalidade = X_train[col].nunique()
    valores_unicos = X_train[col].dropna().unique()
    if cardinalidade <= 10:
        vars_cat_nominal_baixa.append(col)
    else:
        vars_cat_nominal_alta.append(col)
    print(f" -> Coluna '{col}': Cardinalidade = {cardinalidade} | Valores: {valores_unicos}")

# Critério de Decisão:
# - Se tem ordem intrínseca (ex: 'Pobre', 'Médio', 'Rico') -> OrdinalEncoder
# - Se não tem ordem (nominal) e baixa cardinalidade (< 10) -> OneHotEncoder
# - Se alta cardinalidade (> 10) -> TargetEncoder ou RareLabelEncoder
# print(" Decisão -> 'airport', 'waterbody' são Nominais e de baixa cardinalidade: Usar OneHotEncoder\n")


# =============================================================================
# 8. VERIFICAR DADOS DE TEMPO E SAZONALIDADE
# =============================================================================
print("="*60)
print(" 8. VERIFICAR DADOS DE TEMPO E SAZONALIDADE")
print("="*60)
componentes_sazonais_identificados = {}

# Lista para análise: combina datas nativas + data composta de ano e mês (se existir)
datas_para_analise = list(vars_date)
if len(col_ano_venda) > 0 and len(col_mes_venda) > 0:
    datas_para_analise.append('_temp_date_sold')

if len(datas_para_analise) > 0:
    for col in datas_para_analise:
        if col == '_temp_date_sold':
            dt_series = pd.to_datetime(
                X_train[col_ano_venda[0]].astype(int).astype(str) + '-' + 
                X_train[col_mes_venda[0]].astype(int).astype(str) + '-01',
                errors='coerce'
            )
            nome_display = f"venda_combinada ({col_ano_venda[0]} + {col_mes_venda[0]})"
        else:
            dt_series = pd.to_datetime(X_train[col])
            nome_display = col
        
        componentes = {
            'month': ('Mês', dt_series.dt.month),
            'dayofweek': ('Dia da Semana', dt_series.dt.dayofweek),
            'quarter': ('Trimestre', dt_series.dt.quarter),
            'hour': ('Hora', dt_series.dt.hour if (dt_series.dt.hour > 0).any() else None)
        }
        
        sazonalidades_col = []
        print(f" Análise de Sazonalidade para '{nome_display}':")
        
        for comp_key, (nome_pt, comp_series) in componentes.items():
            if comp_series is None or comp_series.nunique() <= 1:
                continue
            
            df_temp = pd.DataFrame({'comp': comp_series, 'target': y_train}).dropna()
            grupos = [g['target'].values for _, g in df_temp.groupby('comp')]
            
            if len(grupos) > 1:
                _, p_val = stats.f_oneway(*grupos)
                if p_val < 0.05:
                    sazonalidades_col.append(comp_key)
                    print(f"   -> [Sazonalidade Detectada] por {nome_pt} (p-value = {p_val:.4e})")
                else:
                    print(f"   -> [Sem Sazonalidade] por {nome_pt} (p-value = {p_val:.4f})")
        
        key_name = f"{col_ano_venda[0]}_{col_mes_venda[0]}" if col == '_temp_date_sold' else col
        componentes_sazonais_identificados[key_name] = sazonalidades_col
else:
    print(" -> Nenhuma coluna de data nativa/composta foi identificada para teste de sazonalidade.")
# =============================================================================
# 9. RESUMO DOS PARÂMETROS GERADOS PARA O PIPELINE
# =============================================================================

print("\n" + "="*60)
print(" 🎯 PARÂMETROS PRONTOS PARA PASSAR AO PIPELINE")
print("="*60)

params_pipeline = {
    'VAR_CONSTANTES': vars_constantes,
    'VAR_DATE': vars_date,                  # Datas originais do dataset
    'VAR_YEAR': vars_year,                  # Colunas de ano (YearBuilt, YrSold, etc.)
    'COMPONENTES_SAZONAIS': componentes_sazonais_identificados,
    'VAR_CAT_MISSING_LABEL': vars_cat_missing_label,
    'VAR_CAT_FREQUENT': vars_cat_frequent,
    'VAR_NUM_MEDIAN': vars_num_median,
    'VAR_OUTLIERS_CAPPING': vars_outliers_capping,
    'VAR_YEO_JOHNSON': vars_alta_assimetria,
    'VAR_CAT_ONEHOT': vars_cat_nominal_baixa,
    'VAR_CAT_RARE_TARGET': vars_cat_nominal_alta
}

for k, v in params_pipeline.items():
    print(f"{k:<23} = {v}")
import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import joblib

# Configura o scikit-learn para manter as saídas sempre como DataFrame do Pandas
sklearn.set_config(transform_output="pandas")


# Para fins de reprodutibilidade
semilla  = 13
np.random.seed( semilla )
random.seed(semilla) 


# =============================================================================
# 1. TRANSFORMADORES CUSTOMIZADOS REUTILIZÁVEIS
# =============================================================================

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Passo 4: Extrai atributos temporais e componentes cíclicos (seno/cosseno)."""
    def __init__(self, date_cols=None):
        self.date_cols = date_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in self.date_cols:
            if col in X_df.columns:
                dt = pd.to_datetime(X_df[col], errors='coerce')
                X_df[f'{col}_year'] = dt.dt.year
                X_df[f'{col}_month'] = dt.dt.month
                X_df[f'{col}_day'] = dt.dt.day
                X_df[f'{col}_dayofweek'] = dt.dt.dayofweek
                
                # Transformação Cíclica para Sazonalidade (Mês e Dia da Semana)
                X_df[f'{col}_sin_month'] = np.sin(2 * np.pi * dt.dt.month / 12)
                X_df[f'{col}_cos_month'] = np.cos(2 * np.pi * dt.dt.month / 12)
                
                # Descarte da coluna original em formato string/data raw
                X_df = X_df.drop(columns=[col])
        return X_df


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Passo 3: Trata outliers limitando valores nos percentis (Clipping / Winsorization)."""
    def __init__(self, lower_percentile=0.01, upper_percentile=0.99, cols=None):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.cols = cols
        self.caps_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        target_cols = self.cols if self.cols is not None else X_df.select_dtypes(include=[np.number]).columns
        for col in target_cols:
            if col in X_df.columns:
                low = X_df[col].quantile(self.lower_percentile)
                high = X_df[col].quantile(self.upper_percentile)
                self.caps_[col] = (low, high)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, (low, high) in self.caps_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=low, upper=high)
        return X_df


class DropConstantFeatures(BaseEstimator, TransformerMixin):
    """Passo 8a: Remove colunas constantes ou sem variância significativa."""
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.to_drop_ = []
        for col in X_df.columns:
            top_freq = X_df[col].value_counts(normalize=True, dropna=False).max()
            if top_freq >= self.threshold:
                self.to_drop_.append(col)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.drop(columns=self.to_drop_, errors='ignore')


class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """Passo 8b: Remove variáveis numéricas altamente correlacionadas entre si."""
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).select_dtypes(include=[np.number])
        if X_df.empty:
            return self
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.drop(columns=self.to_drop_, errors='ignore')


# =============================================================================
# 2. CONFIGURAÇÃO ESPECÍFICA DO SEU DATASET (Altere apenas aqui para cada projeto)
# =============================================================================

# Carregar Dados
path = 'D:/Diego/Curso Udemy/xgboost_curso_/Data_Files/'
data = pd.read_csv( path + 'House_Price.csv')
target_col = 'price'

X = data.drop(columns=[target_col])
y = data[target_col]

# Separar em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=semilla
)

### analisis exploratorio de datos
# identificar quais variáveis são 
# categóricas
vars_cat = [var for var in X_train.columns if X_train[var].dtype == 'O']
# numéricas
vars_num = [var for var in X_train.columns if var not in vars_cat and var != target_col ]


# Verificar dados faltantes
vars_num_with_nan = [ var for var in vars_num if X_train[var].isnull().sum() > 0]

# Separar todas as variáveis categóricas com perda de dados (conjunto de treino)
vars_cat_with_na = [ var for var in vars_cat if X_train[var].isnull().sum() > 0]


# Exibe a porcentagem de valores faltantes por variável
print( X_train[ vars_cat_with_na ].isnull().mean().sort_values(ascending=False) )
print( X_train[ vars_num_with_nan ].isnull().mean().sort_values(ascending=False) )


# Variáveis para imputar com a string 'Missing' (10% é um valor arbitrário)
with_string_missing = [ var for var in vars_cat_with_na if X_train[var].isnull().mean() > 0.1]

# Variáveis para imputar com a categoria mais frequente
with_frequent_category = [ var for var in vars_cat_with_na if X_train[var].isnull().mean() < 0.1]


### extended date dictionary
## observar --> count -- para saber si hay nan
##          --> diferencia entre media y mediana -- medio saber distribucion
##          --> dif entre media, max e min -- outliers
EDD = data.describe()


## verificar coneccion entre variables y label
# for  k in  cat_vars:
#     plt.figure()
#     sns.countplot(x=k,  data = data)
#     plt.figure()
#     sns.jointplot(x=k, y = target, data = data)

# for  k in  num_vars:
#     plt.figure()
#     sns.jointplot(x=k, y = target, data = data)



## onde ha outlier (generalmente conjunto de treino)
num_com = vars_num.copy()
for x in vars_num_with_nan:
    num_com.remove(x) 

vars_num_outlier = []
for num_var in num_com: 
    lower = data[num_var].quantile(0.005)
    upper = data[num_var].quantile(0.995)    
    a1 = data[ num_var ].clip(lower, upper)   
    a2 = data[ num_var ]
    if np.sum(1*( a1 != a2)) >0:
        vars_num_outlier.append(num_var)

# Identificar variáveis numéricas com cauda longa 

# Identificar variáveis temporais

# identificar a variaves a serem transformadas
# Passo 1: Definir Grupos de Variáveis
DATE_VARS = []  # Adicione o nome da coluna se houver data (ex: ['venda_data'])
NUM_VARS = ['crime_rate', 'resid_area', 'air_qual', 'room_num', 'age', 
            'dist1', 'dist2', 'dist3', 'dist4', 'teachers', 'poor_prop', 
            'n_hos_beds', 'n_hot_rooms', 'rainfall', 'parks']

# Variáveis categóricas ordenadas (hierárquicas)
CAT_ORDINAL_VARS = [] 
# Ordens das categorias para OrdinalEncoder (ex: [['Low', 'Medium', 'High']])
ORDINAL_CATEGORIES = []

# Variáveis categóricas nominais (sem ordem)
CAT_NOMINAL_VARS = ['airport', 'waterbody', 'bus_ter']


# =============================================================================
# 3. MONTAGEM DO PIPELINE GENÉRICO MULTI-ETAPAS
# =============================================================================

# Sub-pipeline 1: Processamento de Numéricas
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Passo 2: Mediana para NaNs
    ('outlier_capper', QuantileCapper(lower_percentile=0.01, upper_percentile=0.99)),  # Passo 3: Outliers
    ('scaler', StandardScaler())  # Passo 7: Normalização
])

# Sub-pipeline 2: Processamento de Categóricas Ordinais
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Passo 2: Moda
    ('ordinal_encoder', OrdinalEncoder(categories=ORDINAL_CATEGORIES if ORDINAL_CATEGORIES else 'auto')) # Passo 6
])

# Sub-pipeline 3: Processamento de Categóricas Nominais
nominal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),  # Passo 2: Nova classe 'Missing'
    ('one_hot_encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))  # Passo 6
])

# ColumnTransformer: Aplica as transformações específicas em cada grupo de colunas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, NUM_VARS),
        ('cat_ord', ordinal_pipeline, CAT_ORDINAL_VARS) if CAT_ORDINAL_VARS else ('drop_ord', 'drop', []),
        ('cat_nom', nominal_pipeline, CAT_NOMINAL_VARS) if CAT_NOMINAL_VARS else ('drop_nom', 'drop', [])
    ],
    remainder='passthrough'
)

# Pipeline Principal Unificado
generic_pipeline = Pipeline([
    ('drop_constant', DropConstantFeatures(threshold=0.99)),  # Passo 8a: Remove colunas sem variância (bus_ter)
    ('date_extractor', DatetimeFeatureExtractor(date_cols=DATE_VARS)),  # Passo 4: Trata datas
    ('column_transformer', preprocessor),  # Passos 2, 3, 6, 7
    ('drop_correlated', DropCorrelatedFeatures(threshold=0.85))  # Passo 8b: Filtra redundâncias correlacionadas
])


# =============================================================================
# 4. EXECUÇÃO E APLICAÇÃO DO PIPELINE
# =============================================================================

# Treina as regras APENAS no conjunto de treino
X_train_processed = generic_pipeline.fit_transform(X_train, y_train)

# Aplica as regras aprendidas ao conjunto de teste (Sem Data Leakage)
X_test_processed = generic_pipeline.transform(X_test)

print("--- PROCESSAMENTO CONCLUÍDO COM SUCESSO ---")
print(f"Shape original de X_train: {X_train.shape}")
print(f"Shape processado de X_train: {X_train_processed.shape}")
print("\nColunas finais geradas:")
print(X_train_processed.columns.tolist())

# Salvar o Pipeline para Produção/Implantação
joblib.dump(generic_pipeline, 'generic_preprocessing_pipeline.pkl')
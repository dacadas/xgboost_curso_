import numpy as np
import pandas as pd
import seaborn as sns
import random

from IPython.display import Image
import pydotplus
# from Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score

# Para fins de reprodutibilidade
semilla  = 13
np.random.seed( semilla )
random.seed(semilla) 



# path = 'D:/udemy/xgboost_curso/Data_Files/'


# data = pd.read_csv( path +  'Customer.csv')


# a1 = data.iloc[0]
# sns.histplot(data['Age'], kde = False)


# iris = sns.load_dataset('iris')
# sns.jointplot(x="sepal_length", y = "sepal_width", data = iris)

# #todos contra todos
# sns.pairplot(iris)



path = 'D:/Diego/Curso Udemy/xgboost_curso_/Data_Files/'
data = pd.read_csv( path +  'Movie_regression.csv', header = 0)
target = 'Collection'



cat_vars = [var for var in data.columns if data[var].dtype == 'O']

# # Separar todas as variáveis categóricas com perda de dados (conjunto de treino)
# cat_vars_with_na = [ var for var in cat_vars if data[var].isnull().sum() > 0]

## a2 -- Numéricas 
# Primeiro identificar quais variáveis são numéricas
num_vars = [    var for var in data.columns if var not in cat_vars and var != target]

# Verificar dados faltantes
vars_with_na = [    var for var in num_vars    if data[var].isnull().sum() > 0]

## rellenar nan en variable numerica
data.fillna( {'Time_taken':  data['Time_taken'].mean()}, inplace = True)

## convertir categoricas en numericas
data = pd.get_dummies( data, columns = cat_vars, drop_first=True)


X = data.loc[:, data.columns != target]
y = data[target]

###############################################################################
## Separa em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split( X, # Variáveis preditivas (X)
                                                     y, # Alvo (y)
                                                     test_size    = 0.2, # Proporção do dataset a ser alocada para o conjunto de teste
                                                     random_state = semilla, # Estamos definindo a semente (seed) aqui
                                                    )

## entrenar
reg_tree = tree.DecisionTreeRegressor(  max_depth = 3)
reg_tree.fit( X_train, y_train )

# predecir
y_predict_train = reg_tree.predict( X_train) 
y_predict_test  = reg_tree.predict( X_test) 

# Avaliar 
mse_train = mean_squared_error(y_train, y_predict_train)
mse_test = mean_squared_error(y_test, y_predict_test)

r2_train = r2_score(y_train, y_predict_train)
r2_test  = r2_score(y_test, y_predict_test)


## plotar
# pip install pydotplus
# conda install -c conda-forge graphviz python-graphviz
dotdata = tree.export_graphviz(reg_tree, out_file=None)
graph = pydotplus.graph_from_dot_data(dotdata)
Image(graph.create_png())
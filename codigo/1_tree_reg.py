import numpy as np
import pandas as pd
import seaborn as sns
import random

import matplotlib.pyplot as plt

# from IPython.display import Image
# import pydotplus
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


## entender tree
# score = [35,38,40,45,35,65,70,75,80,85]
# saida = []
# for k in range( 1, len(score)):
#     a1 = score[0:k]
#     a2 = score[k:]
    
#     m1 = np.sum( (a1 - np.mean(a1))**2 )
#     m2 = np.sum( (a2 - np.mean(a2))**2 )
    
#     saida.append([k, m1, m2, m1+m2])
    
# df = pd.DataFrame( saida )    


################################################################################
############ Generacion de arbol  #####################
################################################################################        
# path = 'D:/udemy/xgboost_curso_/Data_Files/'
# # path = 'D:/Diego/Curso Udemy/xgboost_curso_/Data_Files/'
# data = pd.read_csv( path +  'Movie_regression.csv', header = 0)
# target = 'Collection'



# cat_vars = [var for var in data.columns if data[var].dtype == 'O']

# # # Separar todas as variáveis categóricas com perda de dados (conjunto de treino)
# # cat_vars_with_na = [ var for var in cat_vars if data[var].isnull().sum() > 0]

# ## a2 -- Numéricas 
# # Primeiro identificar quais variáveis são numéricas
# num_vars = [    var for var in data.columns if var not in cat_vars and var != target]

# # Verificar dados faltantes
# vars_with_na = [    var for var in num_vars    if data[var].isnull().sum() > 0]

# ## rellenar nan en variable numerica
# data.fillna( {'Time_taken':  data['Time_taken'].mean()}, inplace = True)

# ## convertir categoricas en numericas
# data = pd.get_dummies( data, columns = cat_vars, drop_first=True)


# X = data.loc[:, data.columns != target]
# y = data[target]

# ###############################################################################
# ## Separa em conjunto de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split( X, # Variáveis preditivas (X)
#                                                      y, # Alvo (y)
#                                                      test_size    = 0.2, # Proporção do dataset a ser alocada para o conjunto de teste
#                                                      random_state = semilla, # Estamos definindo a semente (seed) aqui
#                                                     )
# ## entrenar
# reg_tree = tree.DecisionTreeRegressor(  max_depth = 3)
# reg_tree.fit( X_train, y_train )

# # predecir
# y_predict_train = reg_tree.predict( X_train) 
# y_predict_test  = reg_tree.predict( X_test) 

# # Avaliar 
# mse_train = mean_squared_error(y_train, y_predict_train)
# mse_test = mean_squared_error(y_test, y_predict_test)

# r2_train = r2_score(y_train, y_predict_train)
# r2_test  = r2_score(y_test, y_predict_test)

# ## plotar com matplotlib
# plt.figure(figsize=(20, 10)) # Ajuste o tamanho para a árvore não ficar espremida
# tree.plot_tree(reg_tree, 
#                feature_names=X_train.columns, 
#                filled=True, 
#                rounded=True, 
#                fontsize=10)

# # Exibe o gráfico na aba "Plots" do Spyder
# plt.show()
################################################################################
####################   Poda   ##################################
################################################################################ 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree

path = 'D:/udemy/xgboost_curso_/Data_Files/'
path = 'D:/Diego/Curso Udemy/xgboost_curso_/Data_Files/'
data = pd.read_csv( path +  'Movie_regression.csv', header = 0)
target = 'Collection'

cat_vars = [var for var in data.columns if data[var].dtype == 'O']

## a2 -- Numéricas 
# Primeiro identificar quais variáveis são numéricas
num_vars = [var for var in data.columns if var not in cat_vars and var != target]

# Verificar dados faltantes
vars_with_na = [var for var in num_vars if data[var].isnull().sum() > 0]

## rellenar nan en variable numerica
data.fillna( {'Time_taken':  data['Time_taken'].mean()}, inplace = True)

## convertir categoricas en numericas
data = pd.get_dummies( data, columns = cat_vars, drop_first=True)

X = data.loc[:, data.columns != target]
y = data[target]

###############################################################################
## Separa em conjunto de treinamento e teste
semilla = 42 # Apenas garantindo que a variável semilla exista
X_train, X_test, y_train, y_test = train_test_split( X, # Variáveis preditivas (X)
                                                     y, # Alvo (y)
                                                     test_size    = 0.2, # Proporção do dataset a ser alocada para o conjunto de teste
                                                     random_state = semilla, # Estamos definindo a semente (seed) aqui
                                                    )
## entrenar


# =============================================================================
### Crescimiento del arbol --> opcion 1 -- profundidad (max_depth)
# =============================================================================
# max_depth: Define o limite de "andares" da árvore. 
# Sem isso, a árvore cresce até memorizar os dados (overfitting).
# REDUZIR esse valor ajuda a generalizar o modelo e evitar overfitting.
reg_tree = tree.DecisionTreeRegressor( max_depth = 3 ) 
reg_tree.fit( X_train, y_train )

## plotar com matplotlib
plt.figure(figsize=(20, 10)) # Ajuste o tamanho para a árvore não ficar espremida
tree.plot_tree(reg_tree, 
               feature_names=X_train.columns, 
               filled=True, 
               rounded=True, 
               fontsize=10)
plt.show()


# =============================================================================
### Crescimiento del arbol --> opcion 2 -- minimo de observaciones para dividir el nodo
# =============================================================================
# min_samples_split: O nó precisa ter no mínimo 'X' amostras para poder se dividir.
# AUMENTAR esse valor impede que a árvore crie regras baseadas em grupos muito pequenos.
# Isso funciona como um freio e DIMINUI o overfitting.
reg_tree2 = tree.DecisionTreeRegressor( min_samples_split = 40 )
reg_tree2.fit( X_train, y_train )

## plotar com matplotlib
plt.figure(figsize=(20, 10)) 
tree.plot_tree(reg_tree2, 
               feature_names=X_train.columns, 
               filled=True, 
               rounded=True, 
               fontsize=10)
plt.show()


# =============================================================================
### Crescimiento del arbol --> opcion 3 -- minimo de hojas al final
# =============================================================================
# min_samples_leaf: O resultado final de uma quebra (a folha) deve ter pelo menos 'X' amostras.
# AUMENTAR esse valor impede que a árvore crie resultados isolados (ex: folhas com 1 só dado).
# Também funciona como um freio e DIMINUI o overfitting.
reg_tree3 = tree.DecisionTreeRegressor( min_samples_leaf = 25 )
reg_tree3.fit( X_train, y_train )

## plotar com matplotlib
plt.figure(figsize=(20, 10)) 
tree.plot_tree(reg_tree3, 
               feature_names=X_train.columns, 
               filled=True, 
               rounded=True, 
               fontsize=10)
plt.show()
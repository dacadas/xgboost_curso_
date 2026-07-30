################################################################################
####################   Poda   ##################################
################################################################################ 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree

path = 'D:/udemy/xgboost_curso_/Data_Files/'
path = 'D:/Diego/Curso Udemy/xgboost_curso_/Data_Files/'
data = pd.read_csv( path +  'Movie_classification.csv', header = 0)
target = 'Start_Tech_Oscar'

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
reg_tree = tree.DecisionTreeClassifier( max_depth = 5 ) 
reg_tree.fit( X_train, y_train )

# predecir
y_predict_train = reg_tree.predict( X_train) 
y_predict_test  = reg_tree.predict( X_test) 

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


auc_train =  accuracy_score( y_train, y_predict_train)
roc_auc_score_train =  roc_auc_score( y_train, y_predict_train)

cm_train = confusion_matrix( y_train, y_predict_train)
print('train auc: {}'.format((auc_train)))
print('train roc_auc_score: {}'.format((roc_auc_score_train)))
print('train cm_train:\n {}'.format((cm_train)))

# test
auc_test =  accuracy_score( y_test, y_predict_test)
roc_auc_score_test =  roc_auc_score( y_test, y_predict_test)
cm_test = confusion_matrix( y_test, y_predict_test)
print('test auc: {}'.format((roc_auc_score_test)))
print('test roc_auc_score: {}'.format((roc_auc_score_test)))
print('test cm_train:\n {}'.format((cm_test)))



## plotar com matplotlib
plt.figure(figsize=(16, 10)) # Ajuste o tamanho para a árvore não ficar espremida
tree.plot_tree(reg_tree, 
               feature_names=X_train.columns, 
               filled=True, 
               rounded=True, 
               fontsize=10)
plt.show()

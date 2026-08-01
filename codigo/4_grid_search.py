import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from  sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


n_estimadores = 50

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
### crear arbol
# =============================================================================
reg_tree = tree.DecisionTreeClassifier(  ) 

params_grid_bag = {"n_estimators":[10,15,20],
                   
                   }
## aplicar baggin
bag_tree = BaggingClassifier( estimator = reg_tree,
                              n_estimators = n_estimadores,
                              bootstrap = True,
                              random_state = semilla,
                             )
grid_search_bag = GridSearchCV(bag_tree, 
                               params_grid_bag,
                               n_jobs = -1,
                               cv = 5,
                               scoring = 'accuracy'
                               )

grid_search_bag.fit( X_train, y_train )

# separar mejor modelo
bag_tree = grid_search_bag.best_estimator_

# ver mejores parametros
print( grid_search_bag.best_params_ )

# predecir
y_predict_train = bag_tree.predict( X_train) 
y_predict_test  = bag_tree.predict(   X_test) 

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


auc_train =  accuracy_score( y_train, y_predict_train)
roc_auc_score_train =  roc_auc_score( y_train, y_predict_train)

# cm_train = confusion_matrix( y_train, y_predict_train)
# print('train auc: {}'.format((auc_train)))
# print('train roc_auc_score: {}'.format((roc_auc_score_train)))
# print('train cm_train:\n {}'.format((cm_train)))

# test
auc_test =  accuracy_score( y_test, y_predict_test)
roc_auc_score_test =  roc_auc_score( y_test, y_predict_test)
cm_test = confusion_matrix( y_test, y_predict_test)
print('test auc: {}'.format((roc_auc_score_test)))
print('test roc_auc_score: {}'.format((roc_auc_score_test)))
print('test cm_train:\n {}'.format((cm_test)))

############################################################################
print(*30*'#')
print('RandomForest')

params_grid_rf = {"max_features":[4,5,6,7,8,9,10],
                  "min_samples_split":[2,3,5,10]}
                   
                   
## aplicar baggin
RF_tree = RandomForestClassifier(  n_estimators = n_estimadores,                           
                                   random_state = semilla,
                                   n_jobs = -1
                                 )
grid_search_rf = GridSearchCV(RF_tree, 
                               params_grid_rf,
                               n_jobs = -1,
                               cv = 5,
                               scoring = 'accuracy'
                               )
grid_search_rf.fit( X_train, y_train )
# separar mejor modelo
RF_tree = grid_search_rf.best_estimator_

# ver mejores parametros
print( grid_search_rf.best_params_ )


# predecir
y_predict_train = RF_tree.predict( X_train) 
y_predict_test  = RF_tree.predict(   X_test) 

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


############################################################################
print(*30*'#')
print('XGB_tree')

params_grid_XGB = {"max_depth":[7,9],
                  "gamma":[0.1,0.3],
                  "learning_rate":[0.01, 0.1, 0.2],
                  "subsample":[0.7,0.8],
                  "colsample_bytree":[0.7,0.8],
                  "reg_alpha":[  0.1, 0.21],
                  # "n_estimators":[n_estimadores, 5*n_estimadores, 10*n_estimadores]
                  }
                   
                   
## aplicar baggin
XGB_tree = xgb.XGBClassifier(  learning_rate = 0.01,
                                n_estimators = 2*n_estimadores,
                                max_depth = 4,
                                n_jobs = -1,
                            )
grid_search_xgb = GridSearchCV(XGB_tree, 
                               params_grid_XGB,
                               n_jobs = -1,
                               cv = 5,
                               scoring = 'accuracy'
                               )
grid_search_xgb.fit( X_train, y_train )
# separar mejor modelo
XGB_tree = grid_search_xgb.best_estimator_

# ver mejores parametros
print( grid_search_xgb.best_params_ )


# predecir
y_predict_train = XGB_tree.predict( X_train) 
y_predict_test  = XGB_tree.predict(   X_test) 

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
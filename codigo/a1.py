import numpy as np
import pandas as pd
import seaborn as sns

path = 'D:/udemy/xgboost_curso/Data_Files/'
data = pd.read_csv( path +  'Customer.csv')


a1 = data.iloc[0]
sns.histplot(data['Age'], kde = False)


iris = sns.load_dataset('iris')
sns.jointplot(x="sepal_length", y = "sepal_width", data = iris)

#todos contra todos
sns.pairplot(iris)
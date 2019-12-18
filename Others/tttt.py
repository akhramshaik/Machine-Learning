import pandas as pd
from sklearn import tree, model_selection,preprocessing
from sklearn.preprocessing import Imputer
import numpy as np

from sklearn.impute import SimpleImputer
import sklearn
print(sklearn.__version__)




titanic_train = pd.read_csv("C:/Users/akhram/Desktop/Algo/ML/train.csv")
print(titanic_train.info())
print(titanic_train.columns)

imputer = preprocessing.Imputer()
imputer.fit(titanic_train[['Age']])
print(imputer.statistics_)
titanic_train['Age_imputed'] = imputer.transform(titanic_train[['Age']])

print(titanic_train['Age_imputed'])

gender_process =preprocessing.LabelEncoder()
gender_process.fit(titanic_train['Sex'])
titanic_train['Sex'] = gender_process.transform(titanic_train['Sex'])
print(titanic_train['Sex'])


X = titanic_train.iloc[:,:-1].values
y = titanic_train['Survived']
print(titanic_train['Age'].isnull())
print(titanic_train['Age'])

age_imputer_data = SimpleImputer(strategy="most_frequent",missing_values="NaN")
age_imputer_data.fit(titanic_train['Age'])
SimpleImputer(add_indicator=False, copy=True, fill_value=None,missing_values='NaN', strategy='mean', verbose=0)


age_imputer_data.transform(titanic_train)


age_imputer_data = SimpleImputer(strategy="most_frequent",missing_values="number")


print(titanic_train['Age'])

age_imputer_data_1 = SimpleImputer(strategy="most_frequent",missing_values=np.nan)
age_imputer_data_1.fit(titanic_train)
age_imputer_data_1.transform(titanic_train['Age'])
print(titanic_train['Age'])

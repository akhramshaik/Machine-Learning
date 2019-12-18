import pandas as pd
import numpy as np
from sklearn import tree,model_selection,preprocessing
from sklearn.impute import SimpleImputer
import sklearn
print(sklearn.__version__)


df = pd.DataFrame([["a", "x"],
                   [np.nan, "y"],
                   ["a", np.nan],
                   ["b", "y"]], dtype="category")

imp = SimpleImputer(strategy="most_frequent")
print(imp.fit_transform(df))      



excel_data = pd.read_csv('C:/Users/akhram/Desktop/Algo/ML/train.csv')

print(excel_data.info())
print(excel_data.columns)

gender_label_enc = preprocessing.LabelEncoder()
gender_label_enc.fit(excel_data['Sex'])
excel_data['Sex'] = gender_label_enc.transform(excel_data['Sex'])

age_imputer_data = SimpleImputer(missing_values=np.NaN,strategy="most_frequent")
age_imputer_data.fit(excel_data)

print(excel_data['Age'])
print(excel_data['Age_1'])

X_train = excel_data[['Pclass','SibSp','Parch','Sex']] 
y_train = excel_data['Survived']
print(X_train)
print(y_train)

dt_declare = tree.DecisionTreeClassifier()
dt_declare.fit(X_train,y_train)

parameters_set = {'max_depth':[3,4,5,6,7,8,9,10],'min_samples_split':[3, 5, 10],'criterion':['gini', 'entropy']}

dt_grid_estimator = model_selection.GridSearchCV(dt_declare,parameters_set,cv=10,scoring='accuracy')
dt_grid_estimator.fit(X_train,y_train)

print(dt_grid_estimator.cv_results_)
print(dt_grid_estimator.best_params_)
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_estimator_)


print(dt_grid_estimator.score(X_train, y_train))

dt_declare.score(X_train,y_train)

#model_selection.cross_validate()

excel_data_test = pd.read_csv('C:/Users/akhram/Desktop/Algo/ML/test.csv')
X_test = excel_data_test[['Pclass','SibSp','Parch']] 


excel_data_test['Survived'] = dt_declare.predict(X_test)
print(excel_data_test['Survived'])

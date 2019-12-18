import pandas as pd
import os
from sklearn import tree, model_selection
import io
import pydot


titanic_train = pd.DataFrame()


print(titanic_train.info())
print(titanic_train.columns)

X_train = titanic_train[ ['SibSp', 'Parch','Pclass'] ]
y_train = titanic_train['Survived']
dt_estimator = tree.DecisionTreeClassifier()
dt_estimator.fit(X_train, y_train)
print(dt_estimator.tree_)
model_selection.cross_val_score(dt_estimator, X_train, y_train, scoring="accuracy", cv=5).mean()

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(dt_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 

graph.write_pdf("tree.pdf")



import pandas as pd
from sklearn import ensemble, model_selection,tree
import pydot

titanic_train = pd.read_csv('train.csv')

X_train = titanic_train[ ['SibSp', 'Parch'] ]
y_train = titanic_train['Survived']
rf_estimator = ensemble.RandomForestClassifier()
rf_estimator.fit(X_train, y_train)
model_selection.cross_val_score(rf_estimator, X_train, y_train, scoring="accuracy", cv=5).mean()


#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(rf_estimator, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 

graph.write_pdf("tree_rand.pdf")


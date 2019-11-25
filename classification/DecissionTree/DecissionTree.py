import pandas as pd
from sklearn import tree, model_selection, preprocessing

titanic_train = pd.read_csv('E:/Algorithmica/Akhram/train.csv')
print(titanic_train.info())
print(titanic_train.columns)

age_imputer = preprocessing.Imputer()
age_imputer.fit(titanic_train[['Age']])
titanic_train['Age_imputed'] = age_imputer.transform(titanic_train[['Age']])

fare_imputer = preprocessing.Imputer()
fare_imputer.fit(titanic_train[['Fare']])

titanic_train.loc[titanic_train['Embarked'].isnull(), 'Embarked'] = 'S'

sex_encoder = preprocessing.LabelEncoder()
sex_encoder.fit(titanic_train['Sex'])
titanic_train['Sex_encoded'] = sex_encoder.transform(titanic_train['Sex'])

pclass_encoder = preprocessing.LabelEncoder()
pclass_encoder.fit(titanic_train['Pclass'])
titanic_train['Pclass_encoded'] = pclass_encoder.transform(titanic_train['Pclass'])

emb_encoder = preprocessing.LabelEncoder()
emb_encoder.fit(titanic_train['Embarked'])
titanic_train['Embarked_encoded'] = emb_encoder.transform(titanic_train['Embarked'])

#create title feature from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)


title_encoder = preprocessing.LabelEncoder()
title_encoder.fit(titanic_train['Title'])
titanic_train['Title_encoded'] = title_encoder.transform(titanic_train['Title'])

#create family size feature from sibsp, parch
titanic_train['FamilySize'] = titanic_train['SibSp'] +  titanic_train['Parch'] + 1

#create family group feature from family-size
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=5): 
        return 'Medium'
    else: 
        return 'Large'
    
titanic_train['FamilyGroup'] = titanic_train['FamilySize'].map(convert_familysize)

fg_encoder = preprocessing.LabelEncoder()
fg_encoder.fit(titanic_train['FamilyGroup'])
titanic_train['FamilyGroup_encoded'] =fg_encoder.transform(titanic_train['FamilyGroup'])

features = ['SibSp', 'Parch', 'Fare', 'Pclass_encoded', 'Sex_encoded', 'Age_imputed', 'Embarked_encoded', 'Title_encoded', 'FamilySize', 'FamilyGroup_encoded']
X = titanic_train[ features ]
y = titanic_train['Survived']

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)

#grid search based model building
dt_estimator = tree.DecisionTreeClassifier()
dt_grid = {'max_depth': [3,4,5,6,7,8,9], 'criterion':['gini', 'entropy'], 'min_samples_split':[3, 5, 10]}
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator, dt_grid, scoring='accuracy', cv=10)
dt_grid_estimator.fit(X_train, y_train)
print(dt_grid_estimator.best_params_)
print(dt_grid_estimator.best_score_)
print(dt_grid_estimator.best_estimator_)
print(dt_grid_estimator.score(X_train, y_train))

print(dt_grid_estimator.score(X_eval, y_eval))
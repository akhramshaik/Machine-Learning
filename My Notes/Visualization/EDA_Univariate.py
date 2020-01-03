import pandas as pd
import seaborn as sns

titanic_train = pd.read_csv('C:/Users/akhram/Desktop/AIML/Machine Learning/Problems/Titanic/train.csv')

#This is sort of plot using Pandas
pd.crosstab(index=titanic_train["Survived"], columns="count")

#The below are actual plots using sns
sns.countplot(x='Survived',data=titanic_train)

sns.boxplot(x='Fare',data=titanic_train)

#continuous features: visual EDA
titanic_train['Fare'].describe()
sns.boxplot(x='Fare',data=titanic_train)

sns.distplot(titanic_train['Fare'])
sns.distplot(titanic_train['Fare'], hist=False)
sns.distplot(titanic_train['Age'], hist=False)
sns.boxplot(x='Age',data=titanic_train)

sns.distplot(titanic_train['SibSp'], hist=False)
sns.boxplot(x='SibSp',data=titanic_train)



sns.boxplot(x='Survived',y='Fare',data=titanic_train)
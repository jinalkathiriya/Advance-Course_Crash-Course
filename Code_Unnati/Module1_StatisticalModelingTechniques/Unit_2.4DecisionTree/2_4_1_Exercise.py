# Lab - 1 Self Paced Exercise Solution
# Dataset:
# train.csv
# test.csv
# Problem Statement
# For the Titanic challenge we need to guess wheter the individuals from the test dataset had survived or not.

## Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

## Loading data
dataset_train=pd.read_csv("train.csv")
# print(dataset_train.head())
dataset_test=pd.read_csv("test.csv")
# print(dataset_test.head())

## checking Null values
# print(dataset_train.isnull().sum())
# print(dataset_test.isnull().sum())

# print(dataset_train.shape)
# print(dataset_test.shape)

## Data Pre-processing
df_train=dataset_train.fillna(method="ffill")#fill the null values
df_test=dataset_test.fillna(method="ffill")
# print(df_train.isnull().sum())
# print(df_test.isnull().sum())

# print(df_train.describe())
# print(df_test.describe())

## label Encoding
label_encoder=preprocessing.LabelEncoder()

df_train['Pclass_type']=label_encoder.fit_transform(df_train["Pclass"])
df_train['Age_type']=label_encoder.fit_transform(df_train["Age"])
df_train['SibSp_type']=label_encoder.fit_transform(df_train["SibSp"])
df_train['Parch_type']=label_encoder.fit_transform(df_train["Parch"])
df_train['Fare_type']=label_encoder.fit_transform(df_train["Fare"])
df_train['Embarked_type']=label_encoder.fit_transform(df_train["Embarked"])
df_train['Sex_type']=label_encoder.fit_transform(df_train["Sex"])
label_encoder=preprocessing.LabelEncoder()

df_test['Pclass_type']=label_encoder.fit_transform(df_test["Pclass"])
df_test['Age_type']=label_encoder.fit_transform(df_test["Age"])
df_test['SibSp_type']=label_encoder.fit_transform(df_test["SibSp"])
df_test['Parch_type']=label_encoder.fit_transform(df_test["Parch"])
df_test['Fare_type']=label_encoder.fit_transform(df_test["Fare"])
df_test['Embarked_type']=label_encoder.fit_transform(df_test["Embarked"])
df_test['Sex_type']=label_encoder.fit_transform(df_test["Sex"])

## Delete unnecessary Columns(Features)
del df_train['PassengerId']
del df_train['Pclass']
del df_train['Age']
del df_train['Name']
del df_train['SibSp']
del df_train['Cabin']
del df_train['Parch']
del df_train['Fare']
del df_train['Ticket']
del df_train['Sex']
del df_train['Embarked']

del df_test['PassengerId']
del df_test['Pclass']
del df_test['Age']
del df_test['Name']
del df_test['SibSp']
del df_test['Cabin']
del df_test['Parch']
del df_test['Fare']
del df_test['Ticket']
del df_test['Sex']
del df_test['Embarked']

# print(df_train.head())
# print(df_test.head())

## Split into training and test set
#split the data into x and y
x=df_train[['Pclass_type', 'Age_type', 'SibSp_type','Parch_type', 'Fare_type', 'Embarked_type', 'Sex_type']]

y=df_train[['Survived']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

## scaling of X (train and test data)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled = scaler.fit_transform(df_train)


x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled=scaler.transform(x_test)
# print("Mean value:",x_train_scaled.mean(axis=0))
# print("SD value:",x_train_scaled.std(axis=0))

## Model selection and implimentation
from sklearn import tree
clf= tree.DecisionTreeClassifier()
clf= clf.fit(x_train_scaled,y_train)

## Accuracy Check
print(clf.score(x_test_scaled,y_test)*100 )
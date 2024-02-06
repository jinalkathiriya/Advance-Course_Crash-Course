# Lab 1 : Classification Using Support Vector Machine
# Problem Statement
# An Iris Flower Has Three Different Species: Setosa, Versicolor, Virginica. Based on Petal length, Petal width, Sepal length and sepal width we have to predict different species of iris flower.
# Sklearn comes with inbuilt iris dataset. The dataset consists of 150 samples. Students can download inbuilt dataset and they have to perform classification using SVM.

## importing required package
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

## loading iris dataset
iris = load_iris()
# print(dir(iris))

## printing all four features
# print(iris.feature_names)

# print(iris.target_names)

## Creating dataframe using DataFrame constructor
df = pd.DataFrame(iris.data,columns=iris.feature_names)
# print(df.head())

## Adding target feature in existing data frame
df['target'] = iris.target
# print(df.head())

# print(df[df.target==1].head())
# print(df[df.target==2].head())
# print(df[df.target==0].head())

df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
# print(df.head())

df_0 = df[:50] ##setosa
df_1 = df[50:100] ## versicolor
df_2 = df[100:] ## virginica

## Plotting Sepal Length Vs Sepal Width
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.scatter(df_0['sepal length (cm)'], df_0['sepal width (cm)'],color="green",marker='+')
# plt.scatter(df_1['sepal length (cm)'], df_1['sepal width (cm)'],color="blue",marker='.')
# plt.show()

## Plotting Petal Length and Petal Width
# plt.xlabel('Petal Length')
# plt.ylabel('Petal Width')
# plt.scatter(df_0['petal length (cm)'], df_0['petal width (cm)'],color="green",marker='+')
# plt.scatter(df_1['petal length (cm)'], df_1['petal width (cm)'],color="blue",marker='.')
# plt.show()

X = df.drop(['target','flower_name'], axis='columns')
y = df.target

## importing train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(X_train.shape)
# print(X_test.shape)

## Creating an object of SVM class
from sklearn.svm import SVC
iris_svm_model = SVC(kernel="linear",random_state=20)
# print(iris_svm_model.fit(X_train,y_train))
print(iris_svm_model.score(X_test, y_test))
# print(iris_svm_model.predict([[4.5,3.1,1.5,0.5]]))



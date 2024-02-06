# Lab - 1 Self Paced Exercise Solution
# Dataset:
# Inbuild iris dataset

# Problem Statement
# The Iris dataset contains measurements of four features—sepal length, sepal width, petal length, and petal width—for three different species of iris flowers: Iris Setosa, Iris Versicolor, and Iris Virginica.
# The goal of this analysis is to use the K-NN algorithm to Create feature and target variables.Split data into training and test data. Generate a k-NN model using neighbors value.Train or fit the data into the model. Predict the future.

## Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

## Loading data
irisData = load_iris()

## Create feature and target arrays
X = irisData.data
y = irisData.target

## Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
  
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

## Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

## Compute training and test data accuracy
train_accuracy[i] = knn.score(X_train, y_train)
test_accuracy[i] = knn.score(X_test, y_test)

## Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
  
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


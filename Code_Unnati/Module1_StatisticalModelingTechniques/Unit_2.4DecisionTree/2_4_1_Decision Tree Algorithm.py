# Lab-1 Implementing Decision Tree Algorithm using Python

# Importing essential libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics

## Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)

## Exploring Data
# print(X.head())

y = pd.get_dummies(y)

## Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

## Generating Model
dt = DecisionTreeClassifier()

## Train the model using the training sets
print(dt.fit(X_train, y_train))

## Visualize Decision Tree
import matplotlib.pyplot as plt
from sklearn import tree
# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(dt, 
#                    feature_names=iris.feature_names,  
#                    class_names=iris.target_names,
#                    filled=True)

## Predict the response for test dataset
y_pred = dt.predict(X_test)

##Evaluating the Model
# numbers on the diagonal of the confusion matrix correspond to correct predictions

species = np.array(y_test).argmax(axis=1)
predictions = np.array(y_pred).argmax(axis=1)
print(confusion_matrix(species, predictions))
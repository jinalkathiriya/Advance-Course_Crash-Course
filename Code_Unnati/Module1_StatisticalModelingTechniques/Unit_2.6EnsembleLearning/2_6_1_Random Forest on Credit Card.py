# LAB 1 - Demonstrating Random Forest on Credit Card Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from pylab import rcParams
rcParams['figure.figsize'] = 8, 8
# Reading the dataset
data = pd.read_csv("creditcard.csv")
print(data.head())

# Checking the shape of our data
print(data.shape)

# Checking the distribution of two classes in the target variable
data.Class.value_counts()

# Creating the dataset with all independent variables
X = data.iloc[:,:-1]

# Creating the dataset with the dependent variable
Y = data.iloc[:,-1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0, stratify = Y)
print("The shape of train dataset :")
print(X_train.shape)

print("\n The shape of test dataset :")
print(X_test.shape)

print("Distribution of classes of dependent variable in train :")
print(Y_train.value_counts())

print("\n Distribution of classes of dependent variable in test :")
print(Y_test.value_counts())

# Hyperparameter tuning

classifier1 = RandomForestClassifier() # For GBM, use GradientBoostingClassifier()
grid_values = {'n_estimators':[10, 20], 'max_depth':[3, 5]}
classifier = GridSearchCV(classifier1, param_grid = grid_values, scoring = 'roc_auc', cv=5)

# Fit the object to train dataset
classifier.fit(X_train, Y_train)

train_preds =  classifier.predict(X_train)
test_preds  = classifier.predict(X_test)

# Obtain accuracy on train set
accuracy_score(Y_train,train_preds)

# Obtain accuracy on test set
accuracy_score(Y_test,test_preds)

# Calculate roc_auc score on train set
roc_auc_score(Y_train,train_preds)

# Calculate roc_auc score on test set
roc_auc_score(Y_test,test_preds)

# Obtain the confusion matrix on train set
confusion_matrix(Y_train,train_preds)

# Obtain the confusion matrix on test set
confusion_matrix(Y_test,test_preds)

features = X_train.columns
importances = classifier.best_estimator_.feature_importances_ ## if best_estimator not chosen so you will face error
indices = np.argsort(importances)

plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='red', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')

plt.show()




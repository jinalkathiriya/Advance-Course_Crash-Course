# Lab-1 Implementing K-NN Algorithm using Python

## Import Libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

## Loading Data
df = pd.read_csv('teleCust1000t.csv')
# print(df.head())

## Data Visualization & Analysis
# print(df['custcat'].value_counts())

# print(df.hist(column='income', bins=50))

## Feature set X
# print(df.columns)

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
# print(X[0:5])

## Labels
y = df['custcat'].values
# print(y[0:5])

## Normalize Data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# print(X[0:5])

## Test-Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  y_train.shape)
# print ('Test set:', X_test.shape,  y_test.shape)

## Classification : K nearest neighbor (KNN)
from sklearn.neighbors import KNeighborsClassifier

## Training
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# print(neigh)

## Predict
yhat = neigh.predict(X_test)
# print(yhat[0:5])

## Accuracy evaluation
from sklearn import metrics
# print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

## Calculate the accuracy of KNN for different Ks.
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

print(mean_acc)

plt.plot(range(1,Ks),mean_acc,'g')
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# Lab 1 : Logistic Regression on Customer Churn data
# Problem Statement
# Telecom Churn (loss of customers to competition) is a problem for telecom companies 
#         because it is expensive to acquire a new customer and companies want to retain their existing customers.

## Importing warning package to ignore the warnings¶
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

## Load required libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

## Load the dataset
churn_df=pd.read_csv("ChurnData.csv")
print(churn_df)

## Explore the Data
# print(churn_df.info())
# print('\n')
# print(churn_df.describe())

## Select the required columns
# churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
# churn_df['churn']=churn_df['churn'].astype('int')
# print(churn_df.shape)
# print(churn_df.head())

## DataFrame to Numpy
X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless']]
# X=X.drop(['tenure'],axis=1)
Y=churn_df[['churn']]
np_X=np.asanyarray(X)
np_Y=np.asanyarray(Y)

## Machine Learning Started .....
from sklearn.preprocessing import StandardScaler ### Preprocess the data
std_scl=StandardScaler()
std_scl.fit(np_X)
np_X_procs=std_scl.transform(np_X)

## Train / Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(np_X_procs, np_Y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  Y_train.shape)
# print ('Test set:', X_test.shape,  Y_test.shape)

## Model initialization & training
model=LogisticRegression(C=0.01,solver='liblinear',verbose=1)
# print(model.fit(X_train,Y_train))
# Y_pred=model.predict(X_test)

## Model Predictions
Y_pred=model.predict(X_test)
# Y_pred_prob=model.predict_proba(X_test)
# print(Y_pred)
# print('\n')
# print((Y_pred_prob))

## Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("Model achieved a classification accuracy of:",end='\t')
print(accuracy_score(Y_test,Y_pred))
dsp=ConfusionMatrixDisplay(confusion_matrix(Y_test,Y_pred),display_labels=["Yes","No"])
print('\n')
dsp.plot()
print("Model Confusion Matrix")
from sklearn.metrics import jaccard_score
print('\n')
print("Jaccard Similarity Score:", end='\t')
print(jaccard_score(Y_test,Y_pred))

## Generate classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))
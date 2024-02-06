#LAB 2 - Demonstrating XG Boost on Credit Card Dataset

##importing required package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

##Reading the dataset
data = pd.read_csv("creditcard.csv")
data.head()

#Checking the shape of our data
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

#import lightgbm and xgboost 
# import lightgbm as lgb 
import xgboost as xgb

## The data is stored in a DMatrix object 
#label is used to define our outcome variable
#The data is stored in a DMatrix object Data Matrix used in XGBoost. DMatrix is an internal data structure that is used by XGBoost, which is optimized for both memory efficiency and training speed. You can construct DMatrix from multiple different sources of data.
#
dtrain=xgb.DMatrix(X_train,label=Y_train)
dtest=xgb.DMatrix(X_test)

#setting parameters for xgboost
parameters={'max_depth':7, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}

#training our model 
num_round=50
from datetime import datetime 
start = datetime.now() 
xg=xgb.train(parameters,dtrain,num_round) 
stop = datetime.now()

#now predicting our model on train set 
train_class_preds_probs=xg.predict(dtrain) 
#now predicting our model on test set 
test_class_preds_probs =xg.predict(dtest)

#length of pred prob
len(train_class_preds_probs)

#evaluation from thresold value
train_class_preds = []
test_class_preds = []
for i in range(0,len(train_class_preds_probs)):
  if train_class_preds_probs[i] >= 0.5:
    train_class_preds.append(1)
  else:
    train_class_preds.append(0)

for i in range(0,len(test_class_preds_probs)):
  if test_class_preds_probs[i] >= 0.5:
    test_class_preds.append(1)
  else:
    test_class_preds.append(0)

#print the array of pred prob
test_class_preds_probs[:20]

#length of y train
len(Y_train)

#lenght of train class pred
len(train_class_preds)

# Get the accuracy scores
train_accuracy_xgb = accuracy_score(train_class_preds,Y_train)
test_accuracy_xgb = accuracy_score(test_class_preds,Y_test)

print("The accuracy on train data is ", train_accuracy_xgb)
print("The accuracy on test data is ", test_accuracy_xgb)

#model evaluation before gdcv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
test_accuracy_xgb = accuracy_score(test_class_preds,Y_test)
test_precision_xgb = precision_score(test_class_preds,Y_test)
test_recall_score_xgb = recall_score(test_class_preds,Y_test)
test_f1_score_xgb = f1_score(test_class_preds,Y_test)
test_roc_score_xgb = roc_auc_score(test_class_preds,Y_test)

print("The accuracy on test data is ", test_accuracy_xgb)
print("The precision on test data is ", test_precision_xgb)
print("The recall on test data is ", test_recall_score_xgb)
print("The f1 on test data is ", test_f1_score_xgb)
print("The roc_score on train data is ", test_roc_score_xgb)

#grid search cv for xgboost
from xgboost import XGBClassifier
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='accuracy',n_jobs=-1, cv=3, verbose = 2)
gsearch1.fit(X_train, Y_train)

#gdcv score after training model on the data set
gsearch1.best_score_

#optimal xgb
optimal_xgb = gsearch1.best_estimator_

# Get the predicted classes
train_class_preds = optimal_xgb.predict(X_train)
test_class_preds = optimal_xgb.predict(X_test)

# Get the accuracy scores
train_accuracy_xgb_tuned = accuracy_score(train_class_preds,Y_train)
test_accuracy_xgb_tuned = accuracy_score(test_class_preds,Y_test)

print("The accuracy on train data is ", train_accuracy_xgb_tuned)
print("The accuracy on test data is ", test_accuracy_xgb_tuned)

#model score for xgboost
test_accuracy_xgb_tuned = accuracy_score(test_class_preds,Y_test)
test_precision_xgb_tuned = precision_score(test_class_preds,Y_test)
test_recall_score_xgb_tuned = recall_score(test_class_preds,Y_test)
test_f1_score_xgb_tuned = f1_score(test_class_preds,Y_test)
test_roc_score_xgb_tuned = roc_auc_score(test_class_preds,Y_test)

print("The accuracy on test data is ", test_accuracy_xgb_tuned)
print("The precision on test data is ", test_precision_xgb_tuned)
print("The recall on test data is ", test_recall_score_xgb_tuned)
print("The f1 on test data is ", test_f1_score_xgb_tuned)
print("The roc_score on test data is ", test_roc_score_xgb_tuned)

#important features respect to Xgb
pd.DataFrame(optimal_xgb.feature_importances_,
                                
                                    columns=['importance_xgb']).sort_values('importance_xgb',
                                                                        ascending=False)[:10]

# Feature Importance
feature_importances_xgb = pd.DataFrame(optimal_xgb.feature_importances_,
                                    columns=['importance_xgb']).sort_values('importance_xgb',
                                                                        ascending=False)[:10]

#plot for the important feature selection by xgb
plt.subplots(figsize=(17,6))
plt.title("Feature importances")
plt.bar(feature_importances_xgb.index, feature_importances_xgb['importance_xgb'],
        color="b",  align="center")
plt.show()

#y pred after training the model by gdcv
y_preds_proba_xgb = optimal_xgb.predict_proba(X_test)[::,1]

#plot for the auc-roc for xgb
import sklearn.metrics as metrics
y_pred_proba = y_preds_proba_xgb
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

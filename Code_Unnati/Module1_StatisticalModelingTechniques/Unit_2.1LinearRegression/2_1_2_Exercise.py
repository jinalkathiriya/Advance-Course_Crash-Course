# LAB 2 - Demonstrating Ridge Regression on Emission Data
# Problem Statement
# ""For extended data of CO2 emission ("Excercise_Data_1.csv").
# Company needs a Regularized Ridge regression model to predict the emission generate from a newly produced car.

## Load Needful Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import metrics

## Load the dataset
df = pd.read_csv("Excercise_Data_1.csv")
# print(df)

## Select features that we want to use for regression.
# print(df.select_dtypes(exclude=['object']))

## Select integer and float datatype columns
df_num = df.select_dtypes(['int','float'])
# print(df_num)

## Lets Visualize some relation between variables
import matplotlib.pyplot as plt
# print(plt.scatter(df_num['CYLINDERS'],df_num['CO2EMISSIONS']))
# plt.show()

## Select columns of interest from whole data
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(9))

## Generate plot betwen engine size and emission
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

## Lets proceed with selected data
# print(cdf.info())

## Check feature names
# print(cdf.columns)

## Split data into Train Test set
df_train,df_test = train_test_split(cdf, train_size=0.7,test_size = 0.3, random_state=100)
y_train = df_train.CO2EMISSIONS
X_train = df_train.drop("CO2EMISSIONS",1)

y_test  = df_test.CO2EMISSIONS
X_test = df_test.drop("CO2EMISSIONS",1)

## Scale the data
features_select=['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']
scaler = StandardScaler()
X_train[features_select] = scaler.fit_transform(X_train[features_select])
X_test[features_select] = scaler.transform(X_test[features_select])

## Train model and select best alpha values
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train)

print(model_cv.best_params_)
print(model_cv.best_score_)

## Tabulate model performance on investigated alphas
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=100]
print(cv_results)

## Tune to best performing alpha and instantiate Ridge model
alpha = 20
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
print(ridge.coef_)

## lets predict the R-squared value
y_train_pred = ridge.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
print(mean_squared_error(y_train, y_train_pred))

y_test_pred = ridge.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
print(mean_squared_error(y_test, y_test_pred))

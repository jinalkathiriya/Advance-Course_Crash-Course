# Lab 1
# Problem Statement
# ""For extended data of CO2 emission ("Excercise_Data_1.csv").
# Company needs a multiple regression model to predict the emission generate from a newly produced car.


## Load Needful Python libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

## Load the dataset
df = pd.read_csv("Excercise_Data_1.csv")
# print(df)

## Select features that we want to use for regression
# print(df.select_dtypes(exclude=['object']))

## Select integer and float datatype columns
df_num = df.select_dtypes(['int','float'])
# print(df_num)

##Lets Visualize some relation between variables
import matplotlib.pyplot as plt
# print(plt.scatter(df_num['CYLINDERS'],df_num['CO2EMISSIONS']))

## Select columns of interest from whole data
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(9))

## Generate plot betwen engine size and emission
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

## Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

## Verify the pattern consistent in training part
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

## Multiple Regression Model
from sklearn import linear_model
regr = linear_model.LinearRegression()    #classifier
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

## Scikit-learn uses plain Ordinary Least Squares (OLS) method to solve this problem.
from sklearn.metrics import r2_score

y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - y) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , y) )
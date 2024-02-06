# Lab 2 Housing Price Prediction using XG Boost Algorithm
# Problem Statement
# California Housing dataset provides information on California's housing districts. This dataset can be fetched from internet using scikit-learn. This dataset have total 8 attributes and have total instances 20640.The target variable is the median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).
# Your task is to predict housing price using XG boost.

# Importing Required Packages
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

## Loading California housing dataset
#Loading the California housing dataset
data = fetch_california_housing(as_frame=True)
# print(data)

X, y = data.data, data.target
print(x)
print(y)

#Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating an XGBoost regressor
model = xgb.XGBRegressor()

#Training the model on the training data
model.fit(X_train, y_train)

#Making predictions on the test set
predictions = model.predict(X_test)
print(predictions)

# Calculate the mean squared error and R-squared score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


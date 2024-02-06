# LAB 2 - Demonstrating Lasso & Ridge Regression

## Importing warning package to ignore the warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

## Load python libraries and sklearn linear model library
# Importing the required library

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

## Importing and Understanding Data
h_data = pd.read_csv(r"train.csv")
print(h_data.head())

## Inspect the data
# print(h_data.describe(include='all'))

## checking the number of rows and columns
# print(h_data.shape)

# print(h_data.info())

## Checking the Null values
# print(h_data.isnull().sum())

## Checking if there are columns with one unique value
# print(h_data.nunique())

## Checking the value count
# print(h_data.PoolQC.value_counts())

# print(h_data.Alley.value_counts())

# print(h_data.Street.value_counts())

# print(h_data.Utilities.value_counts())

## Checking the percentage of Null values
# df_missing=pd.DataFrame((round(100*(h_data.isnull().sum()/len(h_data.index)), 2)), columns=['missing'])
# print(df_missing.sort_values(by=['missing'], ascending=False).head(20))

## Treating the NaN Values
# h_data['PoolQC'] = h_data['PoolQC'].fillna('NoPool')
# h_data['MiscFeature'] = h_data['MiscFeature'].fillna('None')
# h_data['Alley'] = h_data['Alley'].fillna('NoAlleyAccess')
# h_data['Fence'] = h_data['Fence'].fillna('No_Fence')
# h_data['FireplaceQu'] = h_data['FireplaceQu'].fillna('No_Fireplace')
# h_data['GarageYrBlt'] = h_data['GarageYrBlt'].fillna(0)
# h_data['MasVnrType'] = h_data['MasVnrType'].fillna('None')
# h_data['MasVnrArea'] = h_data['MasVnrArea'].fillna(0)
# h_data['MasVnrArea'] = h_data['MasVnrArea'].fillna(0)
# h_data['Electrical'] = h_data['Electrical'].fillna("Other")

## Dropping the LotFontgage columns as it have more Null values
# print(h_data.drop("LotFrontage",axis = 1, inplace=True))

## Imputing the Nan Values with 'No Basementh_data'
# for col in ('BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtQual','BsmtCond'):
#     h_data[col] = h_data[col].fillna('No_Basement')

## Imputing the NaN values with 'no garage'
# for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
#     h_data[col] = h_data[col].fillna('No_Garage')

## Now verify data enrichness
# print(h_data.info())

## Convert to needful datatypes
# h_data['GarageYrBlt'] = h_data['GarageYrBlt'].astype(int)

## Performing EDA
## Univariate and Bivariate Analysis
# print(plt.scatter(h_data.MasVnrArea,h_data.SalePrice))

# print(sns.distplot(h_data['SalePrice'],color='y'))

# print("Skewness: %f" % h_data['SalePrice'].skew())
# print("Kurtosis: %f" % h_data['SalePrice'].kurt())

## Checking Basement counts
# sns.countplot(x='BsmtCond', data= h_data)
# print(plt.title('Basement Condition'))

# sns.countplot(x='OverallCond', data= h_data).tick_params(axis='x', rotation = 90)
# print(plt.title('Overall Condition'))

# data = pd.concat([h_data['SalePrice'], h_data['GrLivArea']], axis=1)
# data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
# print(plt.title('Gr LivArea vs SalePrice'))

## Checking the outliers
# print(sns.boxplot(x='SalePrice', data=h_data))

# sns.boxplot(x='OverallQual', y='SalePrice', data=h_data)
# print(plt.title("Overall Quality vs SalePrice"))

## Check pairwise scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(h_data[cols], size = 2.5)
# plt.show()

## Checking the correlation matrix
# corrmat = h_data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);
# plt.title("Checking Correlation matrix ")

## Data Preperation
# print(plt.figure(figsize=(1(1,6))))
# print(sns.distplot(np.log(h_data["SalePrice"])))
# plt.show()

## Deriving Variables
# numeric_data = h_data.select_dtypes(include = ['float64','int64'])
# print(numeric_data.columns)

## Converting a Binary varible into numeric datatypes
## mapping it to 0 and 1
# h_data['Street'] = h_data['Street'].apply(lambda x: 1 if x == 'Pave' else 0 )

# h_data['CentralAir'] = h_data['CentralAir'].apply(lambda x : 1 if x == 'Y' else 0)
                                                  
# h_data['PavedDrive'] = h_data['PavedDrive'].apply(lambda x : 1 if x == 'Y' else 0)   

# cat_values = h_data.select_dtypes(include=['object'])
# print(cat_values.head())

## convert into dummies
# data_dummies = pd.get_dummies(cat_values, drop_first=True)
# print(data_dummies.head())

## Droping the 'Id' column
# df = h_data.drop(['Id'],axis=1)

## Droping the original categorical column
# df = df.drop(list(cat_values.columns), axis=1)

## Adding the dummy categorical column to original dataset
# df = pd.concat([df,data_dummies], axis=1)
# print(df.shape)

# print(df)

## Train Test Split
# df_train,df_test = train_test_split(df,train_size=0.7,test_size = 0.3, random_state=100)
# y_train = np.log(df_train.SalePrice)
# X_train = df_train.drop("SalePrice",1)

# y_test= np.log(df_test.SalePrice)
# X_test = df_test.drop("SalePrice",1)
# num_values=X_train.select_dtypes(include=['int64','float64']).columns
# print(num_values)

## Scaling the data
# scaler = StandardScaler()
# X_train[num_values] = scaler.fit_transform(X_train[num_values])
# X_test[num_values] = scaler.transform(X_test[num_values])

## Model Building
## Building a Regression model.
# reg = LinearRegression()
# reg.fit(X_train,y_train)

# len(X_train.columns)

## Calculating the RFE (Recursive Feature Elimination)
# rfe = RFE(reg,n_features_to_select=20)
# rfe = rfe.fit(X_train, y_train)

# col=X_train.columns[rfe.support_]
# print(col)

## Load statistical libraries
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Select top performing features
## Train the model
# X_train_new=X_train[col]
# X_train_new = sm.add_constant(X_train_new)

#create first model
# lr=sm.OLS(y_train,X_train_new)

#fit the model
# lr_model=lr.fit()

#Print the summary 
# lr_model.summary()

## Ridge Regression
## list of alphas to tune
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


## Optimal value of alpha
print(model_cv.best_params_)
print(model_cv.best_score_)

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=100]
print(cv_results) 

##plotting mean test and train scores with alpha
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')
plt.figure(figsize=(16,5))

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()

## final ridge model
alpha = 0.1
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
print(ridge.coef_)

## lets predict the R-squared value
 
y_train_pred = ridge.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))

## Prediction on test set
y_test_pred = ridge.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))

## Printing the RMSE value
mean_squared_error(y_test, y_test_pred)

## Lasso Regression
params = {'alpha': [0.00005, 0.0001, 0.001, 0.008, 0.01]}
lasso = Lasso()

# cross validation
lasso_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

lasso_cv.fit(X_train, y_train)

cv_results_l = pd.DataFrame(lasso_cv.cv_results_)

print(lasso_cv.best_params_)
print(lasso_cv.best_score_)

## final lasso model
alpha = 0.001

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 

## Predict the R-squared value for Train data
y_train_pred = lasso.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))

## Predict the R-squared value for test data
y_test_pred = lasso.predict(X_test)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))

mean_squared_error(y_test, y_test_pred)

print(lasso.coef_)

## plotting mean test and train scoes with alpha
 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results_l['param_alpha'], cv_results_l['mean_train_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper left')
plt.show()

## Finally Conncluded Model
model_cv.best_params_

ridge = Ridge(alpha = 0.1)
ridge.fit(X_train,y_train)

y_pred_train = ridge.predict(X_train)
print(r2_score(y_train,y_pred_train))

y_pred_test = ridge.predict(X_test)
print(r2_score(y_test,y_pred_test))

model_parameter = list(ridge.coef_)
model_parameter.insert(0,ridge.intercept_)
cols = df_train.columns
cols.insert(0,'constant')
ridge_coef = pd.DataFrame(list(zip(cols,model_parameter)))
ridge_coef.columns = ['Feaure','Coef']

## Top Performing features for Regression
ridge_coef.sort_values(by='Coef',ascending=False).head(10)

## Final Model Performance
lasso = Lasso(alpha=0.001)
lasso.fit(X_train,y_train)

y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print(r2_score(y_true=y_train,y_pred=y_train_pred))
print(r2_score(y_true=y_test,y_pred=y_test_pred))



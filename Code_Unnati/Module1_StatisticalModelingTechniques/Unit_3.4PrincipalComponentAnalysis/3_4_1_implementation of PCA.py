# Lab-1 implementation of PCA using Numpy and pandas

## Importing Required Package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Loading the inbuilt dataset
from sklearn.datasets import load_breast_cancer
 # instantiating
cancer = load_breast_cancer(as_frame=True)
# creating dataframe
df = cancer.frame

# print(df)

### checking shape
# print('Original Dataframe shape :',df.shape)

## Input features
X = df[cancer['feature_names']]
# print('Inputs Dataframe shape   :', X.shape)


## Step 1: Standardization

# Mean
X_mean = X.mean()

# Standard deviation
X_std = X.std() 

# Standardization
Z = (X - X_mean) / X_std
# print(Z)

## Step 2: Calculate Covariance Matrix

# covariance
c = Z.cov()

# Plot the covariance matrix
sns.heatmap(c)
# plt.show()

## Step 3: Calculate the eigenvalues and eigenvectors of the covariance matrix

eigenvalues, eigenvectors = np.linalg.eig(c)
# print('Eigen values:\n', eigenvalues)
# print('Eigen values Shape:', eigenvalues.shape)
# print('Eigen Vector Shape:', eigenvectors.shape)

## Step 4: Sort the eigenvalues in descending order and sort the corresponding eigenvectors accordingly

# Index the eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]

# Sort the eigenvalues in descending order
eigenvalues = eigenvalues[idx]

# sort the corresponding eigenvectors accordingly
eigenvectors = eigenvectors[:,idx]

## Step 5: Calculate the cumulative explained variance
explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
# print(explained_var)

## Step 6: Determine the number of principal components
n_components = np.argmax(explained_var >= 0.50) + 1
# print(n_components)

## Step 7: Project the data onto the selected principal components
# PCA component or unit matrix
u = eigenvectors[:,:n_components]
pca_component = pd.DataFrame(u, index = cancer['feature_names'],columns = ['PC1','PC2'])

# Matrix multiplication or dot Product
Z_pca = Z @ pca_component
Z_pca = pd.DataFrame(Z_pca.values,columns = ['PCA1','PCA2'])
print(Z_pca.head())



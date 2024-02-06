# Lab 1 - Self Paced Solution : Analyzing Iris Dataset
# Dataset:
# Inbuild iris dataset

# Problem Statement
# The Iris dataset contains measurements of four features—sepal length, sepal width, petal length, and petal width—for three different species of iris flowers: Iris Setosa, Iris Versicolor, and Iris Virginica.
# The goal of this analysis is to use the K-Means clustering algorithm to segment the iris flowers into clusters based on these four features. The assumption is that similar measurements among flowers might indicate similarities in species, though this is an unsupervised analysis and the algorithm will work without using species labels.

##Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

##Load the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# print(data.head())

## Data Preprocessing
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

## Choosing the Number of Clusters (K)
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(normalized_data)
    inertia.append(kmeans.inertia_)

# plt.plot(range(2, 11), inertia, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.show()

## Applying K-Means Clustering
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
cluster_assignments = kmeans.fit_predict(normalized_data)
data['Cluster'] = cluster_assignments

##Applying K-Means Clustering
plt.scatter(data['sepal length (cm)'], data['petal length (cm)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Clustered Data')
plt.show()



#Lab 1 - Classroom Exercise : Implementing K-means Algorithm

##I mport Libraries
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
os.environ["OMP_NUM_THREADS"] = '1'

from sklearn.cluster import KMeans
 
## Loading Data
df=pd.read_csv("Mall_Customers.csv")
# print(df)
# print(df.head())

##Data Exploration
### Check Null Values
# print(df.isnull().sum())

### Observation: There is no missing values.
### Visual and Statistical Understanding of data
# print(df.columns)

# plt.scatter(df['Age'],df['Spending Score (1-100)'])
# plt.xlabel("Age")
# plt.ylabel("Spending Score")
# plt.show()

### Observation: It seems to purpose two types of Customer
# plt.scatter(df["Age"],df["Annual Income (k$)"])
# plt.xlabel("Age")
# plt.ylabel("Annual Income (k$)")
# plt.show()

### Observation: No Group
# plt.scatter(df["Spending Score (1-100)"], df["Annual Income (k$)"])
# plt.xlabel("Spending Score (1-100)")
# plt.ylabel("Annual Income (k$)")
# plt.show()

### It seems to purpose five Groups

##Choose Relevant Columns
relevant_cols = ["Age", "Annual Income (k$)", 
                 "Spending Score (1-100)"]

customer_df = df[relevant_cols]
# print(customer_df)

## Data Transformation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(customer_df)
scaled_data = scaler.transform(customer_df)
# print(scaled_data)

## Determine the best number of cluster

def find_best_clusters(df, maximum_K):
    clusters_centers = []
    k_values = []
    for k in range(2, maximum_K):
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)

        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)

    return clusters_centers, k_values

clusters_centers, k_values = find_best_clusters(scaled_data, 12)

def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    # plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    # plt.xlabel("Number of Clusters (K)")
    # plt.ylabel("Cluster Inertia")
    # plt.title("Elbow Plot of KMeans")
    # plt.show()

generate_elbow_plot(clusters_centers, k_values)

## Create the final KMeans model
kmeans_model = KMeans(n_clusters = 5)
kmeans_model.fit(scaled_data)

## ### We can access the cluster to which each data point belongs by using the .labels_ attribute.
df["clusters"] = kmeans_model.labels_
# print(df)

## Visualize the clusters
print(plt.scatter(df["Spending Score (1-100)"], 
            df["Annual Income (k$)"], 
            c = df["clusters"]
            ))
plt.show()



# LAB 4 - Machine Learning on IoT Sensor Data
# Problem Statement
# The downloaded data in lab 3 is majorily in form of unsupervised data.
# Try to aggregate data from various sensors and apply Machine Learning Algorithms on them to identify the clusters and data homogeneity.

## Importing warning package to ignore the warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

## Import python Libraries viz, Pandas, Seaborn & Sci-kit learn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

## Configure number of threads for computation
import os
os.environ["OMP_NUM_THREADS"] = '1'

##Load Data downloaded in Lab3, ( Sensor data from Cloud)
df=pd.read_csv("mobile_data.csv")
# print(df)

# print(df.head())

## Explore data to check consistency
## Check Null Values

# print(df.isnull().sum())

## Observation: There is no missing values.
## Perform Visual and Statistical Understanding of data

# plt.scatter(df['X'],df['Y'])
# plt.xlabel("Acc_X")
# plt.ylabel("Acc_Y")
# plt.show()

# plt.scatter(df["Y"],df["Z"])
# plt.xlabel("Acc_Y")
# plt.ylabel("Acc_Z")
# plt.show()

## Choose Relevant Columns
## In this example, we will use the numerical ones: X, Y, and Z direction acceleration values

relevant_cols = ["X","Y","Z"]

final_df = df[relevant_cols]
# print(final_df)

## Data Transformation

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# print(scaler.fit(final_df))

## apply transformation to raw data
scaled_data = scaler.transform(final_df)
# print(scaled_data)

## Determine the best number of cluster
def find_best_clusters(df, maximum_K):
    clusters_centers = []
    k_values = []
    for k in range(1, maximum_K):
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
    return clusters_centers, k_values

clusters_centers, k_values = find_best_clusters(scaled_data, 12)

def generate_elbow_plot(clusters_centers, k_values):

    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()

generate_elbow_plot(clusters_centers, k_values)


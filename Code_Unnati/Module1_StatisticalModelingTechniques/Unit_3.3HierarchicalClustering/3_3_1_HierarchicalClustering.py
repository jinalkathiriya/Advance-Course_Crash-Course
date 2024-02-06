#Lab-1 Hierarchical Clustering

##Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

##Importing the dataset
df = pd.read_csv('Mall_Customers.csv')
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# plt.figure(1 ,figsize = (15 , 6))
# graph = 0 
# for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
#     graph += 1  
#     # ploting graph
#     plt.subplot(1 , 3 , graph)
#     plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
#     sns.histplot(dataset[x] , bins = 18,kde=True)
#     plt.title('Distplot of {}'.format(x))

# plt.show()

### Interested Feature
X = dataset.iloc[:, 3:5].values
print(X.shape)
print(x)

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

# Using the dendogram to find the optimal number of clusters
# In linkage method we can pass single, complete, average,centroid  or ward value.

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.axhline(y=200, color='r', linestyle='--')
plt.show()

### No of cluters is 5

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters (two dimensions only)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


#Lab-2 Breast Cancer Principal Component Analysis

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()

# print(type(cancer_data))

# print(cancer_data.keys())

no_features=len(cancer_data['feature_names'])
# print(no_features)

no_records=len(cancer_data['data'])
# print(no_records)

df = pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
# print(df)

# print(df.head())

## Scaling data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df)
# print(df_scaled)

## Principal Component Analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pc12 = pca.fit_transform(df_scaled)
# print(pc12.shape)

df_pca = pd.DataFrame(pc12,columns=['PC1','PC2'])
# print(df_pca.head())

# plt.figure(figsize=(10,8))
# plt.scatter(
#     x=df_pca['PC1'],
#     y=df_pca['PC2'],
#     c=cancer_data['target'],
#     cmap='Set1')
# plt.xlabel('First Principal Component - PC1')
# plt.ylabel('Second Principal Component - PC2')
# plt.show()

## Logistic Regression (Classification) PCA
X = df_pca
y = cancer_data['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.linear_model import LogisticRegression
logR = LogisticRegression()
logR.fit(X_train,y_train)
predictions = logR.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test,predictions))



## Importing Required Libraries
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

## Loading Sklearn wine dataset
wine_info = load_wine()
X = pd.DataFrame(wine_info.data, columns=wine_info.feature_names)
y=wine_info.target
# print(X.shape)
# print(y.shape)

# print(X.head())

# print(wine_info.target_names)

df = X.join(pd.Series(y, name='class'))
# print(df)

## Importing LDA class from Sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_model = LinearDiscriminantAnalysis()
X_lda = lda_model.fit_transform(X, y)
# print(X_lda.shape)

## Creating scatter plot
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap='rainbow',alpha=0.7,edgecolors='b')
plt.show()



# Lab-1

## Loading dataset
from sklearn.datasets import load_wine
wine = load_wine()
X = wine.data
y = wine.target
# print("Wine dataset size:", X.shape)

## Apply PCA to the Wine data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

## Making the scatterplot
# import matplotlib.pyplot as plt
# plt.figure(figsize=[7, 5])

# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=25, cmap='plasma')
# plt.title('PCA for wine data with 2 components')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.savefig("PCA.png")
# plt.show()

exp_var = sum(pca.explained_variance_ratio_ * 100)
# print('Variance explained:', exp_var)

## Apply LDA to the Wine data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

### Making the scatterplot
# import matplotlib.pyplot as plt
# plt.figure(figsize=[7, 5])

# plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, s=25, cmap='plasma')
# plt.title('LDA for wine data with 2 components')
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.savefig("LDA.png")
# plt.show()

exp_var = sum(lda.explained_variance_ratio_ * 100)
print('Variance explained:', exp_var)
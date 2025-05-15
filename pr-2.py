# Based on the previous code, implements the PCA algorithm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = df1.select_dtypes(include=['int64','float64'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=x_pca, columns=['PC1','PC2'])
print("Explained Variance Ration:", pca.explained_variance_ratio_)
plt.scatter(pca_df["PC1"], pca_df["PC2"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Result")
plt.grid(True)
plt.show()
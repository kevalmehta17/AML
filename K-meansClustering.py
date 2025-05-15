# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 2: Load the customer dataset
# Replace with your actual file path
df = pd.read_csv("C:/Users/keval/Downloads/customer_data.csv")

# Step 3: Explore and clean the data
print(df.head())
print(df.info())
df = df.dropna()  # Drop missing values

# Step 4: Select relevant features (e.g., Annual Income, Spending Score)
# You can change based on your dataset columns
x = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Step 5: Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Step 6: Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(x_scaled)

# Step 7: Add cluster labels to the original DataFrame
df['Cluster'] = y_kmeans

# Step 8: Evaluate cluster quality
sil_score = silhouette_score(x_scaled, y_kmeans)
print(f"\nSilhouette Score: {sil_score:.3f} (higher is better)")

# Step 9: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, label='Centroids')
plt.title("Customer Segments using K-Means")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend()
plt.grid(True)
plt.show()

# Step 10 (Optional): Print cluster-wise customer count
print("\nCustomers in each cluster:")
print(df['Cluster'].value_counts())

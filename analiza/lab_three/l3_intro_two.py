import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=1000, noise=0.05, random_state=0)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', marker='o', edgecolor='k', s=50, alpha=0.6)
plt.title("Scatter Plot of the Moon-Shaped Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
kmeans_centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

agglomerative = AgglomerativeClustering(n_clusters=2, linkage='single')
agglomerative_labels = agglomerative.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=agglomerative_labels, cmap='plasma', marker='o', edgecolor='k', s=50, alpha=0.6)
plt.title("Agglomerative Clustering (Single Linkage)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

dbscan_1 = DBSCAN(eps=0.05, min_samples=5)
dbscan_labels_1 = dbscan_1.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels_1, cmap='tab10', marker='o', edgecolor='k', s=50, alpha=0.6)
plt.title("DBSCAN Clustering (eps=0.05, min_samples=5)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

dbscan_2 = DBSCAN(eps=0.25, min_samples=5)
dbscan_labels_2 = dbscan_2.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels_2, cmap='tab10', marker='o', edgecolor='k', s=50, alpha=0.6)
plt.title("DBSCAN Clustering (eps=0.25, min_samples=5)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

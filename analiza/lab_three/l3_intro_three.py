import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

X, y_true = make_circles(n_samples=1000, noise=0.01, factor=0.5, random_state=0)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c='gray', marker='o', edgecolor='k', s=50, alpha=0.6)
plt.title("Scatter Plot of the Ring-Shaped Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
kmeans_centroids = kmeans.cluster_centers_

print("Centroidy k-średnich:\n", kmeans_centroids)

agglomerative_ward = AgglomerativeClustering(n_clusters=2, linkage='ward')
agglomerative_labels_ward = agglomerative_ward.fit_predict(X)

print("Etykiety klastrów dla metody Warda:\n", agglomerative_labels_ward)

agglomerative_single = AgglomerativeClustering(n_clusters=2, linkage='single')
agglomerative_labels_single = agglomerative_single.fit_predict(X)

print("Etykiety klastrów dla pojedynczego połączenia:\n", agglomerative_labels_single)

dbscan = DBSCAN(eps=0.1, min_samples=8)
dbscan_labels = dbscan.fit_predict(X)

print("Etykiety klastrów DBSCAN:\n", dbscan_labels)

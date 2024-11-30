import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=2000, centers=8, cluster_std=2.5, random_state=42)

sse = []

for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 21), sse, marker='o')
plt.title("Metoda Łokcia")
plt.xlabel("Liczba skupień k")
plt.ylabel("Suma kwadratów odległości (SSE)")
plt.show()

chosen_k_values = [3, 5, 8, 12]

for k in chosen_k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    print(f"Liczba skupień k = {k}, Średni współczynnik sylwetki: {silhouette_avg}")

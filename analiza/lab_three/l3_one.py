import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image = cv2.imread('palm_tree.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Kody RGB dla każdego piksela:")
print(image_rgb)

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("Oryginalne Zdjęcie")
plt.axis('off')
plt.show()

height, width, channels = image_rgb.shape
print(f"Rozmiary zdjęcia: {width} x {height}, Liczba kanałów: {channels}")

pixels = image_rgb.reshape(-1, 3)

kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Etykiety klastrów dla każdego piksela:")
print(labels.reshape(height, width))

print("Współrzędne centroidów:")
print(centroids)

centroids_rounded = np.round(centroids, 0).astype(int)
print("Zaokrąglone centroidy (kolory):")
print(centroids_rounded)

quantized_image = centroids[labels].reshape(height, width, 3).astype(int)

plt.figure(figsize=(8, 6))
plt.imshow(quantized_image)
plt.title("Obraz po Kwantyzacji")
plt.axis('off')
plt.show()

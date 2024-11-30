import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('digits.csv')
df_pixels = data.drop('number_label', axis=1)
df_labels = data['number_label']
if df_pixels.isnull().values.any():
    raise ValueError("The data contains missing values.")
first_image = df_pixels.iloc[0].values.astype(float).reshape(8, 8)
plt.figure(figsize=(4, 4))
plt.imshow(first_image, cmap='gray', interpolation='none')
plt.title(f"Digit: {df_labels.iloc[0]}")
plt.axis('off')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pixels)
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
print(pca_2.components_)

explained_variance_2 = pca_2.explained_variance_ratio_

print(explained_variance_2)

total_explained_variance_2 = np.sum(explained_variance_2)
print(total_explained_variance_2)

pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
components_95 = np.argmax(cumulative_variance >= 0.95) + 1
components_99 = np.argmax(cumulative_variance >= 0.99) + 1

print(components_95)
print(components_99)

plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance * 100, marker='o')
plt.axhline(95, color='r', linestyle='--')
plt.axhline(99, color='g', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Explained Variance by Number of Components')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=df_labels, cmap='tab10', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - 2 Components')
plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
centroids_2d = []
for i in np.unique(df_labels):
    indices = df_labels == i
    centroid = X_pca_2[indices].mean(axis=0)
    centroids_2d.append(centroid)
centroids_2d = np.array(centroids_2d)
distances_2d = pairwise_distances(centroids_2d)
most_distinguishable_2d = np.unravel_index(np.argmax(distances_2d, axis=None), distances_2d.shape)
least_distinguishable_2d = np.unravel_index(
    np.argmin(distances_2d + np.eye(distances_2d.shape[0]) * distances_2d.max()), distances_2d.shape)

print(most_distinguishable_2d)
print(least_distinguishable_2d)

knn_2d = KNeighborsClassifier(n_neighbors=5)
accuracy_2d = cross_val_score(knn_2d, X_pca_2, df_labels, cv=5, scoring='accuracy')

print(accuracy_2d.mean())

pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)
explained_variance_3 = pca_3.explained_variance_ratio_

print(explained_variance_3)

total_explained_variance_3 = np.sum(explained_variance_3)

print(total_explained_variance_3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=df_labels, cmap='tab10', alpha=0.7)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA - 3 Components')
plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
centroids_3d = []
for i in np.unique(df_labels):
    indices = df_labels == i
    centroid = X_pca_3[indices].mean(axis=0)
    centroids_3d.append(centroid)
centroids_3d = np.array(centroids_3d)
distances_3d = pairwise_distances(centroids_3d)
most_distinguishable_3d = np.unravel_index(np.argmax(distances_3d, axis=None), distances_3d.shape)
least_distinguishable_3d = np.unravel_index(
    np.argmin(distances_3d + np.eye(distances_3d.shape[0]) * distances_3d.max()), distances_3d.shape)

print(most_distinguishable_3d)
print(least_distinguishable_3d)

knn_3d = KNeighborsClassifier(n_neighbors=5)
accuracy_3d = cross_val_score(knn_3d, X_pca_3, df_labels, cv=5, scoring='accuracy')

print(accuracy_3d.mean())

pca_2_no_scaling = PCA(n_components=2)
X_pca_2_no_scaling = pca_2_no_scaling.fit_transform(df_pixels)

print(pca_2_no_scaling.components_)

explained_variance_2_no_scaling = pca_2_no_scaling.explained_variance_ratio_

print(explained_variance_2_no_scaling)

total_explained_variance_2_no_scaling = np.sum(explained_variance_2_no_scaling)

print(total_explained_variance_2_no_scaling)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca_2_no_scaling[:, 0], X_pca_2_no_scaling[:, 1], c=df_labels, cmap='tab10', alpha=0.7)
plt.xlabel('Principal Component 1 (No Scaling)')
plt.ylabel('Principal Component 2 (No Scaling)')
plt.title('PCA without Standardization - 2 Components')
plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
centroids_2d_ns = []
for i in np.unique(df_labels):
    indices = df_labels == i
    centroid = X_pca_2_no_scaling[indices].mean(axis=0)
    centroids_2d_ns.append(centroid)
centroids_2d_ns = np.array(centroids_2d_ns)
distances_2d_ns = pairwise_distances(centroids_2d_ns)
most_distinguishable_2d_ns = np.unravel_index(np.argmax(distances_2d_ns, axis=None), distances_2d_ns.shape)
least_distinguishable_2d_ns = np.unravel_index(
    np.argmin(distances_2d_ns + np.eye(distances_2d_ns.shape[0]) * distances_2d_ns.max()), distances_2d_ns.shape)

print(most_distinguishable_2d_ns)
print(least_distinguishable_2d_ns)

knn_2d_ns = KNeighborsClassifier(n_neighbors=5)
accuracy_2d_ns = cross_val_score(knn_2d_ns, X_pca_2_no_scaling, df_labels, cv=5, scoring='accuracy')
print(accuracy_2d_ns.mean())

pca_3_no_scaling = PCA(n_components=3)
X_pca_3_no_scaling = pca_3_no_scaling.fit_transform(df_pixels)
explained_variance_3_no_scaling = pca_3_no_scaling.explained_variance_ratio_
print(explained_variance_3_no_scaling)

total_explained_variance_3_no_scaling = np.sum(explained_variance_3_no_scaling)
print(total_explained_variance_3_no_scaling)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3_no_scaling[:, 0], X_pca_3_no_scaling[:, 1], X_pca_3_no_scaling[:, 2], c=df_labels,
                     cmap='tab10', alpha=0.7)
ax.set_xlabel('Principal Component 1 (No Scaling)')
ax.set_ylabel('Principal Component 2 (No Scaling)')
ax.set_zlabel('Principal Component 3 (No Scaling)')
ax.set_title('PCA without Standardization - 3 Components')
plt.legend(*scatter.legend_elements(), title="Digits", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

centroids_3d_ns = []
for i in np.unique(df_labels):
    indices = df_labels == i
    centroid = X_pca_3_no_scaling[indices].mean(axis=0)
    centroids_3d_ns.append(centroid)
centroids_3d_ns = np.array(centroids_3d_ns)
distances_3d_ns = pairwise_distances(centroids_3d_ns)
most_distinguishable_3d_ns = np.unravel_index(np.argmax(distances_3d_ns, axis=None), distances_3d_ns.shape)
least_distinguishable_3d_ns = np.unravel_index(
    np.argmin(distances_3d_ns + np.eye(distances_3d_ns.shape[0]) * distances_3d_ns.max()), distances_3d_ns.shape)

print(most_distinguishable_3d_ns)
print(least_distinguishable_3d_ns)

knn_3d_ns = KNeighborsClassifier(n_neighbors=5)
accuracy_3d_ns = cross_val_score(knn_3d_ns, X_pca_3_no_scaling, df_labels, cv=5, scoring='accuracy')

print(accuracy_3d_ns.mean())

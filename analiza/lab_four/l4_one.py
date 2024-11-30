import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
            'concavity', 'concave points', 'symmetry', 'fractal_dimension']
stats = ['mean', 'se', 'worst']

columns = ['ID', 'Diagnosis'] + [f'{feature}_{stat}' for feature in features for stat in stats]

data = pd.read_csv('wdbc.csv', header=None, names=columns)

data['Target'] = data['Diagnosis'].map({'M': 1, 'B': 0})
data = data.drop(columns=['ID', 'Diagnosis'])

if data.isnull().values.any():
    raise ValueError("Dane zawierają brakujące wartości. Usuń lub uzupełnij je przed kontynuacją.")

features = data.drop(columns=['Target'])
target = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

target_names = {0: 'Benign', 1: 'Malignant'}
plt.figure(figsize=(8, 6))
for target_value in np.unique(target):
    plt.scatter(X_pca[target == target_value, 0],
                X_pca[target == target_value, 1],
                label=target_names[target_value],
                alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization with Target Classes')
plt.legend()
plt.grid(True)
plt.show()

print("PCA Components (Eigenvectors):\n", pca.components_)

plt.figure(figsize=(14, 6))
sns.heatmap(pca.components_,
            cmap='viridis',
            annot=True,
            fmt='.2f',
            xticklabels=features.columns,
            yticklabels=['PC1', 'PC2'])
plt.xticks(rotation=90)
plt.title('Heatmap of PCA Components')
plt.tight_layout()
plt.show()

explained_variance = pca.explained_variance_ratio_
print("Explained Variance (PC1, PC2):", explained_variance)
print("Total Explained Variance (PC1 + PC2):", np.sum(explained_variance))

pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
components_95 = np.argmax(cumulative_variance >= 0.95) + 1
components_99 = np.argmax(cumulative_variance >= 0.99) + 1
print(f"Number of components for 95% variance: {components_95}")
print(f"Number of components for 99% variance: {components_99}")

plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance, marker='o')
plt.axhline(0.95, color='r', linestyle='--', label='95% Variance')
plt.axhline(0.99, color='g', linestyle='--', label='99% Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Number of Components')
plt.legend()
plt.grid(True)
plt.show()

pca_no_scaling = PCA(n_components=2)
X_pca_no_scaling = pca_no_scaling.fit_transform(features)

plt.figure(figsize=(8, 6))
for target_value in np.unique(target):
    plt.scatter(X_pca_no_scaling[target == target_value, 0],
                X_pca_no_scaling[target == target_value, 1],
                label=target_names[target_value],
                alpha=0.7)
plt.xlabel('Principal Component 1 (No Scaling)')
plt.ylabel('Principal Component 2 (No Scaling)')
plt.title('PCA without Standardization')
plt.legend()
plt.grid(True)
plt.show()

print("PCA Components without Scaling (Eigenvectors):\n", pca_no_scaling.components_)
explained_variance_no_scaling = pca_no_scaling.explained_variance_ratio_
print("Explained Variance without Scaling (PC1, PC2):", explained_variance_no_scaling)
print("Total Explained Variance without Scaling (PC1 + PC2):", np.sum(explained_variance_no_scaling))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('wdbc.csv', header=None)  # specify header if needed
data.columns = ['ID', 'Gender', *[f'Feature_{i}' for i in range(1, data.shape[1] - 2)], 'Target']

data = data.drop(columns=['ID', 'Gender'], errors='ignore')
if data.isnull().values.any():
    print("Data contains missing values. Please handle them before proceeding.")
else:
    print("Data is complete.")

features = data.drop(columns=['Target'], errors='ignore')
scaler = StandardScaler()
X = scaler.fit_transform(features)

cov_matrix = np.cov(X.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

components_to_keep = 2

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

principal_components = sorted_eigenvectors[:, :components_to_keep]

X_transformed = X.dot(principal_components)

plt.figure(figsize=(8, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data in 2D Principal Component Space')
plt.grid(True)
plt.show()

if 'Target' in data.columns:
    targets = data['Target']
    plt.figure(figsize=(8, 6))
    for target in np.unique(targets):
        plt.scatter(X_transformed[targets == target, 0], X_transformed[targets == target, 1], label=f'Target {target}',
                    alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('Principal Component Analysis (PCA) with Target Classes')
    plt.grid(True)
    plt.show()

explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1

print(f"Number of components for 95% variance: {components_95}")
print(f"Number of components for 99% variance: {components_99}")

plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance_ratio, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.title('Explained Variance by Principal Components')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Zadanie 2: Model SVM z większą liczbą punktów
X_full, y_full = make_blobs(n_samples=200, centers=2, cluster_std=0.60, random_state=0)

# Trenowanie i wizualizacja modelu dla 60 punktów
X_60, y_60 = X_full[:60], y_full[:60]
model_svc_60 = SVC(kernel='linear', C=1e10)
model_svc_60.fit(X_60, y_60)

plt.scatter(X_60[:, 0], X_60[:, 1], c=y_60, s=50, cmap='viridis')

# Prosta decyzyjna dla modelu z 60 punktami
w_60 = model_svc_60.coef_[0]
b_60 = model_svc_60.intercept_[0]
x_vals = np.linspace(-2, 4, 100)
y_vals_60 = -(w_60[0] / w_60[1]) * x_vals - b_60 / w_60[1]
plt.plot(x_vals, y_vals_60, 'k-', label='Decyzja (60 punktów)')

# Marginesy dla modelu z 60 punktami
margin_60 = 1 / np.sqrt(np.sum(w_60 ** 2))
plt.plot(x_vals, y_vals_60 + margin_60, 'k--')
plt.plot(x_vals, y_vals_60 - margin_60, 'k--')

# Wektory nośne dla modelu z 60 punktami
plt.scatter(model_svc_60.support_vectors_[:, 0], model_svc_60.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Wektory nośne (60 punktów)')

# Ustawienia wykresu
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Model SVM wytrenowany na 60 punktach')
plt.legend()
plt.show()

# Trenowanie i wizualizacja modelu dla 120 punktów
X_120, y_120 = X_full[:120], y_full[:120]
model_svc_120 = SVC(kernel='linear', C=1e10)
model_svc_120.fit(X_120, y_120)

plt.scatter(X_120[:, 0], X_120[:, 1], c=y_120, s=50, cmap='viridis')

# Prosta decyzyjna dla modelu z 120 punktami
w_120 = model_svc_120.coef_[0]
b_120 = model_svc_120.intercept_[0]
x_vals = np.linspace(-2, 4, 100)
y_vals_120 = -(w_120[0] / w_120[1]) * x_vals - b_120 / w_120[1]
plt.plot(x_vals, y_vals_120, 'k-', label='Decyzja (120 punktów)')

# Marginesy dla modelu z 120 punktami
margin_120 = 1 / np.sqrt(np.sum(w_120 ** 2))
plt.plot(x_vals, y_vals_120 + margin_120, 'k--')
plt.plot(x_vals, y_vals_120 - margin_120, 'k--')

# Wektory nośne dla modelu z 120 punktami
plt.scatter(model_svc_120.support_vectors_[:, 0], model_svc_120.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Wektory nośne (120 punktów)')

# Ustawienia wykresu
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Model SVM wytrenowany na 120 punktach')
plt.legend()
plt.show()
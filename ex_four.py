import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import numpy as np

# 4.1. Generowanie danych - 100 punktów, 2 klastry, odchylenie standardowe 1.2
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=0)

# 4.1. Wykres punktowy X i y
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Wykres punktowy dla danych z make_blobs')
plt.show()

# 4.2. Zmiana rozproszenia na 0.8 i budowa modelu SVM dla dwóch wartości C
X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.8, random_state=0)

# Model SVM z jądrem liniowym, C = 0.1
clf_svm_01 = SVC(kernel='linear', C=0.1)
clf_svm_01.fit(X, y)

# Model SVM z jądrem liniowym, C = 10
clf_svm_10 = SVC(kernel='linear', C=10)
clf_svm_10.fit(X, y)

# Rysowanie granicy decyzyjnej dla modelu z C = 0.1
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Tworzenie siatki punktów do oceny modelu
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_svm_01.decision_function(xy).reshape(XX.shape)

# Rysowanie granicy decyzyjnej i marginesów dla C = 0.1
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Rysowanie wektorów nośnych
ax.scatter(clf_svm_01.support_vectors_[:, 0], clf_svm_01.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Wektory nośne (C=0.1)')

# Ustawienia wykresu
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Granica decyzyjna dla SVM z C=0.1')
plt.legend()
plt.show()

# Rysowanie granicy decyzyjnej dla modelu z C = 10
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Tworzenie siatki punktów do oceny modelu
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_svm_10.decision_function(xy).reshape(XX.shape)

# Rysowanie granicy decyzyjnej i marginesów dla C = 10
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Rysowanie wektorów nośnych
ax.scatter(clf_svm_10.support_vectors_[:, 0], clf_svm_10.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Wektory nośne (C=10)')

# Ustawienia wykresu
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Granica decyzyjna dla SVM z C=10')
plt.legend()
plt.show()
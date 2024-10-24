import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 3.1. Generowanie danych w kształcie okręgów
X, y = make_circles(n_samples=100, factor=0.1, noise=0.1, random_state=0)

# 3.2. Tworzenie i trenowanie klasyfikatora SVM z jądrem liniowym
clf_svm = SVC(kernel='linear')
clf_svm.fit(X, y)

# 3.3. Rysowanie granicy decyzyjnej dla wytrenowanego modelu SVM
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')

# Rysowanie granicy decyzyjnej
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Tworzenie siatki punktów do oceny modelu
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_svm.decision_function(xy).reshape(XX.shape)

# Rysowanie granicy decyzyjnej i marginesów
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Rysowanie wektorów nośnych
ax.scatter(clf_svm.support_vectors_[:, 0], clf_svm.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Wektory nośne')

# Ustawienia wykresu
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Granica decyzyjna dla liniowego SVM')
plt.legend()
plt.show()

# 3.4. Rzutowanie danych na przestrzeń 3D
r = np.exp(-(X ** 2).sum(1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], r, c=y, s=50, cmap='viridis')

# Ustawienia wykresu 3D
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('R (funkcja odległości)')
plt.title('Rzutowanie danych na przestrzeń 3D')
plt.show()

# 3.5. Tworzenie i trenowanie klasyfikatora SVM z jądrem RBF
clf_svm_rbf = SVC(kernel='rbf')
clf_svm_rbf.fit(X, y)

# Rysowanie granicy decyzyjnej dla modelu z jądrem RBF
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')

# Rysowanie granicy decyzyjnej
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Tworzenie siatki punktów do oceny modelu
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_svm_rbf.decision_function(xy).reshape(XX.shape)

# Rysowanie granicy decyzyjnej i marginesów
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Rysowanie wektorów nośnych
ax.scatter(clf_svm_rbf.support_vectors_[:, 0], clf_svm_rbf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Wektory nośne (RBF)')

# Ustawienia wykresu
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Granica decyzyjna dla SVM z jądrem RBF')
plt.legend()
plt.show()
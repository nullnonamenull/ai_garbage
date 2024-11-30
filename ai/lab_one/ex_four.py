import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Wykres punktowy dla danych z make_blobs')
plt.show()

X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.8, random_state=0)

clf_svm_01 = SVC(kernel='linear', C=0.1)
clf_svm_01.fit(X, y)

clf_svm_10 = SVC(kernel='linear', C=10)
clf_svm_10.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_svm_01.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

ax.scatter(clf_svm_01.support_vectors_[:, 0], clf_svm_01.support_vectors_[:, 1], s=100, facecolors='none',
           edgecolors='k', label='Wektory nośne (C=0.1)')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Granica decyzyjna dla SVM z C=0.1')
plt.legend()
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf_svm_10.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

ax.scatter(clf_svm_10.support_vectors_[:, 0], clf_svm_10.support_vectors_[:, 1], s=100, facecolors='none',
           edgecolors='k', label='Wektory nośne (C=10)')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Granica decyzyjna dla SVM z C=10')
plt.legend()
plt.show()

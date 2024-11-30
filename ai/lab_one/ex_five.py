import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
import numpy as np

# 5.1. Generowanie danych sierpowatych - 100 punktów, szum 0.15
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# 5.1. Wykres punktowy X i y
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Wykres punktowy dla danych z make_moons')
plt.show()

# 5.2. Budowa i trenowanie klasyfikatorów SVM

# 5.2.1. Model liniowy dla C = 10
clf_linear = SVC(kernel='linear', C=10)
clf_linear.fit(X, y)

# 5.2.2. Modele z jądrem wielomianowym
# Stopień wielomianu 3, coef0 = 1, C = 5
clf_poly3 = SVC(kernel='poly', degree=3, coef0=1, C=5)
clf_poly3.fit(X, y)

# Stopień wielomianu 10, coef0 = 100, C = 5
clf_poly10 = SVC(kernel='poly', degree=10, coef0=100, C=5)
clf_poly10.fit(X, y)

# 5.2.3. Modele z jądrem RBF
# gamma = 0.1, C = 0.001
clf_rbf_1 = SVC(kernel='rbf', gamma=0.1, C=0.001)
clf_rbf_1.fit(X, y)

# gamma = 0.1, C = 1000
clf_rbf_2 = SVC(kernel='rbf', gamma=0.1, C=1000)
clf_rbf_2.fit(X, y)

# gamma = 5, C = 0.001
clf_rbf_3 = SVC(kernel='rbf', gamma=5, C=0.001)
clf_rbf_3.fit(X, y)

# gamma = 5, C = 1000
clf_rbf_4 = SVC(kernel='rbf', gamma=5, C=1000)
clf_rbf_4.fit(X, y)


# Funkcja do rysowania granicy decyzyjnej
def plot_decision_boundary(clf, X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Tworzenie siatki punktów do oceny modelu
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Rysowanie granicy decyzyjnej i marginesów
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # Rysowanie wektorów nośnych
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k',
               label='Wektory nośne')

    # Ustawienia wykresu
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()


# Rysowanie granic decyzyjnych dla wszystkich modeli
plot_decision_boundary(clf_linear, X, y, 'Granica decyzyjna dla SVM z jądrem liniowym, C=10')
plot_decision_boundary(clf_poly3, X, y, 'Granica decyzyjna dla SVM z jądrem wielomianowym, stopień=3, coef0=1, C=5')
plot_decision_boundary(clf_poly10, X, y, 'Granica decyzyjna dla SVM z jądrem wielomianowym, stopień=10, coef0=100, C=5')
plot_decision_boundary(clf_rbf_1, X, y, 'Granica decyzyjna dla SVM z jądrem RBF, gamma=0.1, C=0.001')
plot_decision_boundary(clf_rbf_2, X, y, 'Granica decyzyjna dla SVM z jądrem RBF, gamma=0.1, C=1000')
plot_decision_boundary(clf_rbf_3, X, y, 'Granica decyzyjna dla SVM z jądrem RBF, gamma=5, C=0.001')
plot_decision_boundary(clf_rbf_4, X, y, 'Granica decyzyjna dla SVM z jądrem RBF, gamma=5, C=1000')
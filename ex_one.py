import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# 1.1. Generowanie danych
X, y = make_blobs(n_samples=50, centers=2, cluster_std=0.60, random_state=0)

# 1.2. Tworzenie wykresu punktowego
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')


# 1.3. Generowanie trzech prostych
def plot_line(w0, w1, label):
    x_vals = np.linspace(-2, 4, 100)
    y_vals = w0 + w1 * x_vals
    plt.plot(x_vals, y_vals, label=label)


plot_line(0.65, 1, 'w0=0.65, w1=1')
plot_line(1.6, 0.5, 'w0=1.6, w1=0.5')
plot_line(2.9, -0.2, 'w0=2.9, w1=-0.2')

# 1.4. Rysowanie marginesów wokół każdej prostej
for w0, w1 in [(0.65, 1), (1.6, 0.5), (2.9, -0.2)]:
    x_vals = np.linspace(-2, 4, 100)
    y_vals = w0 + w1 * x_vals
    distances = np.abs((w1 * X[:, 0] - X[:, 1] + w0) / np.sqrt(w1 ** 2 + 1))
    min_distance = np.min(distances)
    plt.fill_between(x_vals, y_vals - min_distance, y_vals + min_distance, color='gray', alpha=0.2)

# 1.5. Tworzenie i trenowanie klasyfikatora SVC
model_svc = SVC(kernel='linear', C=1e10)
model_svc.fit(X, y)

# 1.6. Wizualizacja działania klasyfikatora SVC
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')

# Rysowanie prostej decyzyjnej
w = model_svc.coef_[0]
b = model_svc.intercept_[0]
x_vals = np.linspace(-2, 4, 100)
y_vals = -(w[0] / w[1]) * x_vals - b / w[1]
plt.plot(x_vals, y_vals, 'k-')

# Rysowanie marginesów liniami przerywanymi
margin = 1 / np.sqrt(np.sum(w ** 2))
y_vals_margin1 = y_vals + margin
y_vals_margin2 = y_vals - margin
plt.plot(x_vals, y_vals_margin1, 'k--')
plt.plot(x_vals, y_vals_margin2, 'k--')

# Rysowanie wektorów nośnych
plt.scatter(model_svc.support_vectors_[:, 0], model_svc.support_vectors_[:, 1], s=100, facecolors='none',
            edgecolors='k', label='Wektory nośne')

# Ustawienia wykresu
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Wykres punktowy dla danych z klasyfikatorem SVC i marginesami')
plt.legend()

plt.show()

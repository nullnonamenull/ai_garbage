import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1.1 Generowanie zbioru danych
X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)

# 1.2 Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Zadanie 2
# 2.1. Zbudowanie i wytrenowanie 2 klasyfikatorów na danych treningowych i wyrysowanie granic decyzyjnych

# 2.1.1 Model drzewa decyzyjnego
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# 2.1.2 Model boostrapowej agregacji
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(X_train, y_train)


# Funkcja do wyrysowania granic decyzyjnych
def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.show()


# Wyrysowanie granic decyzyjnych dla tree_clf
title = "Decision Tree Classifier"
plot_decision_boundary(tree_clf, X, y, title)

# Wyrysowanie granic decyzyjnych dla bag_clf
title = "Bagging Classifier (Bootstrap Aggregation)"
plot_decision_boundary(bag_clf, X, y, title)

# 2.2. Zbudowanie i wytrenowanie 2 klasyfikatorów wzmocnienia adaptacyjnego na danych treningowych i wyrysowanie granic decyzyjnych

# 2.2.1 Model ada_clf_01
ada_clf_01 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=500,
    learning_rate=0.1, random_state=42)
ada_clf_01.fit(X_train, y_train)

# Wyrysowanie granic decyzyjnych dla ada_clf_01
title = "AdaBoost Classifier (learning rate=0.1)"
plot_decision_boundary(ada_clf_01, X, y, title)

# 2.2.2 Model ada_clf_1
ada_clf_1 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=500,
    learning_rate=1, random_state=42)
ada_clf_1.fit(X_train, y_train)

# Wyrysowanie granic decyzyjnych dla ada_clf_1
title = "AdaBoost Classifier (learning rate=1)"
plot_decision_boundary(ada_clf_1, X, y, title)

import matplotlib.pyplot as plt
import numpy as np
from keras.src.datasets import mnist
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 3.1. Wczytanie zbioru danych
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 3.2. Sprawdzenie przykładowej cyfry
plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.show()

# 3.3. Wyświetlenie kilku obrazów ze zbioru danych MNIST
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].imshow(train_images[i], cmap='gray')
    axes[i].set_title(f"Label: {train_labels[i]}")
    axes[i].axis('off')
plt.show()

# 3.4. Preprocessing obrazów
# 4.1. Przekształcenie macierzy 28x28 na wektor 784 cech
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# 4.2. Normalizacja odcienia pikseli do zakresu [0, 1]
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 3.5. Uczenie na małym zbiorze
small_train_images, _, small_train_labels, _ = train_test_split(train_images, train_labels, train_size=500,
                                                                stratify=train_labels, random_state=42)
small_test_images, _, small_test_labels, _ = train_test_split(test_images, test_labels, train_size=50,
                                                              stratify=test_labels, random_state=42)

# 3.5.1. Jeden klasyfikator, np. las losowy składający się z 1000 drzew
small_rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)

# 3.5.2. Uczenie na zbiorze nie większym niż 500 obrazów
small_rf_classifier.fit(small_train_images, small_train_labels)

# 3.5.3. Testowanie modelu na zbiorze nie większym niż 50 obrazów
small_predictions = small_rf_classifier.predict(small_test_images)

# 3.5.4. Predykcja na małym zbiorze
print("Classification Report (Small Random Forest):")
print(classification_report(small_test_labels, small_predictions))

# 3.5.5. Ewaluacja modelu za pomocą macierzy błędów
print("Confusion Matrix (Small Random Forest):")
print(confusion_matrix(small_test_labels, small_predictions))

# 3.5.6. Sprawdzenie, ile obrazów zostało nieprawidłowo sklasyfikowanych
misclassified_indices = np.where(small_test_labels != small_predictions)[0]
print(f"Number of misclassified images: {len(misclassified_indices)}")

# 3.5.7. Wyświetlenie kilku nieprawidłowo sklasyfikowanych obrazów
fig, axes = plt.subplots(1, min(5, len(misclassified_indices)), figsize=(15, 3))
for i, idx in enumerate(misclassified_indices[:5]):
    axes[i].imshow(small_test_images[idx].reshape(28, 28), cmap='gray')
    axes[i].set_title(f"True: {small_test_labels[idx]}, Pred: {small_predictions[idx]}")
    axes[i].axis('off')
plt.show()

# 3.6. Uczenie na oryginalnym zbiorze treningowym
# 3.6.1. Wybór algorytmów klasyfikujących
classifiers = {
    "SVC (linear)": SVC(kernel='linear', random_state=42),
    "KNN (5 neighbors)": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}


# 3.6.2. Trenowanie klasyfikatorów
def train_and_evaluate(classifier_name, classifier, train_images, train_labels, test_images, test_labels):
    print(f"\nTraining {classifier_name}...")
    classifier.fit(train_images, train_labels)
    predictions = classifier.predict(test_images)
    print(f"Classification Report ({classifier_name}):")
    print(classification_report(test_labels, predictions))
    print(f"Confusion Matrix ({classifier_name}):")
    print(confusion_matrix(test_labels, predictions))


# 3.6.3. Predykcja
# 3.6.4. Ewaluacja modeli (raport z klasyfikacji, macierz błędów)
for classifier_name, classifier in classifiers.items():
    train_and_evaluate(classifier_name, classifier, train_images, train_labels, test_images, test_labels)

# 3.6.5. Dyskusja wyników
# Dyskusja wyników:
# Na podstawie raportów z klasyfikacji oraz macierzy błędów można zauważyć, że każdy z klasyfikatorów ma swoje mocne i słabe strony.
# Klasyfikator SVC z jądrem liniowym może osiągać dobrą precyzję, ale jest stosunkowo wolny w trenowaniu na dużych zbiorach danych.
# KNN (k najbliższych sąsiadów) jest prosty do zrozumienia, ale jego wydajność spada wraz ze wzrostem liczby próbek, a czas predykcji jest wysoki.
# Random Forest jest bardzo skuteczny dzięki połączeniu wielu drzew decyzyjnych, co pozwala uzyskać stabilne wyniki.
# Metody zespołowe, takie jak Bagging i Gradient Boosting, również wykazują wysoką skuteczność, przy czym Gradient Boosting często osiąga najlepsze wyniki, ale kosztem dłuższego czasu trenowania.
# Wybór najlepszego modelu zależy od wymagań dotyczących szybkości, precyzji oraz dostępnych zasobów obliczeniowych.

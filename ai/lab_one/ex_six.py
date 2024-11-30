import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 6.1. Otwarcie pliku z danymi i stworzenie obiektu df_wine
df_wine = pd.read_csv('wine_fraud.csv')
df_wine.columns = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
    "quality", "type"
]
print(df_wine.columns)

# Usunięcie spacji z nazw kolumn
df_wine.columns = df_wine.columns.str.strip()

# 6.2. Sprawdzenie podstawowych statystyk
print(df_wine.describe())

# 6.3. Sprawdzenie kompletności danych
print(df_wine.isnull().sum())

# 6.4. Wykres słupkowy ilości wystąpień kategorii dotyczącej atrybutu warunkowego
plt.figure(figsize=(8, 6))
df_wine['quality'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Jakość wina')
plt.ylabel('Liczba próbek')
plt.title('Ilość wystąpień kategorii jakości wina')
plt.show()

# 6.5. Sprawdzenie różnicy między winem czerwonym i białym, jeśli chodzi o przypadki fałszerstwa
red_wine = df_wine[df_wine['type'] == 'red']
white_wine = df_wine[df_wine['type'] == 'white']
fraud_red = red_wine['quality'].value_counts(normalize=True)
fraud_white = white_wine['quality'].value_counts(normalize=True)
print('Procent fałszywych win czerwonych:', fraud_red)
print('Procent fałszywych win białych:', fraud_white)

# 6.6. Procent win czerwonych i białych, które są fałszywe
percent_fraud_red = (red_wine['quality'] == 'Fraud').sum() / len(red_wine) * 100
percent_fraud_white = (white_wine['quality'] == 'Fraud').sum() / len(white_wine) * 100
print(f'Procent win czerwonych fałszywych: {percent_fraud_red:.2f}%')
print(f'Procent win białych fałszywych: {percent_fraud_white:.2f}%')

# 6.7. Korelacja między różnymi cechami chemicznymi wina a jakością wina
# Zamiana kolumny 'quality' na wartości numeryczne dla korelacji
df_wine['quality'] = df_wine['quality'].map({'Legit': 0, 'Fraud': 1})
# Zamiana kolumny 'type' na wartości numeryczne
df_wine['type'] = df_wine['type'].map({'red': 0, 'white': 1})
correlation_matrix = df_wine.corr()
print(correlation_matrix['quality'])

# 6.8. Zamiana kolumny kategorialnej dotyczącej rodzaju wina
df_wine['type'] = df_wine['type'].map({'red': 0, 'white': 1})

# 6.9. Podział danych na macierz cech X oraz etykiety y
X = df_wine.drop(['quality'], axis=1)
y = df_wine['quality']

# 6.10. Podział danych na dane uczące i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

# 6.11. Skalowanie danych → standaryzacja
# Usunięcie kolumny 'type', aby uniknąć problemów ze skalowaniem
X_train = X_train.drop(['type'], axis=1)
X_test = X_test.drop(['type'], axis=1)
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

# 6.12. Stworzenie i trenowanie modelu SVM
model_svc = SVC()
model_svc.fit(scaled_X_train, y_train)

# 6.13. Predykcja dla zbioru testowego
y_pred = model_svc.predict(scaled_X_test)

# 6.14. Wyświetlenie raportu z klasyfikacji oraz macierz błędów
print('Raport z klasyfikacji:')
print(classification_report(y_test, y_pred, zero_division=1))
print('Macierz błędów:')
print(confusion_matrix(y_test, y_pred))

# 6.15. Przeszukiwanie siatki w celu doboru optymalnych parametrów
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(scaled_X_train, y_train)

# 6.16. Wyświetlenie raportu z klasyfikacji oraz macierz błędów dla najlepszego modelu
y_pred_optimized = grid.predict(scaled_X_test)
print('Raport z klasyfikacji dla najlepszego modelu:')
print(classification_report(y_test, y_pred_optimized, zero_division=1))
print('Macierz błędów dla najlepszego modelu:')
print(confusion_matrix(y_test, y_pred_optimized))

# 6.17. Rysowanie macierzy pomyłek dla najlepszego modelu
conf_matrix = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predykowane wartości')
plt.ylabel('Rzeczywiste wartości')
plt.title('Macierz pomyłek dla najlepszego modelu')
plt.show()

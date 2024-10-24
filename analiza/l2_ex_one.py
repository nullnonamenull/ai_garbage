import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1.2. Otwarcie pliku z danymi
file_path = 'data.csv'  # Podaj ścieżkę do pliku z danymi
df = pd.read_csv(file_path, delim_whitespace=True, header=None)

# 1.3. Sprawdzenie podstawowych statystyk
print(df.describe())

# 1.4. Sprawdzenie kompletności danych
print(df.isnull().sum())

# 1.5. Sprawdzenie, czy typy danych są akceptowalne
print(df.dtypes)

# 1.6. Wykresy pudełkowe dla wszystkich kolumn numerycznych
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numerical_cols) > 0:
    df[numerical_cols].boxplot(figsize=(15, 10))
    plt.show()
else:
    print("Brak kolumn numerycznych do wykreślenia.")

# 1.7. Wyznaczenie % wartości odstających dla każdej z kolumn numerycznych
for column in numerical_cols:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).sum()
    percentage = (outliers / len(df)) * 100
    print(f'Odstające wartości dla kolumny {column}: {percentage:.2f}%')

# 1.8. Macierz korelacji i jej wizualizacja
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# 1.9. Wykres pairplot dla wszystkich kolumn numerycznych
if len(numerical_cols) > 0:
    sns.pairplot(df[numerical_cols])
    plt.show()
else:
    print("Brak kolumn numerycznych do wykreślenia pairplot.")

# 1.10. Stworzenie df_selected z najbardziej skorelowanymi cechami
df_selected = df.loc[:, corr_matrix[(corr_matrix[13] > 0.5) | (corr_matrix[13] < -0.5)].index]
sns.pairplot(df_selected)
plt.show()

# 1.11. Przygotowanie zmiennych X i y
X = df.drop(13, axis=1)
y = df[13]

# 1.13. Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# 1.14. Stworzenie modelu regresji liniowej i nauka
model = LinearRegression()
model.fit(X_train, y_train)

# Wyświetlenie współczynników dopasowania
print(f'Nachylenie: {model.coef_}')
print(f'Punkt przecięcia: {model.intercept_}')

# 1.16. Predykcja dla X_test
y_pred = model.predict(X_test)

# 1.17. Wykres punktowy y_test vs y_pred
plt.scatter(y_test, y_pred)
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Prognozowane wartości')
plt.title('y_test vs y_pred')
plt.show()

# 1.18. Ewaluacja modelu
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# 1.19. Regularyzacja
# 1.19.1. Regresja grzbietowa
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f'Współczynniki ridge: {ridge.coef_}')
print(f'Ridge MAE: {mean_absolute_error(y_test, y_pred_ridge)}')
print(f'Ridge MSE: {mean_squared_error(y_test, y_pred_ridge)}')
print(f'Ridge RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge))}')
print(f'Ridge R2: {r2_score(y_test, y_pred_ridge)}')

# 1.19.2. Regresja metodą lasso
lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print(f'Współczynniki lasso: {lasso.coef_}')
print(f'Lasso MAE: {mean_absolute_error(y_test, y_pred_lasso)}')
print(f'Lasso MSE: {mean_squared_error(y_test, y_pred_lasso)}')
print(f'Lasso RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso))}')
print(f'Lasso R2: {r2_score(y_test, y_pred_lasso)}')

# 1.19.3. Regresja elastycznej siatki
elastic = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)
print(f'Współczynniki elastic: {elastic.coef_}')
print(f'ElasticNet MAE: {mean_absolute_error(y_test, y_pred_elastic)}')
print(f'ElasticNet MSE: {mean_squared_error(y_test, y_pred_elastic)}')
print(f'ElasticNet RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_elastic))}')
print(f'ElasticNet R2: {r2_score(y_test, y_pred_elastic)}')

# 1.20. Przygotowanie zmiennych X_selected i y
X_selected = df[[5, 10, 12]]
y = df[13]

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=101)

# Stworzenie modelu regresji liniowej i nauka
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Ewaluacja modelu dla X_selected
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

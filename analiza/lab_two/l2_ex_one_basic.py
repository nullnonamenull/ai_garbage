import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

file_path = 'data.csv'
df = pd.read_csv(file_path, delim_whitespace=True, header=None)

print(df.describe())

print(df.isnull().sum())

print(df.dtypes)

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numerical_cols) > 0:
    df[numerical_cols].boxplot(figsize=(15, 10))
    plt.show()
else:
    print("Brak kolumn numerycznych do wykreślenia.")

for column in numerical_cols:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))).sum()
    percentage = (outliers / len(df)) * 100
    print(f'Odstające wartości dla kolumny {column}: {percentage:.2f}%')

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

if len(numerical_cols) > 0:
    sns.pairplot(df[numerical_cols])
    plt.show()
else:
    print("Brak kolumn numerycznych do wykreślenia pairplot.")

df_selected = df.loc[:, corr_matrix[(corr_matrix[13] > 0.5) | (corr_matrix[13] < -0.5)].index]
sns.pairplot(df_selected)
plt.show()

X = df.drop(13, axis=1)
y = df[13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)

print(f'Nachylenie: {model.coef_}')
print(f'Punkt przecięcia: {model.intercept_}')

y_pred = model.predict(X_test)

# 1.17. Wykres punktowy y_test vs y_pred
plt.scatter(y_test, y_pred)
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Prognozowane wartości')
plt.title('y_test vs y_pred')
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f'Współczynniki ridge: {ridge.coef_}')
print(f'Ridge MAE: {mean_absolute_error(y_test, y_pred_ridge)}')
print(f'Ridge MSE: {mean_squared_error(y_test, y_pred_ridge)}')
print(f'Ridge RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge))}')
print(f'Ridge R2: {r2_score(y_test, y_pred_ridge)}')

lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print(f'Współczynniki lasso: {lasso.coef_}')
print(f'Lasso MAE: {mean_absolute_error(y_test, y_pred_lasso)}')
print(f'Lasso MSE: {mean_squared_error(y_test, y_pred_lasso)}')
print(f'Lasso RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso))}')
print(f'Lasso R2: {r2_score(y_test, y_pred_lasso)}')

elastic = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)
print(f'Współczynniki elastic: {elastic.coef_}')
print(f'ElasticNet MAE: {mean_absolute_error(y_test, y_pred_elastic)}')
print(f'ElasticNet MSE: {mean_squared_error(y_test, y_pred_elastic)}')
print(f'ElasticNet RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_elastic))}')
print(f'ElasticNet R2: {r2_score(y_test, y_pred_elastic)}')

X_selected = df[[5, 10, 12]]
y = df[13]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=101)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

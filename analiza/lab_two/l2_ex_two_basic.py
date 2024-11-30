import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2.1 - Load the data
file_path = 'data.csv'  # Update with the actual path to your data file
df = pd.read_csv(file_path, delim_whitespace=True, header=None)

# Step 2.2 - Define features (X) and target (y)
X = df.drop(13, axis=1)  # Exclude target column 'MEDV'
y = df[13]  # Target variable

# Step 2.3 - Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Step 2.4 - Standardize features
scaler = StandardScaler()  # Step 2.2 - Create scaler object
scaler.fit(X_train)        # Step 2.3 - Fit scaler on training data
scaled_X_train = scaler.transform(X_train)  # Step 2.4 - Transform X_train
scaled_X_test = scaler.transform(X_test)    # Step 2.5 - Transform X_test

# Step 2.6 - Train a linear regression model
std_model = LinearRegression()
std_model.fit(scaled_X_train, y_train)

# Step 2.7 - Display model coefficients
print(f'Standardized Model Coefficients: {std_model.coef_}')
print(f'Standardized Model Intercept: {std_model.intercept_}')

# Step 2.8 - Predict on test data
y_pred_std = std_model.predict(scaled_X_test)

# Step 2.9 - Scatter plot y_test vs y_pred
plt.scatter(y_test, y_pred_std)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Standardized y_test vs y_pred')
plt.show()

# Step 2.10 - Evaluate model
mae_std = mean_absolute_error(y_test, y_pred_std)
mse_std = mean_squared_error(y_test, y_pred_std)
rmse_std = np.sqrt(mse_std)
r2_std = r2_score(y_test, y_pred_std)

print(f'Standardized MAE: {mae_std}')
print(f'Standardized MSE: {mse_std}')
print(f'Standardized RMSE: {rmse_std}')
print(f'Standardized R2: {r2_std}')

# Step 2.11 - Regularization with scaled data (Ridge, Lasso, ElasticNet)
ridge_std = Ridge(alpha=0.5)  # Step 2.11.1 - Ridge regression
ridge_std.fit(scaled_X_train, y_train)
y_pred_ridge = ridge_std.predict(scaled_X_test)
print(f'Ridge Coefficients: {ridge_std.coef_}')
print(f'Ridge MAE: {mean_absolute_error(y_test, y_pred_ridge)}')
print(f'Ridge MSE: {mean_squared_error(y_test, y_pred_ridge)}')
print(f'Ridge RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge))}')
print(f'Ridge R2: {r2_score(y_test, y_pred_ridge)}')

lasso_std = Lasso(alpha=0.5)  # Step 2.11.2 - Lasso regression
lasso_std.fit(scaled_X_train, y_train)
y_pred_lasso = lasso_std.predict(scaled_X_test)
print(f'Lasso Coefficients: {lasso_std.coef_}')
print(f'Lasso MAE: {mean_absolute_error(y_test, y_pred_lasso)}')
print(f'Lasso MSE: {mean_squared_error(y_test, y_pred_lasso)}')
print(f'Lasso RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lasso))}')
print(f'Lasso R2: {r2_score(y_test, y_pred_lasso)}')

elastic_std = ElasticNet(alpha=0.5, l1_ratio=0.5)  # Step 2.11.3 - ElasticNet regression
elastic_std.fit(scaled_X_train, y_train)
y_pred_elastic = elastic_std.predict(scaled_X_test)
print(f'ElasticNet Coefficients: {elastic_std.coef_}')
print(f'ElasticNet MAE: {mean_absolute_error(y_test, y_pred_elastic)}')
print(f'ElasticNet MSE: {mean_squared_error(y_test, y_pred_elastic)}')
print(f'ElasticNet RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_elastic))}')
print(f'ElasticNet R2: {r2_score(y_test, y_pred_elastic)}')

# Step 2.12 - Analysis on selected features ('RM', 'PTRATIO', 'LSTAT')
X_selected = df[[5, 10, 12]]  # Assumes 'RM', 'PTRATIO', 'LSTAT' are columns 5, 10, 12
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=101)

# Standardize selected features
scaler_sel = StandardScaler()
scaler_sel.fit(X_train_sel)
scaled_X_train_sel = scaler_sel.transform(X_train_sel)
scaled_X_test_sel = scaler_sel.transform(X_test_sel)

# Model training and evaluation on selected features
std_model_sel = LinearRegression()
std_model_sel.fit(scaled_X_train_sel, y_train_sel)
y_pred_sel = std_model_sel.predict(scaled_X_test_sel)

# Evaluation metrics for selected features
mae_sel = mean_absolute_error(y_test_sel, y_pred_sel)
mse_sel = mean_squared_error(y_test_sel, y_pred_sel)
rmse_sel = np.sqrt(mse_sel)
r2_sel = r2_score(y_test_sel, y_pred_sel)

print(f'Selected Features MAE: {mae_sel}')
print(f'Selected Features MSE: {mse_sel}')
print(f'Selected Features RMSE: {rmse_sel}')
print(f'Selected Features R2: {r2_sel}')

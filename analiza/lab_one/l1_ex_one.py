import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('iris.csv')
print("Data loaded successfully")

print("Basic statistics:\n", df.describe())

print("Missing values:\n", df.isnull().sum())

features = df.drop(columns=['species'])
species = df['species']

scaler_01 = MinMaxScaler(feature_range=(0, 1))
norm_01 = scaler_01.fit_transform(features)
norm_01_df = pd.DataFrame(norm_01, columns=features.columns)
norm_01_df['species'] = species

scaler_11 = MinMaxScaler(feature_range=(-1, 1))
norm_11 = scaler_11.fit_transform(features)
norm_11_df = pd.DataFrame(norm_11, columns=features.columns)
norm_11_df['species'] = species

scaler_std = StandardScaler()
standardized = scaler_std.fit_transform(features)
standardized_df = pd.DataFrame(standardized, columns=features.columns)
standardized_df['species'] = species

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
fig.suptitle('Petal Length vs Petal Width Differentiated by Species')

for ax, data, title in zip(
        axes,
        [df, norm_01_df, norm_11_df, standardized_df],
        ['Original', '[0, 1] Normalized', '[-1, 1] Normalized', 'Standardized']):
    for species_name in data['species'].unique():
        subset = data[data['species'] == species_name]
        ax.scatter(subset['petal_length'], subset['petal_width'], label=species_name, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Petal Length (cm)')
    ax.set_ylabel('Petal Width (cm)')
    ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
fig.suptitle('Sepal Length vs Sepal Width Differentiated by Species')

for ax, data, title in zip(
        axes,
        [df, norm_01_df, norm_11_df, standardized_df],
        ['Original', '[0, 1] Normalized', '[-1, 1] Normalized', 'Standardized']):
    for species_name in data['species'].unique():
        subset = data[data['species'] == species_name]
        ax.scatter(subset['sepal_length'], subset['sepal_width'], label=species_name, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

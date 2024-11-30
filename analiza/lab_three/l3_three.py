import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('wholesome_customers_data.csv')

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Milk', y='Grocery', hue='Channel', palette='viridis')
plt.title("Wydatki na 'Milk' vs 'Grocery' różnicowane przez 'Channel'")
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Milk', hue='Channel', kde=True)
plt.title("Histogram wydatków na 'Milk' według 'Channel'")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Mapa korelacji wydatków na różne kategorie")
plt.show()

sns.pairplot(df, hue='Region', palette='viridis')
plt.suptitle("Macierz par zmiennych według 'Region'", y=1.02)
plt.show()

df_features = df.drop(columns=['Channel', 'Region'])
scaler = StandardScaler()
df_features_std = scaler.fit_transform(df_features)

outlier_percentages = []
eps_values = np.linspace(0.001, 3, 50)
min_samples = df_features.shape[1] * 2

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df_features_std)
    outliers = np.sum(labels == -1)
    outlier_percent = (outliers / len(labels)) * 100
    outlier_percentages.append(outlier_percent)

plt.figure(figsize=(8, 6))
plt.plot(eps_values, outlier_percentages, marker='o')
plt.title("Procent punktów odstających w zależności od epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Procent punktów odstających")
plt.show()

optimal_eps = eps_values[np.argmin(np.abs(np.array(outlier_percentages) - 10))]
dbscan_final = DBSCAN(eps=optimal_eps, min_samples=min_samples)
dbscan_labels = dbscan_final.fit_predict(df_features_std)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Milk'], y=df['Grocery'], hue=dbscan_labels, palette='tab10')
plt.title("DBSCAN: 'Milk' vs 'Grocery'")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Milk'], y=df['Detergents_Paper'], hue=dbscan_labels, palette='tab10')
plt.title("DBSCAN: 'Milk' vs 'Detergents Paper'")
plt.show()

df['Labels'] = dbscan_labels

df_without_channel_region = df.drop(columns=['Channel', 'Region'])

cluster_means = df_without_channel_region.groupby('Labels').mean()
outlier_mean = df_without_channel_region[df['Labels'] == -1].mean()

print("Średnie wydatków dla klastrów:")
print(cluster_means)
print("\nŚrednie wydatków dla wartości odstających:")
print(outlier_mean)

category_variances = cluster_means.var()
most_varied_category = category_variances.idxmax()
print(f"Kategoria wydatków z największymi różnicami między klastrami: {most_varied_category}")

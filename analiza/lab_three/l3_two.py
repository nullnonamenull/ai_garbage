import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('iris.csv')

print("Czy są braki danych?")
print(df.isnull().sum())

print("\nStatystyki podstawowe:")
print(df.describe())
print("\nInformacje o danych:")
print(df.info())

plt.figure(figsize=(8, 6))
plt.scatter(df['petal_length'], df['petal_width'], color='b', alpha=0.5)
plt.title("Zależność długości płatka od szerokości płatka")
plt.xlabel("Długość płatka")
plt.ylabel("Szerokość płatka")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species')
plt.title("Zależność długości płatka od szerokości płatka według gatunku")
plt.xlabel("Długość płatka")
plt.ylabel("Szerokość płatka")
plt.show()

df_iris = df.drop(columns=['species'])

scaler = StandardScaler()
df_iris_std = pd.DataFrame(scaler.fit_transform(df_iris), columns=df_iris.columns)

normalizer = MinMaxScaler()
df_iris_norm = pd.DataFrame(normalizer.fit_transform(df_iris), columns=df_iris.columns)

plt.figure(figsize=(12, 8))
sns.heatmap(df_iris.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Korelacja - Dane Oryginalne")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df_iris_std.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Korelacja - Dane Zestandaryzowane")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df_iris_norm.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Korelacja - Dane Znormalizowane")
plt.show()

linkages = ['ward', 'single', 'complete']
for linkage in linkages:
    agglomerative = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    agglomerative_labels = agglomerative.fit_predict(df_iris)
    silhouette_avg = silhouette_score(df_iris, agglomerative_labels)
    print(f"Metoda łączenia: {linkage}, Średni współczynnik sylwetki: {silhouette_avg}")

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_iris)
kmeans_labels = kmeans.labels_
silhouette_avg_kmeans = silhouette_score(df_iris, kmeans_labels)
print(f"Średni współczynnik sylwetki dla k-średnich: {silhouette_avg_kmeans}")

for linkage in linkages:
    agglomerative = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    agglomerative_labels = agglomerative.fit_predict(df_iris)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['petal_length'], y=df['petal_width'], hue=agglomerative_labels, palette='viridis')
    plt.title(f"Klasteryzacja Aglomeracyjna - {linkage.capitalize()}")
    plt.xlabel("Długość płatka")
    plt.ylabel("Szerokość płatka")
    plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['petal_length'], y=df['petal_width'], hue=kmeans_labels, palette='viridis')
plt.title("Klasteryzacja K-średnich")
plt.xlabel("Długość płatka")
plt.ylabel("Szerokość płatka")
plt.show()

datasets = {'Zestandaryzowane': df_iris_std, 'Znormalizowane': df_iris_norm}
for name, data in datasets.items():
    print(f"\nKlasteryzacja na danych {name}")

    for linkage in linkages:
        agglomerative = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        agglomerative_labels = agglomerative.fit_predict(data)
        silhouette_avg = silhouette_score(data, agglomerative_labels)
        print(f"Metoda łączenia: {linkage}, Średni współczynnik sylwetki: {silhouette_avg}")

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)
    kmeans_labels = kmeans.labels_
    silhouette_avg_kmeans = silhouette_score(data, kmeans_labels)
    print(f"Średni współczynnik sylwetki dla k-średnich: {silhouette_avg_kmeans}")

import matplotlib.pyplot as plt
import pandas as pd

file_path = 'Zad2_L1.csv'
separator = ';'

columns = [
    "Wavenumber [cm^-1]", "t0", "t1", "t2", "t3", "t4", "t5"
]

df = pd.read_csv(file_path, sep=separator, names=columns, header=2)
df = df.apply(lambda x: x.str.replace(',', '.')).astype(float)

wavenumber_985 = 985.0
index_985 = df.iloc[(df["Wavenumber [cm^-1]"] - wavenumber_985).abs().argsort()[:1]].index[0]

normalized_df = df.copy()

# dlczego  985? do przekminy

# normalizacja -> skalowanie danych -> wszystkie wartości w kolumnie wyrażane w stosunku do jednej wybranej
# aktualizacjka kazdej kolumy
# wartosci kolumny podzielone przez konkretna wartosc znajdujaca sie pod indeksem -> index_985
for col in df.columns[1:]:
    normalized_df[col] = df[col] / df[col][index_985]

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
for col in df.columns[1:]:
    plt.plot(df["Wavenumber [cm^-1]"], df[col], label=col)
plt.xlabel("Wavenumber [cm^-1]")
plt.ylabel("Intensity [a.u.]")
plt.title("Raw Raman Spectra")
plt.legend()

plt.subplot(2, 1, 2)
for col in normalized_df.columns[1:]:
    plt.plot(normalized_df["Wavenumber [cm^-1]"], normalized_df[col], label=col)
plt.xlabel("Wavenumber [cm^-1]")
plt.ylabel("Normalized Intensity [a.u.]")
plt.title("Normalized Raman Spectra")
plt.legend()

plt.tight_layout()
plt.show()

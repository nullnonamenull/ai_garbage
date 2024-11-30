import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('Zad3_L1.csv', delimiter=';')
df.replace(',', '.', regex=True, inplace=True)

wavenumber = df.iloc[:, 0].astype(float)
data = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# np.trapz -> całka metodą trapezów
# lambda -> dla każdej kolumny osobno
# wartości kolumy / wynik np.trapz
# całka z wartości w kolumnie w odniesieniu do zmiennej wavenumber ( -> reprezentacja numerów fal w analizie widmowej )
normalized_data = data.apply(lambda col: col / np.trapz(col, wavenumber), axis=0)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for column in data.columns:
    plt.plot(wavenumber, data[column], label=column)
plt.xlabel('Wavenumber [cm^-1]')
plt.ylabel('Absorbance [a.u.]')
plt.title('Dane surowe')
plt.legend()

plt.subplot(1, 2, 2)
for column in normalized_data.columns:
    plt.plot(wavenumber, normalized_data[column], label=column)
plt.xlabel('Wavenumber [cm^-1]')
plt.ylabel('Znormalizowana absorbancja [a.u.]')
plt.title('Dane znormalizowane')
plt.legend()

plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../datasets/Hitters.csv")
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
print(num_cols)

# print(df[num_cols].head())

# Entfernen von fehlenden Daten
df = df[num_cols]
df.dropna(inplace=True)
print(df.shape)

# Standardisierung
df = StandardScaler().fit_transform(df)

#  PCA - Modell erstellen
pca = PCA(n_components=5)

# Daten transformieren
pca_results = pca.fit_transform(df)

# Ergebnisse in ein neues DataFrame umwandeln
pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])


# print(pca_df.head())
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_))

################################
# Optimum Bileşen Sayısı
################################
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısı")
plt.ylabel("Kümülatif Varyansa Oranı")
plt.show()

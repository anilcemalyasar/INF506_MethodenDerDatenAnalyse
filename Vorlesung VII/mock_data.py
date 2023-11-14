import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import warnings

# Ignoieren aller Warnungen
warnings.filterwarnings("ignore")

# Erzeugen des Datensatzes
X, y  = make_blobs(n_samples=300,  # die Anzahl der Merkmalvektoren
                   n_features=2,   # die Anzahl der Merkmalen
                   centers=4,       # die Anzahl von Clusters
                   random_state=42
                   )

# Umwandlung in ein DataFrame
df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'target':y})
print(df.head())

# Visualisierung mit Hilfe des Scatter Plots
# grouped = df.groupby('target')
# colors = {0:'red', 1:'blue', 2:'green', 3:"black"}
# fig, ax = plt.subplots()
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color=colors[key])
#
# plt.savefig("original_clusters.jpg")
# plt.show()

# KMeans-Modell erstellen und trainieren
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X=X)

# Eğitildikten sonraki küme etiketlerini al
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# İki subplot içeren bir figür oluştur
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# İlk subplot: Orijinal Veri Setinden Gelen Küme Etiketleri
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
axes[0].set_title('Orijinal Veri Setinden Gelen Küme Etiketleri')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature2')

# İkinci subplot: Eğitildikten Sonraki Küme Etiketleri
axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
axes[1].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Trained Centers')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

# Subplot'ları göster
plt.savefig('trained_clusters.png')
plt.show()



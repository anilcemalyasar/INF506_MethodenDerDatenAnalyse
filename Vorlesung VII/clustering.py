import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.datasets import load_iris
import seaborn as sns

# Iris-Datensatz laden
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

true_labels = iris.target

km = KMeans(n_clusters=4, random_state=42)
km.fit(X=iris_df)

iris_df['cluster'] = km.labels_

print(iris_df)
sns.scatterplot(data=iris_df, x="sepal length (cm)", y="sepal width (cm)", hue='cluster')
plt.title('Clusteranalyse des Iris-Datensatzes')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

# Metriken berechnen
ari_score = adjusted_rand_score(labels_true=true_labels, labels_pred=km.labels_)
nmi_score = normalized_mutual_info_score(labels_true=true_labels, labels_pred=km.labels_)
inertia_val = km.inertia_
silhouette_val = silhouette_score(iris_df.iloc[:, :-1], km.labels_)

print(f"Adjusted Rand Index (ARI): {ari_score}")
print(f"Normalized Mutual Information (NMI): {nmi_score}")
print(f"Inertia Value: {inertia_val}")
print(f"Silhoutte Score: {silhouette_val}")

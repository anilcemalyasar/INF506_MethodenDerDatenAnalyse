import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Laden des Iris - Datensatzes
iris = load_iris()
data = iris.data
target = iris.target
print(iris.feature_names)

# Erstellen eines Pandas - Datenrahmens
df = pd.DataFrame(data, columns=iris.feature_names)
df['target'] = target
print(df.head(5))

# Aufteilung in Trainings - und Testdaten
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}", end="\n\n")

# Aggregation und Berechnung des Gesamtumsatzes pro Target
agg_df = df.groupby('target')['sepal length (cm)'].mean().reset_index()
print(agg_df)
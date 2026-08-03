import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\CSV\advertising.csv")
print("\nDataFrame:\n", df)

print(df.isnull().sum())

for i in df.columns:
    print(i, ':', df[i].unique(), '\n')

X = df.drop('Sales', axis=1)
print(X)

### standardize
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
print("\n Scaled_Values:\n", X_scaled)

### Visualize original data
plt.figure(figsize=(8,6))
sns.scatterplot(x=df.TV, y=df.Radio)
plt.title("Original_Data")
plt.show()


######## DBSCAN MODEL
for eps in [0.3, 0.4, 0.5, 0.6, 0.7]:
    dbscan = DBSCAN(eps=0.7, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    df["Clusters"] = labels

    ### Number of clusters found
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Clusters Found:", n_clusters)

    ### Count Noise points
    noise_points = list(labels).count(-1)
    print("Noise Points:", noise_points)

print(f"eps={eps}: Clusters={n_clusters}, Noise={noise_points}")



## Visualize clusters
plt.figure(figsize=(8,6))
# Normal points
plt.scatter(
    df.loc[labels != -1, "TV"],
    df.loc[labels != -1, "Radio"],
    label="Cluster"
)
# Noise points
plt.scatter(
    df.loc[labels == -1, "TV"],
    df.loc[labels == -1, "Radio"],
    marker="x",
    s=100,
    label="Noise"
)
plt.xlabel("TV")
plt.ylabel("Radio")
plt.title("DBSCAN Outlier Detection")
plt.legend()
plt.show()
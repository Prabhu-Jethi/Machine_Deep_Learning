import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\iris\iris.data", header=None)
print("\nIris_DataFrame:\n", df)

df.columns = ['SL', 'SW', 'PL', 'PW', 'Flower']
print("\nColumns_added:\n", df)

print("\nNull_:\n", df.isnull().sum())

for i in df.columns:
    print(i, ':', df[i].unique(), '\n')


le =LabelEncoder()
df['Flower'] = le.fit_transform(df['Flower'])
print("\nEncoded_Columns:\n", df)


X = df.drop('Flower', axis=1)
print("\nDropped_Column:\n", X)


##### Create Dendrogram 
plt.figure(figsize=(12, 6))
Z = linkage(X, method='ward')
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()


##### Hierarchical Clustering
hc = AgglomerativeClustering(
    n_clusters=3,
    metric='euclidean',
    linkage='ward'
)
clusters = hc.fit_predict(X)
print("\nHierarchical-Clustering:\n", clusters[:10])


##### Adding cluster labels
df["Cluster"] = clusters
print("\nCluster_Predictions:\n", df)


##### Visualize clusters
df["Cluster"] = clusters
sns.scatterplot(x=df.SL, y=df.SW, hue=df.Cluster, palette='Set1')
plt.title('Hierarchical Cluster')
plt.show()
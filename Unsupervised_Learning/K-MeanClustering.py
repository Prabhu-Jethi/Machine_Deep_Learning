import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans



##### Collect
df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\heart.xls")
print("\nDataframe:\n", df)


##### Clean
print("\nNull_values:\n", df.isnull().sum())

for i in df.columns:
    print(i, ':', df[i].unique(), '\n')


###### Dropping columns
X = df.drop('target', axis=1)
print("\nX_columns:\n", X)

#####
sc = StandardScaler()
X = sc.fit_transform(X)
print("\nScaled_Values:\n", sc)

##### Model implementing 
# 1. Default approach

km = KMeans(
    n_clusters=2,
    random_state=42
)
km.fit(X)
print("\nK-Mean Clustering:\n", km)

# 2. 'Elbow' method

# wcss = []
# for k in range(1, 11):
#     km = KMeans(
#         n_clusters=k,
#         random_state=42
#     )
#     km.fit(X)
#     wcss.append(km.inertia_)
# plt.plot(range(1, 11), wcss, marker='o')
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("WCSS")
# plt.title("Elbow Method")
# plt.show()

## get cluster labels
labels = km.labels_
print("\nLabels:\n", labels)


####### Visualize clusters
df['Cluster'] = labels
sns.scatterplot(x=df.age, y=df.chol, hue=df.Cluster, palette='Set1')
plt.title('K-Mean Cluster')
plt.show()


###### Compare with actual labels
df['Actual_target_column'] = df['target']
df['Cluster_predicted'] = labels
print("\nComparing Actual vs Cluster predicted values:\n",df[['Actual_target_column', 'Cluster_predicted']])


##### Model final prediction
km_pred = km.predict(X)
print("\nModel_final_prediction:\n", km_pred)


##### Centroids: for model to learn (Average patient profile of group 1 and group 2)
centroid = sc.inverse_transform(km.cluster_centers_)
print("\nCentroid:\n", centroid)

sns.scatterplot(x=df.age, y=df.chol, hue=km_pred, palette='Set1')
sns.scatterplot(x=centroid[:, 0], y=centroid[:, 4], s=100, color='black')
plt.title("Centroid_in_Clusters")
plt.show()
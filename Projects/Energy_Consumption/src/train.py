import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_data():
    ## load the processed data from feature engineering
    data_df = pd.read_csv(r"D:\Python\ML\EVERYDAY_ML_DL\Projects\Energy_Consumption\data\processed_household_power_consumption.csv", sep=',')
    data_df.info()
    return data_df

data_df = load_data()


X = data_df.drop(columns=['is_weekend_True'], axis=1, errors='ignore')

def pipeline(data_df):
    ## pipeline for scaling
    pipe = Pipeline([
        ('scaler', StandardScaler())
    ])
    scaled_data = pipe.fit_transform(X)
    scaled_df = pd.DataFrame(scaled_data, columns=X.columns)
    return pipe, scaled_df

pipe, scaled_df = pipeline(data_df)


### Find optimal number of clusters k 
def find_k():
    wcss = []
    for i in range(1, 11):
        kmean = KMeans(
            n_clusters=i,
            init='k-means++',
            random_state=42,
            n_init='auto'
    )
        kmean.fit(scaled_df)
        wcss.append(kmean.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.show()

    return wcss

wcss = find_k()

def kmean_cluster(scaled_df, k_value=3):
    ## Using K-mean clustering model
    kmeans = KMeans(
        n_clusters=k_value,
        init='k-means++',
        random_state=42,
        n_init='auto',
        verbose=0,
        max_iter=300
    )
    kmeans.fit(scaled_df)

    return kmeans

kmeans = kmean_cluster(scaled_df)


def evaluate_model(kmeans, scaled_df, data_df, X, pipe):
    ## Get cluster labels
    labels = kmeans.labels_

    ## Silhouette score
    sil_score = silhouette_score(scaled_df, labels)
    print(f"\nSilhouette Score: {sil_score:.4f}")

    ## Assign cluster labels to original data
    data_df['Cluster'] = labels

    ## Scatter plot of clusters
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data_df['Global_active_power'], y=data_df['Global_reactive_power'],
                    hue=data_df['Cluster'], palette='deep', alpha=0.6)

    ## Plot centroids (inverse transformed back to original scale)
    centroid = pipe.inverse_transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroid, columns=X.columns)

    sns.scatterplot(x=centroid_df['Global_active_power'], y=centroid_df['Global_reactive_power'],
                    s=150, color='black', marker='X', label='Centroids', legend=False)
    plt.title("K-Means Clustering with Centroids")
    plt.xlabel("Global_active_power")
    plt.ylabel("Global_reactive_power")
    plt.legend()
    plt.grid(True)
    plt.savefig(r"D:\Python\ML\EVERYDAY_ML_DL\Projects\Energy_Consumption\plots\kmeans_clusters.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    ## Bar plot of average Global Active Power per Cluster
    plt.figure(figsize=(8, 5))
    sns.barplot(x=centroid_df.index, y=centroid_df['Global_active_power'], palette='viridis')
    plt.title('Average Global Active Power per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Global Active Power')
    plt.savefig(r"D:\Python\ML\EVERYDAY_ML_DL\Projects\Energy_Consumption\plots\cluster_bar.png",
                dpi=150, bbox_inches='tight')
    plt.show()

    print("\nCluster Centroid Summary:")
    print(centroid_df)

    return centroid_df


def save_model(kmeans, path=r"D:\Python\ML\EVERYDAY_ML_DL\Projects\Energy_Consumption\models\kmean-cluster-model.pkl"):
    joblib.dump(kmeans, path)
    print(f"\nModel saved to: {path}")


if __name__ == "__main__":
    centroid_df = evaluate_model(kmeans, scaled_df, data_df, X, pipe)
    save_model(kmeans)


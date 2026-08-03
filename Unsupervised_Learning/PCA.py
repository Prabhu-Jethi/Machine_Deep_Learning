import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\iris\iris.data", header=None)
print("\nDataframe:\n", df)

df.columns = ['SL', 'SW', 'PL', 'PW', 'Flower']
print("\nColumns_dataframe:\n", df)

print("\nNUll_values:\n", df.isnull().sum())

for i in df.columns:
    print(i, ':', df[i].unique(), '\n')

le = LabelEncoder()
df['Flower'] = le.fit_transform(df['Flower'])
print("\nEncoded:\n", df)

X = df.drop('Flower', axis=1)
print("\nAfter dropping:\n",X)

####### Standardizing 
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
print("\nScaled_X_values:\n", X_scaled)


####### Applying PCA algorithm model
pca = PCA(
    n_components=2
)
X_pca = pca.fit_transform(X_scaled)
print("\nPrincipal_Component_Analysis:\n", X_pca)
print("\nShape:\n", X_pca.shape)

print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio_)

print("\nTotal_variance_Ratio:\n", sum(pca.explained_variance_ratio_))

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Flower'], palette='deep')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Principal Component Analysis on Iris_Set")
plt.show()

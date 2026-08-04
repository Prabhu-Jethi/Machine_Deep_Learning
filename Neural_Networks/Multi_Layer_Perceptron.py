
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

### Reading data
df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\adult\adult.data", header=None, skipinitialspace=True, na_values=['?', ' ?'])
print("\nAdult-Dataset:\n", df)
df.columns = ['Age', 'Workclass', 'fnlgwt', 'Education', 'Education_number', 'Marital-Status', 'Occupation',
               'Relationship', 'Race', 'Gender', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-country', 'Income']
print("\nColumns_Added:\n", df)
print(df.info)
print(df.dtypes)


print("\nMissing_Values:\n", df.isnull().sum())
print("\nRemaining_Missing_Values:", df.dropna(inplace=True))
for i in df.columns:
    print('\n', i, ':', df[i].unique(), '\n')



## Delete Unnecessary-Columns
drop_cols = ['Workclass', 'Education', 'Occupation', 'fnlgwt', 'Native-country']
df.drop(drop_cols, axis=1, inplace=True)
print("\nDropped_Column:\n", drop_cols)

## Ordinal Encoding
income_map = {'<=50K': 0, '>50K': 1}
df['Income'] = df['Income'].map(income_map)

## Label Encode
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

## One hot Encode
cols = ['Marital-Status', 'Relationship', 'Race']
df = pd.get_dummies(df, columns=cols, drop_first=True)

print("\nEncoded_Features:\n", df)


### Splitting
X = df.drop('Income', axis=1)
y = df['Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTrain_and_Test_samples:\n")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


### Standardize
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
print("\nScaled_Values:\n")
print(X_train_scaled)
print(X_test_scaled)


### Model Creation
model = Sequential()
print(model)

## 3 Layers: 
# Input
model.add(Dense(16, input_shape=(X_train_scaled.shape[1],), activation='relu'))
# Hidden
model.add(Dense(8, activation='relu'))
# output
model.add(Dense(1, activation='sigmoid'))

### Model Compilation
model.compile(
    optimizer=Adam(learning_rate=0.05),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

### Model training
fit = model.fit(
    X_train_scaled,
    y_train,
    batch_size = 16,
    epochs = 90,
    verbose=1
)
print(fit.history.keys())


### Evaluate
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("\nLoss:\n", loss)
print("\nAccuracy:\n", accuracy)

# predict probabilities
y_prob = model.predict(X_test_scaled)

# convert probabilities to class labels
y_pred = np.where(y_prob > 0.5, 1, 0).ravel()

print("\nROC_AUC:\n", roc_auc_score(y_test, y_pred))
print("\nAccuracy_Score:\n", accuracy_score(y_test, y_pred))
print("\nRecall_Score:\n", recall_score(y_test, y_pred))
print("\nPrecision_score:\n", precision_score(y_test, y_pred))
print("\nF1 Score:\n", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion_Matrix:\n", cm)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap='hot', fmt='d', annot=True,
             xticklabels=['0', '1'], yticklabels=['1', '0'])
plt.title("ConfusionMatrix")
plt.show()
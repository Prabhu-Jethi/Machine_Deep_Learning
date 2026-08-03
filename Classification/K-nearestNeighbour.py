import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import f1_score, classification_report, accuracy_score, recall_score, precision_score, confusion_matrix


#### Collecting
df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\adult\adult.data", header=None, na_values=['?', ' ?'], skipinitialspace=True)
print("Adult Dataframe:\n", df)

## Add column_names
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income']
print("Added column names:\n", df)


print("Null values:\n", df.isnull().sum())

## Removing Nulls
print(df.dropna())

for i in df.columns:
    print(i, ':', sum(df[i]=='?'))

# print(df.isnull().sum())

for i in df.columns:
    print(i, ':', df[i].unique(), '\n')


### Encoding
le = LabelEncoder()
drop_cols = ['education', 'occupation', 'native_country']
df.drop(drop_cols, axis=1, inplace=True)

workclass_type = {'nan':0, 'State-gov':1, 'Self-emp-not-inc':2, 'Private':3, 'Federal-gov':4, 'Local-gov':5,
                  'Self-emp-inc':6, 'without-pay':7, 'Never-worked':8}
df['workclass'] = df['workclass'].map(workclass_type).fillna(0).astype(int)

binary_cols = ['gender', 'income']
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

oth_cols = ['marital_status', 'relationship', 'race']
for col in oth_cols:
    df[col] = le.fit_transform(df[col])

print("Encoded values:\n", df)


######## Splitting
X = df.drop('income', axis=1)
y = df['income']

print("Input variables:\n", X)
print("Target column:\n", y)


####### Train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)



###### Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Scaled X_train:\n", X_train)
print("Scaled X_test:\n", X_test)




###### Model implementation
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print("K-NearestNeighbour_model:\n", knn)


#### Prediction
ypred = knn.predict(X_test)

print("Predicted X-test:\n", ypred)



##### Evaluation
print("F1: ", f1_score(y_test,ypred))
print("report: ", classification_report(y_test,ypred))
print("recall: ", recall_score(y_test,ypred))
print("precision: ", precision_score(y_test,ypred))
cm = confusion_matrix(y_test, ypred)
sns.heatmap(cm, cmap='hot', annot=True, fmt= 'd',
            xticklabels=['income <= 50k', 'income > 50k'],
            yticklabels=['income > 50k', 'income <= 50k'])
plt.show()


#### visualization
sns.pairplot(df, hue='income')
plt.show()
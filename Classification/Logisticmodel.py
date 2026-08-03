import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, classification_report


#### collecting data

df = pd.read_csv(r"C:\Users\sudip\Downloads\bank-full.csv\bank-full.csv", sep=';', na_values=[' ?', '?']) # separator ';' is used to separate columns 

print("Bank_Marketing DataFrame:\n", df)


###### cleaning 

print("Null values:\n", df.isnull().sum())

for i in df.columns:
    print("Checking Unique values:\n", i, ':', df[i].unique())



######## Encoding

le = LabelEncoder()
df = df.drop('job', axis=1) ## dropped jobs column as it contains many string values which can't be encoded.

#### Used manual mapping for values which must follow specific order like months..
month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8,
               'sept':9, 'oct':10, 'nov':11, 'dec':12}
df['month'] = df['month'].map(month_order)

### checking if it contains any null values or not
print(df['month'].unique())

education_order = {'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3}
df['education'] = df['education'].map(education_order)

bin_cols = ['default', 'housing', 'loan', 'y']
oth_cols = ['marital', 'contact', 'poutcome']

for col in bin_cols:
    df[col] = le.fit_transform(df[col])

for col in oth_cols:
    df[col] = le.fit_transform(df[col])

print("Encoded values:\n", df)


######## Spliting
X = df.drop('y', axis=1)
y = df['y']

### After map encoding months column has some nan values which must be dropped inorder to proceed further
X = X.fillna(X.mean())

print("Input columns:\n", X)
print("Target column:\n", y)


######## Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("X_train data:\n", X_train)
print("X_test data:\n", X_test)
print("y_train data:\n", y_train)
print("y_test data:\n", y_test)


######## Feature Scaling ------> Mean = 0 and Standard division = 1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Scaled X_train:\n", X_train)
print("Scaled X_test:\n", X_test)



######## Model implementing -------> Logistic
lr = LogisticRegression()
lr.fit(X_train, y_train)

print("LogisticRegression Model:\n", lr)



######## Predicting
y_pred = lr.predict(X_test)

print("Prediction of X_test:\n", y_pred)



######### Evaluation
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Recall_Score:\n", rec)
print("F1_Score:\n", f1)
print("Accuracy_Score:\n", ac)
print("Precision_score:\n", pre)
print("Classification_Report:\n", report)

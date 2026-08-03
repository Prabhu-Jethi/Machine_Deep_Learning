import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, classification_report, accuracy_score, recall_score, precision_score

df = pd.read_csv(r"C:\Users\sudip\Downloads\heart.xls")
print("Iris DataFrame:\n", df)


print("Null values:\n", df.isnull().sum())


for i in df.columns:
    print("Unique values:", i, df[i].unique(), '\n')


X = df.drop('target', axis=1)
y = df['target']

print("Input columns:\n", X)
print("Target values:\n", y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Scaled X_train:\n", X_train)
print("Scaled X_test:\n", X_test)


bayes = GaussianNB()
bayes.fit(X_train, y_train)

print("NaiveBayes model:\n", bayes)



ypred = bayes.predict(X_test)

print("Predicting X_test:\n", ypred)


f1 = f1_score(y_test, ypred, average='macro')
rec = recall_score(y_test, ypred, average='macro')
pre = precision_score(y_test, ypred, average='macro')
report = classification_report(y_test, ypred)
ac = accuracy_score(y_test, ypred)

print("f1:\n", f1)
print("Recall:\n", rec)
print("Classification_Report:\n", report)
print("Precision_score:\n", pre)
print("Accuracy_score:\n", ac)
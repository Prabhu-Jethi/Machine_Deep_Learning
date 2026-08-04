import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\heart.xls")
print("DF:\n", df)

print("\nNull:\n", df.isnull().sum())
print(df.dtypes)


le = LabelEncoder()
for i in df.columns:
    df[i] = le.fit_transform(df[i])
print("\nEncoded_DF:\n", df)


X = df.drop('target', axis=1)
y = df['target']
print("\nIndependent variables:\n", X)
print("\nTarget variable:\n", y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nX_train and X_test:\n", X_train, X_test)
print("\ny_train and y_test:\n", y_train, y_test)


gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
print("\nGradient-Boosting-Classifier:\n", gb)

##### comaparison of train and test acc
train_acc = gb.score(X_train, y_train)
test_acc = gb.score(X_test, y_test)
print("\nTrain_acc:\n", train_acc)
print("\nTest_acc:\n", test_acc)


## cross_validating
cv_score = cross_val_score(gb, X, y, cv=4, scoring='accuracy', n_jobs=-1)
print("\nCross-Validation-Score: %.4f +- %.4f" % (cv_score.mean(), cv_score.std()))

## learning curve
train_size, train_score, validation_score = learning_curve(gb, X, y, cv=4, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], n_jobs=-1)
print("\nLearning_curve(size, train_mean, validation_mean):\n")
for s, ts, vs in zip(train_size, train_score.mean(axis=1), validation_score.mean(axis=1)):
    print(int(s), round(ts,3), round(vs,3))


## prediction
ypred = gb.predict(X_test)
print("\nPrediction of X_test:\n", ypred)

## evaluation
acc = accuracy_score(y_test, ypred)
f1 = f1_score(y_test, ypred)
rec = recall_score(y_test, ypred)
pre = precision_score(y_test, ypred)
cm = confusion_matrix(y_test, ypred)

print("\nAccuracy:\n", acc)
print("\nF1:\n", f1)
print("\nRecall:\n", rec)
print("\nPrecision:\n", pre)
print("\nConfusion-Matrix:\n", cm)
sns.heatmap(cm, cmap='viridis', fmt='d', annot=True, xticklabels=['0', '1'], yticklabels=['1', '0'])
plt.show()
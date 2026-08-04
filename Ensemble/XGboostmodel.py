import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix


##### Collecting
df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\bank-full.csv\bank-full.csv", sep=';', na_values=['?', ' ?'])
print("\nDataframe:\n", df)


##### Null check
print("Null:\n", df.isnull().sum())

for i in df.columns:
    print(i, ':', df[i].unique(), '\n')

print(df.dtypes)


# Drop unnecessary columns
drop_col = ['job']
df.drop(drop_col, axis=1, inplace=True)
print("\nDropped_col:\n", drop_col)


######## Encoding

# Map the target column
y_choices = {'no': 0, 'yes': 1}
df['y'] = df['y'].map(y_choices)

le = LabelEncoder()
# binary columns
bin_cols = ['default', 'housing', 'loan']
for col in bin_cols:
    df[col] = le.fit_transform(df[col])

# nominal columns
oth_cols = ['marital', 'education', 'contact', 'poutcome']
df = pd.get_dummies(df, columns=oth_cols, drop_first=True)

month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
               'jul': 7, 'aug': 8, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12}
df['month'] = df['month'].str.lower().map(month_order).fillna(0).astype(int)

print("\nEncoded_Columns:\n", df)

######### Splitting
X = df.drop('y', axis=1)
y = df['y']
print("\nIndependent_columns:\n", X)
print("\nTarget_column:\n", y)



########## Training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print("\nX_train_set:\n", X_train, "\nX_test_set:\n", X_test)
print("\ny_train_set:\n", y_train, "\ny_test_set:\n", y_test)



########## Model Implementation
xgb = XGBClassifier(
    n_estimators = 300,
    max_depth = 4,      #shallow tree
    learning_rate = 0.03,   #slow learning rate
    random_state = 42,
    subsample = 0.7,
    colsample_bytree = 0.7,
    min_child_weight = 5,
    objective = 'binary:logistic',
    scale_pos_weight = 7,       #fixes class imbalance
    n_jobs = -1
)
xgb.fit(X_train, y_train)
print("\nXGBoost Classifier:\n", xgb)


###### Predicting X_testing data
ypred = xgb.predict(X_test)
print("\nPrediction of X_test:\n", ypred)


###### Training accuracy
train_acc = xgb.score(X_train, y_train)
print("\nTraining_accuracy:\n", train_acc)

###### Testing accuracy
test_acc = xgb.score(X_test, y_test)
print("\nTesting_accuracy:\n", test_acc)


########## Cross validation
cv_score = cross_val_score(xgb, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCV_acc: %.4f +- %.4f" % (cv_score.mean(), cv_score.std()))


######### Learning curve
train_sizes, train_scores, validation_scores = learning_curve(xgb, X, y, cv=5, 
                                                              train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0], n_jobs=-1)
print("\nLearning_curve(size, train_mean, validation_mean):\n")
for s, ts, vs in zip(train_sizes, train_scores.mean(axis=1), validation_scores.mean(axis=1)):
    print(int(s), round(ts,3), round(vs,3))


######## Evaluating
print("\nAccuracy:\n", accuracy_score(y_test, ypred))
print("\nF1 Score:\n", f1_score(y_test, ypred, average='macro'))
print("\nRecall Score:\n", recall_score(y_test, ypred, average='macro'))
print("\nPrecision Score:\n", precision_score(y_test, ypred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, ypred))
cm = confusion_matrix(y_test, ypred)
print("\nConfusion Matrix:\n", cm)
sns.heatmap(cm, cmap='hot', annot=True, fmt='d',
            xticklabels=['No', 'Yes'],
            yticklabels=['Yes', 'No'])
plt.show()
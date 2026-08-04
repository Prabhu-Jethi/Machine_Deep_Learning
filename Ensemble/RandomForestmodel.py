import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix

#### Collecting
df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\ai-impact-jobs-layoff-risk-dataset.csv")
print("Dataframe:\n", df)

#### Cleaning
print("Null:\n", df.isnull().sum())

print("\nUnique Values:\n")
for i in df.columns:
    print(i, ':', df[i].unique(), '\n')


#### Encoding
le = LabelEncoder()

# droping str or any other irrelevant columns
drop_cols = ['Job_Role']
df.drop(drop_cols, axis=1, inplace=True)
print("\nDropped_column:\n", drop_cols)

columns = ['Education_Level', 'Industry', 'Company_Size', 'Job_Level', 'AI_Adoption_Level']
for col in columns:
    df[col] = le.fit_transform(df[col])

Layoff_chances = {'Low': 0, 'Medium': 1, 'High': 2}
df['Layoff_Risk'] = df['Layoff_Risk'].map(Layoff_chances)

print("\nEncoded Data:\n", df)


######## Splitting
X = df.drop('Layoff_Risk', axis=1)
y = df['Layoff_Risk']
print("\nIndependent columns:\n", X)
print("\nTarget column:\n", y)



######## Initializing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\nX_set_:\n", X_train, X_test)
print("\ny_set_:\n", y_train, y_test)



####### Model Implementation
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=8,           #limit depth
    min_samples_leaf=16,    #increase to regularize
    min_samples_split=16,   #increase to regularize
    max_features='sqrt',    #reduce features per split
    oob_score=True,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("\nRandom_Forest classifier:\n", rf)


####### Predicting Testing set
ypred = rf.predict(X_test)
print("\nPrediction_of_X_test:\n", ypred)



######## Evaluating

## Checking Training set accuracy...
train_acc = rf.score(X_train, y_train)
print("\nTraining Accuracy:\n", train_acc)

## Testing accuracy...
print("\nTesting Accuracy:\n", rf.score(X_test, y_test))


## Cross validation
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCV acc: %.4f ± %.4f" % (cv_scores.mean(), cv_scores.std()))

## learning curve (training vs validation)
train_sizes, train_scores, val_scores = learning_curve(rf, X, y, cv=5,
                                                        train_sizes=[0.1,0.3,0.5,0.7,1.0],
                                                        n_jobs=-1)
print("\nLearning_Curve(size, train_mean, validation_mean):\n")
for s, ts, vs in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(int(s), round(ts,3), round(vs,3))

## Plotting the learning curve----
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)
plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, 'o-', color='tab:blue', label='Training score')
plt.plot(train_sizes, val_mean, 'o-', color='tab:orange', label='Validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='tab:blue')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='tab:orange')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve - RandomForest')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.show()

print("\nF1_score:\n", f1_score(y_test, ypred, average='macro'))
print("\nRecall_score:\n", recall_score(y_test, ypred, average='macro'))
print("\nPrecision_score:\n", precision_score(y_test, ypred, average='macro'))
print("\nReport:\n", classification_report(y_test, ypred))
cm = confusion_matrix(y_test, ypred)
print("\nConfusion Matrix:\n", cm)
sns.heatmap(cm, cmap='cool', fmt='d', annot=True,
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['High', 'Medium', 'Low'])
plt.title("Confusion_Matrix for Layoff_Risk")
plt.show()
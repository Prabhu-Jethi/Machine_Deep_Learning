import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix


df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\ai-impact-jobs-layoff-risk-dataset.csv")
print("Dataframe:\n", df)

for i in df.columns:
    print(i, ':', df[i].unique())


print("Check for Null:\n", df.isnull().sum())
# print(df.dropna())


le = LabelEncoder()

drop_cols = ['Job_Role']
df.drop(drop_cols, axis=1, inplace=True)

oth_cols = ['Education_Level', 'Industry', 'Company_Size', 'Job_Level', 'AI_Adoption_Level']
for col in oth_cols:
    df[col] = le.fit_transform(df[col])

Layoff_chances = {'Low':0, 'Medium':1, 'High':2}
df['Layoff_Risk'] = df['Layoff_Risk'].map(Layoff_chances)

print("Encoded data:\n", df)



X = df.drop('Layoff_Risk', axis=1)
y = df['Layoff_Risk']
print("Input Columns:\n", X)
print("Target Columns:\n", y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)


# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print("\nScaled_Training_set:\n", X_train, "\nScaled_Testing_set:\n", X_test)


tree = DecisionTreeClassifier(
    criterion='entropy',
    splitter='best',
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42
)
tree.fit(X_train, y_train)
print("\nDecision Tree Classifier:\n", tree)
print("\nTree params:\n", tree.get_params())

######## Checking training accuracy
train_acc = tree.score(X_train, y_train)
print("\nTraining Accuracy:\n", train_acc)

########## Cross validating
cv_scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCV accuracy: %.4f ± %.4f" % (cv_scores.mean(), cv_scores.std()))

train_sizes, train_scores, val_scores = learning_curve(tree, X, y, cv=5, train_sizes=[0.1,0.25,0.5,0.75,1.0], n_jobs=-1)
print("\nLearning curve (size, train_mean, val_mean):\n")
for s, ts, vs in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(int(s), round(ts,3), round(vs,3))

param_grid = {
    'max_depth': [4,6,8,10,None],
    'min_samples_split': [2,5,8,12],
    'min_samples_leaf': [1,2,4,6],
    'ccp_alpha': [0.0, 0.001, 0.01]
}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
gs.fit(X, y)
print("\nBest params (grid):", gs.best_params_)
print("\nBest CV f1_macro:", gs.best_score_)

####### Predicting Testing data
y_pred = tree.predict(X_test)
print("\nPredicted X_test:\n", y_pred)

print("\nTesting_Accuracy:", tree.score(X_test, y_test))
print("\nF1_Score:", f1_score(y_test, y_pred, average='macro'))
print("\nRecall_Score:", recall_score(y_test, y_pred, average='macro'))
print("\nPrecsion_Score:", precision_score(y_test, y_pred, average='macro'))
print("\nClassification_Report\n:", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap='hot', fmt='d', annot=True,
            xticklabels=['Layoffs_High', 'Layoffs_Mid', 'Layoffs_Low'],
            yticklabels=['Layoffs_Mid', 'Layoffs_Low', 'Layoffs_High'])
plt.show()
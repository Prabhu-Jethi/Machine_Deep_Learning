import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import recall_score, f1_score, accuracy_score, roc_auc_score, precision_score, confusion_matrix


##### Collect
df = pd.read_csv(r"C:\Users\sudip\Downloads\DataSets\adult\adult.data", header=None, na_values=['?', ' ?'], skipinitialspace=True)
print("\nAdult Dataframe:\n", df)

df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income']
print("\nAdded columns:\n", df)
print("\nData-Types:\n", df.dtypes)


##### Clean
df.dropna(inplace=True)
print("\nNull_values:\n", df.isnull().sum())

for i in df.columns:
    print(i, ':', sum(df[i] == '?'), '\n')

for i in df.columns:
    print(i, ':', df[i].unique(), '\n')


###### Encoding str to int 
## drop noise cols
drop_cols = ['education', 'occupation', 'native_country']
df.drop(drop_cols, axis=1, inplace=True)
print("\nDropped Columns:\n", drop_cols)

income_col = {'<=50K': 0, '>50K': 1}
df['income'] = df['income'].map(income_col)
print("\nMapped_Target_Column:\n", income_col)

df = pd.get_dummies(
    df,
    columns=[
        'workclass',
        'marital_status',
        'relationship',
        'race',
        'gender'
    ],
    drop_first=True
)
## encode categorized columns
cat_cols = df.select_dtypes(include='object').columns


###### Correlation
co_relation = df.corr()
print("\nPearson_Co-relation_among_X_and_y:\n", co_relation)


####### Split Independent and dependent variables
X = df.drop('income', axis=1)
y = df['income']
print("\nIndependent_Column:\n", X)
print("\nTarget_Column:\n", y)


###### Train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("\nX_train_:\n", X_train, "\nX_test_:\n", X_test)
print("\ny_train_:\n", y_train, "\ny_test_:\n", y_test)


####### ****** Instead of Using Standardization and Model implementation separately, I've used it in a module named 'Pipeline'
####### and included both of them as a single parameters. ******

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(
        kernel='rbf',
        C=1,
        class_weight='balanced',
        gamma='scale',
        probability=True
    ))
])
pipe.fit(X_train, y_train)
print("\nSVM_model:\n", pipe)

### Train vs Test accuracy:
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)
print("\nTraining_accuracy:\n", train_acc)
print("\nTesting_accuracy:\n", test_acc)


##### Cross validation
cv_score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy', n_jobs=-1)
print("\nCross-Validation-Score: %.4f +- %.4f" % (cv_score.mean(), cv_score.std()))


#### learning curve 
train_size, train_score, validation_score = learning_curve(pipe, X, y, cv=4, train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], n_jobs=-1)
print("\nLearning_Curve(size, train_score, validation_score):\n")
for s, ts, vs in zip(train_size, train_score.mean(axis=1), validation_score.mean(axis=1)):
    print(int(s), round(ts, 3), round(vs, 3))


##### Grid search CV
param_grid = {
    'svm__C':[0.1,1,10],
    'svm__gamma':[0.01,'scale'],
    'svm__kernel':['rbf']
}
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=4,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)
print("\nBest Parameters:\n")
print(grid.best_params_)

print("\nBest Cross Validation F1 Score:\n")
print(grid.best_score_)

# Best Model
best_svm = grid.best_estimator_

## Roc-auc calculation
proba = pipe.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, proba)


##### Prediction
y_pred = pipe.predict(X_test)
print("\nPrediction:\n", y_pred)


###### Evaluation
print("\nROC_AUC:\n", roc_auc)
print("\nAccuracy_score:\n", accuracy_score(y_test, y_pred))
print("\nF1_score:\n", f1_score(y_test, y_pred))
print("\nPrecision_score:\n", precision_score(y_test, y_pred))
print("\nRecall_score:\n", recall_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion_matrix:\n", cm)
sns.heatmap(cm, cmap='viridis', fmt='d', annot=True, xticklabels=['<=50K', '>50K'], yticklabels=['>50K', '<=50K'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
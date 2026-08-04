import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from scikeras.wrappers import KerasClassifier


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


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.25)
print("\nTraining_and_Testing Samples:\n")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
print("\nScaled_Features:\n", X_train_scaled, X_test_scaled)

#### Model
def create_model(learning_rate=0.05, neurons=32):

    ## Model creation
    model = Sequential()
    ## Ip
    model.add(Dense(neurons, activation='relu', input_shape=(21,))),
    ## Hidden
    model.add(Dense(16, activation='relu')),
    model.add(Dense(16, activation='relu')),
    ## Op
    model.add(Dense(1, activation='sigmoid'))

    ## Model compilation
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


### Model Wrapping using Scikeras Classifier
model_wrapper = KerasClassifier(
    model=create_model,
    epochs=30,
    batch_size=16,
    verbose=2
)
print(model_wrapper)

### Grid hyperparameters
params_grid = {
    "model__learning_rate": [0.001, 0.01, 0.05], # 3
    "model__neurons": [16, 32],                  # 2
    "batch_size": [16],                          # 1
    "epochs": [30]                               # 1
}                                               # total = 3 * 2 * 1 * 1 = 6 models * 5 cv = 30 neural networks * 30 epochs = 900 epochs

## K-fold
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

## Run gridsearch cv
grid = GridSearchCV(
    estimator=model_wrapper,
    param_grid=params_grid,
    cv=cv_strategy,
    scoring='accuracy',
    verbose=2
)
grid_result = grid.fit(X_train_scaled, y_train)
print("\nGrid-Search:\n", grid_result)
print("\nBest_Accuracy:\n", grid.best_score_)   #Accuracy --> 0.8459
print("\nBest_Parameter:\n", grid.best_params_) #Parameter --> {'batch_size': 16, 'epochs': 30, 'model__learning_rate': 0.001, 'model__neurons': 16}


### Evaluating
# predict probabilities
y_pred = grid.predict(X_test_scaled)

# convert probabilities to class labels
y_prob = grid.predict_proba(X_test_scaled)[:, 1]

print("\nROC_AUC:\n", roc_auc_score(y_test, y_prob))
print("\nAccuracy_Score:\n", accuracy_score(y_test, y_pred))
print("\nRecall_Score:\n", recall_score(y_test, y_pred))
print("\nPrecision_score:\n", precision_score(y_test, y_pred))
print("\nF1 Score:\n", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion_Matrix:\n", cm)
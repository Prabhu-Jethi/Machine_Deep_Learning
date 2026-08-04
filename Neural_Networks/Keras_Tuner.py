import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

import keras_tuner as kt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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

### Model creation, adding layers and Compilation
def create_model(hp):
    model = Sequential()

    model.add(Dense(units=hp.Int("units", min_value=16, max_value=128, step=16),
                    activation='relu',
                    input_shape=(X_train_scaled.shape[1],)
                ))
    model.add(BatchNormalization())     # Stabilizes activation function
    model.add(Dropout(0.3))             # Regularization -> Turns off some neurons
    model.add(Dense(units=hp.Int("hidden-units", 8, 64, 8),
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=hp.Int("hidden-units", 8, 64, 8),
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[0.001, 0.01, 0.05])),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )

    return model

### EarlyStopping Callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
### ModelCheckpoint Callback
checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
### Create Keras_Tuner
tuner = kt.RandomSearch(
    hypermodel=create_model,
    objective='val_accuracy',
    directory='keras_tuner',
    max_trials=10,
    project_name='adult_income'
)
## Search: KerasTuner trains multiple models automatically
tuner.search(
    X_train_scaled,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=2
)
### Best Hyperparameter
best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBest Hyperparameter:\n", best_hp)

### Best Model
best_model = tuner.get_best_models(1)[0]
print("\nBest Models:\n", best_model)

### Train best model again
history = best_model.fit(
    X_train_scaled,
    y_train,
    epochs=30,
    validation_split=0.2,
    batch_size=16,
    callbacks=[early_stop, checkpoint]
)

##Evaluate
loss, accuracy = best_model.evaluate(
    X_test_scaled,
    y_test
)
print("\nLoss:\n", loss)
print("\nAccuracy:\n", accuracy)

## Probability
y_prob = best_model.predict(X_test_scaled)
## Prediction
y_pred = np.where(y_prob > 0.5, 1, 0).ravel()

print("\nROC_AUC:\n", roc_auc_score(y_test, y_prob))
print("\nAccuracy_Score:\n", accuracy_score(y_test, y_pred))
print("\nRecall_Score:\n", recall_score(y_test, y_pred))
print("\nPrecision_score:\n", precision_score(y_test, y_pred))
print("\nF1 Score:\n", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion_Matrix:\n", cm)
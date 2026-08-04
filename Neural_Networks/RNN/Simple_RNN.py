import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Embedding, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

### input dimension
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Training_Samples:", len(X_train))
print("\nTesting_Samples:", len(X_test))


print("\nX_train:\n", X_train[0])
print("\ny_train:\n", y_train[0])

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


##### Pad Sequence: Each review has different lengths. for example- Review 1: 85 lengths, Review 2: 100 lengths
### Neural network requires equal-length inputs
max_length = 200

X_train = pad_sequences(
    X_train,
    maxlen=max_length,
    padding='post',
    truncating='post'
)
X_test = pad_sequences(
    X_test,
    maxlen=max_length,
    padding='post',
    truncating='post'
)
print("\nPadded_Shape_of_X_train:\n", X_train.shape)

print("\nUnique_labels:\n", np.unique(y_train))
print("\nNo._of_labels:\n", np.bincount(y_train))
print("\nNo._of_labels:\n", np.bincount(y_test))

print("\nMax_value_of_X-train:\n", max(max(x) for x in X_train))
print("\nMax_value_of_X-test:\n", max(max(x) for x in X_test))


#### RNN Model
def build_model():

    model = Sequential([
        Input(shape=(max_length,)),
        ### Converting integers to dense vectors
        Embedding(
            input_dim=vocab_size,
            output_dim=32
        ),
        SimpleRNN(
            units=64,
            activation='tanh',
            dropout=0.2,
            kernel_regularizer=l2(1e-4)
        ),
        Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

model = build_model()

early = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2
)

### Train model RNN
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early, reduce_lr],
    verbose=2
)

loss, accuracy = model.evaluate(X_test, y_test)

print("\nLoss:\n", loss)
print("\nAccuracy:\n", accuracy)

## accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

## loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

print(history.history)

predictions = model.predict(X_test[:20])
predicted_labels = (predictions > 0.5).astype(int)

actual = y_test[:20]
predicted = predicted_labels[:20].flatten()

for i in range(10):
    print(f"Actual: {actual[i]}   Predicted: {predicted[i]}")


'''Why overfitting happens so quickly in SimpleRNN

SimpleRNN has a limited memory mechanism.

It often:
1. Memorizes the training reviews.
2. Cannot retain long-term dependencies.
3. Generalizes poorly on long documents like IMDB reviews.

That's why LSTM and GRU were developed.'''
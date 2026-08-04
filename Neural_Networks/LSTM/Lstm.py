import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2


''' Sample Dataset on TEXT GENERATION '''
corpus = [
    "deep learning is amazing",
    "deep learning is powerful",
    "artificial intelligence is the future",
    "machine learning is fun",
    "deep neural networks learn patterns"
]
print(corpus)

#### Using Tokenizers, Convert words into tokens (integers)
token = Tokenizer()
token.fit_on_texts(corpus)
total_words = len(token.word_index) + 1
print("\nTotal_Words:", total_words)
print("\nTokenized Words:\n", token.word_index)

### Creating training sequence
'''Deep
Deep learning
Deep learning is
Deep learning is amazing'''

input_sequences = []
for line in corpus:
    token_list = token.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)
    print("\n", input_sequences)

### Padding
max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_len,
    padding="pre"
)
print("\nPadded_Inputs:\n", input_sequences)


### Split inputs and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = to_categorical(y, num_classes=total_words)

print("\nX-Shape:", X.shape)
print("\ny-Shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#### Model 
def build_model():
    model = Sequential([
        Input(shape=(max_len, )),
        Embedding(
            input_dim=total_words,
            output_dim=128,
            input_length=max_len-1
        ),
        LSTM(
            units=256,
            return_sequences=True,
            dropout=0.2
        ),
        LSTM(
            units=128,
            dropout=0.2
        ),
        Dense(128, activation='relu'),
        Dense(total_words, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

model = build_model()

print("\nOutput shape:\n", model.output_shape)

### Callbacks
early = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.5,
    min_lr=1e-6
)

### Model Training
history = model.fit(
    X_train,
    y_train,
    callbacks=[early, reduce_lr],
    epochs=500,
    verbose=2
)

### Generating Text
seed_text = "artificial"
next_words = 4
for _ in range(next_words):
    token_list = token.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len-1,
        padding="pre"
    )
    predicted = np.argmax(
        model.predict(token_list, verbose=0),
        axis=-1
    )[0]
    output_word = ""
    for word, index in token.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print("\n", seed_text)
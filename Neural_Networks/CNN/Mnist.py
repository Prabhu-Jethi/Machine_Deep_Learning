import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import random
np.random.seed(0)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

''''The MNIST dataset contains handwritten digits from 0 to 9.

-> Training images: 60,000
-> Testing images: 10,000
-> Image size: 28 × 28 pixels
-> Color channels: 1 (Grayscale)
-> Classes: 10

The goal is to build a CNN that can classify handwritten digits.'''

##### Loading MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Training Images :", X_train.shape)
print("Training Labels :", y_train.shape)

print("Testing Images :", X_test.shape)
print("Testing Labels :", y_test.shape)


##### Display sample images
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')
plt.show()

### counts samples of each class 0-9
for i in range(0, 10):
  print("\n", i, ':', len(X_train[y_train==i]))  ### creates boolean mask --> i = 7 --> false, false, true, false, true......... and select only images of 7


for i in range(0, 10):
  images = X_train[y_train == i]
  idx = np.random.randint(len(images))  ### Displays one random image from each class 0-9
  plt.imshow(images[idx], cmap='gray')
  plt.title(f"Digit {i}")
  plt.axis('off')
  plt.show()

#### Normalize images of (0-255) pixels to (0-1)
### 255 -> 1, 128 -> 0.50, 64 -> 0.25
X_train = X_train / 255.0
X_test = X_test / 255.0
print("\nNormalized Images:\n")
print(X_train)
print(X_test)


#### Re-Shape Images
## CNN needs 4D input: (60000, 28, 28) ----> (60000, 28, 28, 1)
## Reshaping includes no_of_images, height, width, channels
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print("\nRe-Shaped X_train:\n", X_train.shape)
print("\nRe-Shaped X_test:\n", X_test.shape)


#### One-Hot Encode labels
## 5 -> [0,0,0,0,0,1,0,0,0,0] , 2 -> [0,0,1,0,0,0,0,0,0,0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("\nEncoded Y values:\n")
print(y_train)
print(y_test)


#### Building model
def build_model(learning_rate=0.001):

    model = Sequential([

        ## 1st convolution layer
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(28, 28, 1)     ### filter = 32, size = (3 x 3) -> (28 - 3 + 1) --> 26 --> output shape: 26 x 26 x 32
        ),                            

        ## 1st max pooling layer
        MaxPooling2D(
            pool_size=(2, 2)        ### (28//2 = 13) --> o/p shape: 13 x 13 x 32 (Max pooling selects maximum from a pool so it get halved)
        ),
        
        ## 2nd convolution
        Conv2D(           
            filters=64,
            kernel_size=(3, 3),     ### 13 - 3 + 1 --> 11 (shape: 11 x 11 x 64)
            activation='relu'
        ),

        ## 2nd Max pooling
        MaxPooling2D(     ### 11//2 = 5 --> op shape: 5 x 5 x 64
            pool_size=(2, 2)
        ),

        ## 3rd convolution 
        Conv2D(
            filters=128,
            kernel_size=(3,3),      ### 5 - 3 + 1 ---> 3 (shape: 3 x 3 x 128)
            activation='relu'
        ),

        ### Flatten Layer
        Flatten(),   ### Converts 5 x 5 x 64 => 1600

        ### Dense Layer
        Dense(128, activation='relu'),    ## learns combination of extracted features

        ### Dropout
        Dropout(0.3),     ## Disables 30% neurons

        ### Output Layer
        Dense(10, activation='softmax')      ## Highest probability = prediction
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    return model

model = build_model()

#### Train model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=2
)

### Learning curve
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train','Validation'])
plt.show()

#### Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)
print("\nTest Loss:\n", loss)
print("\nTest Accuracy:\n", accuracy)


#### Predictions
predictions = model.predict(X_test)

index = np.random.randint(len(X_test))
predicted_digit = np.argmax(predictions[index])
actual_digit = np.argmax(y_test[index])
print("Predicted:", predicted_digit)
print("Actual:", actual_digit)

#### Display Predictions
plt.imshow(X_test[0].reshape(28,28), cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis("off")
plt.show()
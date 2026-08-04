import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast
from tensorflow.keras.regularizers import l2

'''CIFAR-10 is a benchmark dataset for image classification created by the Canadian Institute for Advanced Research (CIFAR).

It contains 60,000 color images divided into 10 classes.

> Training Images: 50,000
> Testing Images: 10,000
> Image Size: 32 × 32 pixels
> Channels: 3 (RGB)
> Classes: 10
> Images per class: 6,000

Unlike MNIST (handwritten digits), CIFAR-10 contains real-world objects.'''

#### Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("\n CIFAR 10 Data:\n")
print("Training Images :", X_train.shape)
print("Training Labels :", y_train.shape)

print("Testing Images :", X_test.shape)
print("Testing Labels :", y_test.shape)

print("\n2D y_train:\n", y_train[:5])  ## 2D array

y_train = y_train.reshape(-1,)
print("\nFlatten y_train:\n", y_train[:5]) ## Convert 2D array to 1D array or (Flatten it)

y_test = y_test.reshape(-1,)
print("\ny_test Flatten:\n", y_test)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#### Display images
def plot_images(X, y, index):
    plt.figure(figsize=(8, 8))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
    plt.show()
plot_images(X_train, y_train, 2)
plot_images(X_train, y_train, 4)

'''Normalize the images to a number from 0 to 1. 
Image has 3 channels (R,G,B) and each value in the channel can range from 0 to 255. 
Hence to normalize in 0-->1 range, we need to divide it by 255'''

#### Normalize
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0
print("\nX_train_normalized:\n", X_train_norm)
print("\nX_test_normalized:\n", X_test_norm)

#### Convert labels to categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
print("\ny_train_categorical:\n", y_train_cat)
print("\ny_test_categorical:\n", y_test_cat)

''' Data Augmenting '''
'''This often increases validation accuracy substantially because the model sees varied versions of the same images.
Like rotated, flipped, zoomed, contrasted images'''

###### METHOD 1: Using ImageDataGenerator (Older API)
''' Train_ac: 69%, loss: 92%
    Validation_ac: 69%, loss: 94%
    Test ac: 74%
    Test loss: 76%
    (Underfit network, generalization to be done for better results. Like re-training or adjusting parameters)'''
# datagen = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     zoom_range=0.1
# )
# datagen.fit(X_train_norm)


####### METHOD 2: Data Augmentation
''' Training accuracy = 70%, loss = 87%
    Validation accuracy = 70%. val_loss = 85%
    Test accuracy = 73%
    Test loss = 79% 
    (overall, a good generalized network with minimal gap between training and validation curves)'''
data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.05),
    RandomZoom(0.05),
    RandomTranslation(0.05, 0.05),
    RandomContrast(0.05)
])
###### Normal CNN Architecture
def build_model(learning_rate=0.001):

    data_augmentation,

    model = Sequential([
        ## ip
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            input_shape=(32, 32, 3),
            padding='same',
            kernel_regularizer=l2(1e-4),
            activation='relu'
        ),
        BatchNormalization(),
        MaxPooling2D(),

        ## hidden
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            kernel_regularizer=l2(1e-4),
            activation='relu'
        ),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            kernel_regularizer=l2(1e-4),
            activation='relu'
        ),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.4),
        
        Flatten(),
        # GlobalAveragePooling2D(),   ## computes avg of entire feature map, few parameters
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        ## op
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

model = build_model()

#### Callbacks
early = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    filepath='best_cifar10_model.keras',
    monitor='val_accuracy',
    save_best_only=True
)

#### Model training
history = model.fit(
    X_train_norm,
    y_train_cat,
    batch_size=32,
    validation_data=(X_test_norm, y_test_cat),
    callbacks=[early, checkpoint],
    epochs=30,
    verbose=2
)

### Learning curve
plt.title("\nLearning_Curve\n")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.show()

#### Evaluate
loss, accuracy = model.evaluate(
    X_test_norm,
    y_test_cat
)
print("\nTest Loss:\n", loss)
print("\nTest Accuracy:\n", accuracy)

#### Predict
predictions = model.predict(X_test_norm)

index = np.random.randint(len(X_test_norm))
predicted = np.argmax(predictions[index])
actual = np.argmax(y_test_cat[index])
print("Predicted:", predicted)
print("Actual:", actual)



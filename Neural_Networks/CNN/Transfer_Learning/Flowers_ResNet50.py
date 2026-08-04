import os
import warnings
warnings.filterwarnings('ignore')
import pathlib
import cv2
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

'''This is a 'Flower-Photos' dataset containing 3670 images of various flowers.
    Total class -> 5
    ('Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips')
    Type -> RGB Images
'''

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file(
    fname='flower_photos',
    origin=dataset_url,
    cache_dir='.',
    untar=True
)
print(data_dir)
data_dir = pathlib.Path(data_dir)
print(data_dir)

data_dir = data_dir / "flower_photos"
print("\nList_of_Classes:\n", os.listdir(data_dir))

image_count = len(list(data_dir.glob("*/*.jpg")))
print("\nTotal_images:", image_count)

# ##### Displaying sample images
# daisy = list(data_dir.glob("daisy/*"))[36]
# img = Image.open(daisy)
# daisy_count = len(list(data_dir.glob("daisy/*")))
# print("\nDaisy_Counts:", daisy_count)
# plt.imshow(img)
# plt.title("\nDaisy\n")
# plt.axis('off')
# plt.show()

# roses = list(data_dir.glob('roses/*'))[9]
# img = Image.open(roses)
# roses_count = len(list(data_dir.glob("roses/*")))
# print("Roses_Counts:", roses_count)
# plt.imshow(img)
# plt.title('\nRoses\n')
# plt.axis('off')
# plt.show()

# dandelion = list(data_dir.glob('dandelion/*'))[12]
# img = Image.open(dandelion)
# dandelion_count = len(list(data_dir.glob("dandelion/*")))
# print("Dandelion_Counts:", dandelion_count)
# plt.imshow(img)
# plt.title('\nDandelion\n')
# plt.axis('off')
# plt.show()

# sunflowers = list(data_dir.glob('sunflowers/*'))[2]
# img = Image.open(sunflowers)
# sunflowers_count = len(list(data_dir.glob("sunflowers/*")))
# print("Sunflowers_Counts:", sunflowers_count)
# plt.imshow(img)
# plt.title('\nSunflowers\n')
# plt.axis('off')
# plt.show()

# tulips = list(data_dir.glob('tulips/*'))[25]
# img = Image.open(tulips)
# tulips_count = len(list(data_dir.glob("tulips/*")))
# print("Tulips_Counts:", tulips_count)
# plt.imshow(img)
# plt.title('\nTulips\n')
# plt.axis('off')
# plt.show()

#### Read images into numpy using computer vision
flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'tulips': list(data_dir.glob('tulips/*'))
}

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4
}
flowers_images_dict['roses'][5]

classes = list(flowers_labels_dict.keys())
counts = [len(flowers_images_dict[c]) for c in classes]
plt.figure(figsize=(8,5))
plt.bar(classes, counts)
plt.title("Flower Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.show()

img = cv2.imread(str(flowers_images_dict['roses'][0]))
print("\nOriginal_image:", img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(img, (160,160))
print("\nResized_image:", resized_img.shape)


X, y = [], []
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (160,160))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

#### Delete org dataset
del X
del y
gc.collect()

X_train = np.array(X_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

##### Normalize
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

print("\nX_train_normalized:\n", X_train)
print("\nX_val_normalized:\n", X_val)
print("\nX_test_normalized:\n", X_test)

#### categorized 'y'
y_train = to_categorical(y_train, 5)
y_val = to_categorical(y_val, 5)
y_test = to_categorical(y_test, 5)
print("\ny_train_categorized:\n", y_train)
print("\ny_val_categorized:\n", y_val)
print("\ny_test_categorized:\n", y_test)

##### Augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomTranslation(0.1, 0.1)
])

plt.figure(figsize=(10,8))
sample = X_train[0]
sample = tf.expand_dims(sample, 0)
for i in range(9):
    plt.subplot(3,3,i+1)
    aug = data_augmentation(sample)
    plt.imshow((aug[0] + 127.5) / 255.0)  # approximate visualization after preprocess_input
    plt.axis("off")
plt.tight_layout()
plt.show()


''' Transfer Learning, In this we don't train model from scratch instead use a pre-trained model. '''
####### Pre-trained model
IMAGE_SHAPE = (160, 160)

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3)
)

##### Phase 1: Feature Extraction (Freeze the backbone). Freeze the model for 4-5 epochs
def phase1_model():
    base_model.trainable = False
    model = Sequential([
        data_augmentation,
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dense(5, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

model = phase1_model()
early = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
plateau = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7
)

history1 = model.fit(
    X_train,
    y_train,
    callbacks=[early, plateau],
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=5,
    verbose=2
)
### Learning curve
plt.title("\nLearning_Curve\n")
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.legend(['Train', 'Validation'])
plt.show()

# tf.keras.backend.clear_session()
# gc.collect()
early = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
plateau = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7
)

####### Phase 2: Fine-Tuning (Unfreeze part of the backbone). Unfreeze model and freeze last 30 layers that are trainable
def phase2_model():
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
    
model = phase2_model()
history2 = model.fit(
    X_train,
    y_train,
    callbacks=[early, plateau],
    batch_size=4,
    validation_data=(X_val, y_val),
    epochs=10,
    initial_epoch=5,
    verbose=2
)

### Learning curve
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
plt.figure(figsize=(8,5))
plt.plot(acc)
plt.plot(val_acc)
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])
plt.show()

loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
plt.figure(figsize=(8,5))
plt.title("\nLearning_Curve_of_phase1_&_phase2\n")
plt.plot(loss)
plt.plot(val_loss)
plt.legend(["Train","Validation"])
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

#### Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test,
)
print("\nTest Loss:\n", loss)
print("\nTest Accuracy:\n", accuracy)

#### Predict
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
true_classes = np.argmax(y_test, axis=1)
class_names = list(flowers_labels_dict.keys())
plt.figure(figsize=(12,10))

for i in range(9):
    plt.subplot(3,3,i+1)
    img = X_test[i]      # Save original images before preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    actual = class_names[true_classes[i]]
    predicted = class_names[pred_classes[i]]
    color = "green" if actual == predicted else "red"
    plt.title(
        f"Actual: {actual}\nPred: {predicted}",
        color=color
    )
    plt.axis("off")
plt.tight_layout()
plt.show()

print(classification_report(
    true_classes,
    pred_classes,
    target_names=class_names
))

#### Cofusion Matrix
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=flowers_labels_dict.keys(),
    yticklabels=flowers_labels_dict.keys()
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
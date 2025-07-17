import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
DATA_DIR = 'data/train' 
CATEGORIES = ['human', 'vehicle']
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30

print(f"\n Using TRAINING data from: {DATA_DIR}")
for category in CATEGORIES:
    folder = os.path.join(DATA_DIR, category)
    if not os.path.exists(folder):
        print(f" ERROR: Folder '{folder}' does not exist. Did you run generate_data.py?")
        exit(1)
    count = len(os.listdir(folder))
    print(f" - {category}: {count} images")
    if count < 500:
        print(f" WARNING: Very few images in {category}. Results may be poor.")

print("\n Loading images...")
data, labels = [], []

for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATA_DIR, category)
    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        data.append(img_array)
        labels.append(label)

data = np.array(data, dtype='float32')
labels = to_categorical(np.array(labels), num_classes=2)

print(f" Loaded {len(data)} images. Human={np.sum(labels[:,0])}, Vehicle={np.sum(labels[:,1])}")

if data.ndim == 3:
    data = np.expand_dims(data, -1)

X_train, X_val, y_train, y_val = train_test_split
(
    data, labels, test_size=0.2, stratify=labels, random_state=SEED
)
print(f"\n Training set: {X_train.shape[0]} images")
print(f" Validation set: {X_val.shape[0]} images")

datagen = ImageDataGenerator
(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.85, 1.15],
)
datagen.fit(X_train)

model = Sequential
([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile
(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n Model Summary:")
model.summary()

early_stop = EarlyStopping
(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\n🚀 Starting training...")
history = model.fit
(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=2
)

print("\n Saving trained model...")
model.save('doppler_classifier.keras')
print("\n Model saved as doppler_classifier.keras")

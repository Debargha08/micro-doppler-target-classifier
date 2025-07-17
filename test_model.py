import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

IMG_SIZE = 128
DATA_DIR = 'data/test' 
CATEGORIES = ['human', 'vehicle']
BATCH_SIZE = 32

print(f"\n Using TEST data from: {DATA_DIR}")

print("\n Loading trained model...")
model = load_model('doppler_classifier.keras')
print(" Model loaded successfully.")

print("\n Loading test images...")
data, labels, paths = [], [], []

for label, category in enumerate(CATEGORIES):
    folder = os.path.join(DATA_DIR, category)
    files = sorted(os.listdir(folder))
    print(f" - {category}: {len(files)} images")
    for filename in files:
        path = os.path.join(folder, filename)
        img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0
        data.append(img_array)
        labels.append(label)
        paths.append(path)

data = np.array(data, dtype='float32')
labels = np.array(labels)

if data.ndim == 3:
    data = np.expand_dims(data, -1)

print(f"\n Total test samples loaded: {len(data)}")

print("\n Making predictions on test set...")
pred_probs = model.predict(data, batch_size=BATCH_SIZE, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)

acc = np.mean(pred_labels == labels)
print(f"\n Overall accuracy on TEST set: {acc*100:.2f}%")

cm = confusion_matrix(labels, pred_labels)
print("\n Confusion Matrix:")
print(cm)

report = classification_report(labels, pred_labels, target_names=CATEGORIES, zero_division=0)
print("\n Classification Report:")
print(report)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\n Showing sample predictions.")
num_samples = 12
plt.figure(figsize=(16, 12))

indices = np.random.choice(len(data), num_samples, replace=False)
for i, idx in enumerate(indices):
    plt.subplot(3, 4, i+1)
    img = data[idx].reshape(IMG_SIZE, IMG_SIZE)
    plt.imshow(img, cmap='gray')
    true_label = CATEGORIES[labels[idx]]
    pred_label = CATEGORIES[pred_labels[idx]]
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\n Test script completed successfully!")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

DATASET1_PATH = 'dataset1'
DATASET2_PATH = 'dataset2'

main_classes = ['bullish', 'bearish', 'neutral']

def load_dataset1(base_path):
    image_paths = []
    labels = []
    for main_class in main_classes:
        class_folder = os.path.join(base_path, main_class)
        if not os.path.isdir(class_folder):
            continue
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_folder, filename))
                labels.append(f"{main_class}_base")
    return image_paths, labels

def load_dataset2(base_path):
    image_paths = []
    labels = []
    for main_class in main_classes:
        main_class_folder = os.path.join(base_path, main_class)
        if not os.path.isdir(main_class_folder):
            continue
        for subtype in os.listdir(main_class_folder):
            subtype_folder = os.path.join(main_class_folder, subtype)
            if not os.path.isdir(subtype_folder):
                continue
            for filename in os.listdir(subtype_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(subtype_folder, filename))
                    labels.append(f"{main_class}_{subtype}")
    return image_paths, labels

print("Loading Dataset1...")
ds1_paths, ds1_labels = load_dataset1(DATASET1_PATH)
print(f"Dataset1 loaded: {len(ds1_paths)} images")

print("Loading Dataset2...")
ds2_paths, ds2_labels = load_dataset2(DATASET2_PATH)
print(f"Dataset2 loaded: {len(ds2_paths)} images")

# Combine both datasets
image_paths = ds1_paths + ds2_paths
combined_labels = ds1_labels + ds2_labels

print(f"Total images combined: {len(image_paths)}")
print(f"Total unique classes: {len(set(combined_labels))}")

# Encode labels
label_encoder = LabelEncoder()
y_labels = label_encoder.fit_transform(combined_labels)
num_classes = len(label_encoder.classes_)

# Preprocess images
def preprocess_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return img

print("Preprocessing images...")
X = np.array([preprocess_image(p) for p in image_paths])
y = tf.keras.utils.to_categorical(y_labels, num_classes)

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_labels
)

# CNN Model
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save model & encoder
model.save('trained_combined_model.h5')
joblib.dump(label_encoder, 'combined_label_encoder.pkl')

print("Training complete. Model and label encoder saved.")

# Evaluate model
print("Calculating full dataset accuracy and classification report...")
predictions = model.predict(X)
pred_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y, axis=1)

# Accuracy
accuracy = np.mean(pred_labels == true_labels)
print(f"✅ Average accuracy on full dataset: {accuracy * 100:.2f}%")

# Classification Report
print("\n📊 Classification Report:")
target_names = label_encoder.classes_
print(classification_report(true_labels, pred_labels, target_names=target_names))

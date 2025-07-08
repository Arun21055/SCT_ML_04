import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set path
datadir = 'leapGestRecog/leapGestRecog'
model_path = "gesture_model.h5"

# Load and preprocess data
def load_data(data_dir, img_size=(64, 64)):
    data = []
    labels = []
    class_names = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            if subfolder not in class_names:
                class_names.append(subfolder)

            for img_name in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, img_size)
                    data.append(img)
                    labels.append(subfolder)

    return np.array(data), np.array(labels), class_names

# Load or train model
if os.path.exists(model_path):
    print("✅ Loading saved model...")
    model = load_model(model_path)
    class_names = ['01_palm','02_l','03_fist','04_fist_moved','05_thumb','06_index','07_ok','08_palm_moved','09_c','10_down']  # Update based on your classes
else:
    print("🔁 No saved model found. Training new model...")
    X, y, class_names = load_data(datadir)
    X = X.astype('float32') / 255.0

    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_index[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                 shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    datagen.fit(X_train)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20,
                        validation_data=(X_test, y_test), callbacks=[early_stop])

    model.save(model_path)
    print(f"💾 Model saved to {model_path}")

# --- Webcam Prediction ---
print("\n🎥 Starting webcam for real-time gesture prediction...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not detected.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    roi = frame[100:300, 100:300]
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    prediction = model.predict(roi_input)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]
    confidence = np.max(prediction)

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f"{predicted_label} ({confidence*100:.1f}%)",
                (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting webcam...")
        break

cap.release()
cv2.destroyAllWindows()

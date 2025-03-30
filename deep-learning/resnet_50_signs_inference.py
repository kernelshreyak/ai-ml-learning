import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

# === CONFIG ===
CAPTURE_INTERVAL = 5  # seconds
MODEL_PATH = '../trained_models_computervision/final_resnet50_signs.keras'
IMG_SIZE = (224, 224)

# Label map for your 6 classes
label_map = {
    0: "Zero", 1: "One", 2: "Two",
    3: "Three", 4: "Four", 5: "Five"
}

# === Load trained model ===
model = load_model(MODEL_PATH)

# === Frame preprocessing ===
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized / 255.0
    rgb = np.stack([normalized] * 3, axis=-1)  # grayscale → RGB
    return np.expand_dims(rgb, axis=0)

# === OpenCV camera capture ===
cap = cv2.VideoCapture(0)
last_capture_time = time.time()

print("🟢 Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show live feed
    cv2.imshow("Live Feed", frame)

    # Capture every N seconds
    if time.time() - last_capture_time >= CAPTURE_INTERVAL:
        preprocessed = preprocess_frame(frame)
        preds = model.predict(preprocessed, verbose=0)
        pred_class = np.argmax(preds[0])
        label = label_map[pred_class]
        confidence = preds[0][pred_class]

        print(f"🧠 Prediction: {label} ({confidence:.2f})")

        last_capture_time = time.time()

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

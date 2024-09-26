from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time

# Load the YOLO TFLite model
tflite_model = YOLO("yolov8n_saved_model/yolov8n_float16.tflite")

# Download the image from the URL
image_url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Convert the image to OpenCV format
image_cv = np.array(image)


start = time.time()
# Run inference
results = tflite_model(image_url)
end = time.time()
print(f"Inference time: {end - start:.2f} seconds")

# Get bounding box data from results
boxes = results[0].boxes.xyxy.cpu().numpy()  # Get xyxy format for bounding boxes
confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs

# Define colors and class labels (You can add actual class names here)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Example colors for different classes
class_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# Draw bounding boxes on the image
for i, box in enumerate(boxes):
    # Extract bounding box coordinates
    x1, y1, x2, y2 = map(int, box)
    
    # Get class ID and confidence score
    class_id = int(class_ids[i])
    confidence = confidences[i]

    # Draw the bounding box
    color = colors[class_id % len(colors)]  # Choose a color based on the class
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)

    # Put class label and confidence score
    label = f"{class_labels[class_id]}: {confidence:.2f}"
    cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the final image with bounding boxes
output_image_path = "output_image_with_boxes.jpg"
cv2.imwrite(output_image_path, image_cv)

print(f"Image saved to {output_image_path}")

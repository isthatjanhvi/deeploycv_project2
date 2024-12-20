import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load the SSD MobileNet V2 model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded successfully!")

# Function to perform object detection
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

# Start webcam feed
cap = cv2.VideoCapture(0)  # 0 is the default webcam; change if using an external webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB (TensorFlow expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    detections = detect_objects(frame_rgb)

    # Draw bounding boxes on the frame
    for i in range(int(detections['detection_scores'][0].shape[0])):
        score = detections['detection_scores'][0][i].numpy()
        if score > 0.5:  # Only consider detections with confidence > 50%
            box = detections['detection_boxes'][0][i].numpy()
            h, w, _ = frame.shape
            y1, x1, y2, x2 = int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)

            # Draw rectangle around detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = detections['detection_classes'][0][i].numpy()
            cv2.putText(frame, f"Object {int(label)}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("SSD MobileNet V2 Object Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Using pre-trained YOLOv5s model


# Function to apply Sobel filter for edge detection
def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in X direction
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in Y direction
    grad_mag = cv2.magnitude(grad_x, grad_y)  # Combine the gradients
    return grad_mag


# Function to perform flag classification based on edge detection and region analysis
def classify_flag(region):
    top_half = region[:region.shape[0] // 2, :]
    bottom_half = region[region.shape[0] // 2:, :]

    top_avg = np.mean(top_half)
    bottom_avg = np.mean(bottom_half)

    # Classification based on intensity
    if top_avg > bottom_avg:
        return "Poland"
    else:
        return "Indonesia"


# Function to detect flags using YOLO and image processing techniques
def detect_indonesian_or_polish_flag(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Use YOLO for object detection
    results = model(image)

    # Get predictions
    predictions = results.pred[0]
    labels = predictions[:, -1].cpu().numpy()  # Object labels
    coords = predictions[:, :-1].cpu().numpy()  # Coordinates for detected objects

    # Loop through the detections
    detected_flag = False
    for label, coord in zip(labels, coords):
        if label == 59:  # YOLO label for flag (based on COCO dataset)
            x1, y1, x2, y2 = map(int, coord)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            detected_flag = True
            print("Flag detected by YOLO!")

    # If flag not detected by YOLO, perform image processing
    if not detected_flag:
        print("No flag detected by YOLO, performing edge detection...")

        # Sobel edge detection
        edges = sobel_edge_detection(image)

        # Thresholding to detect the flag region
        _, thresh = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        # Find contours to extract the flag region
        contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the detected flag region
        flag_region = image[y:y + h, x:x + w]

        # Resize the flag region to a standard size
        flag_region_resized = cv2.resize(flag_region, (100, 60))

        # Classify the flag based on its color distribution
        flag_type = classify_flag(flag_region_resized)
        print(f"Flag detected as: {flag_type}")

        # Draw bounding box around the flag
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Flag Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test with an image path
image_path = 'flag.jpeg'  # Use path to your image
detect_indonesian_or_polish_flag(image_path)

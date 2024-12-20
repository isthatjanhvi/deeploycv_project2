from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('yolov5s.pt')  # Load YOLOv5 model (small version)

# Open video (use 0 for real-time camera or replace 'video.mp4' with your video file)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detects objects
    results = model(frame)

    # Annotate frame with detections
    for result in results:
        boxes = result.boxes  # Access detected boxes
        for box in boxes:
            # Extract box details and draw on frame
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID

            # Draw bounding box and label on the frame
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the video with detections
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

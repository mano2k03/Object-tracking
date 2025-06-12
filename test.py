'''import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Replace with your actual RTSP stream URL from CP Plus camera
rtsp_url = "rtsp://admin:admin@123@192.168.1.32:554/cam/realmonitor?channel=1&subtype=0"

# Connect to the IP camera
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Failed to connect to IP camera.")
    exit()

print("🔍 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # YOLOv8 inference
    results = model(frame)
    result = results[0]

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({cx},{cy})", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.imshow("Object Tracking - IP Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()'''
import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# ✅ Properly encoded RTSP URL
rtsp_url = "rtsp://admin:admin%40123@192.168.1.32:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp_url)

# Wait for the camera to initialize
time.sleep(2)

if not cap.isOpened():
    print("❌ Failed to connect to IP camera.")
    exit()

# Detect flying objects only (bird, airplane)
flying_class_ids = [5, 14]
class_names = model.names

print("🔍 Looking for flying objects (Press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Frame not received. Retrying...")
        continue  # Keep the loop alive

    # Inference
    results = model(frame)
    result = results[0]

    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id in flying_class_ids:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            label = class_names[cls_id]

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} ({cx},{cy})", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.imshow("Flying Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


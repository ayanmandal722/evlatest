import cv2
from ultralytics import YOLO
import time

# Initialize YOLOv11 model (nano for speed, x for accuracy)
model = YOLO("yolo11n.pt")  # or "yolo11x.pt" for higher accuracy

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Optionally, set higher frame size for better accuracy
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 inference for instance segmentation for road scene
    results = model.track(frame, show=False, persist=True, stream=False)

    if results:
        # Each results[0] holds the detection and segmentation
        img_annotated = results[0].plot()
        cv2.imshow("YOLOv11 Road Components, Speed Estimation", img_annotated)
    else:
        cv2.imshow("YOLOv11 Road Components, Speed Estimation", frame)

    # Press 'q' to quit, 's' to save a frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"scene_{int(time.time())}.jpg"
        cv2.imwrite(filename, img_annotated)
        print(f"Frame saved: {filename}")

cap.release()
cv2.destroyAllWindows()

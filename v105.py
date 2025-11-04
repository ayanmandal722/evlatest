`import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
MAX_FPS_YOLO11 = 20       # Switch to YOLOv9 if FPS falls below this
MIN_OBJ_COUNT_YOLO9 = 5   # Switch to YOLOv11 if objects exceed this
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

# Initialize models - Using YOLOv9s (small) which works reliably
model_v9 = YOLO("yolov9s.pt")    # YOLOv9 Small - good balance of speed/accuracy
model_v11 = YOLO("yolo11n.pt")   # YOLOv11 Nano - fastest v11 variant

# Speed estimation tracking
class SpeedTracker:
    def __init__(self):
        self.tracks = {}
        self.max_frames = 30
        self.pixel_to_meter = 0.05  # Calibrate based on camera setup
    def update_track(self, track_id, bbox):
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        current_time = time.time()
        
        if track_id not in self.tracks:
            self.tracks[track_id] = []
        
        self.tracks[track_id].append((center_x, center_y, current_time))
        
        # Keep only recent frames
        if len(self.tracks[track_id]) > self.max_frames:
            self.tracks[track_id] = self.tracks[track_id][-self.max_frames:]
    
    def calculate_speed(self, track_id):
        if track_id not in self.tracks or len(self.tracks[track_id]) < 15:
            return 0
        
        positions = self.tracks[track_id]
        old_pos = positions[0]
        new_pos = positions[-1]
        
        # Calculate pixel distance
        pixel_dist = np.sqrt((new_pos[0] - old_pos[0])**2 + 
                           (new_pos[1] - old_pos[1])**2)
        
        # Calculate time difference
        time_diff = new_pos[2] - old_pos[2]
        
        if time_diff > 0:
            # Convert to real world speed (km/h)
            real_dist = pixel_dist * self.pixel_to_meter
            speed_ms = real_dist / time_diff
            speed_kmh = speed_ms * 3.6
            return max(0, min(200, speed_kmh))  # Clamp between 0-200 km/h
        
        return 0

# Initialize speed tracker
speed_tracker = SpeedTracker()

# Road component colors
COLORS = {
    'person': (0, 255, 255),      # Cyan
    'bicycle': (255, 0, 255),     # Magenta
    'car': (0, 255, 0),           # Green
    'motorcycle': (255, 255, 0),  # Yellow
    'bus': (0, 0, 255),           # Red
    'truck': (255, 0, 0),         # Blue
    'traffic light': (128, 0, 128), # Purple
    'stop sign': (255, 165, 0),   # Orange
}

# Vehicle classes for speed estimation
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

def choose_model(last_fps, obj_count, avg_conf):
    """
    Dynamic model selection based on performance metrics
    """
    # Use YOLOv11 for complex scenes or low confidence
    if obj_count > MIN_OBJ_COUNT_YOLO9 or avg_conf < CONF_THRESHOLD:
        return model_v11, "YOLOv11"
    
    # Use YOLOv9 for simpler scenes or when FPS is acceptable
    if last_fps >= MAX_FPS_YOLO11:
        return model_v9, "YOLOv9s"
    
    # Default to YOLOv9s
    return model_v9, "YOLOv9s"

def process_detections(results, frame):
    """
    Process YOLO results and add speed estimation
    """
    if not results or not results[0].boxes:
        return frame, 0, 1.0
    
    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    
    obj_count = len(confidences)
    avg_conf = np.mean(confidences)
    
    # Get class names
    class_names = [results[0].names[int(cls)] for cls in classes]
    
    # Process tracking data if available
    track_ids = []
    if hasattr(boxes, 'id') and boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)
    
    # Draw detections with speed estimation
    for i, (bbox, conf, cls_name) in enumerate(zip(xyxy, confidences, class_names)):
        if conf < CONF_THRESHOLD:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        color = COLORS.get(cls_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Speed estimation for vehicles
        speed = 0
        if cls_name in VEHICLE_CLASSES and track_ids:
            track_id = track_ids[i] if i < len(track_ids) else i
            speed_tracker.update_track(track_id, bbox)
            speed = speed_tracker.calculate_speed(track_id)
        
        # Create label with speed if available
        label = f"{cls_name}: {conf:.2f}"
        if speed > 0:
            label += f" | {speed:.1f} km/h"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame, obj_count, avg_conf

def detect_road_boundaries(frame):
    """
    Enhanced road boundary detection
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # Focus on lower portion of frame (road area)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(0, height), 
                             (width//2 - 50, height//2), 
                             (width//2 + 50, height//2), 
                             (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough line detection
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, 
                           threshold=100, minLineLength=40, maxLineGap=25)
    
    return lines

def main():
    """
    Main execution loop
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Performance tracking
    last_fps = 30
    last_obj_count = 0
    avg_conf = 1.0
    current_model_name = "YOLOv9s"
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    
    print("🚀 Starting Adaptive YOLO Road Detection System")
    print("📹 Camera initialized")
    print("🤖 Models: YOLOv9s (fast) & YOLOv11n (accurate)")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'r' - Reset speed tracking")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # GET FRAME DIMENSIONS - FIX FOR THE ERROR
        height, width = frame.shape[:2]
        
        start_time = time.time()
        
        # Dynamic model selection
        model, model_name = choose_model(last_fps, last_obj_count, avg_conf)
        
        if model_name != current_model_name:
            print(f"🔄 Switched to {model_name}")
            current_model_name = model_name
        
        # Run inference with tracking
        results = model.track(frame, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD, 
                             persist=True, tracker="bytetrack.yaml")
        
        # Process detections and add speed estimation
        processed_frame, obj_count, avg_conf = process_detections(results, frame)
        
        # Detect road boundaries
        road_lines = detect_road_boundaries(frame)
        if road_lines is not None:
            for line in road_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(processed_frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        
        # Calculate FPS
        inference_time = time.time() - start_time
        current_fps = 1 / inference_time if inference_time > 0 else 0
        
        # Update tracking variables
        last_fps = current_fps
        last_obj_count = obj_count
        
        # Display performance metrics
        metrics_text = [
            f"Model: {current_model_name}",
            f"FPS: {current_fps:.1f}",
            f"Objects: {obj_count}",
            f"Avg Confidence: {avg_conf:.2f}",
            f"Inference: {inference_time*1000:.1f}ms"
        ]
        
        for i, text in enumerate(metrics_text):
            y_pos = 30 + (i * 25)
            cv2.putText(processed_frame, text, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Model status indicator - NOW WITH DEFINED WIDTH
        status_color = (0, 255, 255) if model_name == "YOLOv11" else (255, 255, 0)
        cv2.circle(processed_frame, (width - 30, 30), 15, status_color, -1)
        
        cv2.imshow("Adaptive YOLO Road Detection", processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"road_detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"💾 Saved: {filename}")
        elif key == ord('r'):
            speed_tracker = SpeedTracker()
            print("🔄 Speed tracking reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Application closed")

if __name__ == "__main__":
    main()

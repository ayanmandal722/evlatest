import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
import math

class RoadSegmentationSpeedEstimator:
    def __init__(self):
        # Load YOLOv9 segmentation model
        self.model = YOLO('yolov9c-seg.pt')  # Auto-downloads if not present
        
        # Speed estimation parameters
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.speed_history = defaultdict(lambda: deque(maxlen=10))
        self.meter_per_pixel = 0.05  # Calibrate based on camera height/angle
        self.fps = 30
        
        # Road component class mapping (COCO dataset classes)
        self.road_classes = {
            0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 
            7: 'truck', 1: 'bicycle', 9: 'traffic light',
            11: 'stop sign', 12: 'parking meter'
        }
        
        # Colors for different road components
        self.colors = {
            'car': (0, 255, 0), 'truck': (255, 0, 0), 'bus': (0, 0, 255),
            'motorcycle': (255, 255, 0), 'bicycle': (255, 0, 255),
            'person': (0, 255, 255), 'traffic light': (128, 0, 128),
            'stop sign': (255, 165, 0), 'parking meter': (128, 128, 128)
        }

    def calculate_speed(self, track_id, current_pos):
        """Calculate vehicle speed based on position history"""
        self.track_history[track_id].append((current_pos, time.time()))
        
        if len(self.track_history[track_id]) < 2:
            return 0
        
        # Get positions from 1 second ago (30 frames at 30fps)
        recent_positions = list(self.track_history[track_id])
        if len(recent_positions) >= 30:
            old_pos, old_time = recent_positions[0]
            new_pos, new_time = recent_positions[-1]
            
            # Calculate distance in pixels
            pixel_distance = math.sqrt(
                (new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2
            )
            
            # Convert to real-world distance
            real_distance = pixel_distance * self.meter_per_pixel
            time_diff = new_time - old_time
            
            if time_diff > 0:
                speed_ms = real_distance / time_diff
                speed_kmh = speed_ms * 3.6
                
                # Smooth speed calculation
                self.speed_history[track_id].append(speed_kmh)
                return np.mean(list(self.speed_history[track_id]))
        
        return 0

    def draw_road_components(self, frame, results):
        """Draw segmentation masks and bounding boxes for road components"""
        annotated_frame = frame.copy()
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                class_id = int(box[5])
                confidence = box[4]
                
                if class_id in self.road_classes and confidence > 0.5:
                    class_name = self.road_classes[class_id]
                    color = self.colors.get(class_name, (255, 255, 255))
                    
                    # Draw segmentation mask
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Create colored overlay
                    overlay = annotated_frame.copy()
                    overlay[mask_binary == 1] = color
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, overlay, 0.3, 0)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Calculate speed for vehicles
                    speed = 0
                    if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        track_id = i  # Simple tracking by detection index
                        speed = self.calculate_speed(track_id, (center_x, center_y))
                    
                    # Draw labels
                    label = f"{class_name}: {confidence:.2f}"
                    if speed > 0:
                        label += f" | {speed:.1f} km/h"
                    
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated_frame

    def process_frame(self, frame):
        """Process single frame for road segmentation and speed estimation"""
        # Run YOLOv9 inference
        results = self.model.track(frame, persist=True, classes=list(self.road_classes.keys()))
        
        # Draw road components and speed information
        annotated_frame = self.draw_road_components(frame, results)
        
        # Add road boundary detection (simple lane detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=50)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Draw road boundary lines
                cv2.line(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        return annotated_frame

def main():
    # Initialize the road segmentation system
    road_system = RoadSegmentationSpeedEstimator()
    
    # Open camera (0 for default camera)
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting road segmentation and speed estimation...")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        processed_frame = road_system.process_frame(frame)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('YOLOv9 Road Segmentation & Speed Estimation', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'road_analysis_{int(time.time())}.jpg', processed_frame)
            print("Frame saved!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

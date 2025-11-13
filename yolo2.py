# yolo_tracker.py
from ultralytics import YOLO
from datetime import datetime
import cv2
import logging
import csv
import os
import numpy as np

# === CONFIG ===
DEFAULT_CONFIG = {
    "base_weights": "yolo11n.pt",  # All office objects
    "custom_weights": "C:/Users/Laptopster/extraction_project/tahha_dataset/runs/detect/tahha_only/weights/best.pt",
    "source": 0,
    "save_path": "tahha_demo.mp4",
    "show": True,
    "tahha_conf": 0.5,  # Confidence threshold for Tahha
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class YOLOTracker:
    def __init__(self, config):
        self.config = config
        
        # Load two models
        self.base_model = YOLO(config["base_weights"])
        self.tahha_model = YOLO(config["custom_weights"])
        
        # Base model: keep your 11 office classes
        self.base_model.classes = [
            0, 62, 67, 73, 76, 74, 77, 72, 44, 47, 84
        ]
        
        # Tahha model: only class 0 = tahha
        self.tahha_model.classes = [0]
        
        self.first_seen = {}
        self.last_seen = {}
        self.csv_file = "tracking_log.csv"
        self._setup_csv()

    def _setup_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "class_id", "class_name", "track_id", "event", "x", "y"])

    def _log_to_csv(self, track_id, cls_id, cls_name, event, x, y):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), cls_id, cls_name, track_id, event, x, y])

    def run(self):
        cap = cv2.VideoCapture(self.config["source"])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run both models
            base_results = self.base_model.track(frame, persist=True, verbose=False)
            tahha_results = self.tahha_model.track(frame, persist=True, conf=self.config["tahha_conf"], verbose=False)
            
            annotated_frame = frame.copy()
            tahha_boxes = set()  # Track which boxes are Tahha
            
            # Get Tahha detections
            if tahha_results[0].boxes.id is not None:
                for box in tahha_results[0].boxes.xyxy.cpu().numpy():
                    tahha_boxes.add(tuple(map(int, box)))
            
            # Process base model
            for result in base_results:
                if result.boxes.id is None:
                    continue
                    
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                cls_ids = result.boxes.cls.int().cpu().numpy()
                
                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    box_tuple = (x1, y1, x2, y2)
                    
                    # Check if this box overlaps with Tahha model
                    is_tahha = any(
                        self._iou(box_tuple, t_box) > 0.5
                        for t_box in tahha_boxes
                    )
                    
                    if is_tahha:
                        cls_id = 11
                        cls_name = "tahha"
                        color = (0, 255, 0)  # Green
                    else:
                        cls_name = result.names[cls_id]
                        color = (255, 0, 0)  # Blue
                    
                    # Draw
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{cls_name} #{track_id}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Log ENTER/EXIT
                    if track_id not in self.first_seen:
                        self.first_seen[track_id] = datetime.now()
                        self._log_to_csv(track_id, cls_id, cls_name, "ENTER", center_x, center_y)
                        log.info(f"ENTER: {cls_name} #{track_id}")
                    
                    self.last_seen[track_id] = datetime.now()
            
            # Save video
            if out is None and self.config["save_path"]:
                h, w = annotated_frame.shape[:2]
                out = cv2.VideoWriter(self.config["save_path"], fourcc, 20.0, (w, h))
            if out:
                out.write(annotated_frame)
                
            if self.config["show"]:
                cv2.imshow("YOLO Tracker - Tahha + Office", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # EXIT LOG
        for oid in list(self.last_seen):
            if oid in self.first_seen:
                duration = (self.last_seen[oid] - self.first_seen[oid]).total_seconds()
                log.info(f"EXIT: tahha #{oid} | Duration: {duration:.1f}s" if oid in self.last_seen else "")
                self._log_to_csv(oid, 11, "tahha", "EXIT", 0, 0)
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    def _iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

# === RUN ===
if __name__ == "__main__":
    tracker = YOLOTracker(DEFAULT_CONFIG)
    tracker.run()
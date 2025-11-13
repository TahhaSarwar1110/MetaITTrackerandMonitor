# yolo_tracker.py
from ultralytics import YOLO
from datetime import datetime
import cv2
import logging
import csv
import os

# === CONFIG ===
DEFAULT_CONFIG = {
    "base_weights": "yolo11s.pt",
    "custom_weights": "C:/Users/Laptopster/extraction_project/tahha_dataset/runs/detect/tahha_s_v1/weights/best.pt",
    "source": 0,  # 0 = webcam
    "save_path": "tahha_demo.mp4",
    "show": True,
    "tahha_conf": 0.65,
}

# === LOGGING ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

class YOLOTracker:
    def __init__(self, config):
        self.config = config
        self.base_model = YOLO(config["base_weights"])
        self.tahha_model = YOLO(config["custom_weights"])
        
        # Office classes
        self.base_model.classes = [0, 62, 67, 73, 76, 74, 77, 72, 44, 47, 84]
        self.tahha_model.classes = [0]
        
        self.first_seen = {}
        self.last_seen = {}
        self.csv_file = "tracking_log.csv"
        self._setup_csv()

    def _setup_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "class_id", "class_name", "track_id", "event", "x", "y", "duration_sec"])

    def _log(self, track_id, cls_id, cls_name, event, x, y, duration=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration_str = f"{duration:.1f}" if duration else ""
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, cls_id, cls_name, track_id, event, x, y, duration_str])
        log.info(f"{event}: {cls_name} #{track_id} | {timestamp}")

    def run(self):
        cap = cv2.VideoCapture(self.config["source"])
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        frame_count = 0

        print("Starting tracker... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = datetime.now()

            # Run both models
            base_results = self.base_model.track(frame, persist=True, verbose=False)[0]
            tahha_results = self.tahha_model.track(frame, persist=True, conf=self.config["tahha_conf"], verbose=False)[0]

            annotated_frame = frame.copy()
            tahha_boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) 
                          for b in tahha_results.boxes.xyxy.cpu().numpy()] if tahha_results.boxes else []

            # Process base detections
            if base_results.boxes and base_results.boxes.id is not None:
                boxes = base_results.boxes.xyxy.cpu().numpy()
                track_ids = base_results.boxes.id.int().cpu().numpy()
                cls_ids = base_results.boxes.cls.int().cpu().numpy()

                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    box_tuple = (x1, y1, x2, y2)

                    # Check if overlaps with Tahha model
                    is_tahha = any(self._iou(box_tuple, t) > 0.5 for t in tahha_boxes)

                    if is_tahha:
                        final_cls_id = 11
                        final_cls_name = "tahha"
                        color = (0, 255, 0)
                    else:
                        final_cls_id = int(cls_id)
                        final_cls_name = base_results.names[cls_id]
                        color = (255, 0, 0)

                    # Draw
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{final_cls_name} #{track_id}",
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # ENTER
                    if track_id not in self.first_seen:
                        self.first_seen[track_id] = current_time
                        self._log(track_id, final_cls_id, final_cls_name, "ENTER", center_x, center_y)

                    self.last_seen[track_id] = current_time

            # Save video
            if out is None and self.config["save_path"]:
                h, w = annotated_frame.shape[:2]
                out = cv2.VideoWriter(self.config["save_path"], fourcc, 20.0, (w, h))
                print(f"Recording to {self.config['save_path']}")
            if out:
                out.write(annotated_frame)

            if self.config["show"]:
                cv2.imshow("Tahha + Office Tracker", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # === FINAL EXIT LOGS ===
        for track_id in self.first_seen:
            if track_id in self.last_seen:
                duration = (self.last_seen[track_id] - self.first_seen[track_id]).total_seconds()
                final_cls_name = "tahha" if track_id in [t for t, b in tahha_boxes] else "person"
                self._log(track_id, 11 if final_cls_name == "tahha" else 0, final_cls_name, "EXIT", 0, 0, duration)

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Video saved: {self.config['save_path']}")
        print(f"Log saved: {self.csv_file}")

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
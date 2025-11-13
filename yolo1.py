import os
import csv
import cv2
import time
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO
import argparse

# ============= Configuration =============
DEFAULT_CONFIG = {
    "source": 0,
    "weights": "yolo11n.pt",  
    "tracker_cfg": "bytetrack.yaml",
    "device": "cpu",
    "conf": 0.25,
    "iou": 0.5,
    "show": True,
    "save_path": "",
    "imgsz": 640,
    "half": False,
    "max_display_classes": 10,
    "log_file": "tracking_log.csv",
    "show_trails" : False 
}
# ==================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class RealTimeTrackerPro:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self._initialize_model()
        
        # Detection classes - person + office items
        self.model.classes = [
            0,   # person
            62,  # chair
            67,  # table/desk
            73,  # laptop
            76,  # keyboard
            74,  # mouse
            77,  # cell phone
            72,  # monitor/TV
            44,  # bottle
            47,  # cup
            84,  # book
        ]
        
        # Tracking data
        self.seen_ids = defaultdict(set)
        self.first_seen = {}
        self.last_seen = {}
        self.track_history = defaultdict(list)
        
        # Performance tracking
        self.ema_fps = None
        self.prev_time = time.time()
        self.frame_count = 0
        
        # File handlers
        self.writer = None
        self._init_csv()
        
        log.info("Tracker initialized successfully")

    def _initialize_model(self):
        """Initialize YOLO model with error handling"""
        try:
            log.info(f"Loading model: {self.cfg['weights']}")
            self.model = YOLO(self.cfg['weights'])
            self.model.to(self.cfg['device'])
            
            if self.cfg['half'] and 'cuda' in self.cfg['device']:
                self.model.model.half()
                log.info("FP16 enabled")
                
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise

    def _init_csv(self):
        """Initialize CSV logging"""
        try:
            self.csv_file = open(self.cfg['log_file'], 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                "Timestamp", "Class ID", "Class Name", "Object ID", 
                "Action", "Duration(ms)", "Frame Count"
            ])
            log.info(f"Logging to {self.cfg['log_file']}")
        except Exception as e:
            log.error(f"Failed to initialize CSV: {e}")
            self.csv_writer = None

    def _log_to_csv(self, oid, cid, class_name, action, duration_ms=0):
        """Log event to CSV with timestamp"""
        if not self.csv_writer:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.csv_writer.writerow([
                timestamp, cid, class_name, oid, 
                action, duration_ms, self.frame_count
            ])
            self.csv_file.flush()  # Ensure data is written immediately
        except Exception as e:
            log.error(f"CSV write error: {e}")

    def _update_fps(self):
        """Calculate and update FPS"""
        now = time.time()
        inst_fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        self.ema_fps = inst_fps if self.ema_fps is None else (0.9 * self.ema_fps + 0.1 * inst_fps)
        return self.ema_fps

    def _init_writer(self, frame):
        """Initialize video writer"""
        if not self.cfg["save_path"] or self.writer:
            return
            
        try:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 20  # Conservative FPS for recording
            os.makedirs(os.path.dirname(self.cfg["save_path"]) if os.path.dirname(self.cfg["save_path"]) else ".", exist_ok=True)
            self.writer = cv2.VideoWriter(self.cfg["save_path"], fourcc, fps, (w, h))
            log.info(f"Saving video → {self.cfg['save_path']}")
        except Exception as e:
            log.error(f"Failed to initialize video writer: {e}")

    def _detect_exits(self, current_ids, names):
        """Detect objects that have left the frame"""
        now = datetime.now()
        active_oids = {oid for _, oid in current_ids}
        exited_oids = set(self.first_seen.keys()) - active_oids

        for oid in exited_oids:
            if oid in self.first_seen:
                entry_time = self.first_seen[oid]
                duration = (now - entry_time).total_seconds() * 1000  # Convert to milliseconds
                
                # Find class ID for this object
                class_id = None
                for cid, oids in self.seen_ids.items():
                    if oid in oids:
                        class_id = cid
                        break
                
                if class_id is not None:
                    class_name = names.get(class_id, f"Class_{class_id}")
                    self._log_to_csv(oid, class_id, class_name, "EXIT", duration)
                    log.info(f"EXIT: {class_name} #{oid} | Duration: {duration:.0f}ms")
                    
                    # Clean up tracking data
                    del self.first_seen[oid]
                    del self.last_seen[oid]
                    self.seen_ids[class_id].discard(oid)
                    if oid in self.track_history:
                        del self.track_history[oid]

    def _draw_tracking_info(self, frame, names):
        """Draw tracking information on frame"""
        # FPS counter
        fps = self._update_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Object counts
        y = 65
        cv2.putText(frame, "Active Objects:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

        displayed = 0
        for cid, ids_set in self.seen_ids.items():
            if displayed >= self.cfg["max_display_classes"]:
                break
            if ids_set:  # Only show classes with active objects
                label = names.get(cid, f"Class {cid}")
                cv2.putText(frame, f"{label}: {len(ids_set)}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                y += 20
                displayed += 1

    def _process_detections(self, res, names):
        """Process detection results and update tracking"""
        frame = res.plot()
        now = datetime.now()
        
        current_ids = []
        if hasattr(res.boxes, "id") and res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().tolist()
            clss = res.boxes.cls.int().cpu().tolist()
            confs = res.boxes.conf.float().cpu().tolist()

            for oid, cid, conf in zip(ids, clss, confs):
                self.seen_ids[cid].add(oid)
                current_ids.append((cid, oid))

                # New object detection
                if oid not in self.first_seen:
                    self.first_seen[oid] = now
                    class_name = names.get(cid, f"Class_{cid}")
                    self._log_to_csv(oid, cid, class_name, "ENTER")
                    log.info(f"ENTER: {class_name} #{oid} | Conf: {conf:.2f}")

                self.last_seen[oid] = now

                # Update track history for trail visualization
                if hasattr(res.boxes, "xywh"):
                    box = res.boxes.xywh[ids.index(oid)].cpu()
                    x, y, w, h = box
                    center = (float(x), float(y))
                    self.track_history[oid].append(center)
                    # Keep only last 30 points
                    if len(self.track_history[oid]) > 30:
                        self.track_history[oid].pop(0)

            # Detect exits
            self._detect_exits(current_ids, names)

        return frame

    def run(self):
        """Main tracking loop"""
        log.info("Starting YOLO + ByteTrack with advanced timestamp logging")
        log.info("Press 'q' to quit, 'k' for quick exit")

        try:
            results = self.model.track(
                source=self.cfg["source"],
                tracker=self.cfg["tracker_cfg"],
                stream=True,
                persist=True,
                imgsz=self.cfg["imgsz"],
                device=self.cfg["device"],
                conf=self.cfg["conf"],
                iou=self.cfg["iou"],
                half=self.cfg["half"] and "cuda" in self.cfg["device"],
                verbose=False
            )

            for res in results:
                self.frame_count += 1
                
                # Process detections
                frame = self._process_detections(res, res.names)
                
                # Initialize video writer if needed
                if not self.writer and self.cfg["save_path"]:
                    self._init_writer(frame)
                
                # Draw tracking information
                self._draw_tracking_info(frame, res.names)
                
                # Draw track trails
                for oid, track in self.track_history.items():
                    if len(track) > 1:
                        points = np.array(track, dtype=np.int32)
                        cv2.polylines(frame, [points], isClosed=False, 
                                    color=(0, 255, 255), thickness=2)

                # Display frame
                if self.cfg["show"]:
                    cv2.imshow("YOLO Tracking Pro", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('k'):
                        log.info("User requested exit")
                        break

                # Save frame if recording
                if self.writer:
                    self.writer.write(frame)

        except KeyboardInterrupt:
            log.info("Interrupted by user")
        except Exception as e:
            log.error(f"Tracking error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources"""
        log.info("Cleaning up resources...")
        
        # Close video writer
        if self.writer:
            self.writer.release()
            log.info("Video writer closed")
        
        # Close CSV file
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            log.info("CSV file closed")
        
        # Log final statistics
        total_objects = sum(len(ids) for ids in self.seen_ids.values())
        log.info(f"Tracking completed - Total frames: {self.frame_count}, Unique objects: {total_objects}")
        
        cv2.destroyAllWindows()

# ============= CLI =============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Object Tracking with Timestamping")
    for k, v in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{k}", type=type(v), default=v, help=f"Default: {v}")
    
    args = parser.parse_args()
    config = vars(args)
    
    # Validate source
    if config["source"] == 0:
        # Test camera access
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.error("Cannot access camera. Check if it's being used by another application.")
            exit(1)
        cap.release()
    
    try:
        tracker = RealTimeTrackerPro(config)
        tracker.run()
    except Exception as e:
        log.error(f"Failed to start tracker: {e}")
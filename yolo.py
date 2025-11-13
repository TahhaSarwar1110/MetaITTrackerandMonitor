import cv2
import time
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO
import argparse
from pathlib import Path

# ============= CONFIG =============
DEFAULT_CONFIG = {
    "source": 0,
    "weights": "yolo11n.pt",
    "tracker_cfg": "bytetrack.yaml",
    "device": "cpu",
    "conf": 0.25,
    "iou": 0.5,
    "show": True,
    "save_path": None,
    "imgsz": 640,
    "half": True,
    "max_display_classes": 10,
}
# ==================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class RealTimeTrackerPro:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg["weights"])
        self.model.to(cfg["device"])
        if cfg["half"] and "cuda" in cfg["device"]:
            self.model.model.half()
            log.info("FP16 enabled")
        self.seen_ids = defaultdict(set)
        self.ema_fps = None
        self.prev_time = time.time()
        self.writer = None

    def _update_fps(self):
        now = time.time()
        inst_fps = 1.0 / max(now - self.prev_time, 1e-6)
        self.prev_time = now
        self.ema_fps = inst_fps if self.ema_fps is None else (0.9 * self.ema_fps + 0.1 * inst_fps)
        return self.ema_fps

    def _init_writer(self, frame):
        if not self.cfg["save_path"] or self.writer: return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30  # fallback
        self.writer = cv2.VideoWriter(self.cfg["save_path"], fourcc, fps, (w, h))
        log.info(f"Saving → {self.cfg['save_path']}")

    def run(self):
        log.info("Starting YOLOv12 + ByteTrack (.pt)")

        while True:
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
                    frame = res.plot()  # Annotated frame
                    names = res.names

                    # Init writer on first frame
                    if not self.writer and self.cfg["save_path"]:
                        self._init_writer(frame)

                    # Update IDs
                    if hasattr(res.boxes, "id") and res.boxes.id is not None:
                        ids = res.boxes.id.int().cpu().tolist()
                        clss = res.boxes.cls.int().cpu().tolist()
                        for oid, cid in zip(ids, clss):
                            self.seen_ids[cid].add(oid)

                    # Draw stats
                    fps = self._update_fps()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    y = 65
                    cv2.putText(frame, "Unique IDs:", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 28

                    for i, (cid, ids_set) in enumerate(self.seen_ids.items()):
                        if i >= self.cfg["max_display_classes"]: break
                        label = names.get(cid, f"Class {cid}")
                        cv2.putText(frame, f"{label}: {len(ids_set)}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        y += 22

                    # Output
                    if self.cfg["show"]:
                        cv2.imshow("YOLOv12 + ByteTrack Pro", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    if self.writer:
                        self.writer.write(frame)
                    
                else:
                    continue
                break

            except Exception as e:
                log.error(f"Stream failed: {e}. Reconnecting in 2s...")
                time.sleep(2)
                continue

        # Cleanup
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        log.info("Tracking stopped.")

# ============= CLI =============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k, v in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()

    tracker = RealTimeTrackerPro(vars(args))
    tracker.run()
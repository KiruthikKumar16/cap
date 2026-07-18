"""
YOLOv10 traffic perception pipeline.

Reads a video source (file / camera index / RTSP), runs YOLOv10 detection +
BoT-SORT tracking, and emits per-frame traffic metrics as JSONL:

  * per-region vehicle counts  (region = a calibrated polygon ROI)
  * per-region queue counts     (velocity-based, from tracker IDs)
  * per-region mean waiting time (seconds stationary)
  * per-class vehicle counts    (car / motorcycle / bus / truck / bicycle)
  * FPS, frame index, wall-clock timestamp

Calibration is camera-specific: a 4-point image->ground homography converts
pixel positions to meters (needed for real velocity), and named polygon ROIs
describe where the road lives in this camera's frame. Without a calibration
file the pipeline runs in **pixel mode** (counts are still correct; velocities
are in px/s and clearly flagged via ``units``).

This module is self-contained: no RL / control-layer dependency.
"""

from __future__ import annotations

import argparse
import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from calibration import CalibrationConfig, TrackHistory


# Vehicle classes we count (COCO ids). bicycle(1) included because in
# rickshaw-heavy footage many cycle-rickshaws get detected as bicycle/motorcycle.
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class TrafficMetrics:
    """One record per frame, written to the JSONL output."""
    intersection_id: str
    frame_idx: int
    timestamp: float
    units: str                          # "meters" or "pixels"
    lane_counts: Dict[str, int]         # region -> vehicle count
    lane_queues: Dict[str, int]         # region -> queued-vehicle count
    lane_waiting_times: Dict[str, float]# region -> mean waiting time (s)
    vehicle_counts: Dict[str, int]      # class -> count
    num_objects: int                    # total tracked vehicles this frame
    fps: float


# ---------------------------------------------------------------------------
# Video source with RTSP / camera reconnect robustness
# ---------------------------------------------------------------------------

class VideoLoader:
    """Threaded frame producer. Reconnects on stream drops for live sources."""

    def __init__(self, source, frame_queue: Queue, running,
                 is_live: bool, max_retries: int = 3):
        self.source = source
        self.frame_queue = frame_queue
        self.running = running
        self.is_live = is_live
        self.max_retries = max_retries

    def __call__(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {self.source}")
            self.running[0] = False
            return

        while self.running[0]:
            ret, frame = cap.read()
            if not ret:
                if self.is_live:
                    # Reconnect with backoff for RTSP / IP cameras
                    print("[WARN] Stream dropped; attempting reconnect...")
                    reconnected = False
                    for attempt in range(self.max_retries):
                        time.sleep(2 ** attempt)        # 1, 2, 4 s
                        cap.release()
                        cap = cv2.VideoCapture(self.source)
                        if cap.isOpened() and cap.read()[0]:
                            print(f"[OK] Reconnected (attempt {attempt + 1}).")
                            reconnected = True
                            break
                    if not reconnected:
                        print("[ERROR] Reconnect failed; stopping.")
                        self.running[0] = False
                        break
                else:
                    print("[INFO] End of video stream.")
                    break
            else:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

        cap.release()


# ---------------------------------------------------------------------------
# Calibration UI (interactive)
# ---------------------------------------------------------------------------

class CalibrationUI:
    """
    Click-based calibration on the first frame.

    Stage 1: click 4 road-plane corners (near-left, near-right, far-left,
             far-right) to build the image->ground homography.
    Stage 2: for each named region, click polygon vertices; press 'n' to finish
             the current polygon and advance, 'ESC' to save & exit.
    """

    def __init__(self, frame: np.ndarray, video_stem: str,
                 regions: List[str], road_width_m: float, road_depth_m: float):
        self.frame = frame.copy()
        self.video_stem = video_stem
        self.regions = regions
        self.road_width_m = road_width_m
        self.road_depth_m = road_depth_m
        self.homography_src: List[List[float]] = []
        self.roi_polygons: Dict[str, List[List[float]]] = {}

    def _on_mouse_homo(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.homography_src) < 4:
            self.homography_src.append([float(x), float(y)])
            cv2.circle(self.frame, (x, y), 6, (0, 255, 255), -1)
            if len(self.homography_src) > 1:
                a = self.homography_src[-2]; b = self.homography_src[-1]
                cv2.line(self.frame,
                         (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         (0, 255, 255), 2)

    def _make_poly_mouse(self, name: str, color=(0, 255, 0)):
        pts: List[List[float]] = []
        canvas = self.frame.copy()

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append([float(x), float(y)])
                cv2.circle(canvas, (x, y), 5, color, -1)
                if len(pts) > 1:
                    a = pts[-2]; b = pts[-1]
                    cv2.line(canvas,
                             (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                             color, 2)
                cv2.imshow("calibrate", canvas)
        return pts, canvas, on_mouse

    def run(self) -> CalibrationConfig:
        # Stage 1: 4 homography points
        print("\n[CALIBRATE] Stage 1: click 4 road corners in order:")
        print("            near-left, near-right, far-left, far-right.")
        print("            (bottom-left, bottom-right, top-left, top-right of the road)")
        cv2.namedWindow("calibrate")
        cv2.setMouseCallback("calibrate", self._on_mouse_homo)
        while len(self.homography_src) < 4:
            cv2.imshow("calibrate", self.frame)
            if cv2.waitKey(50) & 0xFF == 27:
                print("[CALIBRATE] Cancelled at homography stage.")
                cv2.destroyWindow("calibrate")
                return CalibrationConfig.template(self.video_stem)
        cv2.destroyWindow("calibrate")

        # Stage 2: ROI polygons
        for name in self.regions:
            print(f"[CALIBRATE] Stage 2: draw polygon for region '{name}'.")
            print(f"            Left-click to add vertices, 'n' to close, 'ESC' to skip.")
            pts, canvas, on_mouse = self._make_poly_mouse(name)
            cv2.namedWindow("calibrate")
            cv2.setMouseCallback("calibrate", on_mouse)
            cv2.imshow("calibrate", canvas)
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('n'):
                    if len(pts) >= 3:
                        self.roi_polygons[name] = pts
                        self.frame = canvas.copy()  # persist onto base frame
                    break
                if key == 27:    # ESC -> skip this region
                    break
            cv2.destroyWindow("calibrate")

        dst = [[0, 0], [self.road_width_m, 0],
               [0, self.road_depth_m], [self.road_width_m, self.road_depth_m]]
        cfg = CalibrationConfig(
            homography_src=self.homography_src,
            homography_dst=dst,
            roi_polygons=self.roi_polygons,
            road_width_m=self.road_width_m,
            road_depth_m=self.road_depth_m,
            intersection_id=self.video_stem,
        )
        return cfg


# ---------------------------------------------------------------------------
# Main perception engine
# ---------------------------------------------------------------------------

class TrafficVisualInference:
    """Perception engine: YOLOv10 + BoT-SORT -> per-frame TrafficMetrics."""

    def __init__(self,
                 model_path: str = "yolov10s.pt",
                 intersection_id: str = "node_1",
                 use_openvino: bool = True,
                 calibration: Optional[CalibrationConfig] = None,
                 queue_velocity_threshold: float = 0.5,
                 queue_min_stationary_sec: float = 1.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # OpenVINO optimization for CPU
        if self.device == "cpu" and use_openvino:
            print("[INFO] CPU detected. Attempting OpenVINO optimization...")
            ov_model_path = model_path.replace(".pt", "_openvino_model")
            if Path(ov_model_path).exists():
                print(f"[INFO] Loading OpenVINO model from {ov_model_path}")
                self.model = YOLO(ov_model_path, task="detect")
            else:
                print(f"[INFO] Exporting {model_path} to OpenVINO...")
                self.model = YOLO(model_path)
                self.model.export(format="openvino", dynamic=True, imgsz=1280)
                self.model = YOLO(ov_model_path, task="detect")
        else:
            self.model = YOLO(model_path)
            if self.device == "cuda":
                print("[INFO] GPU detected. Enabling CUDA optimization...")
                self.model.to(self.device)

        self.intersection_id = intersection_id

        # Calibration (may be None -> pixel mode)
        self.calibration = calibration
        self.units = "meters" if (calibration and calibration.has_homography) else "pixels"
        if self.units == "pixels":
            print("[WARN] No homography calibration -> pixel mode. "
                  "Velocities in px/s; counts still correct.")

        # Per-track state for velocity-based queuing + waiting time
        self.history = TrackHistory(
            queue_velocity_threshold=queue_velocity_threshold,
            queue_min_stationary_sec=queue_min_stationary_sec,
        )
        self.region_names = (calibration.region_names if calibration
                             else ["frame"])

        self.frame_count = 0
        self.running = [False]   # list so the loader thread can mutate it
        self.frame_queue: Queue = Queue(maxsize=5)

        # Anomaly capture rate-limiting
        self._anomaly_last_capture = 0.0
        self._anomaly_streak = 0

    # ----------------- video loading -----------------

    @staticmethod
    def _is_live_source(source: str) -> bool:
        return isinstance(source, str) and (
            source.lower().startswith("rtsp://")
            or source.lower().startswith("http://")
            or source.lower().startswith("https://")
            or source.isdigit()
        )

    def _video_loader(self, source):
        loader = VideoLoader(source, self.frame_queue, self.running,
                             is_live=self._is_live_source(str(source)))
        loader()

    # ----------------- main loop -----------------

    def run_inference(self, source: str, output_path: str = "metrics.jsonl",
                         headless: bool = False, max_frames: Optional[int] = None,
                         imgsz: int = 640, skip_frames: int = 0):
        self.running[0] = True
        loader_thread = threading.Thread(target=self._video_loader, args=(source,))
        loader_thread.daemon = True
        loader_thread.start()

        start_time = time.time()
        print(f"[INFO] Starting inference on '{self.intersection_id}'...")
        print(f"[INFO] Output units: {self.units} | regions: {self.region_names}")
        print(f"[INFO] Writing metrics to {output_path}")
        print(f"[INFO] Image size: {imgsz} | Skip every {skip_frames + 1} frame(s)")
        if headless:
            print("[INFO] Headless mode (no visualization).")

        out_file = open(output_path, "w")
        t0 = time.time()
        frame_counter = 0

        try:
            while self.running[0]:
                if max_frames is not None and self.frame_count >= max_frames:
                    print(f"\n[INFO] Reached --max-frames {max_frames}; stopping.")
                    break
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except Empty:
                    continue

                frame_counter += 1
                # Skip frames if needed
                if skip_frames > 0 and frame_counter % (skip_frames + 1) != 0:
                    continue

                self.frame_count += 1
                # Use video time (frame_count / fps) when available so metrics
                # advance with the video, not wall-clock during offline runs.
                wall_t = time.time() - t0

                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker="botsort.yaml",
                    conf=0.25,  # Slightly higher confidence for faster processing
                    iou=0.5,
                    imgsz=imgsz,
                    classes=list(VEHICLE_CLASSES.keys()),
                    verbose=False,
                    device=self.device,
                )
                result = results[0]
                boxes = result.boxes

                elapsed = time.time() - start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0.0
                metrics = self._analyze_detections(boxes, result, wall_t, fps)

                self._maybe_capture_anomaly(frame, metrics)

                out_file.write(json.dumps(asdict(metrics)) + "\n")
                out_file.flush()

                print(f"ENGINE | f{self.frame_count:>5} | FPS:{fps:5.1f} | "
                      f"obj:{metrics.num_objects:>3} | "
                      f"queues:{metrics.lane_queues} | "
                      f"wt:{ {k: round(v,1) for k,v in metrics.lane_waiting_times.items()} }",
                      end='\r')

                if not headless:
                    annotated = result.plot()
                    self._draw_overlay(annotated)
                    cv2.imshow("Traffic Perception (BoT-SORT + YOLOv10)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running[0] = False
                        break
        finally:
            self.running[0] = False
            out_file.close()
            cv2.destroyAllWindows()
            loader_thread.join(timeout=2.0)

    # ----------------- detection analysis -----------------

    def _analyze_detections(self, boxes, result, wall_t: float,
                            fps: float) -> TrafficMetrics:
        lane_counts = {r: 0 for r in self.region_names}
        lane_queues = {r: 0 for r in self.region_names}
        lane_wait_sum = {r: 0.0 for r in self.region_names}
        vehicle_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
        num_objects = 0

        if boxes is None or len(boxes) == 0:
            self.history.prune(wall_t)
            return TrafficMetrics(
                intersection_id=self.intersection_id,
                frame_idx=self.frame_count,
                timestamp=time.time(),
                units=self.units,
                lane_counts=lane_counts,
                lane_queues=lane_queues,
                lane_waiting_times={r: 0.0 for r in self.region_names},
                vehicle_counts=vehicle_counts,
                num_objects=0,
                fps=fps,
            )

        h, w = result.orig_shape if result.orig_shape else (1, 1)
        queued_wait_by_region: Dict[str, List[float]] = {r: [] for r in self.region_names}

        for box in boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            num_objects += 1
            vehicle_counts[VEHICLE_CLASSES[cls]] += 1

            xywh = box.xywh[0].cpu().numpy()
            x_px, y_px = float(xywh[0]), float(xywh[1])

            # Region assignment (image-space point-in-polygon)
            if self.calibration and self.calibration.roi_polygons:
                region = self.calibration.region_of(x_px, y_px)
            else:
                region = "frame"
            if region is None:
                region = "_outside"
                # still tracked, but not attributed to any region
            if region in lane_counts:
                lane_counts[region] += 1

            # Track-id based velocity / queue
            tid = int(box.id[0]) if box.id is not None else None
            if tid is not None:
                ground = self.calibration.image_to_ground(x_px, y_px) \
                    if self.calibration else np.array([x_px, y_px])
                self.history.update(tid, ground, wall_t)
                if self.history.is_queued(tid, wall_t) and region in lane_queues:
                    lane_queues[region] += 1
                    queued_wait_by_region[region].append(
                        self.history.waiting_time(tid, wall_t))

        self.history.prune(wall_t)

        for r in self.region_names:
            waits = queued_wait_by_region.get(r, [])
            lane_wait_sum[r] = float(np.mean(waits)) if waits else 0.0

        return TrafficMetrics(
            intersection_id=self.intersection_id,
            frame_idx=self.frame_count,
            timestamp=time.time(),
            units=self.units,
            lane_counts=lane_counts,
            lane_queues=lane_queues,
            lane_waiting_times=lane_wait_sum,
            vehicle_counts=vehicle_counts,
            num_objects=num_objects,
            fps=fps,
        )

    # ----------------- visualization + anomaly -----------------

    def _draw_overlay(self, frame: np.ndarray):
        if not self.calibration or not self.calibration.roi_polygons:
            return
        for name, poly in self.calibration.roi_polygons.items():
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255),
                          thickness=2)
            cv2.putText(frame, name, (int(poly[0][0]), int(poly[0][1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _maybe_capture_anomaly(self, frame: np.ndarray, metrics: TrafficMetrics):
        """Throttled active-learning capture on sustained high density."""
        threshold = 15
        streak_required = 5
        cooldown_sec = 30.0

        if metrics.num_objects > threshold:
            self._anomaly_streak += 1
        else:
            self._anomaly_streak = 0
            return

        if self._anomaly_streak < streak_required:
            return
        now = time.time()
        if now - self._anomaly_last_capture < cooldown_sec:
            return

        self._anomaly_last_capture = now
        ts = int(now)
        save_path = f"data/anomalies/capture_{ts}.jpg"
        Path("data/anomalies").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, frame)
        print(f"\n[ACTIVE LEARNING] Captured anomaly frame (obj={metrics.num_objects}): {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_calibration(source: str, output_json: str, model_path: str,
                     use_openvino: bool, regions: List[str],
                     road_width_m: float, road_depth_m: float):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {source} for calibration.")
        return
    # Grab a frame ~2s in so the road isn't empty.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read a frame for calibration.")
        return

    stem = Path(source).stem
    ui = CalibrationUI(frame, stem, regions, road_width_m, road_depth_m)
    cfg = ui.run()
    cfg.save(output_json)
    print(f"\n[OK] Calibration saved to {output_json}")
    print(json.dumps(asdict(cfg), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv10 traffic perception pipeline (velocity-based queues, JSONL output).")
    parser.add_argument("--source", type=str,
                        help="Video file, camera index (e.g. 0), or RTSP url.")
    parser.add_argument("--model", type=str, default="yolov10s.pt",
                        help="YOLO weights (default: yolov10s.pt)")
    parser.add_argument("--output", type=str, default="metrics.jsonl",
                        help="JSONL metrics output path")
    parser.add_argument("--calib", type=str, default=None,
                        help="CalibrationConfig JSON (omit -> pixel mode)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Interactive calibration: click 4 homography pts + ROI polygons, save JSON, exit.")
    parser.add_argument("--regions", type=str, default="approach,queue_zone",
                        help="Comma-separated region names to draw during --calibrate")
    parser.add_argument("--road-width", type=float, default=3.7,
                        help="Real road width in meters (calibration)")
    parser.add_argument("--road-depth", type=float, default=20.0,
                        help="Real road depth in meters (calibration)")
    parser.add_argument("--queue-vel", type=float, default=0.5,
                        help="Velocity threshold (m/s) below which a vehicle is 'stopped'")
    parser.add_argument("--queue-min-stop", type=float, default=1.5,
                        help="Seconds a vehicle must be stopped before counting as queued")
    parser.add_argument("--no-openvino", action="store_true",
                        help="Disable OpenVINO on CPU")
    parser.add_argument("--headless", action="store_true",
                        help="Disable video window")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (useful for verification)")
    parser.add_argument("--intersection-id", type=str, default=None,
                        help="Override intersection_id label (default: source filename)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for inference (default: 640, smaller = faster)")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Process every N-th frame (0 = process all, 1 = every 2nd, etc.)")
    args = parser.parse_args()

    if args.calibrate:
        if not args.source:
            print("--calibrate requires --source.")
            raise SystemExit(1)
        out_json = args.calib or f"{Path(args.source).stem}.calib.json"
        regions = [r.strip() for r in args.regions.split(",") if r.strip()]
        _run_calibration(args.source, out_json, args.model,
                         use_openvino=not args.no_openvino, regions=regions,
                         road_width_m=args.road_width, road_depth_m=args.road_depth)
        raise SystemExit(0)

    if not args.source:
        print("Usage: python yolo_inference.py --source <video|0|rtsp://...> [options]")
        print("       python yolo_inference.py --source <video> --calibrate")
        print("\nOptions:")
        print("  --model PATH         YOLO weights (default yolov10s.pt)")
        print("  --output PATH        JSONL output (default metrics.jsonl)")
        print("  --calib PATH         calibration JSON (omit -> pixel mode)")
        print("  --calibrate          interactive calibration mode")
        print("  --regions A,B        region names for --calibrate (default approach,queue_zone)")
        print("  --queue-vel M/S      stop-velocity threshold (default 0.5)")
        print("  --queue-min-stop S   stationary seconds before 'queued' (default 1.5)")
        print("  --max-frames N       stop after N frames")
        print("  --headless / --no-openvino")
        raise SystemExit(0)

    calibration = CalibrationConfig.load(args.calib) if args.calib else None
    iid = args.intersection_id or Path(args.source).stem
    engine = TrafficVisualInference(
        model_path=args.model,
        intersection_id=iid,
        use_openvino=not args.no_openvino,
        calibration=calibration,
        queue_velocity_threshold=args.queue_vel,
        queue_min_stationary_sec=args.queue_min_stop,
    )
    engine.run_inference(args.source, output_path=args.output,
                         headless=args.headless, max_frames=args.max_frames,
                         imgsz=args.imgsz, skip_frames=args.skip_frames)

"""
Calibration and tracking state for the traffic perception pipeline.

Two responsibilities:

1. ``CalibrationConfig`` -- per-camera geometry: the 4-point image->ground
   homography (pixels -> meters) plus named polygon ROIs that describe where
   the road / stop-line / queue zone live in this specific camera's frame.
   Loaded from / saved to JSON so a camera is calibrated once and reused.

2. ``TrackHistory`` -- per-track-ID state for velocity-based queue detection
   and waiting-time accumulation. A vehicle is "queued" only if it has been
   moving below ``queue_velocity_threshold`` for at least
   ``queue_min_stationary_sec``. Waiting time is the seconds since it first
   became stationary.

This module is deliberately framework-agnostic: it takes (id, ground_pos_m,
timestamp) tuples from whatever detector/tracker feeds it and returns
structured metrics. No YOLO / ultralytics dependency here.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Calibration config
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """
    Per-camera geometry.

    Attributes
    ----------
    homography_src : 4 image points (px) defining the road-plane trapezoid,
        ordered to match ``homography_dst``.
    homography_dst : 4 ground-plane points (meters) the src maps onto. The
        convention used throughout this project is:
            [near-left, near-right, far-left, far-right]  (both src and dst)
        with the rectangle being ``road_width_m`` wide x ``road_depth_m`` deep.
    roi_polygons : named regions, each a list of [x_px, y_px] vertices. A
        detection's image centroid is "in" a region if it lies inside the
        polygon (point-in-polygon test on the raw image, no homography needed).
        Typical names: ``approach`` (whole visible road) and ``queue_zone``
        (band just behind the stop line).
    road_width_m / road_depth_m : real-world size of the calibrated rectangle.
    meters_per_pixel : fallback scale used when no homography is provided
        (rough pixel-mode). ``None`` means "uncalibrated, report pixel units".
    """
    homography_src: List[List[float]] = field(default_factory=list)
    homography_dst: List[List[float]] = field(default_factory=list)
    roi_polygons: Dict[str, List[List[float]]] = field(default_factory=dict)
    road_width_m: float = 3.7
    road_depth_m: float = 20.0
    meters_per_pixel: Optional[float] = None
    intersection_id: str = "node_1"

    # ----- persistence -----

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def template(cls, intersection_id: str = "node_1") -> "CalibrationConfig":
        """An empty config used as a starting point for interactive calibration."""
        return cls(intersection_id=intersection_id)

    # ----- geometry -----

    @property
    def has_homography(self) -> bool:
        return len(self.homography_src) == 4 and len(self.homography_dst) == 4

    @property
    def matrix(self) -> Optional[np.ndarray]:
        """3x3 perspective-transform matrix, or None if uncalibrated."""
        if not self.has_homography:
            return None
        src = np.float32(self.homography_src)
        dst = np.float32(self.homography_dst)
        return cv2.getPerspectiveTransform(src, dst)

    def image_to_ground(self, x_px: float, y_px: float) -> np.ndarray:
        """Project an image point to ground-plane meters [x_m, y_m]."""
        M = self.matrix
        if M is None:
            # Pixel mode: scale by meters_per_pixel if known, else identity.
            if self.meters_per_pixel is not None:
                return np.array([x_px * self.meters_per_pixel,
                                 y_px * self.meters_per_pixel])
            return np.array([x_px, y_px])
        pt = np.array([[[x_px, y_px]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, M)
        return out[0][0]

    def region_of(self, x_px: float, y_px: float) -> Optional[str]:
        """Return the name of the first ROI polygon containing (x_px, y_px)."""
        for name, poly in self.roi_polygons.items():
            if len(poly) < 3:
                continue
            if cv2.pointPolygonTest(np.array(poly, np.float32),
                                    (float(x_px), float(y_px)), False) >= 0:
                return name
        return None

    @property
    def region_names(self) -> List[str]:
        return list(self.roi_polygons.keys()) if self.roi_polygons else ["frame"]


# ---------------------------------------------------------------------------
# Per-track state -> velocity + queue + waiting time
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Live state for a single tracked vehicle ID."""
    positions: deque = field(default_factory=lambda: deque(maxlen=30))   # (x_m, y_m, t)
    first_stationary_t: Optional[float] = None   # when it first slowed below threshold
    velocity: float = 0.0                        # smoothed m/s

    def update(self, ground_pos: np.ndarray, t: float,
               velocity_threshold: float) -> None:
        self.positions.append((float(ground_pos[0]), float(ground_pos[1]), t))

        # Smoothed velocity from the last two samples within the buffer.
        if len(self.positions) >= 2:
            x0, y0, t0 = self.positions[-2]
            x1, y1, t1 = self.positions[-1]
            dt = t1 - t0
            if dt > 1e-3:
                dist = float(np.hypot(x1 - x0, y1 - y0))
                inst = dist / dt
                # Exponential moving average to suppress jitter
                self.velocity = 0.6 * self.velocity + 0.4 * inst

        if self.velocity < velocity_threshold:
            if self.first_stationary_t is None:
                self.first_stationary_t = t
        else:
            # Moving again -> reset stationary clock
            self.first_stationary_t = None

    def is_queued(self, t: float, min_stationary_sec: float,
                  velocity_threshold: float) -> bool:
        if self.first_stationary_t is None:
            return False
        return (t - self.first_stationary_t) >= min_stationary_sec \
            and self.velocity < velocity_threshold

    def waiting_time(self, t: float) -> float:
        if self.first_stationary_t is None:
            return 0.0
        return max(0.0, t - self.first_stationary_t)


class TrackHistory:
    """
    Maintains TrackState for every active track ID and exposes aggregate
    per-region queue / waiting-time metrics.
    """

    def __init__(self,
                 queue_velocity_threshold: float = 0.5,
                 queue_min_stationary_sec: float = 1.5,
                 max_age_sec: float = 5.0):
        self.queue_velocity_threshold = queue_velocity_threshold
        self.queue_min_stationary_sec = queue_min_stationary_sec
        self.max_age_sec = max_age_sec
        self.tracks: Dict[int, TrackState] = {}

    def update(self, track_id: int, ground_pos: np.ndarray, t: float) -> None:
        st = self.tracks.get(track_id)
        if st is None:
            st = TrackState()
            self.tracks[track_id] = st
        st.update(ground_pos, t, self.queue_velocity_threshold)

    def prune(self, t: float) -> None:
        """Drop tracks not seen for max_age_sec."""
        stale = [tid for tid, st in self.tracks.items()
                 if not st.positions or (t - st.positions[-1][2]) > self.max_age_sec]
        for tid in stale:
            del self.tracks[tid]

    def is_queued(self, track_id: int, t: float) -> bool:
        st = self.tracks.get(track_id)
        if st is None:
            return False
        return st.is_queued(t, self.queue_min_stationary_sec,
                            self.queue_velocity_threshold)

    def waiting_time(self, track_id: int, t: float) -> float:
        st = self.tracks.get(track_id)
        if st is None:
            return 0.0
        return st.waiting_time(t)


# ---------------------------------------------------------------------------
# Smoke test: build a synthetic config and exercise TrackHistory
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CalibrationConfig round-trip ===")
    cfg = CalibrationConfig(
        intersection_id="test_node",
        homography_src=[[100, 600], [1100, 600], [400, 300], [800, 300]],
        homography_dst=[[0, 0], [3.7, 0], [0, 20], [3.7, 20]],
        roi_polygons={"approach": [[0, 300], [1280, 300], [1280, 720], [0, 720]]},
    )
    p = Path("/tmp/_calib_smoke.json")
    cfg.save(p)
    cfg2 = CalibrationConfig.load(p)
    assert cfg2.has_homography
    g = cfg2.image_to_ground(600, 600)
    print(f"matrix present: {cfg2.matrix is not None}")
    print(f"image(600,600) -> ground {g}")
    print(f"region of (600,600): {cfg2.region_of(600, 600)}")

    print("\n=== TrackHistory queue / waiting-time ===")
    hist = TrackHistory(queue_velocity_threshold=0.5,
                       queue_min_stationary_sec=1.0)
    # Simulate a vehicle that stops at t=2 and stays stopped
    tid = 42
    t = 0.0
    # moving phase: 1 m/s for 2 seconds (samples every 0.5s)
    for i in range(5):
        hist.update(tid, np.array([i * 0.5, 0.0]), t)
        t += 0.5
    print(f"after moving: queued={hist.is_queued(tid, t)} wait={hist.waiting_time(tid, t):.2f}s")
    # stopped phase
    stop_pos = np.array([2.0, 0.0])
    for i in range(6):
        hist.update(tid, stop_pos, t)   # no movement
        t += 0.5
    print(f"after 3s stopped: queued={hist.is_queued(tid, t)} wait={hist.waiting_time(tid, t):.2f}s")
    assert hist.is_queued(tid, t), "should be queued after stationary period"
    assert hist.waiting_time(tid, t) > 1.0
    print("\n[OK] calibration.py smoke test passed.")

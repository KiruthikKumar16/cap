# Standalone Traffic Perception Pipeline (YOLOv10 + BoT-SORT)

A self-contained computer-vision project that reads a video source (file or
camera), runs YOLOv10 detection with BoT-SORT multi-object tracking, and emits
**per-frame traffic metrics** (vehicle counts and queue estimates per approach
lane) as a JSONL stream for any downstream consumer.

This module has **no dependency on any RL / control layer**. It does one job:
pixels → structured traffic metrics.

---

## 👁️ 1. The Computer Vision Model: YOLOv10 + BoT-SORT

For real-time traffic analysis, the model must balance high accuracy (detecting
small vehicles at a distance) with low latency.

### **A. Object Detection: YOLOv10**
- **Why**: YOLOv10 (released 2024) removes the need for Non-Maximum Suppression
  (NMS), significantly reducing end-to-end latency.
- **Task**: Detects `car`, `motorcycle`, `bus`, and `truck` (COCO classes
  2, 3, 5, 7).
- **Optimization**: Auto-selects CUDA (targets TensorRT) when a GPU is present,
  otherwise exports to and runs an **OpenVINO** CPU model with
  `PERFORMANCE_HINT=LATENCY`.

### **B. Multi-Object Tracking (MOT): BoT-SORT**
- **Why**: Tracking gives each vehicle a persistent ID, which is the basis for
  queue estimation and (in downstream consumers) waiting-time calculation.
- **Task**: Assigns a unique ID to every vehicle via camera motion compensation
  + Re-ID (`botsort.yaml`, `persist=True`).

---

## 📐 2. Geometry: Pixels → Meters

`PerspectiveTransformer` maps bounding-box pixels into real-world meters via
homography, so a "queue" is defined as *vehicles within 5 m of the stop line*
rather than a pixel heuristic.

- `auto_calibrate(frame)` — zero-touch calibration from Hough lane lines.
- `auto_generate_rois(frame)` — geometric ROI heuristic (placeholder for SAM).

Default ROIs cover a 4-way intersection (north / east / south / west).

---

## 📊 3. Output: JSONL Metrics

`yolo_inference.py` writes one JSON record **per frame** to the `--output` file
(default `metrics.jsonl`) and prints a live `FPS | Objects` line to stdout.

Each line is a JSON object:

```json
{
  "intersection_id": "node_1",
  "frame_idx": 42,
  "timestamp": 1721295234.57,
  "lane_counts": {"north": 3, "east": 1, "south": 0, "west": 2},
  "lane_queues":  {"north": 2, "east": 0, "south": 0, "west": 1},
  "num_objects": 6,
  "fps": 17.4
}
```

| Field | Meaning |
| :--- | :--- |
| `intersection_id` | Label passed to the engine (default `node_1`). |
| `frame_idx` | 1-based running frame counter. |
| `timestamp` | Epoch seconds the frame was processed. |
| `lane_counts` | Total vehicles per approach lane within the ROI. |
| `lane_queues` | Vehicles within 5 m of the stop line per approach lane. |
| `num_objects` | Total tracked vehicles this frame. |
| `fps` | Cumulative average FPS since start. |

### Active-learning hook
When any lane count exceeds 10 (high density — possible incident/stall), the
raw frame is saved to `data/anomalies/` for offline review.

---

## 🚀 4. Usage

```bash
python yolo_inference.py --source traffic.mp4
```

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--source` | *(required)* | Video file path or camera index (e.g. `0`). |
| `--model` | `yolov10s.pt` | YOLO weights. |
| `--output` | `metrics.jsonl` | JSONL output path. |
| `--headless` | off | Disable the video window for higher FPS. |
| `--no-openvino` | off | Skip OpenVINO acceleration on CPU. |

```bash
# Headless run with a custom output path
python yolo_inference.py --source rtsp://camera/stream --output runs/cam_01.jsonl --headless
```

---

## 🏗️ 5. Deployment Context

This Python pipeline is the prototyping/reference implementation. For
research-grade multi-camera deployment, the intended production stack is
**NVIDIA DeepStream**:

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Inference Engine** | TensorRT | Runs the YOLOv10 weights at FP16 precision. |
| **Tracking Engine** | NvMultiObjectTracker | Hardware-accelerated BoT-SORT/KLT. |
| **Video Analytics** | GStreamer | Handles RTSP streams from IP cameras without CPU overhead. |
| **Metadata Output** | Kafka/MQTT | Ships counts/queue data to downstream consumers. |

### Actuation (NTCIP 1202)
A downstream controller's action (e.g. "switch to Phase 2") would be
communicated to a physical Traffic Signal Controller (TSC) over **SNMP** as a
"Force-Off" / "Phase Omit" OID command. That actuation layer is **out of scope**
for this perception-only project.

---

## 🛠️ 6. Hardware Recommendations

### **Option A: Edge-First (Recommended)**
- **Hardware**: NVIDIA Jetson AGX Orin.
- **Deployment**: One device per intersection.
- **Benefit**: Zero dependency on a central server; low bandwidth (only
  metadata leaves the edge).

### **Option B: Cloud-Central**
- **Hardware**: NVIDIA L4 or A10G (AWS G5 instance).
- **Deployment**: All cameras stream RTSP to a central server.
- **Benefit**: Easier global coordination, but requires high-bandwidth fiber.

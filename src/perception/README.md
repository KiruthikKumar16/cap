# Real-World Perception Stack: Computer Vision for Traffic Control

To transition the MAPPO-STGNN research into a live urban environment, the SUMO simulation is replaced by a **Deep Learning Perception Pipeline**. This document outlines the optimized-in-class technologies required to generate the 12-dimensional feature vector for our RL agents.

---

## 👁️ 1. The Computer Vision Model: YOLOv10 + ByteTrack

For real-time traffic analysis, the model must balance high accuracy (detecting small vehicles at a distance) with low latency (to maintain the RL control loop).

### **A. Object Detection: YOLOv10**
- **Why**: YOLOv10 (Released 2024) removes the need for Non-Maximum Suppression (NMS), significantly reducing end-to-end latency.
- **Task**: Detects `car`, `truck`, `bus`, `motorcycle`, and `emergency_vehicle`.
- **Optimization**: Export to **TensorRT** for 2x-5x speedup on NVIDIA hardware.

### **B. Multi-Object Tracking (MOT): ByteTrack**
- **Why**: Tracking is essential to calculate **Waiting Time** and **Queue Length**. Standard detection cannot tell if a car has been stopped for 10 seconds or 100 seconds.
- **Task**: Assigns a unique ID to every vehicle. If ID #402 has a velocity of < 0.1m/s for 30 frames, it is added to the `lane_queues` count.

---

## 🏗️ 2. The Implementation Stack (NVIDIA DeepStream)

Running multiple 4K cameras on a standard Python script is inefficient. For research-grade deployment, use **NVIDIA DeepStream**.

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Inference Engine** | TensorRT | Runs the YOLOv10 weights at FP16 precision. |
| **Tracking Engine** | NvMultiObjectTracker | Hardware-accelerated ByteTrack/KLT implementation. |
| **Video Analytics** | GStreamer | Handles RTSP streams from IP cameras without CPU overhead. |
| **Metadata Output** | Kafka/MQTT | Sends the vehicle counts and queue data to the RL controller. |

---

## 📡 3. Actuation: The NTCIP 1202 Protocol

The RL agent's action (e.g., "Switch to Phase 2") must be communicated to the physical **Traffic Signal Controller** (TSC).

1.  **Input**: The Edge device (Jetson) calculates the next action.
2.  **Protocol**: Uses **SNMP** to send an OID request to the TSC.
3.  **Command**: The TSC receives a "Force-Off" or "Phase Omit" command to change the lights.

---

## 📐 4. Data Inputs: From Pixels to Features

Our RL agent expects 12 features. Here is how the CV model populates them:

| Feature | CV Logic |
| :--- | :--- |
| **Signal Phase** | Read via SNMP from the Signal Controller in real-time. |
| **Queue Sum** | Count of tracked objects with `velocity < threshold` within the ROI (Region of Interest). |
| **Waiting Time** | `current_time - first_stop_time` for all tracked IDs in the queue. |
| **Vehicle Counts** | Bounding box counts passing through a "Virtual Tripwire" in the frame. |

---

## 🚀 5. Hardware Recommendations

### **Option A: The Edge-First Approach (Recommended)**
- **Hardware**: NVIDIA Jetson AGX Orin.
- **Deployment**: One device per intersection.
- **Benefit**: Zero dependency on a central server; low bandwidth usage (only sends metadata, not video).

### **Option B: The Cloud-Central Approach**
- **Hardware**: NVIDIA L4 or A10G (AWS G5 instance).
- **Deployment**: All cameras stream RTSP to a central server.
- **Benefit**: Easier to manage global Graph Neural Network (GNN) updates, but requires high-bandwidth fiber optics.

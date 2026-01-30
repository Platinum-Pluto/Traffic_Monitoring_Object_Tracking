# Real Time Traffic Monitoring System

A real-time traffic monitoring system using **YOLO (ONNX)** for detection and **ByteTrack** for tracking, with a **DearPyGui** interface.  
The system displays live video feed, object counts, FPS monitoring, and system metrics (CPU, RAM, GPU).
The system also displays all detected objects and their bounding boxes with ID and confidence value.

---


## Processed Source Videos

### Inference Video 1
[![Traffic Monitoring Processed Source Video 1](https://img.youtube.com/vi/HrwjkkTQc00/0.jpg)](https://youtu.be/HrwjkkTQc00)

### Inference Video 2
[![Traffic Monitoring Processed Source Video 2](https://img.youtube.com/vi/rpLdTrluT6E/0.jpg)](https://youtu.be/rpLdTrluT6E)

---

## Features

- Real-time object detection and tracking of vehicles using YOLO + ByteTrack
- Support for GPU acceleration via CUDA and ONNX Runtime GPU
- Live video feed with bounding boxes, class labels, track IDs and confidence value
- Confidence slider to adjust detection threshold in real-time
- System performance dashboard (CPU, RAM, GPU)
- FPS plotting for performance monitoring
- Start/Stop and Pause/Resume functionality without losing video state

---

## Requirements

- Python 3.10+
- CUDA-compatible GPU for GPU acceleration (optional)
- ONNX Runtime
- PyTorch
- DearPyGui
- OpenCV
- psutil
- GPUtil (optional, for GPU monitoring)

---

## Setup & Installation

### 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/Platinum-Pluto/Traffic_Monitoring_Object_Tracking.git
cd Traffic_Monitoring_Object_Tracking

```

### 2. Environment Setup (Recommended: uv)

Using `uv` allows you to skip manual virtual environment creation and dependency management. Simply run:

```bash
uv sync

```

This command reads the `pyproject.toml` file, creates a synchronized virtual environment, and installs all necessary dependencies automatically.

### 3. Alternative Setup (Standard pip)

If you prefer the traditional method:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

```

---

## Running the Project

To execute the project within the managed environment, use:

```bash
uv run main.py

```

This ensures the code runs with the exact dependencies required for the project.

---
## Remember to place the inference video the project folder.
# Real-Time Biomechanical Kinematics Pipeline (OpenCV / MediaPipe) 🖐️👁️
![Status: Active Development](https://img.shields.io/badge/Status-Active_Development-FF9E00)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat&logo=opencv&logoColor=white)

### Overview
This repository contains the foundational architecture for a computer vision pipeline designed to track dynamic hand and finger kinematics in real-time. By utilizing Google MediaPipe and OpenCV, the system extracts 3D spatial joint coordinates from standard 2D video feeds. 

The long-term objective of this research is to bridge edge AI telemetry with human biomechanical tracking, ultimately eliminating the reliance on physical wearable sensors for haptic and spatial inputs.



### 🎯 Current Focus: Coordinate Extraction
The immediate goal of this module is to establish a robust, low-latency pipeline that can:
* Initialize a stable webcam feed using OpenCV.
* Detect and map the 21 3D hand landmarks natively provided by the MediaPipe framework.
* Output raw spatial coordinates (X, Y, Z) for future geometric delta calculations.

### 🎹 Architecture Roadmap: Fine-Motor Ergonomic Tracking
To validate the coordinate extraction pipeline, the system tracks fine-motor degradation using a pianist's hand posture as the testbed, measured by a "collapsed wrist" geometric threshold (< 15 degrees).

* **Phase 1: Real-Time Telemetry MVP (Current):** Establish the live ingestion pipeline using `cv2.VideoCapture(0)`. This proves the low-latency capability of the edge-AI model to extract real-time `(X,Y,Z)` deltas from a live biomechanical target.
* **Phase 2: Asynchronous Batch Processing (Next Sprint):** Transition the ingestion module to process pre-recorded `.mp4` video files. This allows the system to analyze historical, 2D footage of virtuoso pianists (e.g., Martha Argerich) to establish baseline pedagogical dexterity metrics against our domain expert's requirements.

### 🛠️ Tech Stack & Dependencies
* Python 3.x
* `mediapipe`
* `opencv-python`
* `numpy`

### 🚀 Getting Started
*(Setup instructions and usage documentation will be added as the initial architecture is committed.)*

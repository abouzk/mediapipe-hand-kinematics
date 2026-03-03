# Real-Time Hand Kinematics Tracker 🖐️👁️
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

### 🛠️ Tech Stack & Dependencies
* Python 3.x
* `mediapipe`
* `opencv-python`
* `numpy`

### 🚀 Getting Started
*(Setup instructions and usage documentation will be added as the initial architecture is committed.)*

# 635
# Smart Doorbell with Face Recognition on Raspberry Pi

## Overview
This project implements a **privacy-preserving smart doorbell system** that performs **real-time face detection and recognition entirely on a Raspberry Pi 4** using **ONNX models**. All inference runs **locally on-device**, eliminating cloud dependency, latency, and privacy concerns.
The system uses:
* **YuNet (ONNX)** for face detection
* **SFace (ONNX)** for face recognition
It achieves **15–20 FPS**, **95.1% recognition accuracy**, and **<4% false positives** on CPU-only Raspberry Pi hardware.

## Key Features
* Real-time face detection & recognition on Raspberry Pi
* Fully offline (no cloud, no subscription)
* Lightweight ONNX models with OpenCV DNN
* Classifies visitors as **known** or **unknown**
* Robust to distance, lighting, pose, and multiple faces
* Optional logging / alert extensions

## System Pipeline
```
Camera → YuNet (Detect) → Face Crop → SFace (Embed)
      → Cosine Similarity → Known / Unknown
```

## Models Used
 **YuNet**
  * Anchor-free face detector
  * ~75K parameters, ~149M FLOPs
  * Handles 10–1200 px face sizes

 **SFace**
  * Xception-39 backbone
  * 128-D face embeddings
  * Efficient cosine similarity matching

Both models are deployed via **OpenCV DNN using ONNX**.

## Hardware & Software
**Hardware**
* Raspberry Pi 4 (4GB RAM)
* Raspberry Pi Camera Module V2
**Software**
* Raspberry Pi OS (Bookworm)
* Python 3.8+
* OpenCV 4.5+
* ONNX models (YuNet, SFace)

## Repository Structure
```
models/              # ONNX models
registered_faces/    # Stored embeddings (.npy)
src/
 ├─ capture.py       # Enrollment
 ├─ feature.py       # Embedding extraction
 ├─ recognition.py  # Live inference
 ├─ utils.py
 └─ config.py
```

## Performance
* **FPS:** 15–20
* **Accuracy:** 95.1%
* **TPR:** 94.2%
* **FPR:** 3.8%
* **RAM Usage:** ~182 MB

## Team
* **Adithya Kumar** — Networking, Writing
* **Naga Sujay Kodali** — Setup, Software
* **Indra Murala** — Research, Algorithm Design

## Future Extensions
* Delivery person detection
* Mobile notifications
* Night vision (NoIR + IR LEDs)
* Smart home integration

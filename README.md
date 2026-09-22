# Face Recognition Arduino

A face recognition and access control project using Python, OpenCV,
InsightFace, AuraFace, and Arduino.

## Project Overview

This project started with a classical face recognition approach using
Haar Cascade and LBPH, and was later upgraded to a deep-learning-based
approach using AuraFace, InsightFace, face embeddings, and cosine similarity.

The system detects a face using a webcam, compares it with reference
images, and sends the recognition result to an Arduino through serial
communication.

## Project Versions

### Version 1 - LBPH

The first version uses:

- OpenCV
- Haar Cascade
- LBPH Face Recognizer
- Image preprocessing
- Image augmentation
- Serial communication with Arduino

The system sends:

- `M` → Match
- `N` → No Match

### Version 2 - AuraFace

The upgraded version uses:

- InsightFace
- AuraFace-v1
- Face Embeddings
- Cosine Similarity
- OpenCV
- NumPy
- PySerial
- Hugging Face Hub
- ONNX Runtime

The upgraded version extracts an embedding from each reference image
and compares the embedding of the detected face with the reference
embeddings.

The similarity threshold is configurable.

## Hardware

- Arduino
- Webcam
- Green LED
- Red LED
- Resistors
- USB cable

## Software Requirements

### Version 1

```bash
pip install opencv-contrib-python numpy pyserial

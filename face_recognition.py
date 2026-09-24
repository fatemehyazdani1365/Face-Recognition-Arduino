# -*- coding: utf-8 -*-
"""
High-accuracy face recognition (AuraFace / InsightFace embeddings) -> send result to Arduino over serial.

Upgraded from LBPH to a deep-learning embedding model (as suggested by the professor's
reference notebook), which is far more accurate and robust to lighting/angle changes.

- Loads all reference photos from IMAGES_DIR and extracts one face embedding per photo.
- Opens the webcam and compares each detected face's embedding to the reference embeddings
  using cosine similarity (dot product of normalized vectors).
- Sends 'M' (match) or 'N' (no match) over serial to Arduino.
- Press 'q' in the video window to quit.

Install:
    pip install insightface onnxruntime opencv-python numpy huggingface_hub pyserial
    (use onnxruntime-gpu instead of onnxruntime if you have a CUDA-capable GPU)

Note: on first run, the AuraFace-v1 model (~a few hundred MB) is downloaded automatically
from Hugging Face and cached locally in MODEL_DIR, so the first startup will take longer.
"""

import os
import time
import cv2
import numpy as np
import serial
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis

# ============================== SETTINGS ====================================
SERIAL_PORT = "COM3"          # Arduino port, e.g. "COM3" (Windows) or "/dev/ttyUSB0" (Linux/Mac)
BAUD_RATE = 9600                # must match Serial.begin(...) in the Arduino code
IMAGES_DIR = "images"           # folder with reference photos of the authorized person
CAMERA_INDEX = 0                # webcam index (0 = default camera)
SIMILARITY_THRESHOLD = 0.40     # higher = stricter match (try values between 0.35 and 0.60)
SEND_INTERVAL = 1.5             # min seconds between repeated serial sends of the same result
MODEL_DIR = "models/auraface"   # local cache folder for the downloaded model
# ==============================================================================


def load_face_model():
    """Download (once) and initialize the AuraFace-v1 face analysis model."""
    if not os.path.exists(os.path.join(MODEL_DIR, "models")):
        print("Downloading AuraFace-v1 model from Hugging Face (first run only)...")
        snapshot_download(repo_id="fal/AuraFace-v1", local_dir=MODEL_DIR)
        print("Download finished.")

    app = FaceAnalysis(
        name="auraface",
        root=".",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],  # falls back to CPU automatically
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def load_reference_embeddings(face_app, images_dir):
    """Extract one face embedding per reference photo found in images_dir."""
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

    if not os.path.isdir(images_dir):
        raise SystemExit(f"Folder '{images_dir}' not found. Add reference photos there.")

    files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_ext)]
    if not files:
        raise SystemExit(f"No images found in '{images_dir}'.")

    embeddings = []
    for filename in files:
        img = cv2.imread(os.path.join(images_dir, filename))
        if img is None:
            print(f"Skipping '{filename}': cannot read file.")
            continue

        faces = face_app.get(img)
        if len(faces) == 0:
            print(f"Skipping '{filename}': no face detected.")
            continue

        # use the largest face in the photo
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(face.normed_embedding)
        print(f"Processed '{filename}' OK.")

    if not embeddings:
        raise SystemExit("No usable face found in any reference image.")
    return embeddings


def best_similarity(embedding, reference_embeddings):
    """Cosine similarity (dot product of normalized vectors) against the closest reference photo."""
    return max(float(np.dot(embedding, ref)) for ref in reference_embeddings)


def connect_to_arduino(port, baud):
    """Try connecting to Arduino; return None on failure (program still runs without serial)."""
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # give the Arduino time to reset after the port opens
        print(f"Connected to Arduino on {port}.")
        return ser
    except Exception as e:
        print(f"Warning: could not connect to Arduino ({e}). Continuing without serial.")
        return None


def main():
    print("Loading face recognition model...")
    face_app = load_face_model()

    print(f"Extracting reference embeddings from '{IMAGES_DIR}'...")
    reference_embeddings = load_reference_embeddings(face_app, IMAGES_DIR)
    print(f"Loaded {len(reference_embeddings)} reference embedding(s).")

    ser = connect_to_arduino(SERIAL_PORT, BAUD_RATE)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        raise SystemExit("Could not open camera. Check CAMERA_INDEX.")

    last_sent_state, last_sent_time = None, 0.0
    print("Camera on. Press 'q' in the video window to quit.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            faces = face_app.get(frame)
            current_result = None

            for face in faces:
                sim = best_similarity(face.normed_embedding, reference_embeddings)
                x1, y1, x2, y2 = face.bbox.astype(int)

                if sim >= SIMILARITY_THRESHOLD:
                    current_result, color, text = "match", (0, 255, 0), f"Match ({sim:.2f})"
                else:
                    current_result, color, text = "mismatch", (0, 0, 255), f"Unknown ({sim:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Send serial command only when the result changes or enough time has passed
            if current_result is not None and ser is not None:
                now = time.time()
                if current_result != last_sent_state or (now - last_sent_time) > SEND_INTERVAL:
                    cmd = "M" if current_result == "match" else "N"
                    try:
                        ser.write(cmd.encode())
                    except Exception as e:
                        print("Serial write error:", e)
                    last_sent_state, last_sent_time = current_result, now

            cv2.imshow("Face Recognition - press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if ser is not None:
            ser.close()


if __name__ == "__main__":
    main()

"""
Stage 3 — End-to-end inference pipeline + demo.

  input image -> YOLOv8 detects damage regions
              -> crop each region
              -> MobileNetV2 classifies severity (minor/moderate/severe)
              -> draw box + "<damage_type> | <severity>" label
              -> save annotated demo image

Inference is light: this runs fine on CPU (a few images = seconds).
Use it to produce the demo image shown in the README, and to batch-crop
the dataset for training Stage 2 (see crop_dataset() at the bottom).
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

YOLO_WEIGHTS = "best_model.pt"
SEVERITY_WEIGHTS = "severity_model.h5"
CONF = 0.30

DAMAGE_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
with open("severity_classes.txt") as f:
    SEVERITY_NAMES = [l.strip() for l in f if l.strip()]

yolo = YOLO(YOLO_WEIGHTS)
severity_model = tf.keras.models.load_model(SEVERITY_WEIGHTS)

COLORS = {"minor": (0, 200, 0), "moderate": (0, 165, 255), "severe": (0, 0, 255)}


def assess(image_path, out_path="demo_output.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    results = yolo(image, conf=CONF, verbose=False)
    findings = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue
            cls_id = int(box.cls[0])
            damage = DAMAGE_NAMES[cls_id] if cls_id < len(DAMAGE_NAMES) else str(cls_id)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rs = cv2.resize(crop, (224, 224)).astype("float32")
            pred = severity_model.predict(crop_rs[None, ...], verbose=0)[0]
            severity = SEVERITY_NAMES[int(np.argmax(pred))]

            color = COLORS.get(severity, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{damage} | {severity}"
            cv2.putText(image, label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            findings.append({"damage": damage, "severity": severity,
                             "box": [x1, y1, x2, y2]})

    cv2.imwrite(out_path, image)
    print(f"{len(findings)} damage region(s) found. Annotated image -> {out_path}")
    for f in findings:
        print(f"  - {f['damage']:14s} severity={f['severity']}")
    return findings


def crop_dataset(input_dirs=("train/images", "val/images", "test/images"),
                 out_dir="cropped_damage"):
    """Run the detector over the dataset and save crops for Stage-2 labeling."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for d in input_dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            img = cv2.imread(os.path.join(d, name))
            if img is None:
                continue
            for r in yolo(img, conf=CONF, verbose=False):
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    n += 1
                    cv2.imwrite(os.path.join(out_dir, f"crop_{n}.jpg"), crop)
    print(f"Saved {n} crops to {out_dir}/ — sort into minor/moderate/severe to label.")


if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "test/images/" + os.listdir("test/images")[0]
    assess(img)

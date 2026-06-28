"""
Stage 1 — Train the YOLOv8 damage detector on the CarDD dataset.

Run this on Kaggle/Colab (free T4 GPU). ~20-40 min for 30 epochs of yolov8n
on ~2,800 images. On CPU it is impractical, so use a GPU runtime.

WHY THIS REPLACES THE OLD week4/train_.ipynb:
  The old notebook auto-generated a data.yaml with only 4 class names
  (dent, scratch, crack, glass_break) while the label files use 6 class IDs
  (0..5). That nc/names mismatch made YOLO emit box_loss=0 and mAP=0 every
  epoch — the model never learned to localize. This version uses the correct
  6-class config and points at the full local dataset (2816/810/374 images).
"""

from ultralytics import YOLO
import torch

DATA_YAML = "data.yaml"      # the corrected 6-class config in this folder
BASE_WEIGHTS = "yolov8n.pt"  # COCO-pretrained nano backbone
EPOCHS = 30
IMGSZ = 640
BATCH = 16

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device} (cuda available: {torch.cuda.is_available()})")

model = YOLO(BASE_WEIGHTS)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=device,
    patience=10,        # early stop if val mAP plateaus
    project="runs",
    name="cardd_detector",
    exist_ok=True,
)

# Sanity check: print the final validation metrics. These should be NON-ZERO.
metrics = model.val(data=DATA_YAML, split="val")
print("\n=== Final validation metrics ===")
print(f"mAP50    : {metrics.box.map50:.4f}")
print(f"mAP50-95 : {metrics.box.map:.4f}")
print(f"precision: {metrics.box.mp:.4f}")
print(f"recall   : {metrics.box.mr:.4f}")
print("\nBest weights saved to: runs/cardd_detector/weights/best.pt")
print("Download best.pt and commit it to the repo as best_model.pt")

# Vehicle Damage Detection & Severity Assessment

A two-stage computer-vision pipeline that takes a photo of a damaged vehicle and
returns **what** is damaged and **how badly** — designed for use cases like
automated insurance triage and used-car inspection.

```
 input image ──▶ YOLOv8 detector ──▶ damage regions ──▶ crop each region
                                                              │
                                                              ▼
 annotated output ◀── draw box + label ◀── MobileNetV2 severity classifier
   "scratch | minor"                          (minor / moderate / severe)
```

## Why two stages?

Detection and severity are different problems. A YOLO detector is excellent at
*localizing and classifying* damage types but says nothing about how serious each
instance is. So Stage 1 finds and labels every damage region; Stage 2 crops each
region and runs a dedicated classifier that judges severity. Decoupling them lets
each model specialize and keeps the severity labels (which require human judgment)
on a small, focused set of crops rather than the full image.

## Dataset

[**CarDD**](https://cardd-ustc.github.io/) — Car Damage Detection dataset, converted
to YOLO format.

| Split | Images |
|-------|-------:|
| Train | 2,816 |
| Val   |   810 |
| Test  |   374 |

**6 damage classes:** dent, scratch, crack, glass shatter, lamp broken, tire flat.

**Severity labels:** a curated set of ~195 cropped damage regions hand-sorted into
`minor / moderate / severe`.

## Results

> Run `kaggle/1_train_detector.py` and `2_train_severity.py`, then fill in.

**Stage 1 — YOLOv8 detector (test set)**

| Metric | Value |
|--------|-------|
| mAP@50 | _TBD_ |
| mAP@50-95 | _TBD_ |
| Precision | _TBD_ |
| Recall | _TBD_ |

**Stage 2 — Severity classifier (val set)**

| Metric | Value |
|--------|-------|
| Accuracy | _TBD_ |

**Demo**

![demo](kaggle/demo_output.jpg)

## How to run

The compute (training) needs a GPU — use a **free Colab T4 runtime**. Inference (the
demo) runs fine on CPU.

**Easiest path — one notebook, Run All:** open
[`kaggle/colab_pipeline.ipynb`](kaggle/colab_pipeline.ipynb) in Colab. It:

1. installs deps and mounts your Drive project folder,
2. **validates the recovered `best_model.pt`** and retrains the detector only if it's weak,
3. **auto-labels extra severity crops** (weak supervision) to enlarge the small hand-labeled set,
4. trains the MobileNetV2 severity classifier (validated on your trusted hand-labels),
5. runs the end-to-end demo on test images.

Before running, upload the project to Drive (e.g. `MyDrive/auto_damage/`) with
`best_model.pt`, `data.yaml`, the `train/ val/ test/` folders, and your
`minor/ moderate/ severe/` crops, then set `PROJECT_DIR` in the first cell.

**Or run the scripts individually:**

```bash
python kaggle/1_train_detector.py                  # -> best_model.pt (GPU)
python kaggle/2_train_severity.py                  # -> severity_model.h5
python kaggle/3_pipeline_demo.py path/to/car.jpg   # -> demo_output.jpg
```

### Severity labels: weak supervision

Hand-labeling severity is expensive, so only ~195 crops are hand-labeled. To enlarge the
training set, the notebook **auto-labels** additional crops with a rule over damage type +
relative box size (e.g. glass shatter / lamp broken → severe; small scratch → minor). These
noisy auto-labels are used **only for training**; the 195 hand-labels are held out as the
**trusted validation set**, and the notebook prints a confusion matrix against them so the
real accuracy is visible.

`kaggle/data.yaml` holds the corrected 6-class config. On Colab/Kaggle, set its `path:`
to your dataset directory.

## Tech stack

YOLOv8 (Ultralytics) · TensorFlow / Keras · MobileNetV2 transfer learning ·
OpenCV · Python.

## Project notes

This grew out of a structured 6-week applied-ML course (`week 0`–`week5`), moving
from classical ML (regression, PCA, t-SNE, HOG) to CNNs and object detection, and
culminating in this end-to-end pipeline. Those weekly notebooks are kept in the
repo as a learning record; the production pipeline lives in `kaggle/`.

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

![End-to-end demo: detected damage labelled with type and severity](kaggle/demo_output.jpg)

*The pipeline on a test image — each detected region is boxed and labelled with
its damage type and predicted severity.*

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

Trained on a free Colab T4 GPU. All numbers are on held-out data.

**Stage 1 — YOLOv8n detector (test set, 374 images)**

| Metric | Value |
|--------|-------|
| mAP@50 | **0.707** |
| mAP@50-95 | 0.548 |
| Precision | 0.70 |
| Recall | 0.67 |

Per-class mAP@50 — the model is strongest on structured damage and weakest on
diffuse damage, as expected:

| Class | mAP@50 | | Class | mAP@50 |
|-------|--------|---|-------|--------|
| glass shatter | 0.99 | | dent | 0.56 |
| tire flat | 0.95 | | scratch | 0.56 |
| lamp broken | 0.87 | | crack | 0.32 |

**Stage 2 — MobileNetV2 severity classifier (stratified val set, 39 crops)**

| Metric | Value |
|--------|-------|
| Accuracy | **0.769** |
| Recall — minor / moderate / severe | 0.83 / 0.89 / 0.67 |

Confusion matrix (rows = true, cols = predicted):

```
           minor  moderate  severe
minor   [   10       2        0  ]
moderate[    0       8        1  ]
severe  [    0       6       12  ]
```

Errors concentrate on the severe↔moderate boundary — the most subjective
distinction — rather than collapsing to a single class.

## How to run

Training needs a GPU — use a **free Colab T4 runtime**. Inference (the demo) runs on CPU.

Open [`kaggle/colab_kaggle_pipeline.ipynb`](kaggle/colab_kaggle_pipeline.ipynb) in Colab
and run the cells in order. It pulls the CarDD dataset straight from Kaggle (no large
upload), so you only supply a Kaggle API token and a ~22 MB bundle of the hand-labeled
crops + weights. The cells:

1. set the Kaggle token and download the dataset (2,816 / 810 / 374 images),
2. validate the detector, then train a 6-class YOLOv8n detector,
3. train the MobileNetV2 severity classifier on the hand-labeled crops,
4. report a confusion matrix + per-class recall, and run the end-to-end demo.

The three `kaggle/1_*.py` … `3_*.py` scripts are the same stages as standalone,
modular files (train detector → train severity → run the pipeline demo).

### Severity labels

Severity (minor / moderate / severe) is judged per damage crop by a MobileNetV2
classifier. The labels are ~195 hand-sorted crops — small, so the classifier uses a
**frozen ImageNet backbone + augmentation + class weighting**, and is evaluated on a
**stratified** split (every class present in validation) with a confusion matrix, so the
reported accuracy reflects real per-class performance rather than a majority-class bias.

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

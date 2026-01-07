# Week 3 – Problem Set (CNNs, Encodings & Localization) – Solutions

---

## Problem 1 (Easy)
**What is an image encoding? Why are raw pixel values generally a poor encoding for visual recognition tasks?**

An image encoding is a numerical representation of an image that captures meaningful visual information such as edges, shapes, textures, or semantic patterns in a compact form suitable for machine learning models.

Raw pixel values are a poor encoding because they are highly sensitive to lighting changes, noise, and small translations. They lack structural information, are very high-dimensional, and do not explicitly capture patterns such as edges or shapes, making learning inefficient and poorly generalizable.

---

## Problem 2 (Easy)
**Histogram of Oriented Gradients (HOG) captures edge information. Why is edge information useful for recognizing objects in images?**

Edge information is useful because edges represent object boundaries and shapes, which are among the most stable and discriminative visual cues. Shapes define objects more reliably than raw colors or pixel intensities.

Edges are robust to illumination changes and summarize important geometric structure. By encoding edge orientations, HOG provides a compact representation that preserves essential object structure while reducing noise and redundancy.

---

## Problem 3 (Medium)
**Early convolutional layers in CNNs often learn edge-like filters. Explain why this happens and how it relates to hand-crafted features such as HOG.**

Early CNN layers learn edge-like filters because edges are fundamental visual primitives that appear frequently in natural images. Detecting edges allows the network to build higher-level features such as corners, textures, and object parts in deeper layers.

This behavior is closely related to HOG, which explicitly computes gradient orientations to capture edges. The key difference is that HOG uses fixed, hand-crafted filters, while CNNs automatically learn optimal edge detectors through backpropagation, making them more flexible and task-specific.

---

## Problem 4 (Medium)
**CNN embeddings usually separate image classes better than PCA applied to raw pixels. Give two reasons for this improvement.**

1. CNN embeddings are learned through supervised training and are optimized to separate classes, whereas PCA is an unsupervised linear method that only preserves variance.
2. CNNs learn non-linear, hierarchical representations that capture semantic information, while PCA is limited to linear projections and cannot model complex visual patterns.

---

## Problem 5 (Hard)
**YOLO predicts bounding boxes and class probabilities directly from convolutional feature maps. Explain why learning spatially-aware embeddings is essential for object detection, and why a pipeline based only on global image embeddings (e.g., PCA + classifier) would fail.**

Object detection requires predicting both *what* an object is and *where* it is located. Spatially-aware embeddings preserve location and local context information, enabling the model to predict bounding boxes accurately.

YOLO operates on convolutional feature maps where each spatial location corresponds to a region in the image. This allows simultaneous localization and classification. A pipeline based only on global embeddings would lose spatial information, making it impossible to localize objects or detect multiple objects within the same image.

---

## Key Takeaways
- Image encodings transform raw pixels into meaningful representations
- Edge information is fundamental for visual recognition
- CNNs generalize and improve upon classical features like HOG
- Spatial awareness is critical for object detection tasks

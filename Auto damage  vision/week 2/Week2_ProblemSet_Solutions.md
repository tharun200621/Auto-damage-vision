# Week 2 – Problem Set (PCA & t-SNE) – Solutions

## Problem 1 (Easy)
**Why must data be mean-centered before applying PCA?**

PCA relies on the covariance matrix, which measures variance around the mean.
If data is not mean-centered, the first principal component may incorrectly align
with the mean of the data instead of the direction of maximum variance.

Mean-centering ensures:
- Correct covariance computation
- Meaningful principal components
- No bias due to feature offsets

---

## Problem 2 (Medium)
**PCA preserves variance but not class separability. Explain with an example.**

PCA ignores class labels and only maximizes variance.
If the direction that best separates classes has low variance, PCA may discard it.

**Example:**
Two classes overlap heavily in a high-variance direction but are separable in a
low-variance direction. PCA will keep the high-variance direction, causing class
overlap after projection.

---

## Problem 3 (Hard)
**t-SNE shows clean clusters, but classifiers perform poorly. Explain this paradox.**

t-SNE preserves local neighborhoods but distorts global geometry.
It exaggerates cluster separation for visualization purposes.

Reasons:
- Non-linear embedding
- Distorted inter-cluster distances
- Non-invertible mapping
- Visual clusters do not imply separability in original space

Thus, t-SNE plots can look clean while classifiers struggle.

---

## Key Takeaway
- PCA: variance preservation
- t-SNE: visualization only
- Visual separability ≠ classifier performance

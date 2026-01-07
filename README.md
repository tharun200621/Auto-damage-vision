# Auto-damage-vision
## Week-wise Breakdown

# Week 0: Python & Data Science Foundations

This week focused on building strong fundamentals in Python programming and core data science libraries. The goal was to become comfortable with data handling, visualization, and basic algorithmic thinking before moving on to machine learning and computer vision tasks in later weeks.

---

##  Folder Structure

---

##  PythonBasics

### Files
- `python_basics.ipynb`
- `example.csv`
- `videogamesales.csv`

### Description
This section introduces basic Python programming concepts required for data science and machine learning.

- Learned Python syntax including variables, loops, conditionals, and functions
- Loaded and processed CSV files using Python
- Used `example.csv` as a small dataset to practice:
  - Reading structured data
  - Accessing rows and columns
  - Performing simple computations
- Explored `videogamesales.csv` as a real-world dataset to understand how larger datasets are structured and used in analysis tasks

---

##  LibraryBasics

### Files
- `numpy_basics.ipynb`
- `pandas_basics_.ipynb`
- `matplotlib_basics_.ipynb`
- `output.csv`

### NumPy
- Created and manipulated multi-dimensional arrays
- Performed numerical operations and statistical computations
- Learned vectorized operations for efficient numerical processing

### Pandas
- Loaded CSV files into DataFrames
- Performed data cleaning and preprocessing
- Applied filtering, aggregation, and feature engineering
- Saved processed results into `output.csv`

### Matplotlib
- Visualized data using line plots, bar charts, and histograms
- Understood how visualizations help identify trends and distributions in data

---

##  Assignment: K-Means Clustering

### Files
- `Assignments.ipynb`
- `spice_locations.txt`
- `Lisan_Al_Gaib.pdf`
- `kmeans.png`

### Description
This assignment introduced unsupervised learning through K-Means clustering.

- Used `spice_locations.txt`, containing 2D coordinate data
- Implemented the K-Means algorithm **from scratch using NumPy**
- Steps included:
  - Random initialization of cluster centers
  - Distance computation
  - Label assignment
  - Iterative centroid updates
  - Convergence checking
- Visualized final clusters and centroids
- Saved the clustering result as `kmeans.png`
- `Lisan_Al_Gaib.pdf` documents the algorithmic logic and mathematical intuition behind the implementation

---

## Key Learnings

- Strong foundation in Python programming
- Hands-on experience with NumPy, Pandas, and Matplotlib
- Understanding of data preprocessing and visualization workflows
- Introduction to unsupervised learning via K-Means clustering


---
### Week 1: Linear & Logistic Regression (Supervised Learning)

This week focused on understanding **core supervised learning algorithms** through both
theoretical learning (Andrew Ng’s Machine Learning lectures) and hands-on implementation
using Python and NumPy.

---

#### Linear Regression

**Dataset Used:** `linear_data.csv`  
The dataset contains multiple numerical and categorical features related to **car attributes**, such as:
- Original Price
- Miles Driven
- Mileage
- Tank Size
- Horsepower
- Top Speed
- Acceleration
- Age of the vehicle
- Fuel type
- Transmission type (Manual / Automatic)

**Target Variable:**  
- `Estimated Price`

**Objective:**  
To **predict the estimated resale price of a car** based on its physical and performance-related features.

**Steps Performed:**
- Loaded and explored the dataset using Pandas
- Separated features (`X`) and target (`y`)
- Converted categorical and boolean variables using **one-hot encoding**
- Normalized and converted data to NumPy arrays
- Implemented **Linear Regression from scratch**:
  - Weight and bias initialization
  - Prediction function
  - Custom loss function (Minkowski / L2 norm)
  - Gradient descent optimization
- Trained the model and evaluated performance using **Mean Squared Error (MSE)**
- Visualized **true vs predicted values** to assess regression quality

**Key Learnings:**
- Relationship between features and continuous target variables
- Gradient descent mechanics
- Effect of learning rate and epochs on convergence
- Importance of feature preprocessing

---

#### Logistic Regression

**Dataset Used:** `logistic_data.csv`  
The dataset contains features describing input attributes with a **binary outcome**.

**Target Variable:**  
- Binary class label (0 or 1)

**Objective:**  
To **classify data points into one of two classes** using probabilistic modeling.

**Steps Performed:**
- Loaded and preprocessed the dataset
- Converted categorical features using encoding techniques
- Implemented **Logistic Regression from scratch**:
  - Sigmoid activation function
  - Binary cross-entropy loss
  - Gradient descent optimization
- Trained the classifier on labeled data
- Evaluated classification performance

**Key Learnings:**
- Difference between regression and classification
- Probabilistic interpretation of outputs
- Decision boundaries
- Role of sigmoid function and log-loss

---

#### Additional Work
- Implemented **Genetic Algorithm–based Linear Regression** to explore alternative
  optimization techniques beyond gradient descent
- Strengthened theoretical understanding by aligning implementations with
  Andrew Ng’s Machine Learning course

---

**Outcome:**  
Built a strong foundation in supervised learning, optimization, and model evaluation,
which directly supports later work in dimensionality reduction, computer vision,
and deep learning models.

# Week 2 – Dimensionality Reduction (PCA & t-SNE)

This week focuses on **unsupervised learning and dimensionality reduction techniques**, applied to the **`load_digits` dataset** from scikit-learn. The goal was to understand how high-dimensional data can be compressed and visualized while preserving important structure.

---

## Dataset Used

### `load_digits` Dataset (scikit-learn)

- Source: `sklearn.datasets.load_digits`
- Contains **8×8 grayscale images of handwritten digits (0–9)**
- Each image is flattened into a **64-dimensional feature vector**
- Labels correspond to digit classes (0–9)

This dataset is widely used for:
- Visualizing high-dimensional data
- Testing dimensionality reduction algorithms
- Understanding feature manifolds in image data

---

## Files and Work Done

### 1. `week2_pca_task_completed.ipynb`
**Principal Component Analysis (PCA)**

**What was done:**
- Loaded the `load_digits` dataset
- Standardized features before applying PCA
- Applied PCA to reduce dimensions:
  - From 64 → 2 and 3 dimensions
- Visualized digit distributions in reduced space
- Analyzed:
  - Explained variance ratio
  - Contribution of principal components

**Key learning:**
- PCA as a **linear dimensionality reduction technique**
- Trade-off between variance retention and compression
- Interpretation of eigenvectors and eigenvalues

---

### 2. `week2_tsne_task_completed.ipynb`
**t-SNE (t-Distributed Stochastic Neighbor Embedding)**

**What was done:**
- Used the same `load_digits` dataset
- Applied t-SNE to project 64-dimensional data into 2D
- Visualized clusters corresponding to digit classes
- Tuned parameters such as:
  - Perplexity
  - Learning rate
  - Number of iterations

**Key learning:**
- t-SNE as a **non-linear dimensionality reduction technique**
- Preservation of local neighborhood structure
- Clear visual separation of digit clusters
- Differences between PCA and t-SNE in visualization quality

---

## Comparison: PCA vs t-SNE

| Aspect | PCA | t-SNE |
|------|----|------|
| Type | Linear | Non-linear |
| Preserves | Global variance | Local structure |
| Speed | Fast | Computationally expensive |
| Visualization | Moderate | Very clear clusters |

---

## Concepts Learned

- Curse of dimensionality
- Feature space compression
- Linear vs non-linear embeddings
- Manifold learning intuition
- Visualization of high-dimensional image data

---

## Tools & Libraries Used

- Python
- NumPy
- Matplotlib
- scikit-learn
  - `load_digits`
  - `PCA`
  - `TSNE`

---

## Outcome

By the end of Week 2:
- Gained hands-on experience with **unsupervised learning**
- Understood how digit images form clusters in feature space
- Learned when to use PCA vs t-SNE
- Built strong intuition for **representation learning and visualization**

This week bridges classical ML concepts and modern deep learning representations.


## Week 3 – Feature Extraction, Embeddings & Object Detection

### Dataset Used

For Week 3, we used the **Vehicle Damage Dataset** from Kaggle:

🔗 https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection

This dataset contains thousands of real-world vehicle images showing various types and degrees of damage. It provides a rich benchmark for testing computer vision techniques such as:
- Handcrafted feature extraction (HOG)
- Deep feature embeddings (CNN)
- Object detection (YOLO)

---

## Dataset Preparation

The raw dataset was downloaded from Kaggle and reorganized into a format suitable for deep learning workflows.

### Original Structure
The raw dataset contained all images in a flat directory and additional folders.

### Converted Structure (`cnn_data/`)

This conversion allows:
- Compatibility with TensorFlow/Keras training and embedding extraction
- Clean separation of training and validation data
- Easier use of high-level image pipelines

Both **train** and **validation** subsets were created and labeled accordingly for CNN-based processing.

---

## Files and Work Done

### 1. `Week3_HOG.ipynb`
**Topic:** Handcrafted Feature Extraction using HOG

**What was done:**
- Loaded vehicle damage images from the dataset
- Converted images to grayscale and resized
- Extracted **Histogram of Oriented Gradients (HOG)** features
- Visualized:
  - Original images
  - HOG feature maps
- Applied **PCA/t-SNE** to visualize HOG feature distributions

**Key learning:**
- HOG captures edge–gradient structures
- Useful for classical vision tasks
- Limited in semantic representation compared to CNN embeddings

---

### 2. `Week3_CNN_Embeddings.ipynb`
**Topic:** Deep Feature Extraction using CNNs

**What was done:**
- Loaded data from the `cnn_data/` directory
- Built a simple CNN architecture using TensorFlow/Keras
- Trained CNN briefly to extract features
- Created an **embedding model** by removing the final classifier
- Extracted dense feature vectors for images
- Applied PCA/t-SNE to embed and visualize in 2D

**Key learning:**
- CNN embeddings capture higher-level semantic information
- Embeddings reveal patterns not obvious from raw pixels
- CNNs outperform classical features for complex image data

---

### 3. `Week3_YOLO_Inference.ipynb`
**Topic:** Object Detection using Pretrained YOLOv8

**What was done:**
- Loaded the pretrained **YOLOv8n** weights
- Performed inference on organized image dataset
- Detected objects and bounding boxes
- Saved outputs to `runs/detect/predict/`
- Visualized detections with OpenCV/Matplotlib

**Key learning:**
- Pretrained models can generalize effectively to new datasets
- Object detection complements embedding and feature extraction

---

### 4. `Week3_ProblemSet.md`
Contains written responses to theoretical questions related to:
- Feature extraction mechanics
- Dimensionality reduction
- Embedding interpretation
- Object detection logic

---

## Tools & Libraries Used

- Python
- NumPy, Pandas
- OpenCV
- Matplotlib
- scikit-image (HOG)
- TensorFlow/Keras (CNN embeddings)
- scikit-learn (PCA, t-SNE)
- Ultralytics YOLOv8 (object detection)

---

## Key Concepts Learned

- Difference between handcrafted and learned features
- CNN representation learning
- Embedding visualization techniques
- Practical dataset structuring for deep learning
- Object detection pipelines
- Debugging real data-processing workflows

---

## Summary

In Week 3, we progressed from basic image descriptors (HOG) to deep neural embeddings and modern pretrained object detectors (YOLOv8). The Kaggle vehicle damage dataset provided a realistic environment to evaluate and compare feature extraction and detection methods, laying a strong foundation for later weeks involving model training and performance evaluation.



 

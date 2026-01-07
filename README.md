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





 
# Week 3 – Feature Extraction, CNN Embeddings & Object Detection (YOLO)

This week focuses on **computer vision fundamentals and deep learning–based representations**, applied to a **vehicle damage dataset**. The work progresses from classical feature extraction to deep embeddings and finally real-world object detection using YOLO.

---

## Dataset Preparation

### Original Dataset Structure
The original dataset contained raw vehicle images and auxiliary folders such as:
- `image/` – raw vehicle damage images
- `runs/` – YOLO inference outputs
- `validation/` – validation images and annotations

### Conversion to CNN-Compatible Format
To enable **CNN training and embedding extraction**, the dataset was reorganized into a Keras-friendly directory structure:


This conversion ensures compatibility with:
- `tf.keras.preprocessing.image_dataset_from_directory`
- Standard CNN pipelines
- Embedding extraction workflows

The conversion step was critical for maintaining **clean train/validation splits** and reproducible experiments.

---

## Files and Work Done

### 1. `Week3_HOG.ipynb`
**Classical Feature Extraction using HOG (Histogram of Oriented Gradients)**

**What was done:**
- Loaded vehicle damage images
- Converted images to grayscale
- Resized images to a fixed resolution
- Extracted HOG features using `skimage`
- Visualized:
  - Original images
  - Corresponding HOG representations
- Applied **PCA / t-SNE** to reduce HOG feature dimensionality

**Key learning:**
- Understanding edge-based handcrafted features
- Visual intuition behind gradient-based descriptors
- Limitations of classical features compared to deep embeddings

---

### 2. `Week3_CNN_Embeddings.ipynb`
**Deep Feature Extraction using Convolutional Neural Networks**

**What was done:**
- Loaded data from the newly created `cnn_data/` directory
- Built a simple CNN architecture:

- Trained the CNN for a few epochs (representation learning focus)
- Created an **embedding model** by removing the final classification layer
- Extracted fixed-length feature vectors (embeddings) for each image
- Applied **PCA / t-SNE** to visualize embeddings in 2D space

**Key learning:**
- Difference between classification models and embedding models
- How CNNs learn semantic representations automatically
- Why CNN embeddings outperform handcrafted features

---

### 3. `Week3_YOLO_Inference.ipynb`
**Real-Time Object Detection using YOLOv8**

**What was done:**
- Used pretrained `yolov8n.pt` weights from Ultralytics
- Ran object detection directly on vehicle images
- Performed inference on folders containing images
- Saved detection outputs (bounding boxes + labels) to `runs/detect/`
- Visualized predictions using OpenCV / Matplotlib

**Objects detected include:**
- Cars
- Trucks
- Other vehicles

**Key learning:**
- Difference between classification and object detection
- How pretrained detectors generalize to unseen datasets
- Practical challenges of file paths, inference pipelines, and large datasets

---

### 4. `Week3_ProblemSet.md`
Contains conceptual questions and reasoning related to:
- Feature extraction
- CNN embeddings
- Dimensionality reduction
- Object detection pipelines

---

## Key Concepts Learned

- Handcrafted vs Deep Features
- CNN representation learning
- Embedding extraction and visualization
- Dataset structuring for deep learning
- Real-world object detection using YOLO
- Practical debugging of data pipelines and file systems

---

## Tools & Libraries Used

- Python
- NumPy, Pandas
- OpenCV
- Matplotlib
- scikit-image
- TensorFlow / Keras
- Ultralytics YOLOv8
- scikit-learn (PCA, t-SNE)

---

## Outcome

By the end of Week 3:
- Built intuition from **classical CV → deep learning**
- Learned to **prepare datasets properly for CNNs**
- Extracted meaningful embeddings from images
- Applied **industry-grade object detection models**
- Gained hands-on experience debugging real ML pipelines

This work forms a strong foundation for advanced computer vision and applied deep learning projects.





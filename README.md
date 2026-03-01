# Statistical Modeling & Pattern Recognition - Assignment 1

A comprehensive collection of algorithms and experiments focused on dimensionality reduction and classification. This project was developed as the **1st Assignment** for the **Statistical Modeling and Pattern Recognition (THL311)** course at the **Technical University of Crete** (Spring 2024).

## 📌 Project Overview
This assignment explores fundamental techniques in machine learning and pattern recognition, ranging from unsupervised learning (PCA) to supervised classification (LDA, Bayes). The project involves both theoretical analysis and practical implementation using **MATLAB/Octave** and **Python**.

## 🚀 Key Topics & Implementations

### 1. Principal Component Analysis (PCA)
Implementation of PCA from scratch to perform dimensionality reduction on both tabular data and image datasets.

* **Breast Cancer Dataset:**
    * Analysis of 569 samples with 30 features (benign vs. malignant).
    * **Standardization:** Implementation of `featureNormalize.m` to scale features ($\mu=0, \sigma=1$).
    * **Eigenanalysis:** Computation of the covariance matrix $\Sigma = \frac{1}{m}X^TX$ and its eigenvectors (`myPCA.m`).
    * **Projection & Recovery:** Reducing dimensions from 2D to 1D and 30D to 2D, followed by data reconstruction (`projectData.m`, `recoverData.m`).
* **Faces Dataset:**
    * Application of PCA on a dataset of 5000 face images.
    * Visualization of the top **Eigenfaces** (principal components).
    * Compression of images by retaining only the top 100 components.



### 2. Linear Discriminant Analysis (LDA)
Implementation of Fisher's Linear Discriminant for supervised dimensionality reduction, comparing its performance against PCA.

* **Synthetic Data:**
    * Implementation of binary classification LDA (`fisherLinearDiscriminant.m`).
    * Visual comparison of LDA vs. PCA projections to demonstrate class separability.
* **Iris Dataset:**
    * Extension of LDA to multi-class problems (`myLDA.m`).
    * Calculation of **Within-Class** ($S_w$) and **Between-Class** ($S_b$) scatter matrices.
    * Solving the generalized eigenvalue problem $S_w^{-1}S_b$ to reduce the 4D feature space to 2D.



### 3. Bayesian Classification
Theoretical and practical application of Bayes' Theorem for classification assuming Gaussian distributions.

* **Decision Boundaries:**
    * Calculation and visualization of decision boundaries for 2D Gaussian classes.
    * Analysis of how **Prior Probabilities** ($P(\omega_i)$) shift the decision boundary.
* **MNIST Digit Classification (Python):**
    * Development of a **Naive Bayes Classifier** for handwritten digits (0, 1, 2).
    * **Feature Engineering:**
        * **Aspect Ratio:** Width/Height of the digit's bounding box.
        * **Pixel Density:** Count of non-background pixels.
        * **Centroid:** Center of mass of the digit.
    * **Training:** Estimation of Gaussian parameters ($\mu, \sigma$) for each feature per class.
    * **Evaluation:** Accuracy testing on the MNIST test set (`HW1-Bayes-MNIST.py`).



### 4. Minimum Risk Classification
Theoretical calculation of optimal decision thresholds to minimize loss.

* **Rayleigh Distributions:** Classification of 1D data following Rayleigh distributions.
* **Risk Minimization:** Derivation of the optimal decision threshold $x_0$ given a specific Loss Matrix ($L$).

## ⚠️ Data Availability
**Note:** For reasons of storage, we did not upload the datasets (Faces, MNIST, etc.) to this repository. You will need to obtain the `faces.mat`, `breast_cancer_data.csv`, `fisheriris.mat`, and `mnist_*.csv` files separately and place them in the `data/` directory to run the scripts.

## 🛠️ Tech Stack
* **Languages:** MATLAB / Octave, Python.
* **Libraries:** NumPy, Matplotlib (for Python tasks).
* **Key Scripts:**
    * `ex1_1_pca.m`: Driver script for PCA experiments.
    * `ex1_2_Ida.m`: Driver script for LDA experiments.
    * `myPCA.m`: Core PCA algorithm.
    * `fisherLinearDiscriminant.m`: Core LDA algorithm.
    * `HW1-Bayes-MNIST.py`: Python script for Naive Bayes on MNIST.

---
*Developed for the School of Electrical and Computer Engineering, Technical University of Crete.*

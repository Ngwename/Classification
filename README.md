# Iris Classification Demo (Decision Tree + KNN)

## Table of Contents

* [Overview](#overview)
* [What the Code Does](#what-the-code-does)

  * [1) Decision Tree: Depth vs. Accuracy](#1-decision-tree-depth-vs-accuracy)
  * [2) KNN: k vs. Accuracy (2 Features)](#2-knn-k-vs-accuracy-2-features)
  * [3) KNN Decision Boundary Plots](#3-knn-decision-boundary-plots)
  * [4) KNN Evaluation (4 Features)](#4-knn-evaluation-4-features)
  * [5) ROC Curves + AUC](#5-roc-curves--auc)
* [Requirements](#requirements)
* [How to Run](#how-to-run)
* [Results Interpretation](#results-interpretation)
* [Outputs](#outputs)
* [Future Improvements](#future-improvements)

---

## Overview

This project demonstrates **supervised classification** on the classic **Iris dataset** using two models from scikit-learn:

1. **Decision Tree Classifier** (tested at multiple depths)
2. **k-Nearest Neighbors (KNN)** (tested at multiple k values with decision boundary visualization)

The workflow compares **training vs. test accuracy**, visualizes KNN decision regions, and evaluates model performance using a **confusion matrix**, **classification report**, and **multiclass ROC/AUC**.

---

## What the Code Does

### 1) Decision Tree: Depth vs. Accuracy

* Loads the Iris dataset
* Performs stratified train/test split
* Trains Decision Trees with `max_depth = [1, 2, 3]`
* Prints training and testing accuracy

**Purpose:** demonstrate how model complexity affects bias and variance.

---

### 2) KNN: k vs. Accuracy (2 Features)

* Uses only sepal length and sepal width
* Performs stratified split
* Trains KNN models with `k = [1, 3, 5, 10]`
* Prints training and testing accuracy

**Purpose:** show the bias–variance tradeoff in KNN.

---

### 3) KNN Decision Boundary Plots

* Builds a mesh grid over the feature space
* Predicts class regions for each k
* Plots:

  * decision regions
  * actual data points
* Displays one figure per k value

**Purpose:** visually demonstrate how increasing k smooths the boundary.

---

### 4) KNN Evaluation (4 Features)

* Uses all four Iris features
* Trains KNN with `n_neighbors=5`
* Generates:

  * confusion matrix
  * classification report

**Purpose:** provide deeper evaluation beyond raw accuracy.

---

### 5) ROC Curves + AUC

* Binarizes multiclass labels
* Uses One-vs-Rest KNN
* Computes ROC and AUC per class
* Computes micro-average ROC
* Plots ROC curves

**Purpose:** evaluate probabilistic class separability.

---

## Requirements

Install dependencies:

```bash
pip install scikit-learn numpy matplotlib
```

---

## How to Run

```bash
python IrisDatasetClassification.ipynb
```

The script will print metrics to the console and display several matplotlib figures.

---

## Results Interpretation

### Decision Tree: Depth vs. Accuracy

A decision tree’s **max depth** controls model complexity. Shallow trees tend to underfit (high bias), while deeper trees risk overfitting (high variance).

* **Depth = 1**
  A depth-1 tree is too simple to capture the class boundaries in the Iris data. It makes only one split, creating two regions, which leads to high bias and underfitting. This is reflected in both the low training accuracy (**0.667**) and test accuracy (**0.667**).

* **Depth = 2**
  Increasing the depth to 2 allows the tree to perform more splits and better separate the three Iris classes. The training accuracy becomes very high (**0.971**), but the test accuracy (**0.889**) is noticeably lower, suggesting the model is beginning to overfit slightly compared to depth 1.

* **Depth = 3**
  Increasing the depth to 3 further improves the model’s flexibility and the decision boundaries become more refined. The training accuracy (**0.981**) and test accuracy (**0.978**) are both high and very close to each other, indicating a good balance between bias and variance for this dataset. However, deeper trees would carry increasing risk of overfitting on more complex or noisy data.

---

### KNN: k vs. Accuracy (2 Features)

In KNN, **k** controls how many neighbors vote on the predicted class.

* **k = 1**
  The decision boundary is very irregular because each prediction depends on only the single nearest neighbor. This yields very high training accuracy (**0.943**) but noticeably lower test accuracy (**0.711**), indicating overfitting and high variance.

* **k = 3**
  The boundary becomes smoother as the model averages over more neighbors. The training accuracy (**0.867**) drops compared to k = 1, and the test accuracy (**0.667**) is still relatively low, suggesting the model has not yet reached the optimal bias–variance tradeoff.

* **k = 5**
  The boundary is smoother and more stable. The training accuracy (**0.857**) is slightly lower, but the test accuracy improves significantly (**0.822**), indicating better generalization and a good balance between bias and variance for this dataset.

* **k = 10**
  The boundary becomes much smoother and simpler. Training accuracy drops further (**0.810**) and test accuracy (**0.756**) is lower than at k = 5, suggesting the model is starting to underfit by oversmoothing the class regions and losing important local structure.

---

## Outputs

The script produces:

* Console output of training and test accuracies
* KNN decision boundary plots (one per k)
* Confusion matrix
* Classification report
* Multiclass ROC curves with AUC

---


# Iris Classification Demo (Decision Tree + KNN)

## Table of Contents
- [Overview](#overview)
- [What the Code Does](#what-the-code-does)
  - [1) Decision Tree: Depth vs. Accuracy](#1-decision-tree-depth-vs-accuracy)
  - [2) KNN: k vs. Accuracy (2 Features)](#2-knn-k-vs-accuracy-2-features)
  - [3) KNN Decision Boundary Plots](#3-knn-decision-boundary-plots)
  - [4) KNN Evaluation (4 Features)](#4-knn-evaluation-4-features)
  - [5) ROC Curves + AUC](#5-roc-curves--auc)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Interpretation](#results-interpretation)
- [Outputs](#outputs)
- [Future Improvements](#future-improvements)

---

## Overview

This project demonstrates **supervised classification** on the classic **Iris dataset** using two models from scikit-learn:

1. **Decision Tree Classifier** (tested at multiple depths)  
2. **k-Nearest Neighbors (KNN)** (tested at multiple k values with decision boundary visualization)

The workflow compares **training vs. test accuracy**, visualizes KNN decision regions, and evaluates model performance using a **confusion matrix**, **classification report**, and **multiclass ROC/AUC**.

---

## What the Code Does

### 1) Decision Tree: Depth vs. Accuracy

- Loads the Iris dataset  
- Performs stratified train/test split  
- Trains Decision Trees with `max_depth = [1, 2, 3]`  
- Prints training and testing accuracy  

**Purpose:** demonstrate how model complexity affects bias and variance.

---

### 2) KNN: k vs. Accuracy (2 Features)

- Uses only sepal length and sepal width  
- Performs stratified split  
- Trains KNN models with `k = [1, 3, 5, 10]`  
- Prints training and testing accuracy  

**Purpose:** show the bias–variance tradeoff in KNN.

---

### 3) KNN Decision Boundary Plots

- Builds a mesh grid over the feature space  
- Predicts class regions for each k  
- Plots:
  - decision regions
  - actual data points  
- Displays one figure per k value  

**Purpose:** visually demonstrate how increasing k smooths the boundary.

---

### 4) KNN Evaluation (4 Features)

- Uses all four Iris features  
- Trains KNN with `n_neighbors=5`  
- Generates:
  - confusion matrix  
  - classification report  

**Purpose:** provide deeper evaluation beyond raw accuracy.

---

### 5) ROC Curves + AUC

- Binarizes multiclass labels  
- Uses One-vs-Rest KNN  
- Computes ROC and AUC per class  
- Computes micro-average ROC  
- Plots ROC curves  

**Purpose:** evaluate probabilistic class separability.

---

## Requirements

Install dependencies:

```bash
pip install scikit-learn numpy matplotlib

# brake-signalprocessing
Graph-based classification of synchronization states in friction-induced oscillators
# Graph-Based Classification of Synchronization States

## Overview

This project presents a graph-based framework for classifying synchronization states in coupled oscillatory systems. The approach transforms phase-difference time-series signals into symbolic representations and constructs directed transition graphs to capture temporal dynamics. Graph-theoretic features are then extracted and used for machine learning-based classification. (The whole idea/concept can be seen clearly in the ppt that has been uploaded - Maths-final review.pptx)

---

## Problem Statement

Coupled oscillators exhibit different synchronization behaviors depending on system dynamics:

* Weak (no) synchronization
* Partial synchronization
* Strong synchronization

The objective is to automatically classify these states using structural representations of phase dynamics rather than conventional signal-processing techniques.

---

## Dataset

* Data shape: **(156, 50001)**
* Each row represents the **phase difference** between two oscillators over time
* Three labeled classes:

  * State 1: Weak / No synchronization
  * State 2: Partial synchronization
  * State 3: Strong synchronization

---

## Methodology

### 1. Discretization

Continuous phase values are converted into symbolic sequences using uniform binning:

* Range divided into 16 bins
* Each time point mapped to a discrete symbol (0–15)

---

### 2. Transition Graph Construction

* Each symbol → node
* Transition ( s(t) \rightarrow s(t+1) ) → directed edge
* Edge weight = frequency of transition

This results in a **directed weighted graph** representing phase evolution.

---

### 3. Feature Extraction

From each graph, the following features are computed:

* Number of Nodes
* Number of Edges
* Graph Density
* Mean Degree
* Standard Deviation of Degree
* Self-loop Ratio
* Number of Strongly Connected Components (SCC)
* Transition Entropy

These features capture structural complexity, connectivity, and randomness of the signal.

---

### 4. Classification

Machine learning models used:

* Random Forest
* Support Vector Machine (RBF kernel)
* Logistic Regression

Random Forest achieved the best performance.

---

### 5. Feature Analysis

* **Feature Importance** (Random Forest)
* **Feature Correlation Analysis**

Key findings:

* Structural features (nodes, edges, density) are highly discriminative
* Transition entropy reflects randomness of synchronization
* Self-loop ratio captures stability

---

## Results

* Strong synchronization is classified with high accuracy
* Moderate confusion exists between weak and partial states
* Graph-based features effectively capture synchronization dynamics

---

## Technologies Used

* Python
* NumPy
* NetworkX
* Scikit-learn
* Matplotlib / Seaborn

---

## Key Contributions

* Symbolic graph representation of phase dynamics
* Use of graph-theoretic features for synchronization classification
* Feature-level analysis using correlation and importance
* Comparison of multiple machine learning models

---

## Future Work

* Adaptive discretization methods
* Improved feature selection techniques
* Graph Neural Networks for dynamic representation
* Robustness analysis with noisy signals

---

## Conclusion

## The project demonstrates that synchronization states can be effectively classified using graph-based representations of phase dynamics. Structural features derived from transition graphs provide meaningful insights into the underlying dynamics and enable accurate classification.

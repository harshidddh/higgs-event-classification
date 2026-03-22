# Higgs Boson Event Classification

## Overview
This project applies machine learning techniques to high-energy particle physics data. The objective is to build a binary classifier capable of distinguishing rare Higgs boson decay events ("Signal") from standard collision events ("Background") using simulated data from the ATLAS experiment at CERN.

## Dataset
The data is sourced from the [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson/data). It contains 250,000 collision events, each characterized by 30 kinematic features.
* **Primitive Features (`PRI_`):** Raw quantities measured directly by the particle detector (e.g., transverse momentum, pseudorapidity).
* **Derived Features (`DER_`):** Complex kinematic quantities computed from the primitive features (e.g., invariant mass of particles).
* **Missing Data:** The dataset includes physics-based missing values (encoded as `-999.0`), which occur when specific particles are not present in a given event, rendering certain derived calculations invalid.

## 🌟 Unique Project Features
This project bridges the gap between raw Data Science and Experimental Physics by implementing domain-specific optimizations:

1.  **Kinematic Feature Engineering:** Manually reconstructed the **Invariant Mass ($M$)** and **Transverse Mass ($M_T$)** using relativistic four-momentum conservation. This provides the model with the 125 GeV resonance peak, significantly stabilizing the decision boundary.
    
2.  **Discovery Significance Maximization ($s/\sqrt{b}$):** Optimized the classification threshold not for accuracy, but for statistical significance ($Z$). Achieved a peak discovery potential of **280.45σ** at an optimal threshold of **0.89**.
3.  **Explainable AI (SHAP):** Implemented Shapley Additive Explanations (Game Theory) to prove the model honors physical laws, verifying that reconstructed mass is the primary driver for signal classification.
    

---

## 🧠 Model Architecture & Benchmarks
The phase space was modeled using a progression of algorithms to prove the necessity of non-linear ensemble methods.

| Model | ROC-AUC | Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** (Baseline) | 0.7985 | 73.85% |
| **Decision Tree** | 0.8804 | 81.90% |
| **Random Forest** | 0.8949 | 82.71% |
| **XGBoost (Physics-Enhanced)** | **0.9064** | **83.72%** |

---

## 📂 Repository Structure
```text
higgs-event-classification/
├── data/
│   ├── raw/                 # training.csv (ATLAS simulated data)
│   └── processed/           # Physics-enhanced feature tensors (X_train, y_train)
├── models/                  # Serialized XGBoost JSON, Random Forest, & Scalers
├── plots/                   # ROC, Significance, & SHAP beeswarm plots
├── src/                     
│   ├── data_pipeline.py     # Kinematic reconstruction, cleaning & scaling
│   ├── train_model.py       # Multi-model benchmarking engine
│   ├── evaluate.py          # Significance ($s/\sqrt{b}$) & ROC analysis
│   └── interpretability.py  # SHAP global/local explanations
└── requirements.txt


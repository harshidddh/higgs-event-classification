# Higgs Boson Event Classification

## Overview
This project applies machine learning techniques to high-energy particle physics data. The objective is to build a binary classifier capable of distinguishing rare Higgs boson decay events ("Signal") from standard collision events ("Background") using simulated data from the ATLAS experiment at CERN.

## Dataset
The data is sourced from the [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson/data). It contains 250,000 collision events, each characterized by 30 kinematic features.
* **Primitive Features (`PRI_`):** Raw quantities measured directly by the particle detector (e.g., transverse momentum, pseudorapidity).
* **Derived Features (`DER_`):** Complex kinematic quantities computed from the primitive features (e.g., invariant mass of particles).
* **Missing Data:** The dataset includes physics-based missing values (encoded as `-999.0`), which occur when specific particles are not present in a given event, rendering certain derived calculations invalid.

## Objective
To develop a classification pipeline that successfully handles imbalanced data and physics-specific anomalies, progressing through baseline models to advanced ensemble methods.

## Methodology
1.  **Exploratory Data Analysis (EDA):** Analyzing feature distributions, handling `-999.0` anomalies, and evaluating the severe class imbalance between signal and background events.
2.  **Data Preprocessing:** Feature scaling, imputation strategies for missing derived variables, and data stratification.
3.  **Modeling Progression:**
    * Baseline: Logistic Regression
    * Non-linear handling: Decision Trees & Random Forests
    * Advanced Ensemble: Gradient Boosting (XGBoost/LightGBM)
    * Deep Learning (Optional): Multi-Layer Perceptron (MLP)
4.  **Evaluation:** Models are evaluated primarily on the ROC-AUC score and the Approximate Median Significance (AMS) metric, which is the standard for high-energy physics discoveries.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn

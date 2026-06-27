# ❤️ Heart Disease Risk Prediction

> A machine learning pipeline for predicting heart disease using routine clinical measurements, feature engineering, ensemble learning, and Matthews Correlation Coefficient (MCC) optimization.

---

#  Overview

Heart disease is one of the leading causes of death worldwide. Early identification of patients at risk enables timely intervention and improves clinical outcomes.

This project presents a complete machine learning pipeline for binary heart disease classification using routine clinical measurements and demographic information.

The solution focuses on maximizing the **Matthews Correlation Coefficient (MCC)**, a robust evaluation metric particularly suitable for binary medical classification problems.

---

#  Problem

This is a binary classification task.

| Class | Description |
|------|-------------|
| **0** | No Heart Disease |
| **1** | Heart Disease |

The objective is to predict the probability that a patient has heart disease.

---

#  Dataset

The dataset was originally provided as part of a machine learning competition and **cannot be redistributed**.

Dataset characteristics:

| Property | Value |
|----------|------|
| Training Samples | Competition Dataset |
| Test Samples | Competition Dataset |
| Target | Heart Disease |
| Features | Clinical & Demographic Variables |

### Available Features

The dataset contains routine medical information including:

- Age
- Gender
- Blood Pressure
- Cholesterol
- Blood Sugar
- Heart Rate
- Smoking Status
- Alcohol Intake
- Exercise Level
- Family History
- Diabetes
- Obesity
- Chest Pain Type
- Exercise-Induced Angina
- Stress Level

---

#  Machine Learning Pipeline

```text
Raw Clinical Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Categorical Encoding
        │
        ▼
Robust Feature Scaling
        │
        ▼
Stratified Cross Validation
        │
        ▼
Multiple Machine Learning Models
        │
        ▼
Weighted Soft-Voting Ensemble
        │
        ▼
Probability Prediction
```

---

#  Feature Engineering

More than **40 domain-inspired features** were generated to improve predictive performance.

Examples include:

## Clinical Interactions

- Cholesterol × Blood Pressure
- Cholesterol × Blood Sugar
- Heart Rate × Blood Pressure

## Risk Scores

- Cardiac Risk Score
- Lifestyle Risk Score
- Metabolic Syndrome Indicator
- Multiple Risk Factors Score

## Medical Categories

Blood Sugar:

- Normal
- Prediabetes
- High Blood Sugar

Blood Pressure:

- Normal
- Elevated
- Hypertension Stage 1
- Hypertension Stage 2

Heart Rate:

- Bradycardia
- Normal
- Tachycardia

Stress Level:

- Low
- Moderate
- High

## Polynomial Features

- Cholesterol²
- Blood Pressure²
- Blood Sugar²
- Heart Rate²
- Stress²

## Ratio Features

- Cholesterol / Blood Pressure
- Blood Sugar / Blood Pressure
- Heart Rate / Blood Pressure
- Cholesterol / Heart Rate

## Interaction Features

- Gender × Smoking
- Gender × Diabetes
- Gender × Obesity
- Gender × Blood Pressure
- Gender × Cholesterol

---

#  Models

The final prediction is generated using a weighted ensemble of several machine learning algorithms.

Models included:

- Random Forest
- Gradient Boosting
- Extra Trees
- Logistic Regression
- Multi-Layer Perceptron (MLP)

Each model is trained using **5-fold Stratified Cross Validation**.

The final probability is obtained using a weighted soft-voting ensemble.

---

#  Threshold Optimization

Instead of using the default probability threshold of **0.5**, multiple thresholds are evaluated.

The threshold producing the highest **Matthews Correlation Coefficient (MCC)** on out-of-fold predictions is selected.

---

#  Evaluation Metric

The competition evaluates submissions using the **Maximum Matthews Correlation Coefficient (MCC)**.

MCC considers all components of the confusion matrix:

- True Positives
- True Negatives
- False Positives
- False Negatives

making it significantly more informative than simple accuracy, especially for medical classification problems.

---

#  Project Structure

```text
.
├── model_training.py
├── requirements.txt
└── README.md
```

---

---


#  Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn

---

# Highlights

-  Extensive domain-driven feature engineering
-  Robust categorical encoding
-  Medical risk score generation
-  Weighted soft-voting ensemble
-  5-Fold Stratified Cross Validation
-  RobustScaler preprocessing
-  MCC-based threshold optimization
-  Probability prediction for competition submission

---

# Competition Context

This project was developed for a machine learning competition on binary heart disease prediction.

To comply with the competition rules, the original dataset is **not included** in this repository.

---

#  Future Improvements

Possible future enhancements include:

- XGBoost
- LightGBM
- CatBoost
- Optuna Hyperparameter Optimization
- SHAP Explainability
- Stacking Ensemble
- Bayesian Optimization
- Feature Selection

---

# ⚠️ Disclaimer
his repository is intended for research and educational purposes only.

The developed model is **not** intended for clinical diagnosis or medical decision-making without appropriate medical validation.

This repository is intended for research and educational purposes only.

The developed model is **not** intended for clinical diagnosis or medical decision-making without appropriate medical validation.

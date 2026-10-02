# Superstore Profit Classification & Stacking Pipeline

An end-to-end machine learning project analyzing retail order transactions from `samplesuperstore.csv` to predict profitability tiers using tree-based ensemble models and a stacking classifier.

---

## Overview

The pipeline automates:
1. **Exploratory Data Analysis (EDA):** Correlation analysis and numeric summaries.
2. **Feature Engineering:** Creation of a multi-class target based on profit margins.
3. **Data Preprocessing:** Categorical one-hot encoding and standard numeric scaling.
4. **Model Training:** Training 4–5 tree ensembles alongside a meta-stacking model.
5. **Evaluation & Visualization:** Per-class classification metrics, confusion matrices, and One-vs-Rest ROC-AUC curves.

---

## Target Definition

The target variable `profit_class` is derived from `Profit / Sales`:
- **Class 0 (Loss):** Orders with `Profit < 0`
- **Class 1 (Low Margin):** Orders with `Profit >= 0` and profit margin $< 25\%$
- **Class 2 (High Margin):** Orders with profit margin $\ge 25\%$

*Note: Raw `Profit` is excluded from feature sets to prevent target leakage.*

---

## Features

- **Categorical (One-Hot Encoded):** `Ship Mode`, `Segment`, `Region`, `Category`, `Sub-Category`
- **Numerical (Standard Scaled):** `Sales`, `Quantity`, `Discount`

---

## Models Implemented

- **Base Learners:**
  - Random Forest (`RandomForestClassifier`)
  - Extra Trees (`ExtraTreesClassifier`)
  - XGBoost (`XGBClassifier`) / Gradient Boosting
  - LightGBM (`LGBMClassifier`) / Hist Gradient Boosting
  - CatBoost (`CatBoostClassifier`)
- **Meta Learner (Stacking):**
  - Multinomial Logistic Regression (`StackingClassifier` with 5-fold stratified out-of-fold probability vectors)

---

## Evaluation Metrics

- **Macro & Weighted F1-Scores** (to account for class imbalance in retail losses)
- **Precision & Recall** across each profit tier
- **Normalized Confusion Matrices** (evaluating Class 0 loss detection rate)
- **One-vs-Rest (OvR) ROC Curves & Macro AUC**

---

## Author 

- Hasan Al-Banna
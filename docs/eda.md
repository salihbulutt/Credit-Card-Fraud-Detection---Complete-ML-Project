# 🧪 Exploratory Data Analysis (EDA)

This document summarizes the important findings from the exploratory data analysis performed in `01_eda.ipynb`.

---

## 📌 Dataset Overview

- **Rows:** 284,807  
- **Fraud cases:** 492  
- **Fraud rate:** 0.172%  
- **Features:**  
  - Time, Amount  
  - PCA components: V1–V28  
  - Target variable: Class  

The dataset is a real-world anonymized credit card dataset from ULB.

---

## ⚠️ Class Imbalance

The dataset is **extremely imbalanced**:

- 99.83% legitimate  
- 0.17% fraud  

This imbalance affects:
- Model training  
- Evaluation metrics  
- Threshold selection  

Metrics like PR-AUC and recall become more meaningful than accuracy.

---

## 💰 Amount Feature

- Highly skewed distribution  
- Many small transactions  
- Few very large ones  
- Log transformation (`log1p`) helps reduce skewness  

Fraud transactions show slight differences in distribution but no single threshold distinguishes them.

---

## ⏱ Time Feature

- Represents seconds from the first transaction  
- Converting to `hour` revealed some patterns:
  - Certain hours of the day slightly more fraud-prone  

---

## 🔍 PCA Features (V1–V28)

- All are anonymized PCA components  
- Hard to interpret, but:
  - Some show clear separation between fraud vs non-fraud  
  - Important for predictive modeling  

---

## 🔗 Correlation Insights

- PCA features correlate strongly with each other  
- Amount & Time have no strong correlation with PCA features  
- Target correlations are weak, as expected in fraud datasets  

---

## 📝 EDA Conclusions

1. **Imbalance is the key challenge** → requires special handling.  
2. **Amount is skewed** → log scaling is beneficial.  
3. **PCA features provide the most predictive power.**  
4. **Time may contribute some signal** after feature engineering.  
5. **No missing values** → no cleaning required.  

These insights guide baseline modeling, feature engineering, and sampling strategies.

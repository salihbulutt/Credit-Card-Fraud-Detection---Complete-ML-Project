# 🧱 Feature Engineering Documentation

This document summarizes feature engineering attempts in `03_feature_engineering.ipynb`.

---

## 🎯 Goal

Improve the baseline model through:
- New transformations  
- Resampling strategies  
- Better preprocessing  

---

## 🧮 Feature Engineering Attempts

### 1. Log Transformation of Amount

- Applied `log1p(Amount)`  
- Reduced skewness  
- Slightly improved PR-AUC  
- Not strictly necessary but useful  

---

### 2. Sampling Strategies

Because fraud data is extremely rare (0.17%), resampling helps.

**\*Attempted: SMOTE**

- SMOTE synthesizes minority samples  
- Evaluated with cross-validation  
- Improved PR-AUC over baseline Logistic Regression  

**But caution:**  
- Oversampling can cause overfitting  
- Best used for linear models, not tree ensembles  

---

## 🧪 Results Summary

(Replace with your actual scores)

| Experiment | Model | PR-AUC | Notes |
|-----------|-------|--------|-------|
| Baseline | Logistic Regression | XXX | class_weight=balanced |
| SMOTE | Logistic Regression | XXX | improved recall |
| Log Amount | Logistic Regression | XXX | small improvement |

---

## 📌 Conclusions

- Feature engineering on PCA-heavy anonymized data is limited.
- SMOTE provides improvement but may not generalize well.
- Final model relies on tree methods (RF/XGBoost) that already handle feature interactions.
- Feature engineering played a minor role; modeling was the key improvement area.

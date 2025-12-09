# 🟦 Baseline Model Documentation

This document summarizes the baseline modeling phase in `02_baseline.ipynb`.

---

## 🎯 Objective

Create a **simple, fast, interpretable baseline model** to:
- Provide a reference score
- Compare improvements from later stages
- Validate preprocessing steps

---

## ⚙️ Baseline Model

**Model:** Logistic Regression  
**Reason:**  
- Simple, fast  
- Works well on linearly separable PCA features  
- Provides a good baseline for PR-AUC  

**Preprocessing:**  
- StandardScaler applied to:
  - Time  
  - Amount  
- PCA features passed through unchanged  

**Handling Imbalance:**  
- `class_weight="balanced"`  

---

## 📊 Baseline Scores

(Replace XXX with your actual values)

- **PR-AUC:** `XXX`
- **ROC-AUC:** `XXX`
- **Recall @ 0.5 threshold:** `XXX`
- **Precision @ 0.5 threshold:** `XXX`

---

## 📌 Conclusions

- Logistic Regression performs reasonably well considering dataset imbalance.
- Good reference point for more complex models.
- Recall is low at default threshold—expected for fraud detection.
- Threshold tuning will be necessary in final pipeline.

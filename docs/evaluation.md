# 📈 Final Model Evaluation

This document summarizes the evaluation process from `05_model_evaluation.ipynb`.

---

## 🧪 Final Model: XGBoost

Evaluated on:
- Validation set  
- Test set  

Metrics (update with your values):

- **PR-AUC:** `XXX`
- **ROC-AUC:** `XXX`
- **Recall at business threshold:** `XXX`
- **Precision at business threshold:** ≥ 0.90 (requirement)

---

## 📊 Evaluation Tools

### 1. Precision-Recall Curve

Shows:
- High recall possible at acceptable precision  
- Important due to extreme class imbalance  

### 2. ROC Curve

Shows:
- Good separation  
- Complement to PR-AUC  

### 3. SHAP Analysis

Insights:
- Certain PCA components (e.g., V14, V4, V17) strongly influence prediction  
- Amount and Time have weaker effect  
- Confirms model relies mainly on patterns in anonymized PCA embeddings  

---

## 📦 Threshold Selection

We selected a threshold that satisfies:

```
precision ≥ 0.90
```

From validation PR curve.

Final threshold saved at:
```
models/threshold.json
```

Used by:
```
FraudModelService.predict_single()
```

---

## 🎯 Final Conclusion

- XGBoost performs significantly better than baseline logistic regression.
- Meets business requirements for fraud detection.
- Ready for deployment with Streamlit frontend.

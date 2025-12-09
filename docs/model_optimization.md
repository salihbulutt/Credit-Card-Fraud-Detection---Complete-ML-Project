# 🧩 Model Optimization Documentation

Covers hyperparameter tuning results from `04_model_optimization.ipynb`.

---

## 🎯 Optimization Goal

Identify a powerful model that improves:
- Recall
- PR-AUC
- Stability across folds  
- Business requirement: high precision at a given threshold  

---

## 🧪 Tuned Models

### 1. Random Forest

Parameters tested:
- `n_estimators`
- `max_depth`
- `min_samples_split`

Results:
- Good stability  
- Moderate improvement over baseline  
- Slower inference due to tree count  

---

### 2. XGBoost

Parameters tested:
- `n_estimators`
- `max_depth`
- `learning_rate`

Results:
- Best model overall  
- Highest PR-AUC  
- Faster inference than large RF models  
- Chosen as FINAL MODEL  

---

## 📊 Example Best Parameters (Yours may differ)

**RandomForest best params:**
```python
{
    'model__n_estimators': 400,
    'model__max_depth': 7,
    'model__min_samples_split': 2
}
```

**XGBoost best params:**
```python
{
    'model__n_estimators': 300,
    'model__max_depth': 4,
    'model__learning_rate': 0.05
}
```

---

## 🏆 Final Selected Model: **XGBoost**

Reason:
- Best PR-AUC in validation  
- Good recall improvement  
- Handles non-linear boundaries  
- Works well on PCA-based features  

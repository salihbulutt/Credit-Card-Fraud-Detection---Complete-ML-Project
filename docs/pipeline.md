# 🔄 Final Pipeline Documentation

This describes the complete ML pipeline implemented in `06_pipeline.ipynb` and `src/pipeline.py`.

---

## 🎯 Pipeline Goal

Enable a **single command** to:

1. Load raw data  
2. Split into train/val/test  
3. Build preprocessing  
4. Train final model  
5. Compute validation metrics  
6. Select decision threshold  
7. Save all artifacts  

```
python src/pipeline.py
```

---

## 🧱 Pipeline Components

### 1. Preprocessing
- StandardScaler for Time & Amount  
- PCA components passed through as-is  

### 2. Model
- Final model: **XGBoost**  
- Trained on full training set  

### 3. Threshold Selection
- From validation set PR curve  
- Ensures: precision ≥ 0.90  

### 4. Saved Artifacts
```
models/fraud_model.pkl
models/preprocessor.pkl
models/threshold.json
```

### 5. Deployment
Model is used by:
```
FraudModelService (src/inference.py)
Streamlit App (app/app.py)
```

---

## 📌 Why This Pipeline?

- Simple, reproducible, one-click train  
- Consistent preprocessing across training & inference  
- Includes business logic for thresholding  
- Easily deployable  

---

## 🏁 Final Notes

This pipeline meets all **ML Bootcamp** requirements:
- Problem definition  
- Baseline  
- Feature engineering  
- Optimization  
- Evaluation  
- Final end-to-end pipeline  
- Deployment-ready artifacts  

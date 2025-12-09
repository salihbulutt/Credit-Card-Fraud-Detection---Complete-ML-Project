# 💳 Credit Card Fraud Detection System

## About The Project
This project is an end-to-end machine learning solution developed as part of the **Machine Learning Bootcamp**. It addresses the critical problem of credit card fraud detection in the banking and financial services sector.

### Problem Statement
**Business Problem:**
Credit card fraud costs the financial industry billions annually. Banks need an automated system to:
- Detect fraudulent transactions in real-time
- Minimize false positives (legitimate transactions blocked)
- Maximize fraud detection rate
- Provide explainable predictions for compliance

**Technical Challenge:**
- Extreme class imbalance (0.172% fraud rate = 1:578 ratio)
- Need for high recall without sacrificing precision
- Real-time prediction requirements (<100ms latency)
- PCA-transformed features limit interpretability

---

## 📊 Project Overview

### Sector: Banking & Financial Services
- **Problem Type:** Binary Classification (Fraud / Legitimate)
- **Dataset:** Credit Card Fraud Detection Dataset (Kaggle)
- **Data Size:** 284,807 transactions, 30 features
- **Primary Metric:** PR-AUC (Precision-Recall AUC)
- **Business Metric:** Recall (to catch frauds) & Precision (to minimize false alarms)

### 🧬 Dataset

- **Source:** ULB Machine Learning Group (via Kaggle)
- **Rows:** 284,807
- **Fraud cases:** 492
- **Fraud rate:** ≈ 0.172%
- **Features:**
  - `Time` – seconds elapsed between each transaction and the first transaction  
  - `Amount` – transaction amount  
  - `V1`–`V28` – anonymized PCA components  
  - `Class` – 1 (fraud) / 0 (legitimate)

There are **no missing values**, and most signal is encoded in PCA components.

### 📏 Metrics & Business Constraints

Because of extreme imbalance, **accuracy** is misleading. We focus on:

### Primary metrics

- **PR-AUC (Average Precision)**
- **Recall at a minimum precision level (business threshold)**

Business requirement (example):

> ✅ Precision must be at least **0.90** for fraud predictions.  
> Among the transactions we flag as fraud, at least 90% should truly be fraud.

### Secondary metrics

- ROC-AUC  
- F1-score  
- Confusion matrix  

The **decision threshold** is selected on the validation set to satisfy the precision constraint while maximizing recall.

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── creditcard.csv        # Kaggle dataset (not tracked in git)
│   └── processed/                # (optional) intermediate files
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_baseline.ipynb         # Baseline model
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_optimization.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_pipeline.ipynb         # Runs final pipeline
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, constants, business rules
│   ├── data_prep.py              # Load & split data, preprocessing
│   ├── features.py               # (Optional) extra feature functions
│   ├── models.py                 # Baseline, RF, XGBoost definitions
│   ├── utils.py                  # Metrics, threshold selection, helpers
│   ├── pipeline.py               # Final training pipeline
│   └── inference.py              # Model service for app/API
├── app/
│   ├── __init__.py
│   └── app.py                    # Streamlit frontend
├── models/
│   ├── fraud_model.pkl           # Trained pipeline (preproc + model)
│   ├── preprocessor.pkl
│   └── threshold.json            # Chosen decision threshold
├── docs/
│   ├── eda.md                    # EDA findings
│   ├── baseline.md               # Baseline process & scores
│   ├── feature_eng.md            # Feature engineering attempts
│   ├── model_optimization.md     # Hyperparameter tuning summary
│   ├── evaluation.md             # Final evaluation & business fit
│   └── pipeline.md               # End-to-end pipeline description
└── tests/
    ├── test_pipeline.py          # Checks training pipeline artifacts
    ├── test_inference.py         # Checks inference service
    └── test_imports.py           # Simple import sanity check
```

---

## 🔍 Key Findings & Decisions

### 1. Problem Definition & EDA
Credit card fraud detection with extreme class imbalance (0.172% fraud rate). The goal is to maximize fraud detection (Recall) while maintaining acceptable precision to avoid overwhelming fraud analysts with false positives.
- Inspect dataset structure and distributions
- Analyze class imbalance
- Explore Amount and Time distributions
- Study correlations between features
- Summarize key findings in docs/eda.md

### 2. Baseline Process & Score
- Model: Logistic Regression (class_weight="balanced")
- Preprocessing: scale Time and Amount
- Evaluate ROC-AUC & PR-AUC
- Provide first reference scores
- Document results in docs/baseline.md

### 3. Feature Engineering Experiments & Results

- Try log(Amount+1) to reduce skew
- Experiment with SMOTE oversampling
- Evaluate improvements in PR-AUC
- Document experiments & results in docs/feature_eng.md

### 4. Model Optimization

- Tune:
   - RandomForestClassifier
   - XGBClassifier (XGBoost)
- Use RandomizedSearchCV with StratifiedKFold and scoring = average_precision
- Compare models and pick XGBoost as final
- Document best params & selection in docs/model_optimization.md

### 5. Evaluation

- Evaluate final model on validation & test sets
- Plot ROC and Precision–Recall curves
- Apply SHAP to interpret most influential features
- Analyze and select decision threshold to satisfy precision constraint
- Document business-fit analysis in docs/evaluation.md

### 6. Final Pipeline

- Provide a single entry point (run_training_pipeline) that:
  - Loads data
  - Splits into train/val/test (stratified)
  - Builds preprocessor
  - Trains final XGBoost model
  - Evaluates on validation
  - Chooses threshold for required precision
  - Saves fraud_model.pkl, preprocessor.pkl, threshold.json
- Document end-to-end process in docs/pipeline.md

---

## 🧮 Validation Scheme

- Dataset is split with stratified splits to preserve fraud ratio:
  - Train / Validation / Test
- Additionally:
  - Hyperparameter tuning uses StratifiedKFold cross-validation (k=5)
  - Main scoring metric in tuning: average_precision (PR-AUC)
- Why this scheme?
- Class imbalance is huge; we must maintain the same ratio in each split.
- PR-AUC is more informative than ROC-AUC or accuracy.
- StratifiedKFold ensures stable and reliable evaluation.

---

## 🏆 Final Model vs Baseline (Template)

After you run training and evaluation, fill in your actual numbers below.

Baseline (Logistic Regression)

- PR-AUC: X.XXX
- ROC-AUC: Y.YYY
- Recall @ threshold=0.5: Z.ZZ
- Precision @ threshold=0.5: P.PP

Final Model (XGBoost + tuned threshold)

- PR-AUC: A.AAA
- ROC-AUC: B.BBB
- Recall @ business threshold (precision ≥ 0.90): C.CC
- Precision @ business threshold: ≥ 0.90

Improvement:

- Final model significantly improves PR-AUC and recall compared to baseline.
- Decision threshold is aligned with business requirements (precision constraint).

---

## 🏢 Business Alignment & Productionization

Is the final model business-ready?
- The model:
  - Uses a decision threshold that ensures high precision (few false alerts)
  - Maintains good recall (many frauds caught)
- It can be integrated into:
  - Real-time scoring service (API)
  - Batch scoring pipelines

How would this go to production?

1.Serve the model as an API using the logic in src/inference.py.
2.Connect transaction streams (from core banking) to the model.
3.Log:
- All predictions
- Feedback labels (was it fraud or not?)
4.Monitor:
- Fraud recall on labeled feedback
- Precision on flagged alerts
- Volume of alerts per day
- Drift in feature distributions (Time, Amount, PCA components)
- Model latency / API errors
5.Retrain regularly (e.g., monthly or quarterly) with new data.
  
---

## 🌐 Deployment

### Streamlit Frontend

The project includes a Streamlit app in app/app.py:

Features:

- Single Transaction Mode
  - Enter Time, Amount, and PCA values (or keep zeros for demo)
  - Get fraud probability and decision (fraud / not fraud)
- Batch Mode
  - Upload CSV with Time, Amount, V1–V28
  - Get predictions for all rows
  - Download result as CSV

How to run locally
```bash
# 1. Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle:
#    Place `creditcard.csv` under data/raw/
#    -> data/raw/creditcard.csv

# 4. Train the model (creates files in /models)
python -m src.pipeline

# 5. Run the Streamlit app
streamlit run app/app.py

```
Online Deployment (optional for bootcamp)

You can deploy using Streamlit Cloud:

- Push this repo to GitHub
- Connect Streamlit Cloud to the repo
- Set:
  - Entry point: app/app.py
  - Python version and requirements from requirements.txt
- Add the live URL here:

[Live Demo](https://credit-card-fraud-detection---complete-ml-project-sa4bolihnb3r.streamlit.app/)

---

## 🧪 Tests

Basic tests are provided under tests/:

- test_pipeline.py
  - Runs the training pipeline and checks that:
    - fraud_model.pkl
    - preprocessor.pkl
    - threshold.json
      are created under /models.
- test_inference.py
  - Loads FraudModelService
  - Checks single and batch predictions work as expected.
- test_imports.py
  - Simple sanity check that main modules can be imported.

Run all tests with:
```bash
pytest -q
```

---

## 🧰 Technologies Used

- Language: Python 3.x
- Data / ML:
  - pandas, numpy
  - scikit-learn
  - imbalanced-learn
  - xgboost
  - shap
- Visualization:
  - matplotlib, seaborn
- Serving / App:
  - Streamlit
- Utilities:
  - joblib
  - pytest

---

## 🎓 Key Learnings

1. **Class Imbalance:** Standard accuracy is meaningless; use PR-AUC
2. **Feature Engineering:** Simple features (time, amount) add significant value
3. **Validation Strategy:** Time-based split crucial for realistic estimates
4. **Business Alignment:** Technical metrics must map to business value
5. **Monitoring:** Production ML requires extensive monitoring infrastructure

---

## 🔮 Future Improvements

1. **Advanced Features:**
   - Transaction velocity (transactions per hour)
   - Merchant category codes
   - Geographic location patterns
   - Device fingerprinting

2. **Model Enhancements:**
   - Ensemble methods (stacking)
   - Deep learning (LSTM for sequences)
   - Online learning for real-time adaptation
   - Anomaly detection techniques

3. **Production Optimizations:**
   - Model serving infrastructure (TensorFlow Serving)
   - Feature store (Feast)
   - A/B testing framework
   - Automated retraining pipeline

4. **Business Integration:**
   - Customer risk scoring
   - Dynamic thresholds based on risk appetite
   - Integration with fraud investigation tools
   - Feedback loop from fraud analysts

---

## 📝 Documentation
Detailed documentation available in `/docs`:
- [EDA Findings](docs/eda.md)
- [Feature Engineering](docs/feature_eng.md)
- [Model Evaluation](docs/evaluation.md)
- [Model Optimization](docs/model_optimization.md)

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact
**Salih Bulut**
- Email: salihbulut1@gmail.com
- LinkedIn: [salihbulutt](https://linkedin.com/in/salihbulutt)
- GitHub: [@salihbulutt](https://github.com/salihbulutt)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Dataset provided by [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Machine Learning Bootcamp instructors and community
- Reference implementation inspired by various Kaggle notebooks

---
⭐ If you find this project helpful, please consider giving it a star!

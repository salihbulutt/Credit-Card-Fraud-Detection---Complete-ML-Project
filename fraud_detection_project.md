# Credit Card Fraud Detection

Machine learning system for real-time credit card fraud detection in banking operations. Predicts whether a transaction is fraudulent or legitimate with 92.68% recall. Built on [Kaggle's Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset to replicate production fraud detection infrastructure.

**🌐 [Try the Application](https://your-app-url.streamlit.app/)**

---

## Dataset & Methodology

**Data:** 284K credit card transactions over 2 days (European cardholders, 2013)
- Target: 0.17% fraud rate (highly imbalanced 577:1)
- Features: 30 original → 69 engineered → 35 final
- PCA-transformed features (V1-V28) + Time + Amount

**Pipeline:**
1. Baseline models (Logistic Regression)
2. Incremental feature engineering (Time, Amount, Statistical, Interactions)
3. Class imbalance handling (SMOTE oversampling)
4. Model optimization (XGBoost with hyperparameter tuning)
5. Feature selection (SHAP-based importance filtering)

**Results:**
- Test Recall: 0.6138 (baseline) → 0.9268 (final) [+31.3%]
- Test Precision: 0.7245 → 0.8823 [+15.8%]
- Test F1-Score: 0.6644 → 0.9040 [+24.0%]
- ROC-AUC: 0.9245 → 0.9812 [+5.7%]

**Documentation:**
- [Setup & Installation](/docs/00_setup.md)
- [EDA Findings](/docs/01_eda_findings.md) - Data exploration and class imbalance analysis
- [Baseline Model](/docs/02_baseline_results.md) - Logistic Regression baseline performance
- [Feature Engineering](/docs/03_feature_engineering.md) - 39 engineered features across 4 categories
- [Model Optimization](/docs/04_model_optimization.md) - XGBoost hyperparameter tuning
- [Model Evaluation](/docs/05_evaluation_report.md) - SHAP analysis and feature selection
- [Final Pipeline](/docs/06_final_pipeline.md) - Production-ready training pipeline
- [API Deployment](/docs/api_deployment.md) - FastAPI + Streamlit deployment guide

---

## Project Structure

```
fraud-detection/
├── docs/                               # Documentation files
│   ├── 00_setup.md
│   ├── 01_eda_findings.md
│   ├── 02_baseline_results.md
│   ├── 03_feature_engineering.md
│   ├── 04_model_optimization.md
│   ├── 05_evaluation_report.md
│   ├── 06_final_pipeline.md
│   └── api_deployment.md
│
├── models/
│   └── final/
│       ├── final_model.pkl
│       ├── scaler.pkl
│       └── model_card.json
│
├── notebooks/                          # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_optimization.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_final_pipeline.ipynb
│
├── src/                                # Source code
│   ├── __init__.py
│   ├── config.py                       # Configuration management
│   ├── data_preprocessing.py           # Preprocessing utilities
│   ├── feature_engineering.py          # Feature creation functions
│   ├── inference.py                    # Model inference pipeline
│   ├── pipeline.py                     # Training pipeline
│   └── app.py                          # Streamlit web interface
│
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in `data/raw/`

### Run Pipeline

```bash
# Train complete pipeline
python src/pipeline.py

# Launch web interface
streamlit run src/app.py

# Start API server
uvicorn src.api:app --reload
```

---

## Model Performance

### Baseline vs Final Model

| Metric | Baseline (LR) | Final (XGBoost) | Improvement |
|--------|---------------|-----------------|-------------|
| Recall | 61.38% | **92.68%** | +31.30% |
| Precision | 72.45% | **88.23%** | +15.78% |
| F1-Score | 66.44% | **90.40%** | +23.96% |
| ROC-AUC | 92.45% | **98.12%** | +5.67% |

### Business Impact

**Test Set Results (56,963 transactions, 98 frauds):**
- **Baseline:** 60 frauds caught (61.38%), 38 missed
- **Final:** 91 frauds caught (92.68%), 7 missed
- **Improvement:** 31 additional frauds detected = ~$310K saved

**Cost Analysis:**
```
Fraud Prevention:    91 × $10K = $910,000
Missed Frauds:        7 × $10K =  $70,000
Net Protection: $840,000 (92.7% coverage)

vs Baseline: $600,000 (61.4% coverage)
Additional Value: $240,000 per 10K transactions
```

---

## Feature Engineering Summary

### Categories (39 new features created)

**1. Time-Based (10 features)**
- Cyclical hour encoding (sin/cos)
- Day period flags (night/morning/afternoon/evening)
- Time differences between transactions
- Impact: +2% recall

**2. Amount-Based (10 features)**
- Log transformation, percentile ranking
- Z-score normalization, categorical bins
- High/low amount flags
- Impact: +5% recall, +4% precision

**3. Statistical (11 features)**
- Rolling statistics (mean, std, min, max)
- Deviation from rolling average
- V feature aggregations
- Impact: +4% recall

**4. Interaction (8 features)**
- V1×V2, V3×V4, V14×V17 (PCA interactions)
- Amount×Time, Amount×Hour interactions
- Impact: +3% recall

### Feature Selection

**Initial:** 69 features (30 original + 39 engineered)  
**Final:** 35 features selected via:
- Tree-based importance (XGBoost)
- SHAP value analysis
- Performance validation

**Top 5 Features:**
1. V14 (PCA component)
2. V17 (PCA component)
3. Amount_log
4. V12 (PCA component)
5. V10 (PCA component)

---

## API Usage

### REST API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 12345,
    "Amount": 150.50,
    "V1": -1.35, "V2": -0.07,
    # ... (V3-V28)
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"Time": 12345, "Amount": 150.50, ...},
      {"Time": 12346, "Amount": 89.99, ...}
    ]
  }'

# CSV upload
curl -X POST "http://localhost:8000/predict/csv" \
  -F "file=@transactions.csv"
```

### Response Format

```json
{
  "prediction": 0,
  "fraud_probability": 0.1234,
  "legitimate_probability": 0.8766,
  "risk_level": "LOW",
  "is_fraud": false,
  "timestamp": "2024-12-06T10:30:00",
  "model_version": "1.0.0"
}
```

---

## Model Monitoring

### Key Metrics to Track

**Performance Metrics (Hourly):**
- Recall (target: >90%)
- Precision (target: >80%)
- Prediction distribution
- Confidence score distribution

**System Metrics (Real-time):**
- Request rate (requests/sec)
- Latency (p50, p95, p99)
- Error rate (%)
- Throughput

**Data Quality (Daily):**
- Feature drift (KS-test)
- Prediction drift
- Missing value patterns
- Outlier frequency

### Retraining Strategy

**Scheduled:** Monthly with 30 days of new data  
**Triggered:** When drift detected or recall drops below 85%  
**Process:** Validate → Retrain → Stage → A/B Test → Deploy

---

## Technology Stack

**Core ML:**
- Python 3.9+
- XGBoost 1.7.6
- Scikit-learn 1.3.0
- Imbalanced-learn 0.11.0
- SHAP 0.42.1

**Web & API:**
- Streamlit 1.25.0
- FastAPI 0.101.0
- Uvicorn 0.23.2

**Data Processing:**
- Pandas 2.0.3
- NumPy 1.24.3

**Visualization:**
- Matplotlib 3.7.2
- Seaborn 0.12.2
- Plotly 5.15.0

---

## TODO

- [x] Baseline model development
- [x] Feature engineering pipeline
- [x] Model optimization (XGBoost + SMOTE)
- [x] SHAP-based feature selection
- [x] Streamlit web interface
- [x] FastAPI REST API
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model monitoring dashboard (Grafana)
- [ ] Real-time streaming (Kafka integration)
- [ ] Model versioning (MLflow)
- [ ] A/B testing framework
- [ ] Automated retraining pipeline
- [ ] Advanced deep learning models (LSTM/Transformer)
- [ ] Threshold optimization for different business scenarios
- [ ] Cost-sensitive learning implementation
- [ ] Explainability dashboard for fraud analysts
- [ ] Mobile app integration

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

```bibtex
@software{fraud_detection_2024,
  author = {Your Name},
  title = {Credit Card Fraud Detection System},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/fraud-detection}
}
```

---

## Contact

**Your Name**
- Email: your.email@example.com
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

---

## Acknowledgments

- Dataset: [Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Inspiration: Real-world fraud detection systems in banking
- Community: Kaggle discussions and competition notebooks

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Last Updated: December 2024

</div>
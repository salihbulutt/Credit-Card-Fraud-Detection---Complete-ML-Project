"""
Configuration file for Credit Card Fraud Detection Project
Contains all paths, parameters, and business rules
"""

from pathlib import Path
import os

# ==================== PROJECT PATHS ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== DATA PATHS ====================
RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation.csv"

# ==================== MODEL PATHS ====================
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_SELECTOR_PATH = MODELS_DIR / "feature_selector.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

# ==================== BUSINESS RULES ====================
# Fraud Detection Thresholds
FRAUD_PROBABILITY_THRESHOLD = 0.5  # Main classification threshold
HIGH_RISK_THRESHOLD = 0.8  # Threshold for high-risk transactions
MEDIUM_RISK_THRESHOLD = 0.5  # Threshold for medium-risk transactions

# Business Requirements
MIN_RECALL_REQUIREMENT = 0.80  # Must catch at least 80% of frauds
MIN_PRECISION_REQUIREMENT = 0.85  # At least 85% of alerts should be real frauds
MAX_FALSE_POSITIVE_RATE = 0.02  # Max 2% false positive rate

# Cost-Sensitive Parameters
FRAUD_COST = 100  # Average cost of missing a fraud ($)
FALSE_ALARM_COST = 5  # Cost of investigating a false positive ($)
COST_RATIO = FRAUD_COST / FALSE_ALARM_COST  # 20:1 ratio

# Transaction Amount Thresholds
LARGE_TRANSACTION_THRESHOLD = 1000  # Transactions above this require extra scrutiny
SMALL_TRANSACTION_THRESHOLD = 10  # Very small transactions (different fraud pattern)

# Time-based Rules
SUSPICIOUS_HOURS = list(range(0, 6))  # Midnight to 6 AM (higher fraud rate)
BUSINESS_HOURS = list(range(9, 18))  # 9 AM to 6 PM

# ==================== MODEL PARAMETERS ====================
# Random State for Reproducibility
RANDOM_STATE = 42

# Train-Test Split
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Cross-Validation
CV_FOLDS = 5
CV_STRATEGY = "stratified"  # Options: 'stratified', 'time_series'

# Class Imbalance Handling
IMBALANCE_STRATEGY = "smote"  # Options: 'smote', 'undersample', 'class_weight'
SAMPLING_STRATEGY = 0.3  # Ratio of minority to majority after resampling
SMOTE_K_NEIGHBORS = 5

# Feature Engineering
ENABLE_TIME_FEATURES = True
ENABLE_AMOUNT_FEATURES = True
ENABLE_STATISTICAL_FEATURES = True
ENABLE_INTERACTION_FEATURES = True

# Feature Selection
FEATURE_SELECTION_METHOD = "shap"  # Options: 'shap', 'importance', 'rfe'
N_FEATURES_TO_SELECT = 42  # Number of features to keep
MIN_FEATURE_IMPORTANCE = 0.001  # Minimum importance threshold

# ==================== BASELINE MODEL ====================
BASELINE_MODEL = {
    "name": "LogisticRegression",
    "params": {
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs"
    }
}

# ==================== FINAL MODEL PARAMETERS ====================
FINAL_MODEL = {
    "name": "XGBoost",
    "params": {
        "random_state": RANDOM_STATE,
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 20,  # Adjusted for imbalance
        "gamma": 1,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": -1
    }
}

# Alternative Models to Consider
ALTERNATIVE_MODELS = {
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "LightGBM": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 20,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    },
    "CatBoost": {
        "iterations": 300,
        "depth": 5,
        "learning_rate": 0.01,
        "auto_class_weights": "Balanced",
        "random_state": RANDOM_STATE,
        "verbose": False
    }
}

# ==================== HYPERPARAMETER OPTIMIZATION ====================
OPTUNA_CONFIG = {
    "n_trials": 100,
    "timeout": 3600,  # 1 hour
    "n_jobs": -1,
    "study_name": "fraud_detection_optimization",
    "direction": "maximize",  # Maximize PR-AUC
    "sampler": "TPE"  # Tree-structured Parzen Estimator
}

# Hyperparameter Search Space
HYPERPARAMETER_SPACE = {
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.5, 1, 2],
        "min_child_weight": [1, 3, 5, 7],
        "scale_pos_weight": [10, 15, 20, 25, 30],
        "reg_alpha": [0, 0.01, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0]
    }
}

# ==================== EVALUATION METRICS ====================
PRIMARY_METRIC = "pr_auc"  # Precision-Recall AUC (best for imbalanced data)
SECONDARY_METRICS = [
    "roc_auc",
    "f1_score",
    "precision",
    "recall",
    "average_precision",
    "matthews_corrcoef"
]

# Metrics for Business Reporting
BUSINESS_METRICS = [
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
    "cost_savings",
    "investigation_workload"
]

# ==================== PREPROCESSING PARAMETERS ====================
# Scaling
SCALING_METHOD = "robust"  # Options: 'standard', 'robust', 'minmax'
SCALING_FEATURES = ["Amount", "Time"]  # Features to scale

# Outlier Detection
OUTLIER_METHOD = "iqr"  # Options: 'iqr', 'zscore', 'isolation_forest'
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3

# Missing Value Handling
MISSING_VALUE_STRATEGY = "median"  # Options: 'mean', 'median', 'mode', 'drop'
MISSING_THRESHOLD = 0.5  # Drop features with >50% missing values

# ==================== FEATURE ENGINEERING CONFIG ====================
# Time Features
TIME_FEATURES = [
    "hour_of_day",
    "day_period",  # morning, afternoon, evening, night
    "is_weekend",
    "is_business_hours",
    "is_suspicious_hour"
]

# Amount Features
AMOUNT_FEATURES = [
    "amount_log",
    "amount_zscore",
    "amount_bin",
    "is_large_transaction",
    "is_small_transaction"
]

# Statistical Features
STATISTICAL_FEATURES = [
    "amount_rolling_mean_10",
    "amount_rolling_std_10",
    "transaction_velocity"
]

# Interaction Features (top performing combinations)
INTERACTION_FEATURES = [
    ("V1", "V2"),
    ("V14", "V17"),
    ("V12", "V14"),
    ("Amount", "V14")
]

# ==================== MONITORING & LOGGING ====================
# Logging Configuration
LOG_LEVEL = "INFO"  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "fraud_detection.log"

# Model Monitoring Thresholds
MONITORING_THRESHOLDS = {
    "pr_auc_min": 0.85,  # Alert if PR-AUC drops below this
    "data_drift_psi_max": 0.2,  # Alert if PSI exceeds this
    "prediction_latency_max": 100,  # Max acceptable latency in ms
    "false_positive_rate_max": 0.025,  # Alert if FPR exceeds 2.5%
    "fraud_rate_change_max": 0.5  # Alert if fraud rate changes by >50%
}

# Retraining Triggers
RETRAINING_CONFIG = {
    "schedule": "weekly",  # Options: 'daily', 'weekly', 'monthly'
    "performance_degradation_threshold": 0.05,  # 5% drop in PR-AUC
    "data_drift_threshold": 0.25,  # PSI threshold
    "min_new_samples": 10000  # Minimum new samples before retraining
}

# ==================== API CONFIGURATION ====================
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "Credit Card Fraud Detection API",
    "description": "Real-time fraud detection API for credit card transactions",
    "version": "1.0.0",
    "rate_limit": "100/minute",
    "timeout": 30  # seconds
}

# ==================== STREAMLIT APP CONFIG ====================
STREAMLIT_CONFIG = {
    "page_title": "Fraud Detection System",
    "page_icon": "💳",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": "#FF4B4B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
        "font": "sans serif"
    }
}

# ==================== FEATURE COLUMNS ====================
# Original PCA Features
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

# Original Features
ORIGINAL_FEATURES = PCA_FEATURES + ["Time", "Amount"]

# Target Variable
TARGET_COLUMN = "Class"

# Features to Exclude from Modeling
EXCLUDE_FEATURES = ["Time"]  # Time is used for feature engineering only

# ==================== DOCKER & DEPLOYMENT ====================
DOCKER_CONFIG = {
    "image_name": "fraud-detection",
    "tag": "latest",
    "port": 8501,
    "memory_limit": "4g",
    "cpu_limit": "2"
}

# Environment Variables
ENV_VARS = {
    "MODEL_PATH": str(FINAL_MODEL_PATH),
    "LOG_LEVEL": LOG_LEVEL,
    "API_PORT": str(API_CONFIG["port"])
}

# ==================== HELPER FUNCTIONS ====================
def get_model_config(model_name: str) -> dict:
    """Get model configuration by name"""
    if model_name == "baseline":
        return BASELINE_MODEL
    elif model_name == "final":
        return FINAL_MODEL
    elif model_name in ALTERNATIVE_MODELS:
        return {"name": model_name, "params": ALTERNATIVE_MODELS[model_name]}
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability"""
    if probability >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif probability >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"

def calculate_expected_cost(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Calculate expected costs based on confusion matrix"""
    fraud_losses = fn * FRAUD_COST
    investigation_costs = fp * FALSE_ALARM_COST
    total_cost = fraud_losses + investigation_costs
    
    # Calculate savings (what would have been lost without the model)
    potential_losses = (tp + fn) * FRAUD_COST
    actual_savings = tp * FRAUD_COST - investigation_costs
    
    return {
        "fraud_losses": fraud_losses,
        "investigation_costs": investigation_costs,
        "total_cost": total_cost,
        "potential_losses": potential_losses,
        "actual_savings": actual_savings,
        "savings_rate": (actual_savings / potential_losses) if potential_losses > 0 else 0
    }

# ==================== VALIDATION ====================
def validate_config():
    """Validate configuration settings"""
    assert 0 < TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
    assert 0 < VALIDATION_SIZE < 1, "VALIDATION_SIZE must be between 0 and 1"
    assert CV_FOLDS > 1, "CV_FOLDS must be greater than 1"
    assert 0 < FRAUD_PROBABILITY_THRESHOLD < 1, "FRAUD_PROBABILITY_THRESHOLD must be between 0 and 1"
    assert MIN_RECALL_REQUIREMENT <= 1, "MIN_RECALL_REQUIREMENT must be <= 1"
    assert MIN_PRECISION_REQUIREMENT <= 1, "MIN_PRECISION_REQUIREMENT must be <= 1"
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"Project Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")

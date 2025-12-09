from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"

# ML constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # this is % of train set
TARGET_COL = "Class"

# Columns
TIME_COL = "Time"
AMOUNT_COL = "Amount"

# PCA components V1 … V28
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

FEATURE_COLS = [TIME_COL, AMOUNT_COL] + PCA_FEATURES

# Business constraint
MIN_PRECISION = 0.90

import joblib
import pandas as pd

from .config import (
    MODEL_PATH,
    THRESHOLD_PATH,
    FEATURE_COLS
)
from .utils import load_threshold

class FraudModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.threshold = load_threshold(THRESHOLD_PATH)

    def predict_single(self, features: dict):
        """
        features: dict with keys = FEATURE_COLS
        """
        df = pd.DataFrame([features])[FEATURE_COLS]
        proba = self.model.predict_proba(df)[:, 1][0]
        label = proba >= self.threshold

        return {
            "fraud_probability": float(proba),
            "is_fraud": bool(label),
            "threshold": float(self.threshold)
        }

    def predict_batch(self, df: pd.DataFrame):
        df = df[FEATURE_COLS]
        proba = self.model.predict_proba(df)[:, 1]
        labels = proba >= self.threshold
        return proba, labels

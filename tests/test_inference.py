import pandas as pd
from src.inference import FraudModelService
from src.config import FEATURE_COLS

def test_inference_single_prediction():
    """
    Checks if single prediction call works and returns
    required fields.
    """

    service = FraudModelService()

    # Create minimal valid feature input
    sample = {col: 0.0 for col in FEATURE_COLS}

    result = service.predict_single(sample)

    assert "fraud_probability" in result
    assert "is_fraud" in result
    assert "threshold" in result


def test_batch_prediction():
    """
    Checks batch prediction on a small dummy dataframe.
    """

    service = FraudModelService()

    df = pd.DataFrame([{col: 0.0 for col in FEATURE_COLS} for _ in range(5)])

    proba, labels = service.predict_batch(df)

    assert len(proba) == 5
    assert len(labels) == 5

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from .config import (
    RAW_DATA_PATH,
    TARGET_COL,
    FEATURE_COLS,
    TIME_COL,
    AMOUNT_COL,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)

def load_raw_data():
    return pd.read_csv(RAW_DATA_PATH)

def train_val_test_split(df):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # train vs val
    val_relative = VAL_SIZE / (1 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor():
    """
    Scale Time & Amount, leave PCA features unchanged.
    """
    numeric_scaled = [TIME_COL, AMOUNT_COL]
    passthrough_cols = [c for c in FEATURE_COLS if c not in numeric_scaled]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), numeric_scaled),
            ("pass", "passthrough", passthrough_cols),
        ]
    )
    return preprocessor

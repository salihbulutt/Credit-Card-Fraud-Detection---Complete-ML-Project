from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_baseline_model():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1
    )

def get_rf_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

def get_xgb_model():
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1
    )

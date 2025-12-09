import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline as SKPipeline

from .config import (
    MODELS_DIR,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_PATH,
    MIN_PRECISION
)
from .data_prep import load_raw_data, train_val_test_split, build_preprocessor
from .models import get_xgb_model
from .utils import (
    evaluate_probabilities,
    choose_threshold_for_min_precision,
    save_threshold
)

def run_training_pipeline():
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    print("📌 Loading data...")
    df = load_raw_data()

    print("📌 Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    print("📌 Building preprocessor...")
    preprocessor = build_preprocessor()

    print("📌 Initializing model (XGBoost)...")
    model = get_xgb_model()

    clf = SKPipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("📌 Training model...")
    clf.fit(X_train, y_train)

    print("📌 Validating model...")
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    metrics = evaluate_probabilities(y_val, y_val_proba)
    print("Validation Metrics:", metrics)

    print(f"📌 Selecting best threshold for precision >= {MIN_PRECISION}...")
    threshold, best_recall = choose_threshold_for_min_precision(
        y_val, y_val_proba, MIN_PRECISION
    )
    print(f"Chosen threshold: {threshold:.4f} | Recall: {best_recall:.4f}")

    print("📌 Saving model...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    save_threshold(threshold, THRESHOLD_PATH)

    print("🎉 Training completed. Model saved!")

if __name__ == "__main__":
    run_training_pipeline()

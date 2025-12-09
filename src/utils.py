import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

def evaluate_probabilities(y_true, y_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba)
    }

def choose_threshold_for_min_precision(y_true, y_proba, min_precision):
    """
    Finds highest recall while ensuring precision >= min_precision.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best_threshold = 0.5
    best_recall = 0

    for p, r, t in zip(precisions, recalls, list(thresholds)+[thresholds[-1]]):
        if p >= min_precision and r > best_recall:
            best_threshold = t
            best_recall = r

    return float(best_threshold), float(best_recall)

def save_threshold(threshold, path: Path):
    with open(path, "w") as f:
        json.dump({"threshold": threshold}, f)

def load_threshold(path: Path):
    with open(path) as f:
        return json.load(f)["threshold"]

def print_classification_info(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

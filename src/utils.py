"""
Utility Functions
Common helper functions used across the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef
)
from typing import Dict, Tuple, List
import json
import pickle
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, labels=['Legitimate', 'Fraud'], 
                         figsize=(8, 6), save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        figsize: Figure size
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return cm


def plot_roc_curve(y_true, y_pred_proba, figsize=(8, 6), save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fpr, tpr, thresholds, roc_auc


def plot_precision_recall_curve(y_true, y_pred_proba, figsize=(8, 6), save_path=None):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        figsize: Figure size
        save_path: Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=y_true.mean(), color='red', linestyle='--',
                label=f'Baseline ({y_true.mean():.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return precision, recall, thresholds, pr_auc


def calculate_metrics(y_true, y_pred, y_pred_proba=None) -> Dict:
    """
    Calculate comprehensive metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, matthews_corrcoef
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # Add probability-based metrics if available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics


def print_metrics(metrics: Dict, title="Model Performance"):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print("\n" + "="*70)
    print(f"{title.upper()}")
    print("="*70)
    
    # Main metrics
    print("\nClassification Metrics:")
    print(f"  Accuracy:   {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:  {metrics.get('precision', 0):.4f}")
    print(f"  Recall:     {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:   {metrics.get('f1_score', 0):.4f}")
    print(f"  MCC:        {metrics.get('mcc', 0):.4f}")
    
    # AUC metrics
    if 'roc_auc' in metrics:
        print("\nAUC Metrics:")
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics.get('true_negatives', 0):,}")
    print(f"  False Positives: {metrics.get('false_positives', 0):,}")
    print(f"  False Negatives: {metrics.get('false_negatives', 0):,}")
    print(f"  True Positives:  {metrics.get('true_positives', 0):,}")
    
    # Additional metrics
    print("\nAdditional Metrics:")
    print(f"  Specificity: {metrics.get('specificity', 0):.4f}")
    print(f"  FPR:         {metrics.get('fpr', 0):.4f}")
    print(f"  FNR:         {metrics.get('fnr', 0):.4f}")


def save_model(model, filepath: str):
    """
    Save model to pickle file
    
    Args:
        model: Model object to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved to {filepath}")


def load_model(filepath: str):
    """
    Load model from pickle file
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded from {filepath}")
    return model


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✓ Data saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Data loaded from {filepath}")
    return data


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Find optimal classification threshold
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    # Plot threshold vs score
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, linewidth=2)
    plt.axvline(optimal_threshold, color='red', linestyle='--',
                label=f'Optimal: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric.capitalize()} Score')
    plt.title(f'Threshold Optimization ({metric.upper()})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    return optimal_threshold, best_score


def calculate_business_impact(tp, fp, tn, fn, 
                              fraud_cost=100, 
                              false_alarm_cost=5) -> Dict:
    """
    Calculate business impact metrics
    
    Args:
        tp, fp, tn, fn: Confusion matrix values
        fraud_cost: Cost of missing a fraud
        false_alarm_cost: Cost of false alarm
        
    Returns:
        Dictionary with business metrics
    """
    fraud_losses = fn * fraud_cost
    investigation_costs = fp * false_alarm_cost
    total_cost = fraud_losses + investigation_costs
    
    potential_losses = (tp + fn) * fraud_cost
    actual_savings = tp * fraud_cost - investigation_costs
    
    return {
        'fraud_losses': fraud_losses,
        'investigation_costs': investigation_costs,
        'total_cost': total_cost,
        'potential_losses': potential_losses,
        'actual_savings': actual_savings,
        'savings_rate': (actual_savings / potential_losses) if potential_losses > 0 else 0,
        'cost_per_transaction': total_cost / (tp + fp + tn + fn),
        'roi': (actual_savings / investigation_costs) if investigation_costs > 0 else 0
    }


def create_feature_importance_plot(feature_importance: Dict, 
                                  top_n=20, 
                                  figsize=(10, 8),
                                  save_path=None):
    """
    Create feature importance plot
    
    Args:
        feature_importance: Dictionary of feature: importance
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure
    """
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: abs(x[1]), 
                           reverse=True)[:top_n]
    
    features = [f[0] for f in sorted_features]
    importances = [f[1] for f in sorted_features]
    
    plt.figure(figsize=figsize)
    colors = ['red' if x < 0 else 'green' for x in importances]
    plt.barh(range(len(features)), importances, color=colors, edgecolor='black')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_models(results: Dict[str, Dict], figsize=(14, 6)):
    """
    Compare multiple models visually
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        figsize: Figure size
    """
    metrics_to_plot = ['pr_auc', 'roc_auc', 'f1_score', 'precision', 'recall']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    models = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(models)
    
    for i, (model_name, metrics) in enumerate(results.items()):
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        axes[0].bar(x + i * width, values, width, label=model_name)
    
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Comparison - Metrics')
    axes[0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[0].set_xticklabels(metrics_to_plot, rotation=45)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    ax = plt.subplot(122, projection='polar')
    
    for model_name, metrics in results.items():
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison - Radar')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Utility functions module loaded successfully")
    print("Available functions:")
    print("  - plot_confusion_matrix")
    print("  - plot_roc_curve")
    print("  - plot_precision_recall_curve")
    print("  - calculate_metrics")
    print("  - find_optimal_threshold")
    print("  - calculate_business_impact")
    print("  - create_feature_importance_plot")
    print("  - compare_models")

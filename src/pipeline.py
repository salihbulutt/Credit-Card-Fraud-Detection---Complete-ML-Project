"""
Complete ML Pipeline for Credit Card Fraud Detection
Executes the full training pipeline from data loading to model deployment
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from src.config import *


class FraudDetectionPipeline:
    """
    End-to-end ML Pipeline for Fraud Detection
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_history = {}
        
    def load_data(self, filepath=None):
        """Load raw data"""
        filepath = filepath or RAW_DATA_PATH
        print(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df):,} transactions")
        print(f"✓ Fraud rate: {df[TARGET_COLUMN].mean():.2%}")
        
        return df
    
    def split_data(self, df, test_size=TEST_SIZE, val_size=VALIDATION_SIZE):
        """Split data into train, validation, and test sets"""
        print("\nSplitting data...")
        
        # First split: train+val vs test
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=RANDOM_STATE, stratify=y_temp
        )
        
        print(f"✓ Train set: {len(X_train):,} ({len(X_train)/len(df):.1%})")
        print(f"✓ Validation set: {len(X_val):,} ({len(X_val)/len(df):.1%})")
        print(f"✓ Test set: {len(X_test):,} ({len(X_test)/len(df):.1%})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def engineer_features(self, X):
        """Create engineered features"""
        print("\nEngineering features...")
        X = X.copy()
        
        # Time-based features
        if 'Time' in X.columns:
            X['hour_of_day'] = (X['Time'] / 3600) % 24
            X['is_night'] = ((X['hour_of_day'] >= 0) & (X['hour_of_day'] < 6)).astype(int)
            X['is_business_hours'] = ((X['hour_of_day'] >= 9) & (X['hour_of_day'] < 18)).astype(int)
            print("✓ Created time-based features")
        
        # Amount-based features
        if 'Amount' in X.columns:
            X['amount_log'] = np.log1p(X['Amount'])
            X['is_large_transaction'] = (X['Amount'] > LARGE_TRANSACTION_THRESHOLD).astype(int)
            X['is_small_transaction'] = (X['Amount'] < SMALL_TRANSACTION_THRESHOLD).astype(int)
            
            # Amount statistics
            amount_mean = X['Amount'].mean()
            amount_std = X['Amount'].std()
            X['amount_zscore'] = (X['Amount'] - amount_mean) / amount_std
            print("✓ Created amount-based features")
        
        # Interaction features
        if all(f'V{i}' in X.columns for i in [1, 2, 12, 14, 17]):
            X['V1_V2_interaction'] = X['V1'] * X['V2']
            X['V14_V17_interaction'] = X['V14'] * X['V17']
            X['V12_V14_interaction'] = X['V12'] * X['V14']
            print("✓ Created interaction features")
        
        # Drop Time column (used only for feature engineering)
        if 'Time' in X.columns:
            X = X.drop(columns=['Time'])
        
        print(f"✓ Total features: {X.shape[1]}")
        return X
    
    def preprocess_data(self, X_train, X_val, X_test):
        """Preprocess and scale features"""
        print("\nPreprocessing data...")
        
        # Feature engineering
        X_train = self.engineer_features(X_train)
        X_val = self.engineer_features(X_val)
        X_test = self.engineer_features(X_test)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_names, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)
        
        print("✓ Features scaled using RobustScaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def handle_imbalance(self, X_train, y_train):
        """Handle class imbalance using SMOTE"""
        print("\nHandling class imbalance...")
        print(f"Before SMOTE - Class distribution:")
        print(f"  Legitimate: {(y_train == 0).sum():,} ({(y_train == 0).mean():.2%})")
        print(f"  Fraud: {(y_train == 1).sum():,} ({(y_train == 1).mean():.2%})")
        
        smote = SMOTE(
            sampling_strategy=SAMPLING_STRATEGY,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=RANDOM_STATE
        )
        
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"After SMOTE - Class distribution:")
        print(f"  Legitimate: {(y_resampled == 0).sum():,} ({(y_resampled == 0).mean():.2%})")
        print(f"  Fraud: {(y_resampled == 1).sum():,} ({(y_resampled == 1).mean():.2%})")
        
        return X_resampled, y_resampled
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\nTraining XGBoost model...")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Training parameters
        params = FINAL_MODEL['params'].copy()
        num_rounds = params.pop('n_estimators')
        
        # Training with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=20,
            verbose_eval=50
        )
        
        print(f"✓ Model trained with {self.model.best_ntree_limit} trees")
        
        # Store training history
        self.training_history = evals_result
        
        return self.model
    
    def evaluate_model(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        print(f"\nEvaluating on {dataset_name} set...")
        
        # Predictions
        dtest = xgb.DMatrix(X)
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba >= FRAUD_PROBABILITY_THRESHOLD).astype(int)
        
        # Calculate metrics
        metrics = {
            'pr_auc': average_precision_score(y, y_pred_proba),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate business metrics
        business_metrics = calculate_expected_cost(tp, fp, tn, fn)
        
        print(f"\n{'='*50}")
        print(f"{dataset_name.upper()} SET PERFORMANCE")
        print(f"{'='*50}")
        print(f"PR-AUC:     {metrics['pr_auc']:.4f}")
        print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"F1-Score:   {metrics['f1_score']:.4f}")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Positives:  {tp:,}")
        
        print(f"\nBusiness Metrics:")
        print(f"  Fraud Losses:        ${business_metrics['fraud_losses']:,.2f}")
        print(f"  Investigation Costs: ${business_metrics['investigation_costs']:,.2f}")
        print(f"  Total Cost:          ${business_metrics['total_cost']:,.2f}")
        print(f"  Actual Savings:      ${business_metrics['actual_savings']:,.2f}")
        print(f"  Savings Rate:        {business_metrics['savings_rate']:.1%}")
        
        # Check business requirements
        print(f"\nBusiness Requirements Check:")
        recall_pass = metrics['recall'] >= MIN_RECALL_REQUIREMENT
        precision_pass = metrics['precision'] >= MIN_PRECISION_REQUIREMENT
        print(f"  Recall ≥ {MIN_RECALL_REQUIREMENT:.0%}: {'✓' if recall_pass else '✗'} ({metrics['recall']:.2%})")
        print(f"  Precision ≥ {MIN_PRECISION_REQUIREMENT:.0%}: {'✓' if precision_pass else '✗'} ({metrics['precision']:.2%})")
        
        return metrics, business_metrics
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance"""
        importance = self.model.get_score(importance_type='weight')
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False).head(top_n)
        
        print(f"\nTop {top_n} Most Important Features:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:25s}: {row['importance']:.0f}")
        
        return importance_df
    
    def save_artifacts(self):
        """Save model, scaler, and feature names"""
        print("\nSaving artifacts...")
        
        # Save model
        self.model.save_model(str(FINAL_MODEL_PATH))
        print(f"✓ Model saved to {FINAL_MODEL_PATH}")
        
        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to {SCALER_PATH}")
        
        # Save feature names
        with open(FEATURE_NAMES_PATH, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"✓ Feature names saved to {FEATURE_NAMES_PATH}")
        
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        start_time = datetime.now()
        print("="*70)
        print("CREDIT CARD FRAUD DETECTION - FULL ML PIPELINE")
        print("="*70)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Load data
        df = self.load_data()
        
        # 2. Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
        
        # 3. Preprocess data
        X_train, X_val, X_test = self.preprocess_data(X_train, X_val, X_test)
        
        # 4. Handle imbalance (only on training set)
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        
        # 5. Train model
        self.train_model(X_train_balanced, y_train_balanced, X_val, y_val)
        
        # 6. Evaluate on validation set
        val_metrics, val_business = self.evaluate_model(X_val, y_val, "Validation")
        
        # 7. Evaluate on test set
        test_metrics, test_business = self.evaluate_model(X_test, y_test, "Test")
        
        # 8. Feature importance
        importance_df = self.get_feature_importance()
        
        # 9. Save artifacts
        self.save_artifacts()
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Duration: {duration:.1f} seconds")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nFinal Test Performance:")
        print(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"\nModel artifacts saved to: {MODELS_DIR}")
        print("="*70)
        
        return {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'val_business': val_business,
            'test_business': test_business,
            'feature_importance': importance_df,
            'duration': duration
        }


def main():
    """Main execution function"""
    # Validate configuration
    validate_config()
    
    # Initialize and run pipeline
    pipeline = FraudDetectionPipeline()
    results = pipeline.run_full_pipeline()
    
    return results


if __name__ == "__main__":
    results = main()

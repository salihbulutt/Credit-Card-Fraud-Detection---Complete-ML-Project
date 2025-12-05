"""
Complete ML Pipeline for Credit Card Fraud Detection
Handles data loading, preprocessing, training, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    BASELINE_MODEL_PATH,
    FINAL_MODEL_PATH,
    SCALER_PATH,
    PREPROCESSOR_PATH,
    FEATURE_COLS,
    TARGET_COL,
    RANDOM_STATE,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    CV_FOLDS,
    BASELINE_CONFIG,
    FINAL_MODEL_CONFIG,
    SMOTE_SAMPLING_RATIO,
    PREDICTION_THRESHOLD
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Complete ML Pipeline for Fraud Detection
    """
    
    def __init__(self):
        """Initialize the pipeline"""
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.scaler = None
        self.model = None
        
        logger.info("Pipeline initialized")
    
    def load_data(self, data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
        """
        Load the credit card fraud dataset
        
        Args:
            data_path: Path to the CSV file
            
        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading data from {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Check class distribution
            class_dist = df[TARGET_COL].value_counts()
            logger.info(f"Class distribution:\n{class_dist}")
            fraud_pct = class_dist[1] / len(df) * 100
            logger.info(f"Fraud percentage: {fraud_pct:.4f}%")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {data_path}")
            logger.info("Please download the dataset from Kaggle")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def explore_data(self, df: pd.DataFrame):
        """
        Basic exploratory data analysis
        
        Args:
            df: Input dataframe
        """
        logger.info("=" * 50)
        logger.info("EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 50)
        
        # Basic info
        logger.info(f"\nDataset shape: {df.shape}")
        logger.info(f"Number of features: {len(FEATURE_COLS)}")
        logger.info(f"Target variable: {TARGET_COL}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"\nMissing values:\n{missing[missing > 0]}")
        else:
            logger.info("\nNo missing values found")
        
        # Statistical summary
        logger.info(f"\nAmount statistics:")
        logger.info(df['Amount'].describe())
        
        logger.info(f"\nTime statistics:")
        logger.info(df['Time'].describe())
        
        # Class distribution
        logger.info(f"\nClass distribution:")
        logger.info(df[TARGET_COL].value_counts())
        logger.info(f"Imbalance ratio: {df[TARGET_COL].value_counts()[0] / df[TARGET_COL].value_counts()[1]:.2f}:1")
    
    def split_data(self, df: pd.DataFrame):
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input dataframe
        """
        logger.info("Splitting data...")
        
        # Separate features and target
        X = df[FEATURE_COLS].copy()
        y = df[TARGET_COL].copy()
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Validation set: {self.X_val.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        
        # Check stratification
        logger.info(f"\nTrain fraud %: {self.y_train.sum() / len(self.y_train) * 100:.4f}%")
        logger.info(f"Val fraud %: {self.y_val.sum() / len(self.y_val) * 100:.4f}%")
        logger.info(f"Test fraud %: {self.y_test.sum() / len(self.y_test) * 100:.4f}%")
    
    def preprocess_data(self):
        """
        Preprocess the data with scaling
        """
        logger.info("Preprocessing data...")
        
        # Initialize scalers
        self.scaler = StandardScaler()
        
        # Fit scaler on training data only
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info("Data preprocessing completed")
    
    def train_baseline(self):
        """
        Train baseline model (Logistic Regression)
        """
        logger.info("=" * 50)
        logger.info("TRAINING BASELINE MODEL")
        logger.info("=" * 50)
        
        # Initialize model
        baseline_model = LogisticRegression(**BASELINE_CONFIG['params'])
        
        # Train
        logger.info("Training Logistic Regression...")
        baseline_model.fit(self.X_train_scaled, self.y_train)
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            baseline_model,
            self.X_train_scaled,
            self.y_train,
            cv=cv,
            scoring='recall'
        )
        logger.info(f"CV Recall scores: {cv_scores}")
        logger.info(f"Mean CV Recall: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Evaluate
        self.evaluate_model(baseline_model, "Baseline (Logistic Regression)")
        
        # Save model
        joblib.dump(baseline_model, BASELINE_MODEL_PATH)
        logger.info(f"Baseline model saved to {BASELINE_MODEL_PATH}")
        
        return baseline_model
    
    def train_with_smote(self):
        """
        Train model with SMOTE for handling imbalanced data
        """
        logger.info("=" * 50)
        logger.info("TRAINING WITH SMOTE")
        logger.info("=" * 50)
        
        # Create pipeline with SMOTE
        smote = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_RATIO,
            random_state=RANDOM_STATE
        )
        
        # Apply SMOTE
        X_train_resampled, y_train_resampled = smote.fit_resample(
            self.X_train_scaled,
            self.y_train
        )
        
        logger.info(f"Original training set: {self.X_train_scaled.shape}")
        logger.info(f"Resampled training set: {X_train_resampled.shape}")
        logger.info(f"New class distribution: {pd.Series(y_train_resampled).value_counts()}")
        
        # Train XGBoost
        logger.info("Training XGBoost with resampled data...")
        model = XGBClassifier(**FINAL_MODEL_CONFIG['params'])
        model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate
        self.evaluate_model(model, "XGBoost with SMOTE")
        
        return model
    
    def train_final_model(self):
        """
        Train final optimized model
        """
        logger.info("=" * 50)
        logger.info("TRAINING FINAL MODEL")
        logger.info("=" * 50)
        
        # Use SMOTE
        smote = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_RATIO,
            random_state=RANDOM_STATE
        )
        
        X_train_resampled, y_train_resampled = smote.fit_resample(
            self.X_train_scaled,
            self.y_train
        )
        
        # Initialize final model
        self.model = XGBClassifier(**FINAL_MODEL_CONFIG['params'])
        
        # Train
        logger.info("Training final XGBoost model...")
        self.model.fit(
            X_train_resampled,
            y_train_resampled,
            eval_set=[(self.X_val_scaled, self.y_val)],
            verbose=False
        )
        
        # Evaluate
        self.evaluate_model(self.model, "Final Model (XGBoost)")
        
        # Save model and preprocessors
        joblib.dump(self.model, FINAL_MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info(f"Final model saved to {FINAL_MODEL_PATH}")
        logger.info(f"Scaler saved to {SCALER_PATH}")
        
        return self.model
    
    def evaluate_model(self, model, model_name: str):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            model_name: Name for logging
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_val_pred = model.predict(self.X_val_scaled)
        y_test_pred = model.predict(self.X_test_scaled)
        
        # Probabilities
        y_val_proba = model.predict_proba(self.X_val_scaled)[:, 1]
        y_test_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Apply custom threshold
        y_val_pred_thresh = (y_val_proba >= PREDICTION_THRESHOLD).astype(int)
        y_test_pred_thresh = (y_test_proba >= PREDICTION_THRESHOLD).astype(int)
        
        # Calculate metrics for each set
        sets = [
            ("Train", self.y_train, y_train_pred),
            ("Validation", self.y_val, y_val_pred_thresh),
            ("Test", self.y_test, y_test_pred_thresh)
        ]
        
        for set_name, y_true, y_pred in sets:
            logger.info(f"\n{set_name} Set Metrics:")
            logger.info(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
            logger.info(f"Precision: {precision_score(y_true, y_pred):.4f}")
            logger.info(f"Recall:    {recall_score(y_true, y_pred):.4f}")
            logger.info(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
            
            if set_name != "Train":
                logger.info(f"ROC-AUC:   {roc_auc_score(y_true, y_pred):.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
            logger.info(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
            
            # Business metrics
            if set_name == "Test":
                total_frauds = y_true.sum()
                caught_frauds = cm[1, 1]
                missed_frauds = cm[1, 0]
                false_alarms = cm[0, 1]
                
                logger.info(f"\n Business Impact:")
                logger.info(f"Total fraud cases: {total_frauds}")
                logger.info(f"Caught: {caught_frauds} ({caught_frauds/total_frauds*100:.1f}%)")
                logger.info(f"Missed: {missed_frauds} ({missed_frauds/total_frauds*100:.1f}%)")
                logger.info(f"False alarms: {false_alarms}")
    
    def compare_models(self, baseline, final):
        """
        Compare baseline and final model performance
        """
        logger.info("=" * 50)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 50)
        
        # Predictions
        baseline_pred = baseline.predict(self.X_test_scaled)
        final_pred = final.predict(self.X_test_scaled)
        
        # Apply threshold to final model
        final_proba = final.predict_proba(self.X_test_scaled)[:, 1]
        final_pred_thresh = (final_proba >= PREDICTION_THRESHOLD).astype(int)
        
        # Metrics
        metrics = {
            'Accuracy': accuracy_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1-Score': f1_score,
            'ROC-AUC': roc_auc_score
        }
        
        logger.info("\nMetric Comparison (Test Set):")
        logger.info(f"{'Metric':<15} {'Baseline':<12} {'Final Model':<12} {'Improvement'}")
        logger.info("-" * 55)
        
        for metric_name, metric_func in metrics.items():
            baseline_score = metric_func(self.y_test, baseline_pred)
            final_score = metric_func(self.y_test, final_pred_thresh)
            improvement = ((final_score - baseline_score) / baseline_score) * 100
            
            logger.info(
                f"{metric_name:<15} {baseline_score:<12.4f} "
                f"{final_score:<12.4f} {improvement:>+6.2f}%"
            )
    
    def get_feature_importance(self, n_features: int = 20):
        """
        Get and display feature importance
        
        Args:
            n_features: Number of top features to show
        """
        if self.model is None:
            logger.warning("Model not trained yet")
            return
        
        logger.info("=" * 50)
        logger.info("FEATURE IMPORTANCE")
        logger.info("=" * 50)
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_names = FEATURE_COLS
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        logger.info(f"\nTop {n_features} Most Important Features:")
        logger.info(importance_df.head(n_features).to_string(index=False))
        
        return importance_df
    
    def run_complete_pipeline(self):
        """
        Run the complete ML pipeline
        """
        logger.info("=" * 50)
        logger.info("STARTING COMPLETE ML PIPELINE")
        logger.info("=" * 50)
        
        # 1. Load data
        df = self.load_data()
        
        # 2. EDA
        self.explore_data(df)
        
        # 3. Split data
        self.split_data(df)
        
        # 4. Preprocess
        self.preprocess_data()
        
        # 5. Train baseline
        baseline_model = self.train_baseline()
        
        # 6. Train final model
        final_model = self.train_final_model()
        
        # 7. Compare models
        self.compare_models(baseline_model, final_model)
        
        # 8. Feature importance
        self.get_feature_importance()
        
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)


def main():
    """Main execution function"""
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline()
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()

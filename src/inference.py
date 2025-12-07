"""
Inference Module for Credit Card Fraud Detection
Handles prediction on new transactions using trained model
"""

import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Union, List
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    FINAL_MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES_PATH,
    FRAUD_PROBABILITY_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    get_risk_level
)

class FraudDetector:
    """
    Fraud Detection Inference Class
    Loads trained model and performs predictions on new transactions
    """
    
    def __init__(self, model_path=None, scaler_path=None, feature_names_path=None):
        """
        Initialize the fraud detector
        
        Args:
            model_path: Path to trained model pickle file
            scaler_path: Path to fitted scaler pickle file
            feature_names_path: Path to JSON file with feature names
        """
        self.model_path = model_path or FINAL_MODEL_PATH
        self.scaler_path = scaler_path or SCALER_PATH
        self.feature_names_path = feature_names_path or FEATURE_NAMES_PATH
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scaler, and feature names"""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from {self.model_path}")
            
            # Load scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded from {self.scaler_path}")
            
            # Load feature names
            with open(self.feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"✓ Feature names loaded: {len(self.feature_names)} features")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Required artifact not found: {e}. "
                "Please train the model first using pipeline.py"
            )
        except Exception as e:
            raise Exception(f"Error loading artifacts: {e}")
    
    def preprocess_transaction(self, transaction: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess a single transaction or batch of transactions
        
        Args:
            transaction: Dictionary or DataFrame with transaction features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Convert to DataFrame if dictionary
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.DataFrame):
            df = transaction.copy()
        else:
            raise ValueError("Transaction must be a dictionary or DataFrame")
        
        # Engineer features
        df = self._engineer_features(df)
        
        # Select only required features
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        df = df[self.feature_names]
        
        # Scale features
        df_scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=self.feature_names, index=df.index)
        
        return df_scaled
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw transaction data
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Time-based features
        if 'Time' in df.columns:
            df['hour_of_day'] = (df['Time'] / 3600) % 24
            df['is_night'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] < 6)).astype(int)
            df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] < 18)).astype(int)
        
        # Amount-based features
        if 'Amount' in df.columns:
            df['amount_log'] = np.log1p(df['Amount'])
            df['is_large_transaction'] = (df['Amount'] > 1000).astype(int)
            df['is_small_transaction'] = (df['Amount'] < 10).astype(int)
            
            # Amount z-score (using approximate statistics)
            df['amount_zscore'] = (df['Amount'] - 88.35) / 250.12
        
        # Interaction features (if applicable)
        if all(f'V{i}' in df.columns for i in [1, 2, 12, 14, 17]):
            df['V1_V2_interaction'] = df['V1'] * df['V2']
            df['V14_V17_interaction'] = df['V14'] * df['V17']
            df['V12_V14_interaction'] = df['V12'] * df['V14']
        
        return df
    
    def predict(self, transaction: Union[Dict, pd.DataFrame], 
                return_probability: bool = True) -> Union[int, float, Dict]:
        """
        Predict fraud for a transaction
        
        Args:
            transaction: Transaction data (dict or DataFrame)
            return_probability: If True, return probability; else return class
            
        Returns:
            Prediction (int, float, or dict with detailed results)
        """
        # Preprocess
        X = self.preprocess_transaction(transaction)
        
        # Predict
        if return_probability:
            probability = self.model.predict_proba(X)[:, 1][0]
            return probability
        else:
            prediction = self.model.predict(X)[0]
            return int(prediction)
    
    def predict_with_details(self, transaction: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Predict fraud with detailed output including risk level and confidence
        
        Args:
            transaction: Transaction data
            
        Returns:
            Dictionary with prediction details
        """
        # Get probability
        probability = self.predict(transaction, return_probability=True)
        
        # Determine class
        prediction = 1 if probability >= FRAUD_PROBABILITY_THRESHOLD else 0
        
        # Get risk level
        risk_level = get_risk_level(probability)
        
        # Calculate confidence
        confidence = abs(probability - 0.5) * 2  # 0 to 1 scale
        
        # Get feature importance (if available)
        try:
            X = self.preprocess_transaction(transaction)
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.feature_names,
                    self.model.feature_importances_
                ))
                # Get top 5 features
                top_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            else:
                top_features = None
        except:
            top_features = None
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': float(probability),
            'risk_level': risk_level,
            'confidence': float(confidence),
            'threshold_used': FRAUD_PROBABILITY_THRESHOLD,
            'top_features': top_features,
            'recommendation': self._get_recommendation(risk_level, probability)
        }
    
    def predict_batch(self, transactions: pd.DataFrame, 
                     return_details: bool = False) -> Union[np.ndarray, List[Dict]]:
        """
        Predict fraud for multiple transactions
        
        Args:
            transactions: DataFrame with multiple transactions
            return_details: If True, return detailed results for each transaction
            
        Returns:
            Array of predictions or list of detailed result dictionaries
        """
        if return_details:
            results = []
            for idx, row in transactions.iterrows():
                result = self.predict_with_details(row.to_dict())
                result['transaction_id'] = idx
                results.append(result)
            return results
        else:
            X = self.preprocess_transaction(transactions)
            probabilities = self.model.predict_proba(X)[:, 1]
            return probabilities
    
    def _get_recommendation(self, risk_level: str, probability: float) -> str:
        """
        Get action recommendation based on risk level
        
        Args:
            risk_level: Risk level (HIGH/MEDIUM/LOW)
            probability: Fraud probability
            
        Returns:
            Action recommendation string
        """
        if risk_level == "HIGH":
            return "🚨 BLOCK TRANSACTION - High fraud probability. Immediate manual review required."
        elif risk_level == "MEDIUM":
            return "⚠️ FLAG FOR REVIEW - Medium risk. Additional verification recommended."
        else:
            return "✅ APPROVE - Low fraud risk. Transaction appears legitimate."
    
    def explain_prediction(self, transaction: Union[Dict, pd.DataFrame], 
                          method: str = "feature_importance") -> Dict:
        """
        Explain why a prediction was made
        
        Args:
            transaction: Transaction data
            method: Explanation method ('feature_importance' or 'shap')
            
        Returns:
            Dictionary with explanation
        """
        X = self.preprocess_transaction(transaction)
        prediction = self.predict_with_details(transaction)
        
        explanation = {
            'prediction': prediction,
            'method': method,
            'features': {}
        }
        
        if method == "feature_importance" and hasattr(self.model, 'feature_importances_'):
            # Get feature values and importances
            feature_values = X.iloc[0].to_dict()
            feature_importances = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Combine and sort
            combined = [
                {
                    'feature': feat,
                    'value': float(feature_values[feat]),
                    'importance': float(feature_importances[feat]),
                    'contribution': float(feature_values[feat] * feature_importances[feat])
                }
                for feat in self.feature_names
            ]
            
            explanation['features'] = sorted(
                combined,
                key=lambda x: abs(x['contribution']),
                reverse=True
            )[:10]  # Top 10 features
        
        return explanation
    
    def validate_transaction(self, transaction: Dict) -> Dict:
        """
        Validate transaction data before prediction
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        required_fields = [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_fields = [f for f in required_fields if f not in transaction]
        
        if missing_fields:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required fields: {missing_fields}")
        
        # Check data types and ranges
        if 'Amount' in transaction:
            if not isinstance(transaction['Amount'], (int, float)):
                validation_result['errors'].append("Amount must be numeric")
                validation_result['is_valid'] = False
            elif transaction['Amount'] < 0:
                validation_result['errors'].append("Amount cannot be negative")
                validation_result['is_valid'] = False
            elif transaction['Amount'] > 25000:
                validation_result['warnings'].append("Unusually high amount")
        
        # Check PCA features
        for i in range(1, 29):
            feat = f'V{i}'
            if feat in transaction:
                if not isinstance(transaction[feat], (int, float)):
                    validation_result['errors'].append(f"{feat} must be numeric")
                    validation_result['is_valid'] = False
        
        return validation_result


def predict_single_transaction(transaction: Dict) -> Dict:
    """
    Convenience function to predict a single transaction
    
    Args:
        transaction: Transaction dictionary
        
    Returns:
        Prediction result dictionary
    """
    detector = FraudDetector()
    return detector.predict_with_details(transaction)


# Example usage
if __name__ == "__main__":
    # Example transaction
    example_transaction = {
        'V1': -1.359807134,
        'V2': -0.072781173,
        'V3': 2.536346738,
        'V4': 1.378155224,
        'V5': -0.338320770,
        'V6': 0.462387778,
        'V7': 0.239598554,
        'V8': 0.098697901,
        'V9': 0.363786970,
        'V10': 0.090794172,
        'V11': -0.551599533,
        'V12': -0.617800856,
        'V13': -0.991389847,
        'V14': -0.311169354,
        'V15': 1.468176972,
        'V16': -0.470400525,
        'V17': 0.207971242,
        'V18': 0.025790720,
        'V19': 0.403992960,
        'V20': 0.251412098,
        'V21': -0.018306778,
        'V22': 0.277837576,
        'V23': -0.110473910,
        'V24': 0.066928075,
        'V25': 0.128539358,
        'V26': -0.189114844,
        'V27': 0.133558377,
        'V28': -0.021053053,
        'Time': 0,
        'Amount': 149.62
    }
    
    try:
        # Initialize detector
        print("Initializing Fraud Detector...")
        detector = FraudDetector()
        
        # Validate transaction
        print("\nValidating transaction...")
        validation = detector.validate_transaction(example_transaction)
        print(f"Valid: {validation['is_valid']}")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        
        # Make prediction
        print("\nMaking prediction...")
        result = detector.predict_with_details(example_transaction)
        
        print("\n" + "="*50)
        print("FRAUD DETECTION RESULT")
        print("="*50)
        print(f"Is Fraud: {result['is_fraud']}")
        print(f"Fraud Probability: {result['fraud_probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nRecommendation: {result['recommendation']}")
        
        if result['top_features']:
            print("\nTop Contributing Features:")
            for feat, importance in result['top_features']:
                print(f"  - {feat}: {importance:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the model has been trained first by running pipeline.py")

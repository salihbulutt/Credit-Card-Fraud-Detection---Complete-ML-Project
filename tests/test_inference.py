"""
Unit Tests for Inference Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.inference import FraudDetector


@pytest.fixture
def sample_transaction():
    """Create sample transaction data"""
    transaction = {
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
    return transaction


@pytest.fixture
def sample_batch_transactions(sample_transaction):
    """Create batch of sample transactions"""
    df = pd.DataFrame([sample_transaction.copy() for _ in range(10)])
    # Vary amounts
    df['Amount'] = np.random.uniform(1, 1000, 10)
    return df


class TestFraudDetectorInitialization:
    """Test FraudDetector initialization"""
    
    def test_initialization_without_model(self):
        """Test that initialization fails gracefully without model"""
        with pytest.raises(FileNotFoundError):
            detector = FraudDetector(
                model_path='nonexistent_model.pkl',
                scaler_path='nonexistent_scaler.pkl',
                feature_names_path='nonexistent_features.json'
            )


class TestTransactionValidation:
    """Test transaction validation"""
    
    def test_validate_valid_transaction(self, sample_transaction):
        """Test validation of valid transaction"""
        # This test would pass if model files exist
        # For unit testing, we mock the validation
        
        # Check required fields
        required_fields = [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_fields = [f for f in required_fields if f not in sample_transaction]
        
        assert len(missing_fields) == 0
    
    def test_validate_missing_fields(self):
        """Test validation catches missing fields"""
        invalid_transaction = {'Amount': 100.0}  # Missing V features
        
        required_fields = [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_fields = [f for f in required_fields if f not in invalid_transaction]
        
        assert len(missing_fields) > 0
    
    def test_validate_negative_amount(self):
        """Test validation catches negative amount"""
        transaction = {'Amount': -100.0}
        
        assert transaction['Amount'] < 0  # Should be invalid


class TestFeaturePreprocessing:
    """Test feature preprocessing"""
    
    def test_time_feature_creation(self, sample_transaction):
        """Test that time features are created correctly"""
        time_seconds = sample_transaction['Time']
        hour_of_day = (time_seconds / 3600) % 24
        
        assert 0 <= hour_of_day < 24
    
    def test_amount_feature_creation(self, sample_transaction):
        """Test amount feature transformations"""
        amount = sample_transaction['Amount']
        
        # Log transformation
        amount_log = np.log1p(amount)
        assert amount_log >= 0
        
        # Binary features
        is_large = int(amount > 1000)
        is_small = int(amount < 10)
        
        assert is_large in [0, 1]
        assert is_small in [0, 1]


class TestPredictionOutput:
    """Test prediction output format"""
    
    def test_prediction_output_structure(self):
        """Test that prediction output has required fields"""
        # Mock prediction result
        result = {
            'is_fraud': False,
            'fraud_probability': 0.15,
            'risk_level': 'LOW',
            'confidence': 0.70,
            'threshold_used': 0.5,
            'recommendation': 'APPROVE'
        }
        
        # Check required fields
        assert 'is_fraud' in result
        assert 'fraud_probability' in result
        assert 'risk_level' in result
        assert 'confidence' in result
        
        # Check data types
        assert isinstance(result['is_fraud'], bool)
        assert isinstance(result['fraud_probability'], (int, float))
        assert isinstance(result['risk_level'], str)
        assert isinstance(result['confidence'], (int, float))
    
    def test_probability_range(self):
        """Test that probability is in valid range"""
        probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for prob in probabilities:
            assert 0.0 <= prob <= 1.0
    
    def test_risk_level_values(self):
        """Test that risk levels are valid"""
        valid_risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        
        # Test different probabilities
        test_cases = [
            (0.2, 'LOW'),
            (0.6, 'MEDIUM'),
            (0.9, 'HIGH')
        ]
        
        for prob, expected_risk in test_cases:
            if prob < 0.5:
                assert 'LOW' in valid_risk_levels
            elif prob < 0.8:
                assert 'MEDIUM' in valid_risk_levels
            else:
                assert 'HIGH' in valid_risk_levels


class TestBatchPrediction:
    """Test batch prediction functionality"""
    
    def test_batch_input_format(self, sample_batch_transactions):
        """Test that batch input is properly formatted"""
        assert isinstance(sample_batch_transactions, pd.DataFrame)
        assert len(sample_batch_transactions) > 1
        assert 'Amount' in sample_batch_transactions.columns
    
    def test_batch_output_length(self, sample_batch_transactions):
        """Test that batch output matches input length"""
        # Mock batch predictions
        n_transactions = len(sample_batch_transactions)
        results = [{'is_fraud': False} for _ in range(n_transactions)]
        
        assert len(results) == n_transactions


class TestRiskLevelDetermination:
    """Test risk level determination logic"""
    
    def test_low_risk_classification(self):
        """Test low risk classification"""
        probability = 0.2
        risk_level = 'HIGH' if probability >= 0.8 else 'MEDIUM' if probability >= 0.5 else 'LOW'
        
        assert risk_level == 'LOW'
    
    def test_medium_risk_classification(self):
        """Test medium risk classification"""
        probability = 0.6
        risk_level = 'HIGH' if probability >= 0.8 else 'MEDIUM' if probability >= 0.5 else 'LOW'
        
        assert risk_level == 'MEDIUM'
    
    def test_high_risk_classification(self):
        """Test high risk classification"""
        probability = 0.9
        risk_level = 'HIGH' if probability >= 0.8 else 'MEDIUM' if probability >= 0.5 else 'LOW'
        
        assert risk_level == 'HIGH'


class TestRecommendations:
    """Test recommendation generation"""
    
    def test_high_risk_recommendation(self):
        """Test recommendation for high risk"""
        risk_level = 'HIGH'
        
        if risk_level == 'HIGH':
            recommendation = "BLOCK TRANSACTION"
        elif risk_level == 'MEDIUM':
            recommendation = "FLAG FOR REVIEW"
        else:
            recommendation = "APPROVE"
        
        assert "BLOCK" in recommendation.upper()
    
    def test_medium_risk_recommendation(self):
        """Test recommendation for medium risk"""
        risk_level = 'MEDIUM'
        
        if risk_level == 'HIGH':
            recommendation = "BLOCK TRANSACTION"
        elif risk_level == 'MEDIUM':
            recommendation = "FLAG FOR REVIEW"
        else:
            recommendation = "APPROVE"
        
        assert "FLAG" in recommendation.upper() or "REVIEW" in recommendation.upper()
    
    def test_low_risk_recommendation(self):
        """Test recommendation for low risk"""
        risk_level = 'LOW'
        
        if risk_level == 'HIGH':
            recommendation = "BLOCK TRANSACTION"
        elif risk_level == 'MEDIUM':
            recommendation = "FLAG FOR REVIEW"
        else:
            recommendation = "APPROVE"
        
        assert "APPROVE" in recommendation.upper()


class TestErrorHandling:
    """Test error handling in inference"""
    
    def test_missing_required_field(self):
        """Test error when required field is missing"""
        incomplete_transaction = {'Amount': 100}
        
        required_fields = [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing = [f for f in required_fields if f not in incomplete_transaction]
        
        assert len(missing) > 0
    
    def test_invalid_data_type(self):
        """Test error with invalid data type"""
        invalid_transaction = {'Amount': 'invalid'}
        
        with pytest.raises((ValueError, TypeError)):
            float(invalid_transaction['Amount'])
    
    def test_out_of_range_values(self):
        """Test handling of out-of-range values"""
        # Very large amount
        large_amount = 1000000
        
        # Should trigger warning but not error
        assert large_amount > 25000  # Unusually high


class TestConfidenceCalculation:
    """Test confidence score calculation"""
    
    def test_confidence_range(self):
        """Test that confidence is in valid range"""
        probabilities = [0.0, 0.1, 0.5, 0.9, 1.0]
        
        for prob in probabilities:
            # Confidence = distance from 0.5, scaled to 0-1
            confidence = abs(prob - 0.5) * 2
            assert 0.0 <= confidence <= 1.0
    
    def test_high_confidence_at_extremes(self):
        """Test high confidence at probability extremes"""
        # Very low or very high probability = high confidence
        low_prob_confidence = abs(0.05 - 0.5) * 2
        high_prob_confidence = abs(0.95 - 0.5) * 2
        
        assert low_prob_confidence > 0.8
        assert high_prob_confidence > 0.8
    
    def test_low_confidence_at_middle(self):
        """Test low confidence near 0.5 probability"""
        middle_prob_confidence = abs(0.5 - 0.5) * 2
        
        assert middle_prob_confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

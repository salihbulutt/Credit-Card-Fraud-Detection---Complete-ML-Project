"""
Unit Tests for Data Preprocessing Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample fraud detection data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time': np.random.randint(0, 172800, n_samples),
        'Amount': np.random.exponential(100, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    # Add V1-V28 features
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """Create DataPreprocessor instance"""
    return DataPreprocessor()


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor.scaler is None
        assert preprocessor.feature_columns is None
    
    def test_check_data_quality(self, preprocessor, sample_data):
        """Test data quality check"""
        quality_report = preprocessor.check_data_quality(sample_data)
        
        assert 'total_rows' in quality_report
        assert 'total_columns' in quality_report
        assert 'missing_values' in quality_report
        assert 'duplicate_rows' in quality_report
        assert quality_report['total_rows'] == len(sample_data)
        assert quality_report['total_columns'] == len(sample_data.columns)
    
    def test_handle_missing_values_no_missing(self, preprocessor, sample_data):
        """Test handling of data with no missing values"""
        result = preprocessor.handle_missing_values(sample_data)
        assert result.isnull().sum().sum() == 0
        assert len(result) == len(sample_data)
    
    def test_handle_missing_values_with_missing(self, preprocessor, sample_data):
        """Test handling of data with missing values"""
        # Introduce missing values
        sample_data_missing = sample_data.copy()
        sample_data_missing.loc[0:10, 'Amount'] = np.nan
        
        result = preprocessor.handle_missing_values(sample_data_missing)
        assert result['Amount'].isnull().sum() == 0
    
    def test_remove_duplicates(self, preprocessor, sample_data):
        """Test duplicate removal"""
        # Add duplicate rows
        sample_with_dupes = pd.concat([sample_data, sample_data.iloc[:10]])
        
        result = preprocessor.remove_duplicates(sample_with_dupes)
        assert len(result) == len(sample_data)
    
    def test_split_features_target(self, preprocessor, sample_data):
        """Test feature-target split"""
        X, y = preprocessor.split_features_target(sample_data)
        
        assert 'Class' not in X.columns
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert preprocessor.feature_columns is not None
    
    def test_train_test_split_data(self, preprocessor, sample_data):
        """Test train-test-val split"""
        X, y = preprocessor.split_features_target(sample_data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_split_data(
            X, y, test_size=0.2, val_size=0.1
        )
        
        # Check sizes
        total_size = len(X_train) + len(X_val) + len(X_test)
        assert total_size == len(X)
        
        # Check stratification (approximately)
        assert abs(y_train.mean() - y.mean()) < 0.01
        assert abs(y_val.mean() - y.mean()) < 0.01
        assert abs(y_test.mean() - y.mean()) < 0.01
    
    def test_scale_features(self, preprocessor, sample_data):
        """Test feature scaling"""
        X, y = preprocessor.split_features_target(sample_data)
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.train_test_split_data(X, y)
        
        X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(
            X_train, X_val, X_test, method='robust'
        )
        
        # Check scaler was fitted
        assert preprocessor.scaler is not None
        
        # Check shapes preserved
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Check scaling (mean should be close to 0, std close to 1 for robust scaler)
        assert abs(X_train_scaled.mean().mean()) < 1
    
    def test_invalid_scaling_method(self, preprocessor, sample_data):
        """Test invalid scaling method raises error"""
        X, y = preprocessor.split_features_target(sample_data)
        X_train, X_val, X_test, _, _, _ = preprocessor.train_test_split_data(X, y)
        
        with pytest.raises(ValueError):
            preprocessor.scale_features(X_train, X_val, X_test, method='invalid')
    
    def test_missing_target_column(self, preprocessor, sample_data):
        """Test error when target column is missing"""
        sample_data_no_target = sample_data.drop(columns=['Class'])
        
        with pytest.raises(ValueError):
            preprocessor.split_features_target(sample_data_no_target)


class TestDataValidation:
    """Test data validation functions"""
    
    def test_fraud_rate_reasonable(self, sample_data):
        """Test that fraud rate is within reasonable bounds"""
        fraud_rate = sample_data['Class'].mean()
        assert 0 < fraud_rate < 0.5  # Fraud should be minority class
    
    def test_no_negative_amounts(self, sample_data):
        """Test that amounts are non-negative"""
        assert (sample_data['Amount'] >= 0).all()
    
    def test_time_range_valid(self, sample_data):
        """Test that time values are within expected range"""
        assert sample_data['Time'].min() >= 0
        assert sample_data['Time'].max() <= 172800  # 2 days in seconds
    
    def test_v_features_exist(self, sample_data):
        """Test that V1-V28 features exist"""
        v_features = [f'V{i}' for i in range(1, 29)]
        assert all(feat in sample_data.columns for feat in v_features)
    
    def test_no_inf_values(self, sample_data):
        """Test that there are no infinite values"""
        assert not np.isinf(sample_data.select_dtypes(include=[np.number])).any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

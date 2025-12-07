"""
Data Preprocessing Module
Handles all data loading, cleaning, and preprocessing operations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    VALIDATION_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    SCALING_METHOD,
    MISSING_VALUE_STRATEGY,
    MISSING_THRESHOLD
)


class DataPreprocessor:
    """
    Data preprocessing pipeline for fraud detection
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load raw data from CSV file
        
        Args:
            filepath: Path to CSV file (default: RAW_DATA_PATH)
            
        Returns:
            DataFrame with loaded data
        """
        filepath = filepath or RAW_DATA_PATH
        
        print(f"Loading data from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            print(f"✓ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found at {filepath}. "
                "Please download the dataset from: "
                "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
                "and place it in data/raw/creditcard.csv"
            )
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Check data quality and return report
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        cols_with_missing = missing_pct[missing_pct > 0]
        
        if len(cols_with_missing) > 0:
            quality_report['columns_with_missing'] = cols_with_missing.to_dict()
        
        return quality_report
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Check missing values
        missing_pct = (df.isnull().sum() / len(df))
        
        # Drop columns with too many missing values
        cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
        if cols_to_drop:
            print(f"Dropping columns with >{MISSING_THRESHOLD*100}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Fill remaining missing values
        if df.isnull().sum().sum() > 0:
            print(f"Filling missing values using strategy: {MISSING_VALUE_STRATEGY}")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if MISSING_VALUE_STRATEGY == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif MISSING_VALUE_STRATEGY == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif MISSING_VALUE_STRATEGY == 'mode':
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0])
            elif MISSING_VALUE_STRATEGY == 'drop':
                df = df.dropna()
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        
        if removed_rows > 0:
            print(f"✓ Removed {removed_rows:,} duplicate rows")
        
        return df
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split features and target variable
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
        
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def train_test_split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = None,
        val_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Test set proportion (default: TEST_SIZE)
            val_size: Validation set proportion (default: VALIDATION_SIZE)
            random_state: Random state (default: RANDOM_STATE)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        test_size = test_size or TEST_SIZE
        val_size = val_size or VALIDATION_SIZE
        random_state = random_state or RANDOM_STATE
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\nData split completed:")
        print(f"  Train: {len(X_train):,} samples ({len(X_train)/(len(X_train)+len(X_val)+len(X_test)):.1%})")
        print(f"  Val:   {len(X_val):,} samples ({len(X_val)/(len(X_train)+len(X_val)+len(X_test)):.1%})")
        print(f"  Test:  {len(X_test):,} samples ({len(X_test)/(len(X_train)+len(X_val)+len(X_test)):.1%})")
        print(f"\nFraud rates:")
        print(f"  Train: {y_train.mean():.4%}")
        print(f"  Val:   {y_val.mean():.4%}")
        print(f"  Test:  {y_test.mean():.4%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(
        self, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame, 
        X_test: pd.DataFrame,
        method: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified method
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            method: Scaling method ('standard', 'robust', 'minmax')
            
        Returns:
            Tuple of (scaled X_train, X_val, X_test)
        """
        method = method or SCALING_METHOD
        
        # Initialize scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            X_val_scaled, 
            columns=X_val.columns, 
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        print(f"✓ Features scaled using {method.upper()} scaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series
    ):
        """
        Save processed datasets to CSV files
        
        Args:
            X_train, X_val, X_test: Feature DataFrames
            y_train, y_val, y_test: Target Series
        """
        # Combine features and target
        train_df = X_train.copy()
        train_df[TARGET_COLUMN] = y_train
        
        val_df = X_val.copy()
        val_df[TARGET_COLUMN] = y_val
        
        test_df = X_test.copy()
        test_df[TARGET_COLUMN] = y_test
        
        # Save to CSV
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        val_df.to_csv(VALIDATION_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        
        print(f"\n✓ Processed data saved:")
        print(f"  Train: {TRAIN_DATA_PATH}")
        print(f"  Val:   {VALIDATION_DATA_PATH}")
        print(f"  Test:  {TEST_DATA_PATH}")
    
    def get_basic_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get basic statistical summary of the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with statistics
        """
        return df.describe()
    
    def preprocess_pipeline(
        self, 
        filepath: str = None,
        save: bool = True
    ) -> Tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to raw data file
            save: Whether to save processed data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("="*70)
        print("DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # 1. Load data
        df = self.load_data(filepath)
        
        # 2. Check data quality
        print("\n1. Data Quality Check")
        quality_report = self.check_data_quality(df)
        print(f"   Total rows: {quality_report['total_rows']:,}")
        print(f"   Total columns: {quality_report['total_columns']}")
        print(f"   Duplicates: {quality_report['duplicate_rows']:,}")
        print(f"   Memory: {quality_report['memory_usage_mb']:.2f} MB")
        
        # 3. Handle missing values
        print("\n2. Handling Missing Values")
        df = self.handle_missing_values(df)
        print("   ✓ No missing values" if df.isnull().sum().sum() == 0 else "   ✓ Missing values handled")
        
        # 4. Remove duplicates
        print("\n3. Removing Duplicates")
        df = self.remove_duplicates(df)
        
        # 5. Split features and target
        print("\n4. Splitting Features and Target")
        X, y = self.split_features_target(df)
        print(f"   Features: {X.shape[1]}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        # 6. Train-test-val split
        print("\n5. Creating Train/Val/Test Split")
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_split_data(X, y)
        
        # 7. Save processed data (optional)
        if save:
            print("\n6. Saving Processed Data")
            self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_processed_data() -> Tuple:
    """
    Load preprocessed data from saved files
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        val_df = pd.read_csv(VALIDATION_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        X_train = train_df.drop(columns=[TARGET_COLUMN])
        y_train = train_df[TARGET_COLUMN]
        
        X_val = val_df.drop(columns=[TARGET_COLUMN])
        y_val = val_df[TARGET_COLUMN]
        
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_test = test_df[TARGET_COLUMN]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except FileNotFoundError:
        raise FileNotFoundError(
            "Processed data not found. Please run preprocessing pipeline first."
        )


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_pipeline()

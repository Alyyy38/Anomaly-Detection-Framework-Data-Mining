"""
Data preprocessing pipeline for credit card fraud detection
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocessing pipeline for credit card fraud data"""
    
    def __init__(self, normalize: bool = True, scaler_type: str = "standard"):
        """
        Args:
            normalize: Whether to normalize features
            scaler_type: Type of scaler ("standard" or "robust")
        """
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_cols = None
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Extract feature column names (exclude Time and Class)"""
        exclude_cols = ['Time', 'Class']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_cols = feature_cols
        return feature_cols
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        if df.isnull().sum().sum() > 0:
            print(f"Found {df.isnull().sum().sum()} missing values")
            # Fill with median for numerical columns
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col].fillna(df[col].median(), inplace=True)
        return df
    
    def normalize_features(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[list] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Normalize features using StandardScaler or RobustScaler
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe (optional)
            test_df: Test dataframe (optional)
            feature_cols: List of feature columns to normalize
            
        Returns:
            Tuple of normalized dataframes
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns(train_df)
        
        # Initialize scaler
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Fit on training data
        train_features = train_df[feature_cols].values
        self.scaler.fit(train_features)
        
        # Transform all datasets
        train_df_normalized = train_df.copy()
        train_df_normalized[feature_cols] = self.scaler.transform(train_features)
        
        if val_df is not None:
            val_df_normalized = val_df.copy()
            val_df_normalized[feature_cols] = self.scaler.transform(val_df[feature_cols].values)
        else:
            val_df_normalized = None
        
        if test_df is not None:
            test_df_normalized = test_df.copy()
            test_df_normalized[feature_cols] = self.scaler.transform(test_df[feature_cols].values)
        else:
            test_df_normalized = None
        
        print(f"Features normalized using {self.scaler_type} scaler")
        return train_df_normalized, val_df_normalized, test_df_normalized
    
    def balance_dataset(
        self,
        df: pd.DataFrame,
        target_col: str = 'Class',
        method: str = 'undersample',
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Balance the dataset to handle class imbalance
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            method: 'undersample' or 'oversample'
            random_state: Random seed
            
        Returns:
            Balanced dataframe
        """
        # Separate majority and minority classes
        majority_class = df[df[target_col] == 0]
        minority_class = df[df[target_col] == 1]
        
        print(f"\nBefore balancing:")
        print(f"Majority (Normal): {len(majority_class)}")
        print(f"Minority (Fraud): {len(minority_class)}")
        
        if method == 'undersample':
            # Undersample majority class
            majority_downsampled = resample(
                majority_class,
                replace=False,
                n_samples=len(minority_class) * 10,  # Keep 10:1 ratio
                random_state=random_state
            )
            balanced_df = pd.concat([majority_downsampled, minority_class])
        
        elif method == 'oversample':
            # Oversample minority class
            minority_upsampled = resample(
                minority_class,
                replace=True,
                n_samples=len(majority_class) // 10,  # 10:1 ratio
                random_state=random_state
            )
            balanced_df = pd.concat([majority_class, minority_upsampled])
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        print(f"\nAfter balancing:")
        print(f"Normal: {len(balanced_df[balanced_df[target_col] == 0])}")
        print(f"Fraud: {len(balanced_df[balanced_df[target_col] == 1])}")
        
        return balanced_df
    
    def preprocess(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        balance: bool = False,
        balance_method: str = 'undersample'
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Complete preprocessing pipeline
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            balance: Whether to balance the dataset
            balance_method: Method for balancing
            
        Returns:
            Tuple of preprocessed dataframes
        """
        # Handle missing values
        train_df = self.handle_missing_values(train_df)
        if val_df is not None:
            val_df = self.handle_missing_values(val_df)
        if test_df is not None:
            test_df = self.handle_missing_values(test_df)
        
        # Balance dataset if requested (only on training set)
        if balance:
            train_df = self.balance_dataset(train_df, method=balance_method)
        
        # Normalize features
        if self.normalize:
            train_df, val_df, test_df = self.normalize_features(
                train_df, val_df, test_df
            )
        
        return train_df, val_df, test_df


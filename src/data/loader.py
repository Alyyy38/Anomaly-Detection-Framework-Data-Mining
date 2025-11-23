"""
Data loader for Credit Card Fraud Detection dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class CreditCardDataset(Dataset):
    """PyTorch Dataset for Credit Card Fraud Detection"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, window_size: int = 10, stride: int = 1):
        """
        Args:
            data: Normalized feature array [n_samples, n_features]
            labels: Binary labels [n_samples]
            window_size: Size of sliding window
            stride: Stride for sliding window
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        
        # Create sliding windows
        self.windows, self.window_labels = self._create_windows()
    
    def _create_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from time series data"""
        windows = []
        window_labels = []
        
        for i in range(0, len(self.data) - self.window_size + 1, self.stride):
            window = self.data[i:i + self.window_size]
            # Label is 1 if any point in window is anomalous
            label = int(np.any(self.labels[i:i + self.window_size]))
            windows.append(window)
            window_labels.append(label)
        
        return np.array(windows), np.array(window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = torch.FloatTensor(self.windows[idx])
        label = torch.LongTensor([self.window_labels[idx]])[0]
        return window, label


class CreditCardDataLoader:
    """Data loader for Credit Card Fraud Detection dataset"""
    
    def __init__(self, data_path: str = "creditcard.csv"):
        """
        Args:
            data_path: Path to creditcard.csv file
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file"""
        print(f"Loading data from {self.data_path}...")
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data)} samples")
        print(f"Features: {self.raw_data.shape[1] - 1}")  # Exclude target
        print(f"Fraud cases: {self.raw_data['Class'].sum()} ({self.raw_data['Class'].mean()*100:.2f}%)")
        return self.raw_data
    
    def split_data(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets
        
        Args:
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
            test_ratio: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.raw_data is None:
            self.load_data()
        
        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        # First split: train vs (val + test)
        train_size = train_ratio
        temp_size = val_ratio + test_ratio
        
        train_df, temp_df = train_test_split(
            self.raw_data,
            test_size=temp_size,
            random_state=random_state,
            stratify=self.raw_data['Class']
        )
        
        # Second split: val vs test
        val_size = val_ratio / temp_size
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_size,
            random_state=random_state,
            stratify=temp_df['Class']
        )
        
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df
        
        print(f"\nData splits:")
        print(f"Train: {len(train_df)} samples ({len(train_df)/len(self.raw_data)*100:.1f}%)")
        print(f"  - Fraud: {train_df['Class'].sum()} ({train_df['Class'].mean()*100:.2f}%)")
        print(f"Val: {len(val_df)} samples ({len(val_df)/len(self.raw_data)*100:.1f}%)")
        print(f"  - Fraud: {val_df['Class'].sum()} ({val_df['Class'].mean()*100:.2f}%)")
        print(f"Test: {len(test_df)} samples ({len(test_df)/len(self.raw_data)*100:.1f}%)")
        print(f"  - Fraud: {test_df['Class'].sum()} ({test_df['Class'].mean()*100:.2f}%)")
        
        return train_df, val_df, test_df
    
    def get_data_loaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list,
        window_size: int = 10,
        batch_size: int = 256,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            feature_cols: List of feature column names
            window_size: Size of sliding window
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Extract features and labels
        X_train = train_df[feature_cols].values
        y_train = train_df['Class'].values
        
        X_val = val_df[feature_cols].values
        y_val = val_df['Class'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['Class'].values
        
        # Create datasets
        train_dataset = CreditCardDataset(X_train, y_train, window_size=window_size)
        val_dataset = CreditCardDataset(X_val, y_val, window_size=window_size)
        test_dataset = CreditCardDataset(X_test, y_test, window_size=window_size)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"\nDataLoaders created:")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader


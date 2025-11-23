"""
Geometric masking augmentation for time series anomaly detection
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import random


class GeometricMasking:
    """Geometric masking augmentation for time series data"""
    
    def __init__(
        self,
        time_mask_ratio: Tuple[float, float] = (0.1, 0.3),
        feature_mask_ratio: Tuple[float, float] = (0.1, 0.2),
        augmentation_prob: float = 0.5
    ):
        """
        Args:
            time_mask_ratio: Range of timesteps to mask (min, max)
            feature_mask_ratio: Range of features to mask (min, max)
            augmentation_prob: Probability of applying augmentation
        """
        self.time_mask_ratio = time_mask_ratio
        self.feature_mask_ratio = feature_mask_ratio
        self.augmentation_prob = augmentation_prob
    
    def random_time_mask(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask timesteps in the sequence
        
        Args:
            x: Input tensor [batch, seq_len, features]
            mask_ratio: Ratio of timesteps to mask (if None, random in range)
            
        Returns:
            Tuple of (masked_x, mask) where mask is 1 for masked positions
        """
        batch_size, seq_len, features = x.shape
        
        if mask_ratio is None:
            mask_ratio = random.uniform(*self.time_mask_ratio)
        
        num_mask = int(seq_len * mask_ratio)
        masked_x = x.clone()
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        for i in range(batch_size):
            # Randomly select timesteps to mask
            mask_indices = torch.randperm(seq_len)[:num_mask]
            mask[i, mask_indices] = True
            # Zero out masked timesteps
            masked_x[i, mask_indices] = 0.0
        
        return masked_x, mask
    
    def random_feature_mask(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask features across all timesteps
        
        Args:
            x: Input tensor [batch, seq_len, features]
            mask_ratio: Ratio of features to mask (if None, random in range)
            
        Returns:
            Tuple of (masked_x, mask) where mask is 1 for masked features
        """
        batch_size, seq_len, features = x.shape
        
        if mask_ratio is None:
            mask_ratio = random.uniform(*self.feature_mask_ratio)
        
        num_mask = int(features * mask_ratio)
        masked_x = x.clone()
        mask = torch.zeros(batch_size, features, dtype=torch.bool, device=x.device)
        
        for i in range(batch_size):
            # Randomly select features to mask
            mask_indices = torch.randperm(features)[:num_mask]
            mask[i, mask_indices] = True
            # Zero out masked features across all timesteps
            masked_x[i, :, mask_indices] = 0.0
        
        return masked_x, mask
    
    def time_warp(self, x: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
        """
        Apply time warping augmentation
        
        Args:
            x: Input tensor [batch, seq_len, features]
            sigma: Standard deviation for warping
            
        Returns:
            Warped tensor
        """
        batch_size, seq_len, features = x.shape
        warped_x = x.clone()
        
        for i in range(batch_size):
            # Generate random warping curve
            warp = torch.cumsum(
                torch.ones(seq_len, device=x.device) + 
                torch.randn(seq_len, device=x.device) * sigma,
                dim=0
            )
            warp = (warp / warp[-1] * (seq_len - 1)).long()
            warp = torch.clamp(warp, 0, seq_len - 1)
            
            # Apply warping
            warped_x[i] = x[i, warp]
        
        return warped_x
    
    def mixup(
        self,
        x: torch.Tensor,
        alpha: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation for time series
        
        Args:
            x: Input tensor [batch, seq_len, features]
            alpha: Beta distribution parameter
            
        Returns:
            Tuple of (mixed_x, permuted_indices, lambda)
        """
        batch_size = x.shape[0]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        perm = torch.randperm(batch_size, device=x.device)
        
        # Mix samples
        mixed_x = lam * x + (1 - lam) * x[perm]
        
        return mixed_x, perm, lam
    
    def augment(
        self,
        x: torch.Tensor,
        augmentation_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply random augmentation
        
        Args:
            x: Input tensor [batch, seq_len, features]
            augmentation_type: Type of augmentation ('time_mask', 'feature_mask', 
                             'time_warp', 'mixup', or None for random)
            
        Returns:
            Tuple of (augmented_x, mask) where mask is None for non-masking augmentations
        """
        if random.random() > self.augmentation_prob:
            return x, None
        
        if augmentation_type is None:
            augmentation_type = random.choice([
                'time_mask', 'feature_mask', 'time_warp', 'mixup'
            ])
        
        if augmentation_type == 'time_mask':
            return self.random_time_mask(x)
        elif augmentation_type == 'feature_mask':
            return self.random_feature_mask(x)
        elif augmentation_type == 'time_warp':
            return self.time_warp(x), None
        elif augmentation_type == 'mixup':
            mixed_x, _, _ = self.mixup(x)
            return mixed_x, None
        else:
            return x, None
    
    def create_positive_pairs(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create positive pairs for contrastive learning
        
        Args:
            x: Input tensor [batch, seq_len, features]
            
        Returns:
            Tuple of (x1, x2) - two augmented versions of the same input
        """
        x1, _ = self.augment(x)
        x2, _ = self.augment(x)
        return x1, x2


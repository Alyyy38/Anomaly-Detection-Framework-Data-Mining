"""
Contrastive learning module for anomaly detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ContrastiveLearningModule(nn.Module):
    """Contrastive learning module using NT-Xent loss"""
    
    def __init__(
        self,
        input_dim: int = 64,
        projection_dim: int = 128,
        temperature: float = 0.07,
        use_memory_bank: bool = False,
        memory_bank_size: int = 4096
    ):
        """
        Args:
            input_dim: Dimension of input features (latent space)
            projection_dim: Dimension of projection space
            temperature: Temperature parameter for softmax
            use_memory_bank: Whether to use memory bank for negatives
            memory_bank_size: Size of memory bank
        """
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_memory_bank = use_memory_bank
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Memory bank (optional)
        if use_memory_bank:
            self.register_buffer(
                'memory_bank',
                torch.randn(memory_bank_size, projection_dim)
            )
            self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
    
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project latent features to contrastive space
        
        Args:
            z: Latent features [batch, input_dim]
            
        Returns:
            Projected features [batch, projection_dim]
        """
        return F.normalize(self.projection_head(z), dim=1)
    
    def cosine_similarity(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two feature sets
        
        Args:
            z1: Features 1 [batch, projection_dim]
            z2: Features 2 [batch, projection_dim]
            
        Returns:
            Similarity matrix [batch, batch]
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        return torch.matmul(z1, z2.t())
    
    def nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute NT-Xent (InfoNCE) loss
        
        Args:
            z1: Projected features from first augmentation [batch, projection_dim]
            z2: Projected features from second augmentation [batch, projection_dim]
            negatives: Optional negative samples [num_negatives, projection_dim]
            
        Returns:
            Contrastive loss value
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Create labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z1, z2.t()) / self.temperature
        
        # If using memory bank, add negative samples
        if negatives is not None and self.use_memory_bank:
            # Compute similarity with negatives
            neg_similarity = torch.matmul(z1, negatives.t()) / self.temperature
            # Concatenate with positive similarities
            similarity_matrix = torch.cat([similarity_matrix, neg_similarity], dim=1)
        
        # Compute loss for z1 -> z2
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        
        # Compute loss for z2 -> z1
        similarity_matrix_21 = similarity_matrix.t()
        loss_21 = F.cross_entropy(similarity_matrix_21, labels)
        
        # Average
        loss = (loss_12 + loss_21) / 2
        
        return loss
    
    def update_memory_bank(self, features: torch.Tensor):
        """
        Update memory bank with new features
        
        Args:
            features: Projected features [batch, projection_dim]
        """
        if not self.use_memory_bank:
            return
        
        batch_size = features.shape[0]
        ptr = int(self.memory_ptr)
        
        # Replace memory bank entries
        if ptr + batch_size <= self.memory_bank.shape[0]:
            self.memory_bank[ptr:ptr + batch_size] = features.detach()
            ptr = (ptr + batch_size) % self.memory_bank.shape[0]
        else:
            # Handle wrap-around
            remaining = self.memory_bank.shape[0] - ptr
            self.memory_bank[ptr:] = features.detach()[:remaining]
            self.memory_bank[:batch_size - remaining] = features.detach()[remaining:]
            ptr = batch_size - remaining
        
        self.memory_ptr[0] = ptr
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        update_memory: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning
        
        Args:
            z1: Latent features from first augmentation [batch, input_dim]
            z2: Latent features from second augmentation [batch, input_dim]
            update_memory: Whether to update memory bank
            
        Returns:
            Tuple of (projected_z1, projected_z2, contrastive_loss)
        """
        # Project to contrastive space
        proj_z1 = self.project(z1)
        proj_z2 = self.project(z2)
        
        # Get negatives from memory bank if available
        negatives = None
        if self.use_memory_bank:
            # Sample random negatives from memory bank
            num_negatives = min(256, self.memory_bank.shape[0])
            neg_indices = torch.randperm(self.memory_bank.shape[0])[:num_negatives]
            negatives = self.memory_bank[neg_indices]
        
        # Compute contrastive loss
        loss = self.nt_xent_loss(proj_z1, proj_z2, negatives)
        
        # Update memory bank
        if update_memory:
            self.update_memory_bank(proj_z1)
        
        return proj_z1, proj_z2, loss


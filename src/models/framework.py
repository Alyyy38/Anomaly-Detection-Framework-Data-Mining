"""
Unified Anomaly Detection Framework
Combining Transformer, GAN, and Contrastive Learning
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .transformer import TransformerEncoderDecoder
from .gan import Generator, Discriminator
from .contrastive import ContrastiveLearningModule
from ..utils.augmentation import GeometricMasking


class AnomalyDetectionFramework(nn.Module):
    """Unified framework combining all components"""
    
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 10,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 64,
        projection_dim: int = 128,
        temperature: float = 0.07,
        time_mask_ratio: Tuple[float, float] = (0.1, 0.3),
        feature_mask_ratio: Tuple[float, float] = (0.1, 0.2),
        augmentation_prob: float = 0.5
    ):
        """
        Args:
            input_dim: Number of input features
            seq_len: Sequence length
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            latent_dim: Latent space dimension
            projection_dim: Contrastive projection dimension
            temperature: Contrastive temperature
            time_mask_ratio: Time masking ratio range
            feature_mask_ratio: Feature masking ratio range
            augmentation_prob: Augmentation probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Transformer encoder-decoder
        self.transformer = TransformerEncoderDecoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            latent_dim=latent_dim
        )
        
        # GAN components
        self.generator = Generator(
            latent_dim=latent_dim,
            output_seq_len=seq_len,
            output_features=input_dim
        )
        
        self.discriminator = Discriminator(
            input_seq_len=seq_len,
            input_features=input_dim
        )
        
        # Contrastive learning
        self.contrastive = ContrastiveLearningModule(
            input_dim=latent_dim,
            projection_dim=projection_dim,
            temperature=temperature
        )
        
        # Augmentation
        self.augmentation = GeometricMasking(
            time_mask_ratio=time_mask_ratio,
            feature_mask_ratio=feature_mask_ratio,
            augmentation_prob=augmentation_prob
        )
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, features]
            training: Whether in training mode
            return_all: Whether to return all intermediate outputs
            
        Returns:
            Dictionary containing all outputs and losses
        """
        batch_size = x.shape[0]
        device = x.device
        
        results = {}
        
        # 1. Apply geometric masking (during training)
        if training:
            x_aug, mask = self.augmentation.augment(x)
            if mask is None:
                # Create a time-based mask if augmentation didn't return one
                mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        else:
            x_aug = x
            mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        
        # 2. Encode with Transformer
        encoded, latent, reconstructed, attention = self.transformer(x_aug, return_attention=False)
        results['encoded'] = encoded
        results['latent'] = latent
        results['reconstructed'] = reconstructed
        results['attention'] = attention
        
        # 3. Contrastive learning (create positive pairs)
        latent1 = None
        latent2 = None
        if training:
            x1, x2 = self.augmentation.create_positive_pairs(x)
            _, latent1 = self.transformer.encode(x1)
            _, latent2 = self.transformer.encode(x2)
            proj1, proj2, contrastive_loss = self.contrastive(latent1, latent2, update_memory=True)
            results['contrastive_loss'] = contrastive_loss
        else:
            results['contrastive_loss'] = torch.tensor(0.0, device=device)
        
        # 4. Generate with GAN
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        generated = self.generator(noise)
        results['generated'] = generated
        
        # 5. Discriminator outputs
        disc_real, disc_real_features = self.discriminator(x)
        disc_fake, disc_fake_features = self.discriminator(generated.detach())
        disc_recon, _ = self.discriminator(reconstructed.detach())
        
        results['disc_real'] = disc_real
        results['disc_fake'] = disc_fake
        results['disc_recon'] = disc_recon
        
        # 6. Anomaly scores
        # Reconstruction error
        recon_error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        results['reconstruction_error'] = recon_error
        
        # Discriminator score (lower = more anomalous)
        disc_score = torch.sigmoid(disc_real).squeeze()
        results['discriminator_score'] = disc_score
        
        # Combined anomaly score
        anomaly_score = recon_error - disc_score  # Higher = more anomalous
        results['anomaly_score'] = anomaly_score
        
        if return_all:
            results['x_aug'] = x_aug
            results['mask'] = mask
            if training:
                results['latent1'] = latent1
                results['latent2'] = latent2
        
        return results
    
    def detect_anomaly(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies in input
        
        Args:
            x: Input tensor [batch, seq_len, features]
            threshold: Anomaly threshold (if None, uses median)
            
        Returns:
            Tuple of (predictions, scores)
        """
        self.eval()
        with torch.no_grad():
            results = self.forward(x, training=False)
            scores = results['anomaly_score']
            
            if threshold is None:
                # Use median as threshold (can be set from validation set)
                threshold = torch.median(scores)
            
            predictions = (scores > threshold).long()
        
        return predictions, scores
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get anomaly scores without thresholding
        
        Args:
            x: Input tensor [batch, seq_len, features]
            
        Returns:
            Anomaly scores [batch]
        """
        self.eval()
        with torch.no_grad():
            results = self.forward(x, training=False)
            return results['anomaly_score']


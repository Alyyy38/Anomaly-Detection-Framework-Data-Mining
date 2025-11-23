"""
Combined loss functions for anomaly detection framework
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from ..models import gan


class CombinedLoss(nn.Module):
    """Combined loss function for the anomaly detection framework"""
    
    def __init__(
        self,
        alpha: float = 1.0,  # Reconstruction
        beta: float = 0.5,   # Contrastive
        gamma: float = 0.3,  # GAN generator
        delta: float = 0.2,  # GAN discriminator
        lambda_gp: float = 10.0,  # Gradient penalty weight
        use_mask: bool = True
    ):
        """
        Args:
            alpha: Weight for reconstruction loss
            beta: Weight for contrastive loss
            gamma: Weight for GAN generator loss
            delta: Weight for GAN discriminator loss
            lambda_gp: Weight for gradient penalty
            use_mask: Whether to use masking for reconstruction loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_gp = lambda_gp
        self.use_mask = use_mask
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def reconstruction_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss (optionally masked)
        
        Args:
            x: Original input [batch, seq_len, features]
            reconstructed: Reconstructed input [batch, seq_len, features]
            mask: Optional mask [batch, seq_len] where True indicates masked positions
            
        Returns:
            Reconstruction loss
        """
        if mask is not None and self.use_mask:
            # Only compute loss on masked regions
            batch_size, seq_len, features = x.shape
            # Handle different mask shapes
            if mask.dim() == 2:
                # Check if mask is [batch, seq_len] or [batch, features]
                if mask.shape[1] == seq_len:
                    # Time mask: [batch, seq_len] -> [batch, seq_len, features]
                    mask_expanded = mask.unsqueeze(-1).expand(batch_size, seq_len, features)
                elif mask.shape[1] == features:
                    # Feature mask: [batch, features] -> [batch, seq_len, features]
                    mask_expanded = mask.unsqueeze(1).expand(batch_size, seq_len, features)
                else:
                    # Fallback: use all positions
                    mask_expanded = torch.ones_like(x, dtype=torch.bool)
            else:
                mask_expanded = mask if mask.shape == x.shape else torch.ones_like(x, dtype=torch.bool)
            
            loss = self.mse_loss(reconstructed, x) * mask_expanded.float()
            # Average over masked positions only
            num_masked = mask_expanded.sum().clamp(min=1)
            loss = loss.sum() / num_masked
        else:
            # Compute loss on all positions
            loss = self.mse_loss(reconstructed, x).mean()
        
        return loss
    
    def contrastive_loss(self, contrastive_loss: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss (already computed in contrastive module)
        
        Args:
            contrastive_loss: Pre-computed contrastive loss
            
        Returns:
            Contrastive loss
        """
        return contrastive_loss
    
    def gan_generator_loss(
        self,
        disc_fake: torch.Tensor,
        disc_recon: torch.Tensor
    ) -> torch.Tensor:
        """
        GAN generator loss (Wasserstein loss)
        
        Args:
            disc_fake: Discriminator output on generated samples [batch, 1]
            disc_recon: Discriminator output on reconstructed samples [batch, 1]
            
        Returns:
            Generator loss
        """
        # Maximize discriminator score (negative of discriminator loss)
        loss_fake = -disc_fake.mean()
        loss_recon = -disc_recon.mean()
        return (loss_fake + loss_recon) / 2
    
    def gan_discriminator_loss(
        self,
        disc_real: torch.Tensor,
        disc_fake: torch.Tensor,
        disc_recon: torch.Tensor,
        x: torch.Tensor,
        generated: torch.Tensor,
        reconstructed: torch.Tensor,
        discriminator: nn.Module
    ) -> torch.Tensor:
        """
        GAN discriminator loss with gradient penalty
        
        Args:
            disc_real: Discriminator output on real samples [batch, 1]
            disc_fake: Discriminator output on generated samples [batch, 1]
            disc_recon: Discriminator output on reconstructed samples [batch, 1]
            x: Real samples [batch, seq_len, features]
            generated: Generated samples [batch, seq_len, features]
            reconstructed: Reconstructed samples [batch, seq_len, features]
            discriminator: Discriminator model
            
        Returns:
            Discriminator loss
        """
        device = x.device
        
        # Wasserstein loss: maximize real, minimize fake
        loss_real = -disc_real.mean()
        loss_fake = disc_fake.mean()
        loss_recon = disc_recon.mean()
        
        # Gradient penalty
        gp = gan.gradient_penalty(discriminator, x, generated, device)
        
        # Total discriminator loss
        loss = loss_real + loss_fake + loss_recon + self.lambda_gp * gp
        
        return loss, gp
    
    def forward(
        self,
        results: Dict[str, torch.Tensor],
        x: torch.Tensor,
        discriminator: Optional[nn.Module] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            results: Dictionary from model forward pass
            x: Original input [batch, seq_len, features]
            discriminator: Discriminator model (required for discriminator loss)
            training: Whether in training mode
            
        Returns:
            Dictionary containing all losses
        """
        losses = {}
        
        # 1. Reconstruction loss
        reconstructed = results['reconstructed']
        mask = results.get('mask', None)
        recon_loss = self.reconstruction_loss(x, reconstructed, mask)
        losses['reconstruction'] = recon_loss
        
        # 2. Contrastive loss
        contrastive_loss = results.get('contrastive_loss', torch.tensor(0.0, device=x.device))
        losses['contrastive'] = contrastive_loss
        
        # 3. GAN losses (only during training)
        if training and discriminator is not None:
            # Generator loss
            gen_loss = self.gan_generator_loss(
                results['disc_fake'],
                results['disc_recon']
            )
            losses['gan_generator'] = gen_loss
            
            # Discriminator loss
            disc_loss, gp = self.gan_discriminator_loss(
                results['disc_real'],
                results['disc_fake'],
                results['disc_recon'],
                x,
                results['generated'],
                reconstructed,
                discriminator
            )
            losses['gan_discriminator'] = disc_loss
            losses['gradient_penalty'] = gp
        else:
            losses['gan_generator'] = torch.tensor(0.0, device=x.device)
            losses['gan_discriminator'] = torch.tensor(0.0, device=x.device)
            losses['gradient_penalty'] = torch.tensor(0.0, device=x.device)
        
        # 4. Total loss
        total_loss = (
            self.alpha * losses['reconstruction'] +
            self.beta * losses['contrastive'] +
            self.gamma * losses['gan_generator'] +
            self.delta * losses['gan_discriminator']
        )
        losses['total'] = total_loss
        
        return losses
    
    def update_weights(
        self,
        epoch: int,
        max_epochs: int,
        strategy: str = 'linear'
    ):
        """
        Adaptively update loss weights during training
        
        Args:
            epoch: Current epoch
            max_epochs: Total number of epochs
            strategy: Weight update strategy ('linear', 'cosine', 'step')
        """
        if strategy == 'linear':
            # Gradually increase contrastive weight
            progress = epoch / max_epochs
            self.beta = 0.5 * (1 + progress)
        elif strategy == 'cosine':
            # Cosine annealing for GAN weights
            progress = epoch / max_epochs
            self.gamma = 0.3 * (1 + 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159))))
            self.delta = 0.2 * (1 + 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159))))
        elif strategy == 'step':
            # Step function: increase contrastive after 50% of training
            if epoch > max_epochs // 2:
                self.beta = 1.0


"""
GAN components for anomaly detection
"""
import torch
import torch.nn as nn
from typing import Tuple


class Generator(nn.Module):
    """Generator network for GAN"""
    
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (256, 512, 256),
        output_seq_len: int = 10,
        output_features: int = 30,
        dropout: float = 0.1
    ):
        """
        Args:
            latent_dim: Dimension of latent vector
            hidden_dims: Hidden layer dimensions
            output_seq_len: Output sequence length
            output_features: Number of output features
            dropout: Dropout rate
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_seq_len = output_seq_len
        self.output_features = output_features
        
        # Build generator layers
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Output layer
        output_dim = output_seq_len * output_features
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())  # Normalize output to [-1, 1]
        
        self.model = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate time series from latent vector
        
        Args:
            z: Latent vector [batch, latent_dim]
            
        Returns:
            Generated time series [batch, seq_len, features]
        """
        batch_size = z.shape[0]
        output = self.model(z)  # [batch, seq_len * features]
        output = output.view(batch_size, self.output_seq_len, self.output_features)
        return output


class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    
    def __init__(
        self,
        input_seq_len: int = 10,
        input_features: int = 30,
        channels: Tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.1,
        use_spectral_norm: bool = True
    ):
        """
        Args:
            input_seq_len: Input sequence length
            input_features: Number of input features
            channels: Convolutional channels
            dropout: Dropout rate
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        self.input_seq_len = input_seq_len
        self.input_features = input_features
        
        # Build discriminator layers
        layers = []
        in_channels = input_features
        
        for out_channels in channels:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.extend([
                conv,
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discriminate real vs fake
        
        Args:
            x: Input time series [batch, seq_len, features]
            
        Returns:
            Tuple of (logits, features)
        """
        # Convert to [batch, features, seq_len] for Conv1d
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        
        # Convolutional layers
        features = self.conv_layers(x)  # [batch, channels, seq_len']
        
        # Global average pooling
        pooled = self.global_pool(features)  # [batch, channels, 1]
        pooled = pooled.squeeze(-1)  # [batch, channels]
        
        # Classification
        logits = self.fc(pooled)  # [batch, 1]
        
        return logits, pooled


def gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute gradient penalty for Wasserstein GAN
    
    Args:
        discriminator: Discriminator model
        real_samples: Real samples [batch, seq_len, features]
        fake_samples: Fake samples [batch, seq_len, features]
        device: Device
        
    Returns:
        Gradient penalty value
    """
    batch_size = real_samples.shape[0]
    alpha = torch.rand(batch_size, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    # Get discriminator output
    disc_interpolates, _ = discriminator(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=(1, 2)) - 1) ** 2).mean()
    
    return gradient_penalty


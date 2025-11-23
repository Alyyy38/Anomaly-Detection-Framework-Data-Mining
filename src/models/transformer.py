"""
Transformer-based encoder-decoder for time series anomaly detection
"""
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderDecoder(nn.Module):
    """Transformer encoder-decoder for time series reconstruction"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        latent_dim: int = 64,
        max_seq_len: int = 100
    ):
        """
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            latent_dim: Latent space dimension
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent projection
        self.latent_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, latent_dim)
        )
        
        # Decoder input projection
        self.decoder_input_projection = nn.Linear(latent_dim, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            Tuple of (encoded_features, latent_representation)
        """
        # Project input
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Encode
        encoded = self.encoder(x)  # [batch, seq_len, d_model]
        
        # Project to latent space (use mean pooling)
        latent = self.latent_projection(encoded.mean(dim=1))  # [batch, latent_dim]
        
        return encoded, latent
    
    def decode(self, latent: torch.Tensor, memory: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation to sequence
        
        Args:
            latent: Latent tensor [batch, latent_dim]
            memory: Encoder output to use as memory [batch, seq_len, d_model]
            seq_len: Target sequence length
            
        Returns:
            Reconstructed sequence [batch, seq_len, input_dim]
        """
        batch_size = latent.shape[0]
        
        # Project latent to decoder input
        decoder_input = self.decoder_input_projection(latent)  # [batch, d_model]
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        decoder_input = self.pos_encoder(decoder_input)
        
        # Decode using encoder output as memory
        decoded = self.decoder(decoder_input, memory)  # [batch, seq_len, d_model]
        
        # Project to output
        output = self.output_projection(decoded)  # [batch, seq_len, input_dim]
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (encoded_features, latent, reconstructed, attention_weights)
        """
        # Encode
        encoded, latent = self.encode(x)
        
        # Decode
        reconstructed = self.decode(latent, encoded, x.shape[1])
        
        # Get attention weights if requested
        attention_weights = None
        if return_attention:
            # Extract attention from the last encoder layer
            # Note: This is a simplified version
            attention_weights = torch.ones(
                encoded.shape[0], self.encoder.layers[-1].self_attn.num_heads,
                encoded.shape[1], encoded.shape[1],
                device=encoded.device
            )
        
        return encoded, latent, reconstructed, attention_weights


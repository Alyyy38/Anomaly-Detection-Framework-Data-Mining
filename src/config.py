"""
Configuration file for Anomaly Detection Framework
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Model architecture configurations"""
    # Transformer
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    latent_dim: int = 64
    
    # GAN
    gan_latent_dim: int = 64
    generator_hidden_dims: Tuple[int, ...] = (256, 512, 256)
    discriminator_channels: Tuple[int, ...] = (64, 128, 256)
    
    # Contrastive Learning
    projection_dim: int = 128
    temperature: float = 0.07
    use_memory_bank: bool = False
    memory_bank_size: int = 4096


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizers
    transformer_lr: float = 1e-4
    generator_lr: float = 2e-4
    discriminator_lr: float = 2e-4
    
    # Loss weights
    alpha: float = 1.0  # Reconstruction
    beta: float = 0.5   # Contrastive
    gamma: float = 0.3  # GAN generator
    delta: float = 0.2  # GAN discriminator
    
    # Training settings
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 10
    save_checkpoint_every: int = 5
    
    # Scheduler
    scheduler_type: str = "cosine"  # "cosine" or "step"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1


@dataclass
class DataConfig:
    """Data preprocessing configurations"""
    # Dataset paths
    data_path: str = "creditcard.csv"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Time series windowing
    window_size: int = 10
    stride: int = 1
    
    # Augmentation
    masking_ratio: float = 0.2
    time_mask_ratio: Tuple[float, float] = (0.1, 0.3)
    feature_mask_ratio: Tuple[float, float] = (0.1, 0.2)
    augmentation_prob: float = 0.5
    
    # Preprocessing
    normalize: bool = True
    handle_imbalance: bool = True
    contamination_rate: float = 0.05  # Expected anomaly rate in training


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    # System
    device: str = "cuda"  # "cuda" or "cpu"
    num_workers: int = 4
    seed: int = 42
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    results_dir: str = "results"
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()


# Default configuration instance
default_config = Config()


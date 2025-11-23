"""
Main training script for Anomaly Detection Framework
"""
import argparse
import torch
import numpy as np
import random
from pathlib import Path

from src.config import Config, default_config
from src.data import CreditCardDataLoader, DataPreprocessor
from src.models import AnomalyDetectionFramework
from src.training import Trainer


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train Anomaly Detection Framework')
    parser.add_argument('--data_path', type=str, default='creditcard.csv',
                        help='Path to creditcard.csv file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--window_size', type=int, default=None,
                        help='Window size for time series')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (JSON)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(default_config.seed)
    
    # Load or create config
    config = default_config
    if args.config:
        # Load from JSON if provided
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            # Update config (simplified - would need proper merging)
            pass
    
    # Override config with command line arguments
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.window_size:
        config.data.window_size = args.window_size
    if args.device:
        config.device = args.device
    if args.data_path:
        config.data.data_path = args.data_path
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    loader = CreditCardDataLoader(data_path=config.data.data_path)
    raw_data = loader.load_data()
    
    # Split data
    train_df, val_df, test_df = loader.split_data(
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        random_state=config.seed
    )
    
    # Preprocess data
    print("\n" + "="*50)
    print("PREPROCESSING DATA")
    print("="*50)
    preprocessor = DataPreprocessor(
        normalize=config.data.normalize,
        scaler_type='standard'
    )
    
    train_df, val_df, test_df = preprocessor.preprocess(
        train_df,
        val_df,
        test_df,
        balance=config.data.handle_imbalance,
        balance_method='undersample'
    )
    
    # Get feature columns
    feature_cols = preprocessor.get_feature_columns(train_df)
    input_dim = len(feature_cols)
    
    print(f"\nInput dimension: {input_dim}")
    print(f"Window size: {config.data.window_size}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = loader.get_data_loaders(
        train_df,
        val_df,
        test_df,
        feature_cols,
        window_size=config.data.window_size,
        batch_size=config.training.batch_size,
        num_workers=config.num_workers
    )
    
    # Create model
    print("\n" + "="*50)
    print("INITIALIZING MODEL")
    print("="*50)
    model = AnomalyDetectionFramework(
        input_dim=input_dim,
        seq_len=config.data.window_size,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=0.1,
        latent_dim=config.model.latent_dim,
        projection_dim=config.model.projection_dim,
        temperature=config.model.temperature,
        time_mask_ratio=config.data.time_mask_ratio,
        feature_mask_ratio=config.data.feature_mask_ratio,
        augmentation_prob=config.data.augmentation_prob
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    trainer.train()
    
    print("\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == '__main__':
    main()


"""
Inference script for anomaly detection
"""
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import Config, default_config
from src.data import CreditCardDataLoader, DataPreprocessor
from src.models import AnomalyDetectionFramework
from src.utils.metrics import AnomalyMetrics


def load_model(checkpoint_path: str, device: torch.device) -> AnomalyDetectionFramework:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', default_config)
    
    # Get model dimensions from config or checkpoint
    # For now, we'll need to specify or load from config
    input_dim = 30  # Default for credit card dataset (V1-V28 + Amount)
    seq_len = config.data.window_size if hasattr(config, 'data') else 10
    
    # Create model
    model = AnomalyDetectionFramework(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=config.model.d_model if hasattr(config, 'model') else 128,
        nhead=config.model.nhead if hasattr(config, 'model') else 8,
        num_layers=config.model.num_layers if hasattr(config, 'model') else 4,
        dim_feedforward=config.model.dim_feedforward if hasattr(config, 'model') else 512,
        dropout=0.1,
        latent_dim=config.model.latent_dim if hasattr(config, 'model') else 64,
        projection_dim=config.model.projection_dim if hasattr(config, 'model') else 128,
        temperature=config.model.temperature if hasattr(config, 'model') else 0.07
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def predict_from_csv(
    model: AnomalyDetectionFramework,
    data_path: str,
    preprocessor: DataPreprocessor,
    feature_cols: list,
    window_size: int,
    threshold: float = None,
    device: torch.device = None
):
    """Predict anomalies from CSV file"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Preprocess
    df_processed, _, _ = preprocessor.preprocess(df, None, None, balance=False)
    
    # Extract features
    X = df_processed[feature_cols].values
    
    # Create windows
    windows = []
    for i in range(0, len(X) - window_size + 1):
        window = X[i:i + window_size]
        windows.append(window)
    
    windows = np.array(windows)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(windows).to(device)
    
    # Predict
    with torch.no_grad():
        predictions, scores = model.detect_anomaly(X_tensor, threshold=threshold)
    
    return predictions.cpu().numpy(), scores.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to output CSV file')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Anomaly threshold (if None, uses median)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(
        args.device if args.device else
        ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Load preprocessor (would need to be saved/loaded properly in production)
    preprocessor = DataPreprocessor(normalize=True)
    
    # For now, assume standard feature columns
    # In production, these should be saved with the model
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    window_size = config.data.window_size if hasattr(config, 'data') else 10
    
    # Predict
    print(f"Processing data from {args.data_path}...")
    predictions, scores = predict_from_csv(
        model,
        args.data_path,
        preprocessor,
        feature_cols,
        window_size,
        threshold=args.threshold,
        device=device
    )
    
    # Save results
    results_df = pd.DataFrame({
        'anomaly_score': scores,
        'prediction': predictions
    })
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Print summary
    num_anomalies = predictions.sum()
    print(f"\nSummary:")
    print(f"Total samples: {len(predictions)}")
    print(f"Anomalies detected: {num_anomalies} ({num_anomalies/len(predictions)*100:.2f}%)")
    print(f"Mean anomaly score: {scores.mean():.4f}")
    print(f"Std anomaly score: {scores.std():.4f}")


if __name__ == '__main__':
    main()


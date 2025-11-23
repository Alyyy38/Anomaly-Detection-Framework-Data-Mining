# Anomaly Detection Framework

A robust anomaly detection system combining **Transformers + GANs + Contrastive Learning + Geometric Masking** for Credit Card Fraud Detection.

## 🎯 Project Overview

This framework implements a state-of-the-art anomaly detection system that combines multiple deep learning techniques:

- **Transformer Encoder-Decoder**: For learning temporal patterns and reconstruction
- **Generative Adversarial Networks (GANs)**: For learning normal data distributions
- **Contrastive Learning**: For learning robust representations
- **Geometric Masking**: For data augmentation and improved generalization

## 📋 Features

- **Multi-component Architecture**: Combines transformer, GAN, and contrastive learning
- **Geometric Augmentation**: Time masking, feature masking, time warping, and mixup
- **Comprehensive Training**: Multi-optimizer setup with learning rate scheduling
- **Robust Evaluation**: Extensive metrics including ROC-AUC, PR-AUC, F1-score
- **Production Ready**: Inference script and API-ready code

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd anomaly-detection-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the dataset (`creditcard.csv`) is in the project root directory.

## 📁 Project Structure

```
anomaly-detection-framework/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data
├── src/
│   ├── config.py         # Configuration settings
│   ├── data/
│   │   ├── loader.py      # Data loading utilities
│   │   └── preprocessor.py # Data preprocessing
│   ├── models/
│   │   ├── transformer.py # Transformer encoder-decoder
│   │   ├── gan.py         # GAN components
│   │   ├── contrastive.py # Contrastive learning
│   │   └── framework.py   # Unified framework
│   ├── training/
│   │   ├── trainer.py     # Training pipeline
│   │   └── losses.py      # Loss functions
│   └── utils/
│       ├── augmentation.py # Geometric masking
│       └── metrics.py      # Evaluation metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── checkpoints/          # Model checkpoints
├── runs/                 # TensorBoard logs
├── train.py             # Training script
├── inference.py         # Inference script
└── requirements.txt     # Dependencies
```

## 🎓 Usage

### Training

Train the model using the main training script:

```bash
python train.py --epochs 100 --batch_size 256 --window_size 10
```

**Arguments:**
- `--data_path`: Path to creditcard.csv (default: `creditcard.csv`)
- `--epochs`: Number of training epochs (default: from config)
- `--batch_size`: Batch size (default: from config)
- `--window_size`: Time series window size (default: 10)
- `--device`: Device to use (`cuda` or `cpu`)
- `--resume`: Path to checkpoint to resume from

**Example:**
```bash
python train.py --epochs 50 --batch_size 128 --device cuda
```

### Inference

Run inference on new data:

```bash
python inference.py --checkpoint checkpoints/best.pth --data_path test_data.csv --output predictions.csv
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint (required)
- `--data_path`: Path to input CSV file (required)
- `--output`: Path to output CSV file (default: `predictions.csv`)
- `--threshold`: Anomaly threshold (optional, uses median if not provided)
- `--device`: Device to use (`cuda` or `cpu`)

### Jupyter Notebooks

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`):
   - Explore dataset statistics
   - Visualize class distribution
   - Test preprocessing pipeline

2. **Model Training** (`notebooks/02_model_training.ipynb`):
   - Step-by-step training process
   - Monitor training progress
   - Save checkpoints

3. **Evaluation** (`notebooks/03_evaluation.ipynb`):
   - Evaluate model performance
   - Generate metrics and visualizations
   - Find optimal thresholds

## ⚙️ Configuration

Configuration is managed in `src/config.py`. Key parameters:

### Model Configuration
- `d_model`: Transformer model dimension (default: 128)
- `nhead`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer layers (default: 4)
- `latent_dim`: Latent space dimension (default: 64)

### Training Configuration
- `batch_size`: Batch size (default: 256)
- `num_epochs`: Number of epochs (default: 100)
- `learning_rate`: Learning rate (default: 1e-4)
- `alpha`, `beta`, `gamma`, `delta`: Loss weights

### Data Configuration
- `window_size`: Time series window size (default: 10)
- `masking_ratio`: Geometric masking ratio (default: 0.2)
- `augmentation_prob`: Augmentation probability (default: 0.5)

## 📊 Model Architecture

### Components

1. **Transformer Encoder-Decoder**
   - Encodes input sequences to latent representations
   - Decodes latent vectors to reconstruct sequences
   - Uses positional encoding and multi-head attention

2. **GAN (Generator + Discriminator)**
   - Generator: Generates synthetic time series from noise
   - Discriminator: Distinguishes real from fake samples
   - Uses Wasserstein loss with gradient penalty

3. **Contrastive Learning Module**
   - Projects latent features to contrastive space
   - Uses NT-Xent (InfoNCE) loss
   - Creates positive pairs through augmentation

4. **Geometric Masking**
   - Random time step masking
   - Feature masking
   - Time warping
   - Mixup augmentation

### Loss Function

Total loss combines multiple components:

```
L_total = α * L_reconstruction + β * L_contrastive + γ * L_gan_gen + δ * L_gan_disc
```

Where:
- `L_reconstruction`: MSE loss on masked regions
- `L_contrastive`: NT-Xent contrastive loss
- `L_gan_gen`: Generator Wasserstein loss
- `L_gan_disc`: Discriminator loss with gradient penalty

## 📈 Evaluation Metrics

The framework computes comprehensive metrics:

- **Classification Metrics**: Precision, Recall, F1-Score, Accuracy
- **AUC Metrics**: ROC-AUC, PR-AUC
- **Confusion Matrix**: True/False Positives/Negatives
- **Anomaly Scores**: Reconstruction error, discriminator score, combined score

## 🔧 Development

### Running Tests

```bash
# Add tests when available
pytest tests/
```

### Code Structure

- **Modular Design**: Each component is independently testable
- **Type Hints**: Full type annotations for better IDE support
- **Documentation**: Comprehensive docstrings

## 📝 Results

Expected performance on Credit Card Fraud Detection dataset:

- **F1-Score**: > 0.85
- **ROC-AUC**: > 0.92
- **False Positive Rate**: < 5%

*Note: Actual results depend on hyperparameters and training configuration.*

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Reduce `window_size`
   - Use gradient accumulation

2. **Slow Training**
   - Enable mixed precision training
   - Use multiple GPUs
   - Reduce model size (`d_model`, `num_layers`)

3. **Poor Performance**
   - Adjust loss weights (`alpha`, `beta`, `gamma`, `delta`)
   - Tune masking ratios
   - Increase training epochs

## 📚 References

- Transformer Architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- GANs: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- Contrastive Learning: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
- Time Series Anomaly Detection: Various papers on transformer-based anomaly detection

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ using PyTorch**


# Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure `creditcard.csv` is in the project root.

## Training

Quick training command:
```bash
python train.py --epochs 50 --batch_size 128
```

Full training with all options:
```bash
python train.py --epochs 100 --batch_size 256 --window_size 10 --device cuda
```

## Inference

Run inference on new data:
```bash
python inference.py --checkpoint checkpoints/best.pth --data_path test_data.csv --output predictions.csv
```

## Using Jupyter Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open notebooks in order:
   - `notebooks/01_data_exploration.ipynb` - Explore the dataset
   - `notebooks/02_model_training.ipynb` - Train the model
   - `notebooks/03_evaluation.ipynb` - Evaluate performance

## Monitoring Training

View TensorBoard logs:
```bash
tensorboard --logdir runs/
```

## Key Files

- `train.py` - Main training script
- `inference.py` - Inference script
- `src/config.py` - Configuration settings
- `src/models/framework.py` - Main model architecture
- `src/training/trainer.py` - Training pipeline

## Tips

- Start with smaller batch sizes (64-128) if you have limited GPU memory
- Use `--window_size 5` for faster training on smaller datasets
- Monitor validation loss to avoid overfitting
- Adjust loss weights in `src/config.py` if needed


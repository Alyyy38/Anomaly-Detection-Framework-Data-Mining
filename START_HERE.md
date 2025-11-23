# Quick Start Commands

## Start Jupyter Notebook
```powershell
python -m jupyter notebook
```

## Start JupyterLab (alternative)
```powershell
python -m jupyter lab
```

## Run Training
```powershell
python train.py --epochs 10 --batch_size 128
```

## Test Data Loading
```powershell
python -c "from src.data import CreditCardDataLoader; loader = CreditCardDataLoader('creditcard.csv'); data = loader.load_data(); print(f'Loaded {len(data)} samples')"
```

## Test Model Creation
```powershell
python -c "from src.models import AnomalyDetectionFramework; model = AnomalyDetectionFramework(input_dim=30, seq_len=10); print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')"
```

## Fix PATH Issue (Optional)
If you want to use `jupyter` directly instead of `python -m jupyter`, add this to your PATH:
```
C:\Users\Home\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts
```

Or create an alias in PowerShell:
```powershell
Set-Alias jupyter "python -m jupyter"
```


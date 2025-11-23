# After Restart - Complete Installation

## ✅ Long Path Support Enabled

Windows Long Path support has been successfully enabled. You need to **restart your computer** for the change to take effect.

## After Restart

### 1. Verify Long Path Support
```powershell
reg query "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled
```
Should show: `LongPathsEnabled    REG_DWORD    0x1`

### 2. Retry Failed Package Installations

The packages that failed due to long paths should now install successfully:

```powershell
cd "U:\Documents\Data Mining\DM_Project"
python -m pip install --user plotly flask fastapi kaggle
```

Or install all remaining packages:
```powershell
python -m pip install --user -r requirements.txt
```

### 3. Test Your Installation

```powershell
python -c "import torch; import pandas; import numpy; import sklearn; import matplotlib; import tensorboard; print('✅ All core packages working!')"
```

### 4. Start Using the Project

Once packages are installed, you can:

**Run training:**
```powershell
python train.py --epochs 10 --batch_size 128
```

**Start Jupyter:**
```powershell
jupyter notebook
```

**Or use the notebooks:**
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_model_training.ipynb`
- `notebooks/03_evaluation.ipynb`

## Notes

- The `--user` flag installs packages to your user directory, avoiding permission issues
- After restart, the long path errors should be resolved
- If you still see warnings about scripts not on PATH, they're cosmetic and won't affect functionality


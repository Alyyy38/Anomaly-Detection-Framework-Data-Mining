# Fix Windows Long Path and PATH Issues

## Issue 1: Enable Windows Long Path Support

The error occurs because Windows has a 260-character path limit. To fix:

### Option A: Enable via Registry (Requires Admin)

1. Press `Win + R`, type `regedit`, press Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Find `LongPathsEnabled` (or create it as DWORD if missing)
4. Set value to `1`
5. Restart your computer

### Option B: Enable via PowerShell (Requires Admin)

Run PowerShell as Administrator and execute:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Then restart your computer.

### Option C: Use Shorter Installation Path (Quick Fix)

Install packages to a shorter path:
```powershell
pip install --target C:\py_packages -r requirements.txt
```

Then add to your Python path:
```python
import sys
sys.path.insert(0, r'C:\py_packages')
```

## Issue 2: Add Scripts to PATH

To fix the PATH warnings, add the Scripts directory to your PATH:

1. Press `Win + X` → System → Advanced system settings
2. Click "Environment Variables"
3. Under "User variables", select "Path" → Edit
4. Add: `C:\Users\Home\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts`
5. Click OK on all dialogs
6. Restart your terminal

## Quick Workaround: Install with --user flag

This installs to a shorter path:
```powershell
pip install --user -r requirements.txt
```

## Alternative: Use Virtual Environment in Short Path

Create a venv in a shorter path:
```powershell
cd C:\
python -m venv anomaly_detection
.\anomaly_detection\Scripts\activate
cd "U:\Documents\Data Mining\DM_Project"
pip install -r requirements.txt
```


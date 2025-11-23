# Applications and Dataset Adaptation Guide

## 1. Real-World Applications of This Framework

### Financial Services
- **Credit Card Fraud Detection** (Current use case)
  - Real-time transaction monitoring
  - Detecting unauthorized card usage
  - Identifying suspicious spending patterns
  
- **Banking Anomaly Detection**
  - Unusual account activity
  - Money laundering detection
  - Account takeover detection
  - Suspicious wire transfers

- **Insurance Fraud**
  - Claim fraud detection
  - Policy abuse identification
  - Unusual claim patterns

### Cybersecurity
- **Network Intrusion Detection**
  - Malicious network traffic
  - DDoS attack detection
  - Unauthorized access attempts
  - Botnet activity detection

- **System Security**
  - Unusual login patterns
  - Privilege escalation detection
  - Data exfiltration attempts
  - Insider threat detection

### Healthcare
- **Medical Anomaly Detection**
  - Unusual patient vital signs
  - Medical device malfunction
  - Anomalous lab results
  - Disease outbreak detection

- **Healthcare Fraud**
  - Billing fraud detection
  - Prescription abuse
  - Unusual treatment patterns

### Manufacturing & IoT
- **Industrial Equipment Monitoring**
  - Predictive maintenance
  - Equipment failure detection
  - Quality control anomalies
  - Production line defects

- **IoT Sensor Anomalies**
  - Sensor malfunction detection
  - Environmental anomaly detection
  - Smart home anomaly detection

### E-commerce & Retail
- **Retail Fraud**
  - Return fraud detection
  - Price manipulation
  - Inventory anomalies
  - Unusual purchasing patterns

- **Recommendation Systems**
  - Detecting fake reviews
  - Identifying bot behavior
  - Unusual user activity

### Transportation
- **Traffic Anomaly Detection**
  - Accident detection
  - Traffic flow anomalies
  - Vehicle breakdown detection

- **Logistics**
  - Route anomaly detection
  - Delivery time anomalies
  - Supply chain disruptions

### Energy Sector
- **Power Grid Monitoring**
  - Power outage detection
  - Energy theft detection
  - Grid instability detection

- **Renewable Energy**
  - Solar panel malfunction
  - Wind turbine anomalies
  - Energy production anomalies

## 2. Adapting to Other Datasets

### ✅ YES, the framework CAN work on other anomaly detection datasets!

The framework is designed to be **dataset-agnostic** and can be adapted to various anomaly detection tasks.

### Key Adaptable Components

#### 1. **Data Loader** (`src/data/loader.py`)
- Currently: CSV-based credit card data
- Adaptable to: Any tabular, time series, or sequential data
- Modification needed: Update `CreditCardDataLoader` or create new loader

#### 2. **Preprocessor** (`src/data/preprocessor.py`)
- Currently: Handles numerical features, normalization
- Adaptable to: Different feature types (categorical, text, images)
- Modification needed: Add encoding for categorical/text features

#### 3. **Model Architecture** (`src/models/framework.py`)
- Currently: Works with 29 features, window size 10
- Adaptable to: Any input dimension and sequence length
- Modification needed: Change `input_dim` and `seq_len` parameters

#### 4. **Window Size**
- Currently: 10 timesteps
- Adaptable to: Any sequence length
- Modification needed: Adjust `window_size` in config

## 3. Steps to Adapt to a New Dataset

### Step 1: Understand Your Dataset
```python
# Check your dataset structure
import pandas as pd
df = pd.read_csv('your_dataset.csv')
print(df.info())
print(df.head())
print(f"Anomaly rate: {df['label'].mean()*100:.2f}%")
```

### Step 2: Create/Modify Data Loader
```python
# Example: Adapt for a new dataset
class YourDatasetLoader:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_data(self):
        # Load your dataset
        return pd.read_csv(self.data_path)
    
    def get_feature_columns(self, df):
        # Exclude target and non-feature columns
        exclude = ['label', 'id', 'timestamp']  # Adjust as needed
        return [col for col in df.columns if col not in exclude]
```

### Step 3: Update Configuration
```python
# In src/config.py or train.py
config.data.data_path = "your_dataset.csv"
config.data.window_size = 10  # Adjust based on your data
config.model.input_dim = your_feature_count
```

### Step 4: Handle Different Data Types

#### For Categorical Features:
```python
# Add to preprocessor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_categorical(self, df, categorical_cols):
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df
```

#### For Time Series Data:
```python
# Already supported! Just adjust window_size
config.data.window_size = 20  # For longer sequences
```

#### For Image Data:
```python
# Would need CNN-based encoder instead of Transformer
# But framework structure remains similar
```

### Step 5: Adjust Hyperparameters
```python
# Based on your dataset characteristics
config.training.batch_size = 64  # Adjust based on dataset size
config.data.window_size = 15     # Adjust based on sequence length
config.model.d_model = 128       # Adjust based on complexity
```

## 4. Example: Adapting to Outlier Detection Datasets

### Common Outlier Detection Datasets:

1. **KDD Cup 1999** (Network Intrusion)
   - Features: 41 features (mixed types)
   - Adaptation: Handle categorical + numerical features

2. **Shuttle Dataset**
   - Features: 9 numerical features
   - Adaptation: Direct use, just change input_dim=9

3. **Forest Cover Type**
   - Features: 54 features (mixed types)
   - Adaptation: Encode categorical features

4. **Thyroid Disease**
   - Features: 21 features (mixed types)
   - Adaptation: Handle missing values + encoding

### Quick Adaptation Example:

```python
# For a new outlier detection dataset
from src.data import DataPreprocessor
from src.models import AnomalyDetectionFramework

# 1. Load your dataset
loader = YourDatasetLoader('outlier_dataset.csv')
data = loader.load_data()

# 2. Get features (excluding target)
feature_cols = loader.get_feature_columns(data)
input_dim = len(feature_cols)

# 3. Create model with correct dimensions
model = AnomalyDetectionFramework(
    input_dim=input_dim,        # Your feature count
    seq_len=10,                  # Adjust based on your needs
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    latent_dim=64
)

# 4. Train as usual
python train.py --data_path outlier_dataset.csv --window_size 10
```

## 5. Framework Strengths for Different Datasets

### ✅ Works Well For:
- **Tabular data** (like credit card fraud)
- **Time series data** (sensor readings, logs)
- **Sequential data** (user behavior, transactions)
- **Multivariate data** (multiple features)

### ⚠️ Requires Modification For:
- **Image data** (need CNN encoder)
- **Text data** (need text tokenization)
- **Graph data** (need GNN components)
- **Very high-dimensional data** (may need dimensionality reduction)

### 🔧 Easy Adaptations:
- Different feature counts → Change `input_dim`
- Different sequence lengths → Change `window_size`
- Different data types → Modify preprocessor
- Different imbalance ratios → Adjust loss weights

## 6. Pre-trained Model Transfer

The framework can also benefit from:
- **Transfer Learning**: Pre-train on large dataset, fine-tune on specific domain
- **Domain Adaptation**: Train on one domain, adapt to similar domain
- **Few-shot Learning**: Use contrastive learning for limited labeled data

## 7. Performance Considerations

### For Small Datasets (<10K samples):
- Reduce model size (`d_model=64`, `num_layers=2`)
- Use smaller batch size (`batch_size=32`)
- Reduce window size if applicable

### For Large Datasets (>1M samples):
- Increase batch size (`batch_size=512`)
- Use GPU acceleration
- Consider distributed training

### For High-Dimensional Data (>100 features):
- Consider feature selection
- Use dimensionality reduction (PCA)
- Increase model capacity

## 8. Example Use Cases by Dataset Type

| Dataset Type | Example | Adaptation Needed |
|-------------|---------|-------------------|
| Tabular | Credit Card Fraud | ✅ Ready to use |
| Time Series | Sensor Data | ✅ Adjust window_size |
| Network Logs | KDD Cup | 🔧 Handle categorical |
| Images | Medical Images | ⚠️ Need CNN encoder |
| Text | Log Analysis | ⚠️ Need tokenization |
| Multivariate | IoT Sensors | ✅ Ready to use |

## 9. Quick Start for New Dataset

```python
# Minimal adaptation script
import pandas as pd
from src.data import DataPreprocessor
from src.models import AnomalyDetectionFramework

# Load your dataset
df = pd.read_csv('new_dataset.csv')

# Identify features and target
feature_cols = [col for col in df.columns if col != 'label']
target_col = 'label'

# Get dimensions
input_dim = len(feature_cols)
print(f"Input dimension: {input_dim}")

# Create model
model = AnomalyDetectionFramework(
    input_dim=input_dim,
    seq_len=10,  # Adjust as needed
    # ... other parameters
)

# Train
python train.py --data_path new_dataset.csv
```

## 10. Conclusion

**The framework is highly adaptable!** With minimal modifications, it can work on:
- ✅ Most tabular anomaly detection datasets
- ✅ Time series anomaly detection
- ✅ Sequential pattern anomaly detection
- ✅ Multivariate anomaly detection

The key is understanding your dataset structure and adjusting:
1. **Input dimensions** (number of features)
2. **Window size** (sequence length)
3. **Preprocessing** (handling different data types)
4. **Hyperparameters** (based on dataset size and complexity)

The modular design makes it easy to swap components (e.g., different encoders, preprocessors) while keeping the core framework intact.


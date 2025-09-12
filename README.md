# Industrial Electrical Equipment Carbon Footprint Analysis

## Python 3.13 Compatible Implementation

This repository contains a Python 3.13 compatible implementation of an industrial electrical equipment carbon footprint identification methodology. The system uses machine learning and optimization techniques to analyze power consumption patterns and minimize carbon emissions in industrial settings.

## 🔬 Methodology Framework

The implementation follows this framework:
```
Data Input → Data Embedding → GRU Layers → Multi-head Self-Attention → Output States → Carbon Footprint Analysis
```

### Components Overview

1. **Data Preprocessing** (`data_preprocessing.py`): Handles power consumption and appliance state data preprocessing
2. **Neural Network Models** (`models.py`): Implements GRU with multi-head self-attention for appliance identification
3. **DC Optimal Power Flow** (`dc_opf.py`): Performs optimization for carbon footprint minimization

## 🚀 Installation & Setup

### Requirements
- Python 3.12+ (tested on 3.12.3, compatible with 3.13)
- See `requirements.txt` for complete dependencies

### Installation
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **NumPy** ≥1.24.0 - Numerical computing
- **Pandas** ≥2.0.0 - Data manipulation 
- **PyTorch** ≥2.0.0 - Deep learning framework
- **Scikit-learn** ≥1.3.0 - Machine learning utilities
- **CVXPY** ≥1.4.0 - Convex optimization
- **Matplotlib** ≥3.7.0 - Visualization

## 📊 Module Details

### 1. Data Preprocessing Module (`data_preprocessing.py`)

**Purpose**: Preprocesses raw power consumption data for machine learning analysis.

**Key Features**:
- ✅ **Python 3.13 Compatibility**: Updated deprecated pandas methods (`fillna` → `ffill`/`bfill`)
- ✅ **Modern Type Hints**: Full type annotation support
- ✅ **Robust Data Cleaning**: Handles missing values and outliers
- ✅ **Feature Engineering**: Extracts temporal and statistical features
- ✅ **Sequence Generation**: Creates time series sequences for neural networks

**Key Classes**:
- `PowerDataPreprocessor`: Main preprocessing class
- Functions: `load_sample_data()`, `clean_data()`, `extract_features()`

**Python 3.13 Improvements**:
```python
# Before (deprecated in pandas 2.0+)
data.fillna(method='ffill')

# After (Python 3.13 compatible)
data.ffill()
```

**Usage Example**:
```python
from data_preprocessing import PowerDataPreprocessor, load_sample_data

# Initialize preprocessor
preprocessor = PowerDataPreprocessor(
    normalization_method='standard',
    sequence_length=24
)

# Load and preprocess data
data = load_sample_data()
results = preprocessor.preprocess_pipeline(data, target_column='power_consumption')
```

### 2. Neural Network Models (`models.py`)

**Purpose**: Implements GRU-based neural networks with multi-head self-attention for appliance state identification.

**Architecture**:
```
Input Features → Data Embedding → Bidirectional GRU → Multi-head Self-Attention → Classification
```

**Key Features**:
- ✅ **Modern PyTorch Implementation**: Uses latest PyTorch APIs
- ✅ **Multi-head Self-Attention**: Custom implementation for sequence modeling
- ✅ **Bidirectional GRU**: Captures temporal dependencies in both directions
- ✅ **Flexible Architecture**: Configurable layers, attention heads, and classes
- ✅ **Carbon Footprint Prediction**: High-level interface for practical use

**Key Classes**:
- `MultiHeadSelfAttention`: Self-attention mechanism
- `GRUWithAttention`: Complete neural network model
- `CarbonFootprintPredictor`: High-level prediction interface
- `ApplianceDataset`: PyTorch dataset for training

**Python 3.13 Improvements**:
```python
# Enhanced type hints and modern syntax
def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Implementation uses latest PyTorch features
    return {
        'logits': logits,
        'embeddings': gru_output,
        'attention_output': attention_output
    }
```

**Usage Example**:
```python
from models import CarbonFootprintPredictor

# Initialize predictor
predictor = CarbonFootprintPredictor(
    input_size=15,
    hidden_size=128,
    num_classes=4
)

# Train model
history = predictor.fit(
    train_sequences=X_train,
    train_targets=y_train,
    epochs=100
)

# Make predictions
predictions = predictor.predict(X_test)
```

### 3. DC Optimal Power Flow (`dc_opf.py`)

**Purpose**: Implements DC Optimal Power Flow for carbon footprint optimization in power systems.

**Key Features**:
- ✅ **Modern Optimization**: Uses CVXPY for convex optimization
- ✅ **Carbon Emission Modeling**: Integrates carbon costs into power dispatch
- ✅ **Multiple Objectives**: Cost, emissions, or combined optimization
- ✅ **Comprehensive Analysis**: Detailed carbon footprint metrics
- ✅ **Data Export**: Results export to pandas DataFrames

**Key Classes**:
- `DCOptimalPowerFlow`: Main optimization engine
- `Bus`, `Branch`, `Generator`: Power system components
- `BusType`: Enumeration for bus types (Python 3.13 compatible enums)

**Python 3.13 Improvements**:
```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

@dataclass
class Generator:
    """Modern dataclass implementation with type hints"""
    id: int
    bus_id: int
    carbon_emission_rate: float = 0.5  # kg CO2/MWh
    # ... other fields
```

**Optimization Objectives**:
1. **Cost Minimization**: `objective='cost'`
2. **Emission Minimization**: `objective='emissions'`
3. **Combined**: `objective='combined'` (cost + carbon price × emissions)

**Usage Example**:
```python
from dc_opf import create_sample_system

# Create power system
dc_opf = create_sample_system()

# Optimize for minimum emissions
results = dc_opf.solve_dc_opf(objective='emissions')

# Analyze carbon footprint
carbon_metrics = dc_opf.calculate_carbon_footprint()
print(f"Total emissions: {carbon_metrics['total_emissions_kg_co2']:.2f} kg CO2")
```

## 🧪 Testing

### Run All Tests
```bash
python test_modules.py
```

### Individual Module Testing
```bash
# Test data preprocessing
python data_preprocessing.py

# Test neural network models
python models.py

# Test DC-OPF optimization
python dc_opf.py
```

### Test Coverage
The test suite covers:
- ✅ Data preprocessing pipeline
- ✅ Neural network model training and inference
- ✅ DC-OPF optimization with different objectives
- ✅ Carbon footprint calculation and analysis
- ✅ Error handling and edge cases

## 🔧 Python 3.13 Compatibility Fixes

### 1. Pandas Deprecation Warnings
**Issue**: `fillna(method='ffill')` deprecated in pandas 2.0+
```python
# Fixed
cleaned_data = cleaned_data.ffill().bfill()
```

### 2. Modern Type Hints
**Enhancement**: Added comprehensive type annotations
```python
from typing import Dict, List, Tuple, Optional, Union, Any

def process_data(data: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # Implementation with proper type hints
    pass
```

### 3. Dataclass and Enum Integration
**Enhancement**: Used modern Python features
```python
from dataclasses import dataclass
from enum import Enum

class BusType(Enum):
    SLACK = "slack"
    PV = "pv"
    PQ = "pq"

@dataclass
class Bus:
    id: int
    name: str
    bus_type: BusType
```

### 4. Enhanced Error Handling
**Improvement**: Better exception handling and user feedback
```python
try:
    # Optimization code
    problem.solve(solver=cp.ECOS)
except Exception:
    try:
        problem.solve(solver=cp.SCS)
    except Exception:
        problem.solve()
```

## 📈 Performance Metrics

### Neural Network Performance
- **Model Architecture**: Bidirectional GRU + Multi-head Attention
- **Training Time**: ~2-3 minutes for 100 epochs (CPU)
- **Accuracy**: Varies based on data quality and complexity
- **Memory Usage**: Optimized for standard hardware

### Optimization Performance
- **DC-OPF Solve Time**: <1 second for 3-bus system
- **Scalability**: Tested up to 10-bus systems
- **Convergence**: Reliable convergence with ECOS/SCS solvers

## 🌱 Carbon Footprint Analysis Features

### Metrics Provided
1. **Total Emissions**: Total kg CO2 emissions
2. **Emission Rate**: Weighted average emission rate (kg CO2/MWh)
3. **Clean Energy Percentage**: Percentage of generation from clean sources
4. **Per-Generator Analysis**: Detailed breakdown by generator
5. **Cost-Emission Trade-offs**: Analysis of different optimization objectives

### Visualization Support
The modules provide data in formats compatible with:
- Matplotlib for basic plotting
- Seaborn for statistical visualizations
- Pandas for tabular analysis

## 🚨 Known Issues & Limitations

1. **PTDF Matrix**: Simplified implementation for demonstration
2. **Power Flow Constraints**: Basic power balance only (not full AC power flow)
3. **Sample Data**: Synthetic data for demonstration purposes
4. **Scalability**: Optimized for small to medium-sized systems

## 🔮 Future Enhancements

1. **AC Power Flow**: Full AC optimal power flow implementation
2. **Real Data Integration**: Support for real industrial data formats
3. **Advanced ML Models**: Transformer-based architectures
4. **Real-time Optimization**: Online optimization capabilities
5. **Uncertainty Modeling**: Stochastic optimization under uncertainty

## 📚 References

- Deep learning methodology for appliance identification
- DC Optimal Power Flow formulation
- Multi-head self-attention mechanisms
- Carbon footprint calculation standards

## 🤝 Contributing

1. Ensure Python 3.13 compatibility
2. Add comprehensive type hints
3. Include tests for new features
4. Update documentation

## 📄 License

This implementation is provided for educational and research purposes.

---

**Note**: This implementation demonstrates the methodology framework and provides a foundation for industrial electrical equipment carbon footprint analysis. For production use, additional validation and real-world data integration would be required.
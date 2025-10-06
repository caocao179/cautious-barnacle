"""
Data Preprocessing Module for Industrial Electrical Equipment Carbon Footprint Analysis

This module contains preprocessing methods for power consumption and appliance state data.
Python 3.13 compatible implementation with proper type hints and modern syntax.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings


class PowerDataPreprocessor:
    """
    Preprocessor for power consumption data from industrial electrical equipment.
    
    This class handles data cleaning, normalization, and feature extraction
    for carbon footprint analysis.
    """
    
    def __init__(self, 
                 normalization_method: str = 'standard',
                 sequence_length: int = 100,
                 sampling_rate: float = 1.0):
        """
        Initialize the power data preprocessor.
        
        Args:
            normalization_method: 'standard' or 'minmax' normalization
            sequence_length: Length of sequences for time series modeling
            sampling_rate: Sampling rate for data (Hz)
        """
        self.normalization_method = normalization_method
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        
        # Initialize scalers - Python 3.13 compatible initialization
        if normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {normalization_method}")
        
        self.is_fitted = False
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw power consumption data.
        
        Args:
            data: Raw power consumption DataFrame
            
        Returns:
            Cleaned DataFrame with outliers removed and missing values handled
        """
        cleaned_data = data.copy()
        
        # Remove duplicate timestamps
        if 'timestamp' in cleaned_data.columns:
            cleaned_data = cleaned_data.drop_duplicates(subset=['timestamp'])
            cleaned_data = cleaned_data.sort_values('timestamp')
        
        # Handle missing values using forward fill then backward fill
        cleaned_data = cleaned_data.ffill().bfill()
        
        # Remove outliers using IQR method for power columns
        power_columns = [col for col in cleaned_data.columns if 'power' in col.lower()]
        
        for col in power_columns:
            if cleaned_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them to preserve temporal structure
                cleaned_data[col] = cleaned_data[col].clip(lower_bound, upper_bound)
        
        return cleaned_data
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract relevant features from power consumption data.
        
        Args:
            data: Cleaned power consumption DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        features_data = data.copy()
        
        # Extract time-based features if timestamp exists
        if 'timestamp' in features_data.columns:
            features_data['timestamp'] = pd.to_datetime(features_data['timestamp'])
            features_data['hour'] = features_data['timestamp'].dt.hour
            features_data['day_of_week'] = features_data['timestamp'].dt.dayofweek
            features_data['month'] = features_data['timestamp'].dt.month
            
            # Create cyclical features for better model performance
            features_data['hour_sin'] = np.sin(2 * np.pi * features_data['hour'] / 24)
            features_data['hour_cos'] = np.cos(2 * np.pi * features_data['hour'] / 24)
            features_data['day_sin'] = np.sin(2 * np.pi * features_data['day_of_week'] / 7)
            features_data['day_cos'] = np.cos(2 * np.pi * features_data['day_of_week'] / 7)
        
        # Calculate power-related features
        power_columns = [col for col in features_data.columns if 'power' in col.lower()]
        
        if power_columns:
            # Moving averages
            for col in power_columns:
                features_data[f'{col}_ma_5'] = features_data[col].rolling(window=5, min_periods=1).mean()
                features_data[f'{col}_ma_15'] = features_data[col].rolling(window=15, min_periods=1).mean()
                
                # Power derivatives (rate of change)
                features_data[f'{col}_diff'] = features_data[col].diff().fillna(0)
                
                # Power variance in sliding window
                features_data[f'{col}_std_5'] = features_data[col].rolling(window=5, min_periods=1).std().fillna(0)
        
        return features_data
    
    def normalize_data(self, data: pd.DataFrame, 
                      fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Normalize numerical features in the dataset.
        
        Args:
            data: DataFrame to normalize
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Tuple of (normalized DataFrame, list of normalized column names)
        """
        normalized_data = data.copy()
        
        # Identify numerical columns for normalization
        numerical_columns = normalized_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        # Exclude timestamp-related columns from normalization
        exclude_columns = ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        numerical_columns = [col for col in numerical_columns if col not in exclude_columns]
        
        if numerical_columns:
            if fit:
                normalized_data[numerical_columns] = self.scaler.fit_transform(
                    normalized_data[numerical_columns]
                )
                self.is_fitted = True
            else:
                if not self.is_fitted:
                    raise ValueError("Scaler not fitted. Call with fit=True first.")
                normalized_data[numerical_columns] = self.scaler.transform(
                    normalized_data[numerical_columns]
                )
        
        return normalized_data, numerical_columns
    
    def create_sequences(self, data: pd.DataFrame, 
                        target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for time series modeling.
        
        Args:
            data: Preprocessed DataFrame
            target_column: Name of target column for supervised learning
            
        Returns:
            Tuple of (feature sequences, target sequences if target_column provided)
        """
        # Ensure all data is numeric and drop non-numeric columns
        numeric_data = data.select_dtypes(include=[np.number]).copy()
        
        # Convert to numpy array for sequence creation
        if target_column and target_column in numeric_data.columns:
            features = numeric_data.drop(columns=[target_column]).values.astype(np.float32)
            targets = numeric_data[target_column].values.astype(np.int64)
            
            # Ensure targets are valid (non-negative and within bounds)
            targets = np.clip(targets, 0, None)  # Remove negative values
        else:
            features = numeric_data.values.astype(np.float32)
            targets = None
        
        # Create sequences
        X_sequences = []
        y_sequences = [] if targets is not None else None
        
        for i in range(len(features) - self.sequence_length + 1):
            X_sequences.append(features[i:i + self.sequence_length])
            if targets is not None:
                y_sequences.append(targets[i + self.sequence_length - 1])
        
        X_sequences = np.array(X_sequences, dtype=np.float32)
        y_sequences = np.array(y_sequences, dtype=np.int64) if y_sequences else None
        
        return X_sequences, y_sequences
    
    def split_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                  test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature sequences
            y: Target sequences (optional)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of split data (X_train, X_test, y_train, y_test) or (X_train, X_test) if no targets
        """
        if y is not None:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            return train_test_split(X, test_size=test_size, random_state=random_state)
    
    def preprocess_pipeline(self, data: pd.DataFrame, 
                           target_column: Optional[str] = None,
                           fit: bool = True) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Raw power consumption DataFrame
            target_column: Name of target column for supervised learning
            fit: Whether to fit scalers (True for training data)
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        # Step 1: Clean data
        cleaned_data = self.clean_data(data)
        
        # Step 2: Extract features
        features_data = self.extract_features(cleaned_data)
        
        # Step 3: Normalize data
        normalized_data, normalized_columns = self.normalize_data(features_data, fit=fit)
        
        # Step 4: Create sequences
        X_sequences, y_sequences = self.create_sequences(normalized_data, target_column)
        
        # Step 5: Split data
        if y_sequences is not None:
            X_train, X_test, y_train, y_test = self.split_data(X_sequences, y_sequences)
            split_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        else:
            X_train, X_test = self.split_data(X_sequences)
            split_data = {
                'X_train': X_train,
                'X_test': X_test
            }
        
        return {
            'processed_data': normalized_data,
            'sequences': X_sequences,
            'targets': y_sequences,
            'split_data': split_data,
            'normalized_columns': normalized_columns,
            'metadata': {
                'sequence_length': self.sequence_length,
                'normalization_method': self.normalization_method,
                'feature_count': X_sequences.shape[-1] if len(X_sequences) > 0 else 0
            }
        }


def load_sample_data() -> pd.DataFrame:
    """
    Generate sample power consumption data for testing.
    
    Returns:
        Sample DataFrame with power consumption data
    """
    np.random.seed(42)
    
    # Generate timestamps
    timestamps = pd.date_range(start='2023-01-01', end='2023-01-31', freq='h')
    
    # Generate synthetic power consumption data
    n_points = len(timestamps)
    
    # Base consumption with daily and weekly patterns
    time_hours = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values
    
    base_consumption = 100 + 50 * np.sin(2 * np.pi * time_hours / 24)  # Daily pattern
    weekly_pattern = 20 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly pattern
    noise = np.random.normal(0, 10, n_points)  # Random noise
    
    power_consumption = base_consumption + weekly_pattern + noise
    
    # Add some equipment state indicators
    equipment_states = np.random.choice(['idle', 'low', 'medium', 'high'], n_points, 
                                       p=[0.3, 0.3, 0.3, 0.1])
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'power_consumption': power_consumption,
        'equipment_state': equipment_states,
        'temperature': 20 + 10 * np.random.normal(0, 1, n_points),
        'humidity': 50 + 20 * np.random.normal(0, 1, n_points)
    })


if __name__ == "__main__":
    # Example usage
    print("Testing PowerDataPreprocessor with Python 3.13 compatibility...")
    
    # Load sample data
    sample_data = load_sample_data()
    print(f"Generated sample data with shape: {sample_data.shape}")
    
    # Initialize preprocessor
    preprocessor = PowerDataPreprocessor(
        normalization_method='standard',
        sequence_length=24,  # 24 hours for daily patterns
        sampling_rate=1.0
    )
    
    # Run preprocessing pipeline
    results = preprocessor.preprocess_pipeline(sample_data, target_column='power_consumption')
    
    print(f"Preprocessing completed successfully!")
    print(f"Feature sequences shape: {results['sequences'].shape}")
    print(f"Target sequences shape: {results['targets'].shape if results['targets'] is not None else 'N/A'}")
    print(f"Number of normalized columns: {len(results['normalized_columns'])}")
    print(f"Training data shape: {results['split_data']['X_train'].shape}")
    print(f"Test data shape: {results['split_data']['X_test'].shape}")
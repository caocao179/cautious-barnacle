"""
Test suite for Industrial Electrical Equipment Carbon Footprint Analysis modules.
Python 3.13 compatible test cases.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from data_preprocessing import PowerDataPreprocessor, load_sample_data
from models import GRUWithAttention, CarbonFootprintPredictor, ApplianceDataset
from dc_opf import DCOptimalPowerFlow, Bus, Branch, Generator, BusType, create_sample_system


class TestDataPreprocessing:
    """Test cases for data preprocessing module."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization with different parameters."""
        # Test standard normalization
        preprocessor = PowerDataPreprocessor(normalization_method='standard')
        assert preprocessor.normalization_method == 'standard'
        assert preprocessor.sequence_length == 100
        
        # Test minmax normalization
        preprocessor = PowerDataPreprocessor(normalization_method='minmax', sequence_length=50)
        assert preprocessor.normalization_method == 'minmax'
        assert preprocessor.sequence_length == 50
        
        # Test invalid normalization method
        with pytest.raises(ValueError):
            PowerDataPreprocessor(normalization_method='invalid')
    
    def test_sample_data_generation(self):
        """Test sample data generation."""
        data = load_sample_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'timestamp' in data.columns
        assert 'power_consumption' in data.columns
        assert 'equipment_state' in data.columns
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        preprocessor = PowerDataPreprocessor()
        data = load_sample_data()
        
        # Add some missing values
        data.iloc[10:15, 1] = np.nan
        
        cleaned_data = preprocessor.clean_data(data)
        assert not cleaned_data.isna().any().any()
        assert len(cleaned_data) <= len(data)  # May remove rows
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        preprocessor = PowerDataPreprocessor()
        data = load_sample_data()
        
        features_data = preprocessor.extract_features(data)
        
        # Check that time-based features were added
        expected_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        for feature in expected_features:
            assert feature in features_data.columns
        
        # Check that moving averages were added
        power_cols = [col for col in data.columns if 'power' in col.lower()]
        for col in power_cols:
            assert f'{col}_ma_5' in features_data.columns
            assert f'{col}_ma_15' in features_data.columns
    
    def test_normalization(self):
        """Test data normalization."""
        preprocessor = PowerDataPreprocessor()
        data = load_sample_data()
        features_data = preprocessor.extract_features(data)
        
        normalized_data, normalized_columns = preprocessor.normalize_data(features_data, fit=True)
        
        # Check that numerical columns were normalized
        assert len(normalized_columns) > 0
        assert preprocessor.is_fitted
        
        # Test transform mode
        normalized_data2, _ = preprocessor.normalize_data(features_data, fit=False)
        np.testing.assert_array_equal(normalized_data[normalized_columns].values,
                                    normalized_data2[normalized_columns].values)
    
    def test_sequence_creation(self):
        """Test sequence creation for time series modeling."""
        preprocessor = PowerDataPreprocessor(sequence_length=10)
        data = load_sample_data()
        
        # Test without targets
        X_sequences, y_sequences = preprocessor.create_sequences(data)
        expected_n_sequences = len(data) - 10 + 1
        assert X_sequences.shape[0] == expected_n_sequences
        assert X_sequences.shape[1] == 10
        assert y_sequences is None
        
        # Test with targets
        X_sequences, y_sequences = preprocessor.create_sequences(data, target_column='power_consumption')
        assert X_sequences.shape[0] == expected_n_sequences
        assert y_sequences.shape[0] == expected_n_sequences
    
    def test_complete_pipeline(self):
        """Test the complete preprocessing pipeline."""
        preprocessor = PowerDataPreprocessor(sequence_length=24)
        data = load_sample_data()
        
        results = preprocessor.preprocess_pipeline(data, target_column='power_consumption')
        
        # Check all required keys are present
        required_keys = ['processed_data', 'sequences', 'targets', 'split_data', 'normalized_columns', 'metadata']
        for key in required_keys:
            assert key in results
        
        # Check split data structure
        split_data = results['split_data']
        assert 'X_train' in split_data
        assert 'X_test' in split_data
        assert 'y_train' in split_data
        assert 'y_test' in split_data
        
        # Check metadata
        metadata = results['metadata']
        assert 'sequence_length' in metadata
        assert 'normalization_method' in metadata
        assert 'feature_count' in metadata


class TestModels:
    """Test cases for neural network models."""
    
    def test_gru_with_attention_initialization(self):
        """Test GRU with attention model initialization."""
        model = GRUWithAttention(
            input_size=10,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            num_classes=4
        )
        
        assert model.input_size == 10
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.num_classes == 4
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = GRUWithAttention(input_size=10, hidden_size=32, num_classes=4)
        
        # Create sample input
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, 10)
        
        outputs = model(x)
        
        # Check output structure
        assert 'logits' in outputs
        assert 'embeddings' in outputs
        assert 'attention_output' in outputs
        assert 'pooled_features' in outputs
        
        # Check output shapes
        assert outputs['logits'].shape == (batch_size, 4)
        assert outputs['embeddings'].shape[0] == batch_size
        assert outputs['attention_output'].shape[0] == batch_size
        assert outputs['pooled_features'].shape == (batch_size, 64)  # bidirectional
    
    def test_appliance_dataset(self):
        """Test appliance dataset class."""
        sequences = np.random.randn(100, 20, 10).astype(np.float32)
        targets = np.random.randint(0, 4, 100)
        
        dataset = ApplianceDataset(sequences, targets)
        
        assert len(dataset) == 100
        
        # Test data loading
        seq, target = dataset[0]
        assert isinstance(seq, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert seq.shape == (20, 10)
        
        # Test dataset without targets
        dataset_no_targets = ApplianceDataset(sequences)
        seq, target = dataset_no_targets[0]
        assert target is None
    
    def test_carbon_footprint_predictor(self):
        """Test carbon footprint predictor."""
        predictor = CarbonFootprintPredictor(
            input_size=10,
            hidden_size=32,
            num_classes=4
        )
        
        # Create sample data
        n_samples = 50
        sequences = np.random.randn(n_samples, 20, 10).astype(np.float32)
        targets = np.random.randint(0, 4, n_samples)
        
        # Split data
        train_size = int(0.8 * n_samples)
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]
        val_sequences = sequences[train_size:]
        val_targets = targets[train_size:]
        
        # Test training (just a few epochs)
        history = predictor.fit(
            train_sequences=train_sequences,
            train_targets=train_targets,
            val_sequences=val_sequences,
            val_targets=val_targets,
            epochs=2,
            batch_size=8,
            verbose=False
        )
        
        assert 'train_loss' in history
        assert 'train_accuracy' in history
        assert len(history['train_loss']) == 2
        
        # Test prediction
        predictions = predictor.predict(val_sequences)
        assert predictions.shape == (len(val_sequences),)
        assert all(0 <= p < 4 for p in predictions)
        
        # Test embedding extraction
        embeddings = predictor.get_embeddings(val_sequences[:5])
        assert embeddings.shape == (5, 64)  # hidden_size * 2 for bidirectional


class TestDCOptimalPowerFlow:
    """Test cases for DC Optimal Power Flow module."""
    
    def test_bus_creation(self):
        """Test bus creation and management."""
        dc_opf = DCOptimalPowerFlow()
        
        bus = Bus(
            id=1, name="Test Bus", bus_type=BusType.SLACK,
            voltage_magnitude=1.0, real_power_demand=100.0
        )
        
        dc_opf.add_bus(bus)
        assert 1 in dc_opf.buses
        assert dc_opf.buses[1].name == "Test Bus"
        assert dc_opf.buses[1].bus_type == BusType.SLACK
    
    def test_branch_creation(self):
        """Test branch creation and management."""
        dc_opf = DCOptimalPowerFlow()
        
        branch = Branch(
            id=1, from_bus=1, to_bus=2,
            resistance=0.01, reactance=0.1,
            power_rating=100.0
        )
        
        dc_opf.add_branch(branch)
        assert 1 in dc_opf.branches
        assert dc_opf.branches[1].from_bus == 1
        assert dc_opf.branches[1].to_bus == 2
    
    def test_generator_creation(self):
        """Test generator creation and management."""
        dc_opf = DCOptimalPowerFlow()
        
        generator = Generator(
            id=1, bus_id=1,
            real_power_min=0.0, real_power_max=100.0,
            reactive_power_min=-50.0, reactive_power_max=50.0,
            carbon_emission_rate=0.5
        )
        
        dc_opf.add_generator(generator)
        assert 1 in dc_opf.generators
        assert dc_opf.generators[1].bus_id == 1
        assert dc_opf.generators[1].carbon_emission_rate == 0.5
    
    def test_sample_system_creation(self):
        """Test sample system creation."""
        dc_opf = create_sample_system()
        
        assert len(dc_opf.buses) == 3
        assert len(dc_opf.branches) == 3
        assert len(dc_opf.generators) == 3
        
        # Check slack bus exists
        slack_buses = [bus for bus in dc_opf.buses.values() if bus.bus_type == BusType.SLACK]
        assert len(slack_buses) == 1
    
    def test_admittance_matrix_construction(self):
        """Test admittance matrix construction."""
        dc_opf = create_sample_system()
        
        Y = dc_opf.build_admittance_matrix()
        
        assert Y.shape == (3, 3)
        assert not np.iscomplex(Y).any()  # Should be real for DC analysis
        
        # Check symmetry
        np.testing.assert_array_almost_equal(Y, Y.T)
    
    def test_dc_opf_optimization(self):
        """Test DC-OPF optimization."""
        dc_opf = create_sample_system()
        
        # Test cost minimization
        results_cost = dc_opf.solve_dc_opf(objective='cost')
        assert results_cost['status'] == 'optimal'
        assert 'total_cost' in results_cost
        assert 'total_emissions' in results_cost
        assert 'generator_dispatch' in results_cost
        
        # Test emission minimization
        results_emissions = dc_opf.solve_dc_opf(objective='emissions')
        assert results_emissions['status'] == 'optimal'
        
        # Emissions should be lower in emission optimization
        assert results_emissions['total_emissions'] <= results_cost['total_emissions']
        
        # Test combined optimization
        results_combined = dc_opf.solve_dc_opf(objective='combined', carbon_price=50.0)
        assert results_combined['status'] == 'optimal'
    
    def test_carbon_footprint_calculation(self):
        """Test carbon footprint calculation."""
        dc_opf = create_sample_system()
        dc_opf.solve_dc_opf(objective='cost')
        
        carbon_metrics = dc_opf.calculate_carbon_footprint()
        
        assert 'total_emissions_kg_co2' in carbon_metrics
        assert 'emissions_per_generator' in carbon_metrics
        assert 'emission_rate_weighted_avg' in carbon_metrics
        assert 'clean_energy_percentage' in carbon_metrics
        
        assert carbon_metrics['total_emissions_kg_co2'] >= 0
        assert 0 <= carbon_metrics['clean_energy_percentage'] <= 100
    
    def test_results_export(self):
        """Test results export to DataFrames."""
        dc_opf = create_sample_system()
        dc_opf.solve_dc_opf(objective='cost')
        
        dataframes = dc_opf.export_results_to_dataframe()
        
        assert 'generators' in dataframes
        assert 'buses' in dataframes
        assert 'branches' in dataframes
        
        # Check generator DataFrame
        gen_df = dataframes['generators']
        assert len(gen_df) == 3
        required_cols = ['generator_id', 'power_output_mw', 'cost_usd', 'emissions_kg_co2']
        for col in required_cols:
            assert col in gen_df.columns
        
        # Check bus DataFrame
        bus_df = dataframes['buses']
        assert len(bus_df) == 3
        
        # Check branch DataFrame
        branch_df = dataframes['branches']
        assert len(branch_df) == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
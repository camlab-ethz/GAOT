"""
Unit tests for DataProcessor module.
Tests data loading, preprocessing, normalization, and data loader creation.
"""
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os
# Add project root to path to enable proper package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from new_src.datasets.data_processor import DataProcessor
from test.datasets.test_utils import (
    MockDatasetFactory, get_test_config_fx, get_test_config_vx,
    assert_tensor_shape, assert_dataloader_properties, validate_data_loader_output
)


class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = MockDatasetFactory()
        self.factory.setup()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.factory.cleanup()
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        dataset_config, _, metadata = get_test_config_fx()
        processor = DataProcessor(
            dataset_config=dataset_config,
            metadata=metadata,
            dtype=torch.float32
        )
        
        assert processor.dataset_config == dataset_config
        assert processor.metadata == metadata
        assert processor.dtype == torch.float32
        assert processor.u_mean is None
        assert processor.u_std is None
        assert processor.c_mean is None
        assert processor.c_std is None
        assert processor.coord_scaler is None

    def test_load_and_process_data_fx_2d(self):
        """Test loading and processing fixed coordinates 2D data."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(
            coord_dim=2, n_samples=10, n_nodes=64
        )
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Load and process data
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Check coordinate mode detection
        assert not is_variable_coords, "Should detect fixed coordinates"
        
        # Check data splits structure
        assert 'train' in data_splits
        assert 'val' in data_splits
        assert 'test' in data_splits
        
        # Check train split
        train_data = data_splits['train']
        assert 'c' in train_data and 'u' in train_data and 'x' in train_data
        
        # Check shapes
        c_train = train_data['c']
        u_train = train_data['u']
        x_train = train_data['x']
        
        assert c_train is not None, "Condition data should exist"
        assert_tensor_shape(c_train, (6, 64, 1), "c_train")  # 6 train samples
        assert_tensor_shape(u_train, (6, 64, 1), "u_train")
        assert_tensor_shape(x_train, (64, 2), "x_train")  # Fixed coords: [nodes, coord_dim]
        
        # Check normalization statistics were computed
        assert processor.u_mean is not None
        assert processor.u_std is not None
        assert processor.c_mean is not None
        assert processor.c_std is not None
    
    def test_load_and_process_data_vx_2d(self):
        """Test loading and processing variable coordinates 2D data."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_vx_dataset(
            coord_dim=2, n_samples=10, n_nodes=64
        )
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Load and process data
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Check coordinate mode detection
        assert is_variable_coords, "Should detect variable coordinates"
        
        # Check train split shapes
        train_data = data_splits['train']
        c_train = train_data['c']
        u_train = train_data['u'] 
        x_train = train_data['x']
        
        assert_tensor_shape(c_train, (6, 64, 1), "c_train")
        assert_tensor_shape(u_train, (6, 64, 1), "u_train")
        assert_tensor_shape(x_train, (6, 64, 2), "x_train")  # Variable coords: [samples, time, nodes, coord_dim]
    
    def test_load_and_process_data_3d(self):
        """Test loading and processing 3D coordinates data."""
        _, dataset_config, metadata = self.factory.create_fx_dataset(
            coord_dim=3, n_samples=10, n_nodes=64
        )
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Load and process data
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Check coordinate dimension
        train_data = data_splits['train']
        x_train = train_data['x']
        
        assert x_train.shape[-1] == 3, f"Should have 3D coordinates, got {x_train.shape[-1]}D"
    
    def test_load_and_process_data_no_condition(self):
        """Test loading data without condition variables."""
        _, dataset_config, metadata = self.factory.create_no_condition_dataset()
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Load and process data
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Check that condition data is None
        train_data = data_splits['train']
        assert train_data['c'] is None, "Condition data should be None"
        
        # Check that normalization stats for conditions are None
        assert processor.c_mean is None
        assert processor.c_std is None
    
    def test_generate_latent_queries_2d(self):
        """Test generation of 2D latent queries."""
        dataset_config, _, metadata = get_test_config_fx(coord_dim=2)
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Generate latent queries
        token_size = (8, 8)
        latent_queries = processor.generate_latent_queries(token_size)
        
        # Check shape and properties
        assert_tensor_shape(latent_queries, (64, 2), "latent_queries_2d")  # 8*8=64 queries
        
        # Check that coordinates are in expected range (after scaling)
        assert torch.all(latent_queries >= -1.1), "Scaled coordinates should be >= -1"
        assert torch.all(latent_queries <= 1.1), "Scaled coordinates should be <= 1"
        
        # Check that coordinate scaler was created
        assert processor.coord_scaler is not None
    
    def test_generate_latent_queries_3d(self):
        """Test generation of 3D latent queries."""
        dataset_config, _, metadata = get_test_config_fx(coord_dim=3)
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Generate latent queries
        token_size = (4, 4, 4)
        latent_queries = processor.generate_latent_queries(token_size)
        
        # Check shape
        assert_tensor_shape(latent_queries, (64, 3), "latent_queries_3d")  # 4*4*4=64 queries
        assert processor.coord_scaler is not None
    
    def test_create_data_loaders_fx(self):
        """Test creation of data loaders for fixed coordinates."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(n_samples=10)
        
        processor = DataProcessor(dataset_config, metadata)
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Create data loaders
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords
        )
        
        # Check loader properties
        assert_dataloader_properties(loaders['train'], expected_batch_size=2, expected_length=3, dataset_type="train")
        assert_dataloader_properties(loaders['val'], expected_batch_size=2, expected_length=1, dataset_type="val")
        assert_dataloader_properties(loaders['test'], expected_batch_size=2, expected_length=1, dataset_type="test")
        
        # Check data loader output format
        validate_data_loader_output(loaders['train'], coord_mode='fx', has_condition=True)
    
    def test_create_data_loaders_vx(self):
        """Test creation of data loaders for variable coordinates."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_vx_dataset(n_samples=10)
        
        processor = DataProcessor(dataset_config, metadata)
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Mock graph data for VX mode with CSR format
        def create_mock_csr(n_neighbors, n_nodes):
            return {
                'neighbors_index': torch.zeros(n_neighbors, dtype=torch.long),
                'neighbors_row_splits': torch.arange(n_nodes + 1, dtype=torch.long)
            }
        
        mock_encoder_graphs = {
            'train': [[create_mock_csr(10, 64)] for _ in range(6)],  # 6 train samples
            'val': [[create_mock_csr(5, 64)] for _ in range(2)],     # 2 val samples  
            'test': [[create_mock_csr(8, 64)] for _ in range(2)]     # 2 test samples
        }
        mock_decoder_graphs = {
            'train': [[create_mock_csr(15, 64)] for _ in range(6)],
            'val': [[create_mock_csr(12, 64)] for _ in range(2)],
            'test': [[create_mock_csr(10, 64)] for _ in range(2)]
        }
        
        # Create data loaders with mock graphs
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords,
            encoder_graphs=mock_encoder_graphs,
            decoder_graphs=mock_decoder_graphs
        )
        
        # Check loader properties
        assert_dataloader_properties(loaders['train'], expected_batch_size=2, expected_length=3, dataset_type="train")
        
        # Check data loader output format
        validate_data_loader_output(loaders['train'], coord_mode='vx', has_condition=True)
    
    def test_data_normalization_consistency(self):
        """Test that data normalization is consistent across splits."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(n_samples=10)
        
        processor = DataProcessor(dataset_config, metadata)
        data_splits, _ = processor.load_and_process_data()
        
        # Get original data statistics
        u_train_orig = data_splits['train']['u']
        u_val_orig = data_splits['val']['u']
        u_test_orig = data_splits['test']['u']
        
        # Check that all splits are normalized (mean ≈ 0, std ≈ 1 for training data)
        train_mean = torch.mean(u_train_orig)
        train_std = torch.std(u_train_orig)
        
        assert abs(train_mean.item()) < 0.1, f"Training data should be normalized, got mean {train_mean.item()}"
        assert abs(train_std.item() - 1.0) < 0.1, f"Training data should be normalized, got std {train_std.item()}"
        
        # Check that normalization stats are reasonable
        assert processor.u_mean is not None and torch.isfinite(processor.u_mean).all()
        assert processor.u_std is not None and torch.all(processor.u_std > 0)
        
        if processor.c_mean is not None:
            assert torch.isfinite(processor.c_mean).all()
        if processor.c_std is not None:
            assert torch.all(processor.c_std > 0)
    
    def test_split_indices_generation(self):
        """Test data split indices generation."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(n_samples=10)
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Test split generation
        train_indices, val_indices, test_indices = processor._get_split_indices(10)
        
        # Check sizes
        assert len(train_indices) == 6, f"Expected 6 train samples, got {len(train_indices)}"
        assert len(val_indices) == 2, f"Expected 2 val samples, got {len(val_indices)}"  
        assert len(test_indices) == 2, f"Expected 2 test samples, got {len(test_indices)}"
        
        # Check no overlap
        all_indices = np.concatenate([train_indices, val_indices, test_indices])
        assert len(set(all_indices)) == len(all_indices), "Indices should not overlap"
        assert set(all_indices) == set(range(10)), "All indices should be covered"
    
    def test_error_handling_missing_file(self):
        """Test error handling for missing dataset file."""
        dataset_config, _, metadata = get_test_config_fx()
        dataset_config.base_path = "/nonexistent/path"
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            processor.load_and_process_data()
    
    def test_error_handling_insufficient_samples(self):
        """Test error handling when requesting more samples than available."""
        # Create small dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(n_samples=5)
        
        # Request more samples than available
        dataset_config.train_size = 10  # More than the 5 available
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Should raise assertion error
        with pytest.raises(AssertionError, match="Sum of train, val, and test sizes exceeds total samples"):
            processor.load_and_process_data()
    

if __name__ == "__main__":
    pytest.main([__file__])
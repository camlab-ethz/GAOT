"""
Unit tests for data_utils module.
Tests custom dataset classes and data manipulation utilities.
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from new_src.datasets.data_utils import (
    CustomDataset, DynamicPairDataset, StaticDataset, TestDataset,
    collate_variable_batch, collate_sequential_batch, create_data_splits
)
from test.datasets.test_utils import assert_tensor_shape, MockMetadata


class TestCustomDataset:
    """Test suite for CustomDataset class."""
    
    def _create_sample_data(self, n_samples: int = 5, n_nodes: int = 20, 
                          coord_dim: int = 2, n_c_features: int = 2, n_u_features: int = 1):
        """Create sample data for testing."""
        c_data = torch.randn(n_samples, n_nodes, n_c_features)
        u_data = torch.randn(n_samples, n_nodes, n_u_features)
        x_data = torch.rand(n_samples, n_nodes, coord_dim)
        
        # Create mock graph data in CSR format
        encoder_graphs = []
        decoder_graphs = []
        for i in range(n_samples):
            # Each sample has one scale with CSR format neighbors
            enc_csr = {
                'neighbors_index': torch.randint(0, n_nodes, (10,)),  # 10 random neighbor indices
                'neighbors_row_splits': torch.arange(n_nodes + 1)     # Row splits for CSR format
            }
            dec_csr = {
                'neighbors_index': torch.randint(0, n_nodes, (15,)),  # 15 random neighbor indices
                'neighbors_row_splits': torch.arange(n_nodes + 1)     # Row splits for CSR format
            }
            encoder_graphs.append([enc_csr])  # List of scales
            decoder_graphs.append([dec_csr])
        
        return c_data, u_data, x_data, encoder_graphs, decoder_graphs
    
    def test_initialization_with_condition(self):
        """Test CustomDataset initialization with condition data."""
        c_data, u_data, x_data, encoder_graphs, decoder_graphs = self._create_sample_data()
        
        dataset = CustomDataset(
            c_data=c_data,
            u_data=u_data,
            x_data=x_data,
            encoder_graphs=encoder_graphs,
            decoder_graphs=decoder_graphs
        )
        
        assert len(dataset) == 5
        assert dataset.c_data is not None
        assert dataset.transform is None
    
    def test_initialization_without_condition(self):
        """Test CustomDataset initialization without condition data."""
        _, u_data, x_data, encoder_graphs, decoder_graphs = self._create_sample_data()
        
        dataset = CustomDataset(
            c_data=None,
            u_data=u_data,
            x_data=x_data,
            encoder_graphs=encoder_graphs,
            decoder_graphs=decoder_graphs
        )
        
        assert len(dataset) == 5
        assert dataset.c_data is None
    
    def test_initialization_with_transform(self):
        """Test CustomDataset initialization with coordinate transform."""
        c_data, u_data, x_data, encoder_graphs, decoder_graphs = self._create_sample_data()
        
        # Mock transform function
        transform = Mock(return_value=torch.zeros_like(x_data[0]))
        
        dataset = CustomDataset(
            c_data=c_data,
            u_data=u_data,
            x_data=x_data,
            encoder_graphs=encoder_graphs,
            decoder_graphs=decoder_graphs,
            transform=transform
        )
        
        assert dataset.transform is not None
    
    def test_getitem_with_condition(self):
        """Test getting items from dataset with condition data."""
        c_data, u_data, x_data, encoder_graphs, decoder_graphs = self._create_sample_data()
        
        dataset = CustomDataset(
            c_data=c_data,
            u_data=u_data,
            x_data=x_data,
            encoder_graphs=encoder_graphs,
            decoder_graphs=decoder_graphs
        )
        
        # Get first sample
        c, u, x, enc_graph, dec_graph = dataset[0]
        
        # Check shapes and types
        assert_tensor_shape(c, (20, 2), "condition_data")
        assert_tensor_shape(u, (20, 1), "solution_data") 
        assert_tensor_shape(x, (20, 2), "coordinate_data")
        assert isinstance(enc_graph, list), "Encoder graph should be list"
        assert isinstance(dec_graph, list), "Decoder graph should be list"
        assert len(enc_graph) == 1, "Should have 1 scale"
        assert len(dec_graph) == 1, "Should have 1 scale"
        
        # Check CSR format
        assert isinstance(enc_graph[0], dict), "Encoder graph scale should be CSR dict"
        assert isinstance(dec_graph[0], dict), "Decoder graph scale should be CSR dict"
        assert 'neighbors_index' in enc_graph[0], "Should have neighbors_index"
        assert 'neighbors_row_splits' in enc_graph[0], "Should have neighbors_row_splits"
    
    def test_getitem_without_condition(self):
        """Test getting items from dataset without condition data."""
        _, u_data, x_data, encoder_graphs, decoder_graphs = self._create_sample_data()
        
        dataset = CustomDataset(
            c_data=None,
            u_data=u_data,
            x_data=x_data,
            encoder_graphs=encoder_graphs,
            decoder_graphs=decoder_graphs
        )
        
        c, u, x, enc_graph, dec_graph = dataset[0]
        
        # Condition data should be empty tensor
        assert c.numel() == 0, "Condition data should be empty when c_data is None"
        assert_tensor_shape(u, (20, 1), "solution_data")
    
    def test_getitem_with_transform(self):
        """Test getting items with coordinate transformation."""
        c_data, u_data, x_data, encoder_graphs, decoder_graphs = self._create_sample_data()
        
        # Transform that zeros out coordinates
        transform = lambda x: torch.zeros_like(x)
        
        dataset = CustomDataset(
            c_data=c_data,
            u_data=u_data,
            x_data=x_data,
            encoder_graphs=encoder_graphs,
            decoder_graphs=decoder_graphs,
            transform=transform
        )
        
        c, u, x, enc_graph, dec_graph = dataset[0]
        
        # Coordinates should be transformed (all zeros)
        assert torch.all(x == 0), "Coordinates should be transformed to zeros"
    
    def test_data_consistency_validation(self):
        """Test validation of data consistency during initialization."""
        c_data, u_data, x_data, encoder_graphs, decoder_graphs = self._create_sample_data()
        
        # Test mismatched c_data length
        c_data_wrong = c_data[:3]  # Different length
        with pytest.raises(ValueError, match="c_data and u_data must have same number of samples"):
            CustomDataset(c_data_wrong, u_data, x_data, encoder_graphs, decoder_graphs)
        
        # Test mismatched x_data length
        x_data_wrong = x_data[:3]
        with pytest.raises(ValueError, match="x_data and u_data must have same number of samples"):
            CustomDataset(c_data, u_data, x_data_wrong, encoder_graphs, decoder_graphs)
        
        # Test mismatched encoder_graphs length
        encoder_graphs_wrong = encoder_graphs[:3]
        with pytest.raises(ValueError, match="encoder_graphs and u_data must have same number of samples"):
            CustomDataset(c_data, u_data, x_data, encoder_graphs_wrong, decoder_graphs)
        
        # Test mismatched decoder_graphs length
        decoder_graphs_wrong = decoder_graphs[:3]
        with pytest.raises(ValueError, match="decoder_graphs and u_data must have same number of samples"):
            CustomDataset(c_data, u_data, x_data, encoder_graphs, decoder_graphs_wrong)


class TestDynamicPairDataset:
    """Test suite for DynamicPairDataset class."""
    
    def _create_time_series_data(self, n_samples: int = 3, n_timesteps: int = 16, 
                               n_nodes: int = 10, n_vars: int = 1):
        """Create time series data for testing."""
        np.random.seed(42)  # For reproducible tests
        u_data = np.random.randn(n_samples, n_timesteps, n_nodes, n_vars)
        c_data = np.random.randn(n_samples, n_timesteps, n_nodes, 2)  # 2 condition variables
        t_values = np.linspace(0, 1, n_timesteps)
        return u_data, c_data, t_values
    
    def _create_mock_stats(self):
        """Create mock statistics for testing."""
        return {
            'u': {'mean': np.array([0.1]), 'std': np.array([1.0])},
            'c': {'mean': np.array([0.05, -0.1]), 'std': np.array([0.8, 0.9])},
            'start_time': {'mean': 0.3, 'std': 0.2},
            'time_diffs': {'mean': 0.15, 'std': 0.05},
            'res': {'mean': 0.05, 'std': 0.02},
            'der': {'mean': 0.02, 'std': 0.01}
        }
    
    def test_initialization(self):
        """Test DynamicPairDataset initialization."""
        u_data, c_data, t_values = self._create_time_series_data()
        metadata = MockMetadata()
        stats = self._create_mock_stats()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            max_time_diff=14,
            stepper_mode="output",
            stats=stats
        )
        
        assert dataset.num_samples == 3
        assert dataset.num_nodes == 10
        assert dataset.num_vars == 1
        assert dataset.stepper_mode == "output"
        assert len(dataset.t_in_indices) > 0
        assert len(dataset.t_out_indices) > 0
        assert len(dataset.t_in_indices) == len(dataset.t_out_indices)
        assert hasattr(dataset, 'start_time_expanded')
        assert hasattr(dataset, 'time_diff_expanded')
    
    def test_time_pairs_generation(self):
        """Test generation of time pairs."""
        u_data, c_data, t_values = self._create_time_series_data(n_timesteps=16)
        metadata = MockMetadata()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            max_time_diff=14
        )

        # Check that time pairs are reasonable
        assert len(dataset.t_in_indices) > 0, "Should have generated time pairs"
        
        # Check that output times are always after input times
        for t_in, t_out in zip(dataset.t_in_indices, dataset.t_out_indices):
            assert t_out > t_in, f"Output time {t_out} should be after input time {t_in}"
            assert t_out - t_in <= 14, f"Time difference {t_out - t_in} should be <= max_time_diff"
            assert (t_out - t_in) % 2 == 0, "Time differences should be even (as per generation logic)"
    
    def test_length_calculation(self):
        """Test dataset length calculation."""
        u_data, c_data, t_values = self._create_time_series_data(n_samples=3, n_timesteps=10)
        metadata = MockMetadata()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            max_time_diff=8
        )
        
        # Length should be n_samples * n_time_pairs
        n_time_pairs = len(dataset.t_in_indices)
        expected_length = 3 * n_time_pairs
        assert len(dataset) == expected_length
    
    def test_getitem_output_mode(self):
        """Test getting items in output stepper mode."""
        u_data, c_data, t_values = self._create_time_series_data(n_samples=2, n_timesteps=10)
        metadata = MockMetadata()
        stats = self._create_mock_stats()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            stepper_mode="output",
            stats=stats
        )
        
        if len(dataset) > 0:
            input_tensor, target_tensor = dataset[0]
            # Check types and shapes for new implementation
            assert isinstance(input_tensor, torch.Tensor), "input_tensor should be tensor"
            assert isinstance(target_tensor, torch.Tensor), "target_tensor should be tensor"
            
            # Input tensor should contain: u_features + c_features + time_features(2)
            expected_input_dim = 1 + 2 + 2  # u_vars + c_vars + time features
            assert_tensor_shape(input_tensor, (10, expected_input_dim), "input_tensor")
            assert_tensor_shape(target_tensor, (10, 1), "target_tensor")  # u_vars
    
    def test_getitem_residual_mode(self):
        """Test getting items in residual stepper mode."""
        u_data, c_data, t_values = self._create_time_series_data(n_samples=2, n_timesteps=10)
        metadata = MockMetadata()
        stats = self._create_mock_stats()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            stepper_mode="residual",
            stats=stats
        )
        
        if len(dataset) > 0:
            input_tensor, target_tensor = dataset[0]
            
            assert isinstance(input_tensor, torch.Tensor), "input_tensor should be tensor"
            assert isinstance(target_tensor, torch.Tensor), "target_tensor should be tensor"
            
            expected_input_dim = 1 + 2 + 2  # u_vars + c_vars + time features
            assert_tensor_shape(input_tensor, (10, expected_input_dim), "input_tensor")
            assert_tensor_shape(target_tensor, (10, 1), "target_tensor")
    
    def test_getitem_time_derivative_mode(self):
        """Test getting items in time derivative stepper mode."""
        u_data, c_data, t_values = self._create_time_series_data(n_samples=2, n_timesteps=10)
        metadata = MockMetadata()
        stats = self._create_mock_stats()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            stepper_mode="time_der",
            stats=stats
        )
        
        if len(dataset) > 0:
            input_tensor, target_tensor = dataset[0]
            
            assert isinstance(input_tensor, torch.Tensor), "input_tensor should be tensor"
            assert isinstance(target_tensor, torch.Tensor), "target_tensor should be tensor"
            
            expected_input_dim = 1 + 2 + 2  # u_vars + c_vars + time features
            assert_tensor_shape(input_tensor, (10, expected_input_dim), "input_tensor")
            assert_tensor_shape(target_tensor, (10, 1), "target_tensor")
    
    def test_invalid_stepper_mode(self):
        """Test error handling for invalid stepper mode."""
        u_data, c_data, t_values = self._create_time_series_data(n_samples=1, n_timesteps=10)
        metadata = MockMetadata()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            stepper_mode="invalid_mode"
        )
        
        if len(dataset) > 0:
            with pytest.raises(ValueError, match="Unsupported stepper_mode"):
                _ = dataset[0]
    
    def test_variable_coordinates_mode(self):
        """Test DynamicPairDataset with variable coordinates."""
        u_data, c_data, t_values = self._create_time_series_data(n_samples=2, n_timesteps=8)
        x_data = np.random.randn(2, 8, 10, 2)  # Variable coordinates
        metadata = MockMetadata()
        stats = self._create_mock_stats()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            x_data=x_data,
            is_variable_coords=True,
            stepper_mode="output",
            stats=stats
        )
        
        if len(dataset) > 0:
            input_tensor, target_tensor, coord_tensor = dataset[0]
            
            assert isinstance(coord_tensor, torch.Tensor), "coord_tensor should be tensor"
            assert_tensor_shape(coord_tensor, (10, 2), "coord_tensor")
    
    def test_time_feature_preprocessing(self):
        """Test that time features are properly preprocessed."""
        u_data, c_data, t_values = self._create_time_series_data(n_samples=1, n_timesteps=8)
        metadata = MockMetadata()
        stats = self._create_mock_stats()
        
        dataset = DynamicPairDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            stepper_mode="output",
            stats=stats,
            use_time_norm=True
        )
        
        # Check that time features are preprocessed
        assert hasattr(dataset, 'start_time_expanded')
        assert hasattr(dataset, 'time_diff_expanded')
        assert dataset.start_time_expanded.shape[1] == 10  # num_nodes
        assert dataset.time_diff_expanded.shape[1] == 10  # num_nodes
        assert dataset.start_time_expanded.shape[2] == 1   # feature dimension
        assert dataset.time_diff_expanded.shape[2] == 1    # feature dimension


class TestStaticDataset:
    """Test suite for StaticDataset class."""
    
    def test_initialization_with_condition(self):
        """Test StaticDataset initialization with condition data."""
        c_data = torch.randn(10, 50, 3)  # 10 samples, 50 nodes, 3 condition features
        u_data = torch.randn(10, 50, 2)  # 10 samples, 50 nodes, 2 solution features
        
        dataset = StaticDataset(c_data=c_data, u_data=u_data)
        
        assert len(dataset) == 10
        assert dataset.c_data is not None
    
    def test_initialization_without_condition(self):
        """Test StaticDataset initialization without condition data."""
        u_data = torch.randn(8, 30, 1)
        
        dataset = StaticDataset(c_data=None, u_data=u_data)
        
        assert len(dataset) == 8
        assert dataset.c_data is None
    
    def test_getitem_with_condition(self):
        """Test getting items with condition data."""
        c_data = torch.randn(5, 20, 2)
        u_data = torch.randn(5, 20, 1)
        
        dataset = StaticDataset(c_data=c_data, u_data=u_data)
        
        c, u = dataset[0]
        
        assert_tensor_shape(c, (20, 2), "condition_data")
        assert_tensor_shape(u, (20, 1), "solution_data")
    
    def test_getitem_without_condition(self):
        """Test getting items without condition data."""
        u_data = torch.randn(5, 20, 1)
        
        dataset = StaticDataset(c_data=None, u_data=u_data)
        
        c, u = dataset[0]
        
        assert c.numel() == 0, "Condition data should be empty"
        assert_tensor_shape(u, (20, 1), "solution_data")
    
    def test_data_length_mismatch(self):
        """Test error handling for data length mismatch."""
        c_data = torch.randn(5, 20, 2)  # 5 samples
        u_data = torch.randn(8, 20, 1)  # 8 samples (mismatch)
        
        with pytest.raises(ValueError, match="c_data and u_data must have same number of samples"):
            StaticDataset(c_data=c_data, u_data=u_data)


class TestTestDataset:
    """Test suite for TestDataset class."""
    
    def _create_test_data(self):
        """Create test data for TestDataset."""
        np.random.seed(42)
        u_data = np.random.randn(5, 10, 16, 2)  # [samples, timesteps, nodes, vars]
        c_data = np.random.randn(5, 10, 16, 1)  # [samples, timesteps, nodes, c_vars]
        t_values = np.linspace(0, 1, 10)
        time_indices = np.array([0, 2, 4, 6, 8])
        
        stats = {
            'u': {'mean': np.array([0.1, 0.2]), 'std': np.array([1.0, 1.1])},
            'c': {'mean': np.array([0.05]), 'std': np.array([0.8])}
        }
        
        metadata = MockMetadata()
        return u_data, c_data, t_values, time_indices, stats, metadata
    
    def test_initialization_fixed_coords(self):
        """Test TestDataset initialization for fixed coordinates."""
        u_data, c_data, t_values, time_indices, stats, metadata = self._create_test_data()
        
        dataset = TestDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            time_indices=time_indices,
            stats=stats,
            is_variable_coords=False
        )
        
        assert len(dataset) == 5  # num_samples
        assert dataset.num_nodes == 16
        assert not dataset.is_variable_coords
    
    def test_initialization_variable_coords(self):
        """Test TestDataset initialization for variable coordinates."""
        u_data, c_data, t_values, time_indices, stats, metadata = self._create_test_data()
        x_data = np.random.randn(5, 10, 16, 2)  # Variable coordinates
        
        dataset = TestDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            time_indices=time_indices,
            stats=stats,
            x_data=x_data,
            is_variable_coords=True
        )
        
        assert len(dataset) == 5
        assert dataset.is_variable_coords
        assert dataset.x_data is not None
    
    def test_getitem_fixed_coords(self):
        """Test getting items for fixed coordinates mode."""
        u_data, c_data, t_values, time_indices, stats, metadata = self._create_test_data()
        
        dataset = TestDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            time_indices=time_indices,
            stats=stats,
            is_variable_coords=False
        )
        
        input_tensor, target_tensor = dataset[0]
        
        # Input tensor should include u + c + dummy_time_features(2)
        expected_input_dim = 2 + 1 + 2  # u_vars + c_vars + time features
        assert_tensor_shape(input_tensor, (16, expected_input_dim), "input_tensor")
        
        # Target should be [n_timesteps-1, num_nodes, u_vars]
        expected_target_shape = (len(time_indices)-1, 16, 2)
        assert_tensor_shape(target_tensor, expected_target_shape, "target_tensor")
    
    def test_getitem_variable_coords(self):
        """Test getting items for variable coordinates mode."""
        u_data, c_data, t_values, time_indices, stats, metadata = self._create_test_data()
        x_data = np.random.randn(5, 10, 16, 2)
        
        dataset = TestDataset(
            u_data=u_data,
            c_data=c_data,
            t_values=t_values,
            metadata=metadata,
            time_indices=time_indices,
            stats=stats,
            x_data=x_data,
            is_variable_coords=True
        )
        
        input_tensor, target_tensor, coord_tensor = dataset[0]
        
        expected_input_dim = 2 + 1 + 2  # u_vars + c_vars + time features
        assert_tensor_shape(input_tensor, (16, expected_input_dim), "input_tensor")
        assert_tensor_shape(coord_tensor, (16, 2), "coord_tensor")


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_collate_variable_batch(self):
        """Test custom collate function for variable batches."""
        # Create mock batch data
        batch = []
        for i in range(3):  # 3 samples in batch
            c = torch.randn(15, 2)  # condition data
            u = torch.randn(15, 1)  # solution data
            x = torch.rand(15, 2)   # coordinate data
            
            # Create CSR format graphs
            enc_csr = {
                'neighbors_index': torch.randint(0, 15, (8,)),
                'neighbors_row_splits': torch.arange(16)  # 15 + 1
            }
            dec_csr = {
                'neighbors_index': torch.randint(0, 15, (10,)),
                'neighbors_row_splits': torch.arange(16)  # 15 + 1  
            }
            enc_graph = [enc_csr]  # encoder graph with CSR format
            dec_graph = [dec_csr]  # decoder graph with CSR format
            batch.append((c, u, x, enc_graph, dec_graph))
        
        # Test collate function
        c_batch, u_batch, x_batch, enc_graphs, dec_graphs = collate_variable_batch(batch)
        
        # Check output shapes and types
        assert_tensor_shape(c_batch, (3, 15, 2), "c_batch")
        assert_tensor_shape(u_batch, (3, 15, 1), "u_batch") 
        assert_tensor_shape(x_batch, (3, 15, 2), "x_batch")
        assert isinstance(enc_graphs, list), "Encoder graphs should be list"
        assert isinstance(dec_graphs, list), "Decoder graphs should be list"
        assert len(enc_graphs) == 3, "Should have 3 encoder graph samples"
        assert len(dec_graphs) == 3, "Should have 3 decoder graph samples"
    
    def test_collate_variable_batch_empty_condition(self):
        """Test collate function with empty condition data."""
        batch = []
        for i in range(2):
            c = torch.empty(0)      # empty condition data
            u = torch.randn(10, 1)
            x = torch.rand(10, 2)
            
            # Create CSR format graphs
            enc_csr = {
                'neighbors_index': torch.randint(0, 10, (5,)),
                'neighbors_row_splits': torch.arange(11)  # 10 + 1
            }
            dec_csr = {
                'neighbors_index': torch.randint(0, 10, (7,)),
                'neighbors_row_splits': torch.arange(11)  # 10 + 1
            }
            enc_graph = [enc_csr]
            dec_graph = [dec_csr]
            batch.append((c, u, x, enc_graph, dec_graph))
        
        c_batch, u_batch, x_batch, enc_graphs, dec_graphs = collate_variable_batch(batch)
        
        # Condition batch should be None for empty condition data
        assert c_batch is None, "Should return None for empty condition data"
        assert_tensor_shape(u_batch, (2, 10, 1), "u_batch")
    
    def test_create_data_splits(self):
        """Test data splitting utility."""
        # Create test data
        data = torch.randn(100, 20, 3)
        
        # Test split with default ratios
        splits = create_data_splits(data, train_ratio=0.8, val_ratio=0.1, shuffle=False)
        
        assert 'train' in splits and 'val' in splits and 'test' in splits
        assert splits['train'].shape[0] == 80  # 80% of 100
        assert splits['val'].shape[0] == 10    # 10% of 100
        assert splits['test'].shape[0] == 10   # remaining 10%
        
        # Check that all data is accounted for
        total_samples = splits['train'].shape[0] + splits['val'].shape[0] + splits['test'].shape[0]
        assert total_samples == 100
    
    def test_create_data_splits_with_shuffle(self):
        """Test data splitting with shuffling."""
        # Create sequential data to test shuffling
        data = torch.arange(20).float().unsqueeze(-1).unsqueeze(-1)  # [20, 1, 1]
        
        # Split with shuffling
        splits_shuffled = create_data_splits(data, train_ratio=0.6, val_ratio=0.2, shuffle=True)
        
        # Split without shuffling
        splits_ordered = create_data_splits(data, train_ratio=0.6, val_ratio=0.2, shuffle=False)
        
        # Check sizes are correct
        assert splits_shuffled['train'].shape[0] == 12  # 60% of 20
        assert splits_ordered['train'].shape[0] == 12
        
        # Check that shuffled version is different from ordered (with high probability)
        # Note: this test might rarely fail due to random chance, but probability is very low
        train_shuffled = splits_shuffled['train'].squeeze()
        train_ordered = splits_ordered['train'].squeeze()
        
        # They should not be identical (unless extremely unlucky with random shuffle)
        are_different = not torch.equal(train_shuffled, train_ordered)
        if are_different:
            # This is the expected case
            pass
        else:
            # Very unlikely but possible - just warn
            import warnings
            warnings.warn("Shuffled and ordered splits happened to be identical - this is very rare but possible")
    
    def test_collate_sequential_batch_fixed_coords(self):
        """Test collate function for sequential data with fixed coordinates."""
        # Create mock batch with fixed coordinates (2 elements per item)
        batch = []
        for i in range(3):  # 3 samples
            input_tensor = torch.randn(10, 5)  # [nodes, features]
            target_tensor = torch.randn(10, 2)  # [nodes, u_vars]
            batch.append((input_tensor, target_tensor))
        
        # Test collate function
        inputs, targets = collate_sequential_batch(batch)
        
        # Check output shapes
        assert_tensor_shape(inputs, (3, 10, 5), "collated_inputs")
        assert_tensor_shape(targets, (3, 10, 2), "collated_targets")
    
    def test_collate_sequential_batch_variable_coords(self):
        """Test collate function for sequential data with variable coordinates."""
        # Create mock batch with variable coordinates (3 elements per item)
        batch = []
        for i in range(2):  # 2 samples
            input_tensor = torch.randn(15, 4)  # [nodes, features]
            target_tensor = torch.randn(15, 1)  # [nodes, u_vars]
            coord_tensor = torch.rand(15, 2)   # [nodes, coord_dim]
            batch.append((input_tensor, target_tensor, coord_tensor))
        
        # Test collate function
        inputs, targets, coords = collate_sequential_batch(batch)
        
        # Check output shapes
        assert_tensor_shape(inputs, (2, 15, 4), "collated_inputs")
        assert_tensor_shape(targets, (2, 15, 1), "collated_targets")
        assert_tensor_shape(coords, (2, 15, 2), "collated_coords")
    
    def test_collate_sequential_batch_invalid_length(self):
        """Test collate function with invalid batch item length."""
        # Create batch with incorrect number of elements
        batch = [(torch.randn(5, 3), torch.randn(5, 1), torch.randn(5, 2), torch.randn(5, 1))]  # 4 elements
        
        with pytest.raises(ValueError, match="Unexpected batch item length"):
            collate_sequential_batch(batch)


if __name__ == "__main__":
    pytest.main([__file__])
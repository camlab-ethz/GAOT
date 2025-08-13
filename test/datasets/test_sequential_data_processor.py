"""
Unit tests for sequential data processor.
Tests time-dependent data loading, processing, and DataLoader creation.
"""
import pytest
import numpy as np
import torch
import os
import tempfile
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from new_src.datasets.sequential_data_processor import SequentialDataProcessor
from test.datasets.test_utils import assert_tensor_shape, MockMetadata


class TestSequentialDataProcessor:
    """Test suite for SequentialDataProcessor class."""
    
    def setup_method(self):
        """Set up test data and configurations."""
        # Mock dataset configuration
        self.dataset_config = Mock()
        self.dataset_config.base_path = "/mock/path"
        self.dataset_config.name = "test_sequential"
        self.dataset_config.train_ratio = 0.8
        self.dataset_config.val_ratio = 0.1
        self.dataset_config.test_ratio = 0.1
        self.dataset_config.shuffle_split = True
        self.dataset_config.batch_size = 4
        self.dataset_config.shuffle = True
        self.dataset_config.num_workers = 0
        self.dataset_config.train = True
        
        # Sequential-specific config
        self.dataset_config.max_time_diff = 14
        self.dataset_config.stepper_mode = "output"
        self.dataset_config.use_time_norm = True
        self.dataset_config.use_metadata_stats = False
        self.dataset_config.sample_rate = 1.0
        self.dataset_config.use_sparse = False
        
        # Mock metadata
        self.metadata = MockMetadata()
        self.metadata.group_u = "u"
        self.metadata.group_c = "c"
        self.metadata.group_x = "x"
        self.metadata.domain_t = [0.0, 1.0]
        self.metadata.domain_x = [[0.0, 0.0], [1.0, 1.0]]
        self.metadata.fix_x = True
        self.metadata.active_variables = [0, 1]
        
        # Test data dimensions
        self.num_samples = 20
        self.num_timesteps = 16
        self.num_nodes = 64
        self.u_vars = 2
        self.c_vars = 1
        self.coord_dim = 2
    
    def _create_mock_netcdf_data(self):
        """Create mock NetCDF data for testing."""
        # Solution data [samples, timesteps, nodes, variables]
        u_data = np.random.randn(self.num_samples, self.num_timesteps, self.num_nodes, self.u_vars)
        
        # Condition data [samples, timesteps, nodes, c_variables]
        c_data = np.random.randn(self.num_samples, self.num_timesteps, self.num_nodes, self.c_vars)
        
        # Coordinate data [nodes, coord_dim] for fixed coordinates
        x_data = np.random.uniform(0, 1, (self.num_nodes, self.coord_dim))
        
        return u_data, c_data, x_data
    
    def _create_temporary_netcdf_file(self, u_data, c_data, x_data):
        """Create a temporary NetCDF file with test data."""
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "test_sequential.nc")
        
        # Create dataset
        ds = xr.Dataset({
            'u': (['sample', 'time', 'node', 'u_var'], u_data),
            'c': (['sample', 'time', 'node', 'c_var'], c_data),
            'x': (['node', 'coord'], x_data)
        })
        
        ds.to_netcdf(temp_file)
        return temp_file, temp_dir
    
    def test_initialization(self):
        """Test SequentialDataProcessor initialization."""
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        assert processor.dataset_config == self.dataset_config
        assert processor.metadata == self.metadata
        assert processor.max_time_diff == 14
        assert processor.stepper_mode == "output"
        assert processor.use_time_norm == True
        assert processor.use_metadata_stats == False
        assert processor.sample_rate == 1.0
        assert processor.t_values is None
        assert processor.stats is None
    
    def test_initialization_with_defaults(self):
        """Test initialization with default configuration values."""
        # Config without sequential-specific attributes
        minimal_config = Mock()
        minimal_config.base_path = "/path"
        minimal_config.name = "test"
        
        processor = SequentialDataProcessor(minimal_config, self.metadata)
        
        # Should use defaults
        assert processor.max_time_diff == 14
        assert processor.stepper_mode == "output"
        assert processor.use_time_norm == True
        assert processor.use_metadata_stats == False
        assert processor.sample_rate == 1.0
    
    @patch('xarray.open_dataset')
    def test_load_raw_sequential_data_basic(self, mock_open_dataset):
        """Test basic loading of sequential data."""
        u_data, c_data, x_data = self._create_mock_netcdf_data()
        
        # Mock xarray dataset
        mock_ds = Mock()
        mock_ds.__enter__ = Mock(return_value=mock_ds)
        mock_ds.__exit__ = Mock(return_value=None)
        mock_ds.__getitem__.side_effect = lambda key: Mock(values={"u": u_data, "c": c_data, "x": x_data}[key])
        mock_open_dataset.return_value = mock_ds
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        with patch('os.path.exists', return_value=True):
            raw_data = processor._load_raw_sequential_data()
        
        assert 'u' in raw_data
        assert 'c' in raw_data
        assert 'x' in raw_data
        assert 't' in raw_data
        
        np.testing.assert_array_equal(raw_data['u'], u_data[..., self.metadata.active_variables])
        np.testing.assert_array_equal(raw_data['c'], c_data)
        assert len(raw_data['t']) == self.num_timesteps
        assert processor.t_values is not None
    
    @patch('xarray.open_dataset')
    def test_load_raw_sequential_data_no_condition(self, mock_open_dataset):
        """Test loading sequential data without condition data."""
        u_data, _, x_data = self._create_mock_netcdf_data()
        
        # Metadata without condition group
        metadata_no_c = MockMetadata()
        metadata_no_c.group_u = "u"
        metadata_no_c.group_c = None
        metadata_no_c.group_x = "x"
        metadata_no_c.domain_t = [0.0, 1.0]
        metadata_no_c.fix_x = True
        metadata_no_c.active_variables = [0, 1]
        
        mock_ds = Mock()
        mock_ds.__enter__ = Mock(return_value=mock_ds)
        mock_ds.__exit__ = Mock(return_value=None)
        mock_ds.__getitem__.side_effect = lambda key: Mock(values={"u": u_data, "x": x_data}[key])
        mock_open_dataset.return_value = mock_ds
        
        processor = SequentialDataProcessor(self.dataset_config, metadata_no_c)
        
        with patch('os.path.exists', return_value=True):
            raw_data = processor._load_raw_sequential_data()
        
        assert raw_data['c'] is None
        assert raw_data['u'] is not None
    
    def test_load_raw_sequential_data_file_not_found(self):
        """Test error handling when dataset file is not found."""
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Dataset file not found"):
                processor._load_raw_sequential_data()
    
    def test_load_sequential_coordinate_data_provided(self):
        """Test loading coordinate data when provided in dataset."""
        u_data, _, x_data = self._create_mock_netcdf_data()
        
        mock_ds = Mock()
        mock_ds.__getitem__.return_value = Mock(values=x_data)
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        result = processor._load_sequential_coordinate_data(mock_ds, u_data)
        
        # Should expand to [1, 1, num_nodes, coord_dim] for fixed coordinates
        assert result.shape == (1, 1, self.num_nodes, self.coord_dim)
    
    def test_load_sequential_coordinate_data_variable_coords(self):
        """Test loading variable coordinate data."""
        u_data, _, _ = self._create_mock_netcdf_data()
        x_data_variable = np.random.uniform(0, 1, (self.num_samples, self.num_timesteps, self.num_nodes, self.coord_dim))
        
        # Metadata for variable coordinates
        metadata_vx = MockMetadata()
        metadata_vx.group_x = "x"
        metadata_vx.fix_x = False
        
        mock_ds = Mock()
        mock_ds.__getitem__.return_value = Mock(values=x_data_variable)
        
        processor = SequentialDataProcessor(self.dataset_config, metadata_vx)
        result = processor._load_sequential_coordinate_data(mock_ds, u_data)
        
        # Should preserve original shape for variable coordinates
        assert result.shape == x_data_variable.shape
    
    def test_load_sequential_coordinate_data_generated(self):
        """Test generating coordinate data from domain when not provided."""
        u_data, _, _ = self._create_mock_netcdf_data()
        
        # Metadata without coordinate group
        metadata_no_x = MockMetadata()
        metadata_no_x.group_x = None
        metadata_no_x.domain_x = [[0.0, 0.0], [1.0, 1.0]]
        
        processor = SequentialDataProcessor(self.dataset_config, metadata_no_x)
        result = processor._load_sequential_coordinate_data(None, u_data)
        
        # Should generate coordinates with shape [1, 1, num_nodes, 2]
        assert result.shape == (1, 1, self.num_nodes, 2)
        
        # Check coordinate values are within domain
        coords = result[0, 0]  # [num_nodes, 2]
        assert np.all(coords >= 0.0)
        assert np.all(coords <= 1.0)
    
    def test_determine_coordinate_mode_fixed(self):
        """Test determination of fixed coordinate mode."""
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        # Fixed coordinates: x_data shape [1, 1, num_nodes, coord_dim]
        x_data_fixed = np.random.uniform(0, 1, (1, 1, self.num_nodes, self.coord_dim))
        raw_data = {'x': x_data_fixed}
        
        is_variable = processor._determine_coordinate_mode(raw_data)
        assert is_variable == False
    
    def test_determine_coordinate_mode_variable(self):
        """Test determination of variable coordinate mode."""
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        # Variable coordinates: x_data shape [num_samples, num_timesteps, num_nodes, coord_dim]
        x_data_variable = np.random.uniform(0, 1, (self.num_samples, self.num_timesteps, self.num_nodes, self.coord_dim))
        raw_data = {'x': x_data_variable}
        
        is_variable = processor._determine_coordinate_mode(raw_data)
        assert is_variable == True
    
    def test_split_and_normalize_sequential_data_fixed_coords(self):
        """Test splitting and normalizing sequential data with fixed coordinates."""
        u_data, c_data, x_data = self._create_mock_netcdf_data()
        x_data_fixed = x_data[None, None, ...]  # [1, 1, num_nodes, coord_dim]
        
        raw_data = {
            'u': u_data,
            'c': c_data,
            'x': x_data_fixed,
            't': np.linspace(0, 1, self.num_timesteps)
        }
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        # Mock statistics computation
        with patch.object(processor, '_compute_sequential_stats', return_value={'u': {'mean': np.array([0.0, 0.0]), 'std': np.array([1.0, 1.0])}}):
            data_splits = processor._split_and_normalize_sequential_data(raw_data, is_variable_coords=False)
        
        # Check split structure
        assert 'train' in data_splits and 'val' in data_splits and 'test' in data_splits
        
        # Check data shapes
        train_data = data_splits['train']
        assert train_data['u'].shape[0] == int(0.8 * self.num_samples)  # Train samples
        assert train_data['u'].shape[1] <= self.dataset_config.max_time_diff + 1  # Limited timesteps
        assert train_data['u'].shape[2] == self.num_nodes
        assert train_data['u'].shape[3] == self.u_vars
        
        # Fixed coordinates should be same for all splits
        assert np.array_equal(train_data['x'], data_splits['val']['x'])
        assert np.array_equal(train_data['x'], data_splits['test']['x'])
    
    def test_split_and_normalize_sequential_data_variable_coords(self):
        """Test splitting and normalizing sequential data with variable coordinates."""
        u_data, c_data, _ = self._create_mock_netcdf_data()
        x_data_variable = np.random.uniform(0, 1, (self.num_samples, self.num_timesteps, self.num_nodes, self.coord_dim))
        
        raw_data = {
            'u': u_data,
            'c': c_data,
            'x': x_data_variable,
            't': np.linspace(0, 1, self.num_timesteps)
        }
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        # Mock statistics computation
        with patch.object(processor, '_compute_sequential_stats', return_value={'u': {'mean': np.array([0.0, 0.0]), 'std': np.array([1.0, 1.0])}}):
            data_splits = processor._split_and_normalize_sequential_data(raw_data, is_variable_coords=True)
        
        # Check that coordinate data is split properly
        train_data = data_splits['train']
        assert train_data['x'].shape[0] == int(0.8 * self.num_samples)
        assert train_data['x'].shape[1] <= self.dataset_config.max_time_diff + 1
        assert train_data['x'].shape[2] == self.num_nodes
        assert train_data['x'].shape[3] == self.coord_dim
    
    def test_split_and_normalize_sequential_data_time_limiting(self):
        """Test that time steps are properly limited by max_time_diff."""
        u_data, c_data, x_data = self._create_mock_netcdf_data()
        x_data_fixed = x_data[None, None, ...]
        
        raw_data = {
            'u': u_data,
            'c': c_data,
            'x': x_data_fixed,
            't': np.linspace(0, 1, self.num_timesteps)
        }
        
        # Set smaller max_time_diff
        self.dataset_config.max_time_diff = 8
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        with patch.object(processor, '_compute_sequential_stats', return_value={'u': {'mean': np.array([0.0, 0.0]), 'std': np.array([1.0, 1.0])}}):
            data_splits = processor._split_and_normalize_sequential_data(raw_data, is_variable_coords=False)
        
        # Check that timesteps are limited
        train_data = data_splits['train']
        assert train_data['u'].shape[1] == 9  # max_time_diff + 1
        assert len(train_data['t']) == 9
        assert len(processor.t_values) == 9
    
    def test_compute_sequential_stats(self):
        """Test computation of sequential statistics."""
        u_data, c_data, _ = self._create_mock_netcdf_data()
        t_values = np.linspace(0, 1, self.num_timesteps)
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        # Mock the compute_sequential_stats function
        with patch('new_src.datasets.sequential_data_processor.compute_sequential_stats') as mock_compute:
            mock_stats = {
                'u': {'mean': np.array([0.1, 0.2]), 'std': np.array([1.0, 1.1])},
                'c': {'mean': np.array([0.05]), 'std': np.array([0.8])},
                'start_time': {'mean': 0.3, 'std': 0.2},
                'time_diffs': {'mean': 0.15, 'std': 0.05}
            }
            mock_compute.return_value = mock_stats
            
            result = processor._compute_sequential_stats(u_data, c_data, t_values)
            
            # Check that function was called with correct parameters
            mock_compute.assert_called_once_with(
                u_train=u_data,
                c_train=c_data,
                t_values=t_values,
                metadata=processor.metadata,
                max_time_diff=processor.max_time_diff,
                sample_rate=processor.sample_rate,
                use_metadata_stats=processor.use_metadata_stats,
                use_time_norm=processor.use_time_norm
            )
            
            assert result == mock_stats
    
    def test_create_sequential_data_loaders_fixed_coords(self):
        """Test creating data loaders for fixed coordinates mode."""
        # Create mock data splits
        u_train = np.random.randn(16, 10, 64, 2)
        c_train = np.random.randn(16, 10, 64, 1)
        x_coord = np.random.uniform(0, 1, (64, 2))
        t_values = np.linspace(0, 1, 10)
        
        data_splits = {
            'train': {'u': u_train, 'c': c_train, 'x': x_coord, 't': t_values},
            'val': {'u': u_train[:8], 'c': c_train[:8], 'x': x_coord, 't': t_values},
            'test': {'u': u_train[:4], 'c': c_train[:4], 'x': x_coord, 't': t_values}
        }
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        processor.stats = {'u': {'mean': np.array([0.0, 0.0]), 'std': np.array([1.0, 1.0])}}
        
        loaders = processor.create_sequential_data_loaders(data_splits, is_variable_coords=False)
        
        # Check loader structure
        assert 'train' in loaders and 'val' in loaders and 'test' in loaders
        assert isinstance(loaders['train'], DataLoader)
        assert isinstance(loaders['val'], DataLoader)
        assert isinstance(loaders['test'], DataLoader)
        
        # Check batch size is correct
        assert loaders['train'].batch_size == self.dataset_config.batch_size
    
    def test_create_sequential_data_loaders_variable_coords(self):
        """Test creating data loaders for variable coordinates mode."""
        # Create mock data splits with variable coordinates
        u_train = np.random.randn(16, 10, 64, 2)
        c_train = np.random.randn(16, 10, 64, 1)
        x_train = np.random.uniform(0, 1, (16, 10, 64, 2))
        t_values = np.linspace(0, 1, 10)
        
        data_splits = {
            'train': {'u': u_train, 'c': c_train, 'x': x_train, 't': t_values},
            'val': {'u': u_train[:8], 'c': c_train[:8], 'x': x_train[:8], 't': t_values},
            'test': {'u': u_train[:4], 'c': c_train[:4], 'x': x_train[:4], 't': t_values}
        }
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        processor.stats = {'u': {'mean': np.array([0.0, 0.0]), 'std': np.array([1.0, 1.0])}}
        
        loaders = processor.create_sequential_data_loaders(data_splits, is_variable_coords=True)
        
        # Check that loaders are created
        assert 'train' in loaders and 'val' in loaders and 'test' in loaders
        assert isinstance(loaders['train'], DataLoader)
        assert isinstance(loaders['val'], DataLoader)
        assert isinstance(loaders['test'], DataLoader)
    
    def test_create_sequential_data_loaders_no_training(self):
        """Test creating data loaders when training is disabled."""
        self.dataset_config.train = False
        
        data_splits = {
            'train': {'u': np.random.randn(4, 8, 32, 1), 'c': None, 'x': np.random.uniform(0, 1, (32, 2)), 't': np.linspace(0, 1, 8)},
            'val': {'u': np.random.randn(4, 8, 32, 1), 'c': None, 'x': np.random.uniform(0, 1, (32, 2)), 't': np.linspace(0, 1, 8)},
            'test': {'u': np.random.randn(4, 8, 32, 1), 'c': None, 'x': np.random.uniform(0, 1, (32, 2)), 't': np.linspace(0, 1, 8)}
        }
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        processor.stats = {'u': {'mean': np.array([0.0]), 'std': np.array([1.0])}}
        
        loaders = processor.create_sequential_data_loaders(data_splits, is_variable_coords=False)
        
        # Training and validation loaders should be None
        assert loaders['train'] is None
        assert loaders['val'] is None
        assert loaders['test'] is not None
        assert isinstance(loaders['test'], DataLoader)
    
    def test_load_and_process_data_integration(self):
        """Test complete load and process data workflow."""
        u_data, c_data, x_data = self._create_mock_netcdf_data()
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        # Mock the internal methods
        with patch.object(processor, '_load_raw_sequential_data') as mock_load:
            with patch.object(processor, '_determine_coordinate_mode') as mock_determine:
                with patch.object(processor, '_split_and_normalize_sequential_data') as mock_split:
                    
                    # Setup mocks
                    raw_data = {'u': u_data, 'c': c_data, 'x': x_data[None, None, ...], 't': np.linspace(0, 1, self.num_timesteps)}
                    mock_load.return_value = raw_data
                    mock_determine.return_value = False  # Fixed coordinates
                    mock_split.return_value = {'train': {}, 'val': {}, 'test': {}}
                    
                    # Call the method
                    data_splits, is_variable_coords = processor.load_and_process_data()
                    
                    # Verify calls
                    mock_load.assert_called_once()
                    mock_determine.assert_called_once_with(raw_data)
                    mock_split.assert_called_once_with(raw_data, False)
                    
                    # Check return values
                    assert isinstance(data_splits, dict)
                    assert is_variable_coords == False
    
    def test_poseidon_sparse_data_handling(self):
        """Test handling of Poseidon sparse datasets."""
        # Setup for Poseidon dataset
        self.dataset_config.name = "test_poseidon"
        self.dataset_config.use_sparse = True
        
        # Create larger dataset that will be truncated
        u_data_large = np.random.randn(self.num_samples, self.num_timesteps, 10000, self.u_vars)
        c_data_large = np.random.randn(self.num_samples, self.num_timesteps, 10000, self.c_vars)
        x_data_large = np.random.uniform(0, 1, (10000, self.coord_dim))
        
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        processor.poseidon_datasets = ["test_poseidon"]  # Add to poseidon datasets list
        
        mock_ds = Mock()
        mock_ds.__enter__ = Mock(return_value=mock_ds)
        mock_ds.__exit__ = Mock(return_value=None)
        mock_ds.__getitem__.side_effect = lambda key: Mock(values={
            "u": u_data_large, "c": c_data_large, "x": x_data_large
        }[key])
        
        with patch('xarray.open_dataset', return_value=mock_ds):
            with patch('os.path.exists', return_value=True):
                raw_data = processor._load_raw_sequential_data()
        
        # Check that data was truncated to 9216 nodes
        assert raw_data['u'].shape[2] == 9216
        assert raw_data['c'].shape[2] == 9216
        assert raw_data['x'].shape[2] == 9216


class TestSequentialDataProcessorErrors:
    """Test error handling in SequentialDataProcessor."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.dataset_config = Mock()
        self.dataset_config.base_path = "/mock/path"
        self.dataset_config.name = "test"
        
        self.metadata = MockMetadata()
        self.metadata.domain_t = None  # Will cause error
    
    def test_missing_domain_t_error(self):
        """Test error when domain_t is None."""
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        mock_ds = Mock()
        mock_ds.__enter__ = Mock(return_value=mock_ds)
        mock_ds.__exit__ = Mock(return_value=None)
        mock_ds.__getitem__.return_value = Mock(values=np.random.randn(10, 8, 64, 2))
        
        with patch('xarray.open_dataset', return_value=mock_ds):
            with patch('os.path.exists', return_value=True):
                with pytest.raises(ValueError, match="metadata.domain_t is None"):
                    processor._load_raw_sequential_data()
    
    def test_variable_coords_shape_mismatch(self):
        """Test error when variable coordinates have wrong shape."""
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        u_data = np.random.randn(10, 8, 64, 2)  # 10 samples
        x_data_wrong = np.random.randn(5, 8, 64, 2)  # Only 5 samples (mismatch)
        
        metadata_vx = MockMetadata()
        metadata_vx.group_x = "x"
        metadata_vx.fix_x = False
        processor.metadata = metadata_vx
        
        mock_ds = Mock()
        mock_ds.__getitem__.return_value = Mock(values=x_data_wrong)
        
        with pytest.raises(ValueError, match="Variable coordinates must have same number of samples"):
            processor._load_sequential_coordinate_data(mock_ds, u_data)
    
    def test_invalid_grid_size_error(self):
        """Test error when nodes don't form square grid."""
        processor = SequentialDataProcessor(self.dataset_config, self.metadata)
        
        # Use non-square number of nodes
        u_data_nonsquare = np.random.randn(10, 8, 63, 2)  # 63 is not a perfect square
        
        metadata_no_x = MockMetadata()
        metadata_no_x.group_x = None
        metadata_no_x.domain_x = [[0.0, 0.0], [1.0, 1.0]]
        processor.metadata = metadata_no_x
        
        with pytest.raises(ValueError, match="Cannot create square grid from 63 nodes"):
            processor._load_sequential_coordinate_data(None, u_data_nonsquare)


if __name__ == '__main__':
    pytest.main([__file__])
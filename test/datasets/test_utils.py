"""
Testing utilities for dataset modules.
"""
import os
import tempfile
import numpy as np
import torch
import xarray as xr
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from new_src.datasets.dataset import Metadata
from new_src.core.default_configs import DatasetConfig, ModelConfig, ModelArgsConfig
from new_src.model.layers.magno import MAGNOConfig
from new_src.model.layers.attn import TransformerConfig


@dataclass 
class MockMetadata:
    """Mock metadata for testing."""
    group_u: str = "u"
    group_c: str = "c" 
    group_x: Optional[str] = None
    fix_x: bool = True
    domain_x: List[List[float]] = None
    active_variables: List[int] = None
    names: Dict[str, List[str]] = None
    signed: Dict[str, List[bool]] = None
    
    def __post_init__(self):
        if self.active_variables is None:
            self.active_variables = [0]
        if self.names is None:
            self.names = {'u': ['u_field'], 'c': ['c_field']}
        if self.signed is None:
            self.signed = {'u': [True], 'c': [False]}
        if self.domain_x is None:
            self.domain_x = [[0.0, 0.0], [1.0, 1.0]]


def create_mock_netcdf_dataset(file_path: str, n_samples: int = 10, n_nodes: int = 100,
                              coord_dim: int = 2, n_u_channels: int = 1, n_c_channels: int = 1,
                              variable_coords: bool = False, has_condition: bool = True):
    """
    Create a mock NetCDF dataset for testing.
    
    Args:
        file_path: Path to save the NetCDF file
        n_samples: Number of samples
        n_nodes: Number of nodes per sample
        coord_dim: Coordinate dimension (2 or 3)
        n_u_channels: Number of solution channels
        n_c_channels: Number of condition channels
        variable_coords: Whether coordinates vary across samples
        has_condition: Whether to include condition data
    """
    # Create coordinate data
    if variable_coords:
        # Variable coordinates: different for each sample
        if coord_dim == 2:
            x_data = np.random.uniform(0, 1, (n_samples, 1, n_nodes, 2))
        else:  # 3D
            x_data = np.random.uniform(0, 1, (n_samples, 1, n_nodes, 3))
    else:
        # Fixed coordinates: same for all samples
        if coord_dim == 2:
            x_coords = np.random.uniform(0, 1, (n_nodes, 2))
            x_data = np.tile(x_coords[None, None, :, :], (n_samples, 1, 1, 1))  # [1, 1, n_nodes, coord_dim]
        else:  # 3D
            x_coords = np.random.uniform(0, 1, (n_nodes, 3))
            x_data = np.tile(x_coords[None, None, :, :], (n_samples, 1, 1, 1))
    
    # Create solution data
    u_data = np.random.randn(n_samples, 1, n_nodes, n_u_channels)
    
    # Create condition data if needed
    if has_condition:
        c_data = np.random.randn(n_samples, 1, n_nodes, n_c_channels)
    
    # Create xarray dataset
    data_vars = {
        'u': (['sample', 'time', 'node', 'u_channel'], u_data)
    }
    
    if has_condition:
        data_vars['c'] = (['sample', 'time', 'node', 'c_channel'], c_data)
    
    if variable_coords or not variable_coords:  # Always include coordinates for testing
        data_vars['x'] = (['sample', 'time', 'node', 'coord'], x_data)
    
    # Create coordinates
    coords = {
        'sample': range(n_samples),
        'time': [0],  # Static datasets have single timestep
        'node': range(n_nodes),
        'u_channel': range(n_u_channels),
        'coord': range(coord_dim)
    }
    
    if has_condition:
        coords['c_channel'] = range(n_c_channels)
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Save to NetCDF
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    ds.to_netcdf(file_path)
    
    return file_path


def get_test_config_fx(coord_dim: int = 2) -> Tuple[DatasetConfig, ModelConfig, MockMetadata]:
    """Get test configuration for fixed coordinates mode."""
    
    # Create dataset config
    dataset_config = DatasetConfig()
    dataset_config.name = "test_fx_dataset"
    dataset_config.metaname = "test/fx"
    dataset_config.train_size = 6
    dataset_config.val_size = 2
    dataset_config.test_size = 2
    dataset_config.batch_size = 2
    dataset_config.num_workers = 0  # Avoid multiprocessing in tests
    dataset_config.shuffle = False
    dataset_config.rand_dataset = False
    
    # Create model config
    magno_config = MAGNOConfig()
    magno_config.coord_dim = coord_dim
    magno_config.radius = 0.1
    magno_config.scales = [1.0]
    magno_config.neighbor_search_method = "native"
    
    transformer_config = TransformerConfig()
    transformer_config.patch_size = 4
    
    model_args = ModelArgsConfig()
    model_args.magno = magno_config
    model_args.transformer = transformer_config
    
    model_config = ModelConfig()
    model_config.name = "gaot"
    if coord_dim == 2:
        model_config.latent_tokens_size = (8, 8)
    else:
        model_config.latent_tokens_size = (4, 4, 4)
    model_config.args = model_args
    
    # Create metadata
    metadata = MockMetadata()
    metadata.fix_x = True
    metadata.group_x = "x"
    if coord_dim == 3:
        metadata.domain_x = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    
    return dataset_config, model_config, metadata


def get_test_config_vx(coord_dim: int = 2) -> Tuple[DatasetConfig, ModelConfig, MockMetadata]:
    """Get test configuration for variable coordinates mode."""
    
    dataset_config, model_config, metadata = get_test_config_fx(coord_dim)
    
    # Modify for variable coordinates
    dataset_config.name = "test_vx_dataset"
    dataset_config.metaname = "test/vx"
    
    metadata.fix_x = False
    
    return dataset_config, model_config, metadata


class MockDatasetFactory:
    """Factory for creating mock datasets with different configurations."""
    
    def __init__(self):
        self.temp_dir = None
        self.created_files = []
    
    def setup(self):
        """Setup temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir
    
    def cleanup(self):
        """Clean up created test files."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_fx_dataset(self, coord_dim: int = 2, n_samples: int = 10, 
                         n_nodes: int = 64) -> Tuple[str, DatasetConfig, MockMetadata]:
        """Create fixed coordinates dataset."""
        if self.temp_dir is None:
            self.setup()
        
        file_path = os.path.join(self.temp_dir, "test_fx_dataset.nc")
        create_mock_netcdf_dataset(
            file_path, n_samples=n_samples, n_nodes=n_nodes,
            coord_dim=coord_dim, variable_coords=False
        )
        
        dataset_config, model_config, metadata = get_test_config_fx(coord_dim)
        dataset_config.base_path = self.temp_dir
        
        self.created_files.append(file_path)
        return file_path, dataset_config, metadata
    
    def create_vx_dataset(self, coord_dim: int = 2, n_samples: int = 10,
                         n_nodes: int = 64) -> Tuple[str, DatasetConfig, MockMetadata]:
        """Create variable coordinates dataset."""
        if self.temp_dir is None:
            self.setup()
        
        file_path = os.path.join(self.temp_dir, "test_vx_dataset.nc")
        create_mock_netcdf_dataset(
            file_path, n_samples=n_samples, n_nodes=n_nodes,
            coord_dim=coord_dim, variable_coords=True
        )
        
        dataset_config, model_config, metadata = get_test_config_vx(coord_dim)
        dataset_config.base_path = self.temp_dir
        
        self.created_files.append(file_path)
        return file_path, dataset_config, metadata
    
    def create_no_condition_dataset(self, coord_dim: int = 2) -> Tuple[str, DatasetConfig, MockMetadata]:
        """Create dataset without condition data."""
        if self.temp_dir is None:
            self.setup()
        
        file_path = os.path.join(self.temp_dir, "test_no_c_dataset.nc")
        create_mock_netcdf_dataset(
            file_path, coord_dim=coord_dim, has_condition=False
        )
        
        dataset_config, model_config, metadata = get_test_config_fx(coord_dim)
        dataset_config.base_path = self.temp_dir
        dataset_config.name = "test_no_c_dataset"
        metadata.group_c = None
        
        self.created_files.append(file_path)
        return file_path, dataset_config, metadata


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                       tensor_name: str = "tensor"):
    """Assert that tensor has expected shape."""
    actual_shape = tuple(tensor.shape)
    assert actual_shape == expected_shape, \
        f"{tensor_name} shape mismatch: expected {expected_shape}, got {actual_shape}"


def assert_dataloader_properties(loader, expected_batch_size: int, 
                                expected_length: int, dataset_type: str = "unknown"):
    """Assert basic properties of a data loader."""
    assert loader is not None, f"{dataset_type} loader should not be None"
    assert hasattr(loader, '__len__'), f"{dataset_type} loader should have length"
    assert len(loader) == expected_length, \
        f"{dataset_type} loader length: expected {expected_length}, got {len(loader)}"
    assert loader.batch_size == expected_batch_size, \
        f"{dataset_type} batch size: expected {expected_batch_size}, got {loader.batch_size}"


def validate_data_loader_output(loader, coord_mode: str, has_condition: bool = True):
    """Validate that data loader produces expected output format."""
    sample_batch = next(iter(loader))
    
    if coord_mode == 'fx':
        # Fixed coordinates: (c_data, u_data) 
        assert len(sample_batch) == 2, f"FX mode should return 2 items, got {len(sample_batch)}"
        c_batch, u_batch = sample_batch
        
        if has_condition:
            assert c_batch.numel() > 0, "Condition data should not be empty when has_condition=True"
        else:
            assert c_batch.numel() == 0, "Condition data should be empty when has_condition=False"
        
        assert u_batch.dim() == 3, f"Solution data should be 3D [batch, nodes, channels], got {u_batch.dim()}D"
        
    elif coord_mode == 'vx':
        # Variable coordinates: (c_data, u_data, x_data, encoder_graphs, decoder_graphs)
        assert len(sample_batch) == 5, f"VX mode should return 5 items, got {len(sample_batch)}"
        c_batch, u_batch, x_batch, encoder_graphs, decoder_graphs = sample_batch
        
        if has_condition:
            assert c_batch.numel() > 0, "Condition data should not be empty when has_condition=True"
        else:
            assert c_batch.numel() == 0, "Condition data should be empty when has_condition=False"

        assert u_batch.dim() == 3, f"Solution data should be 3D [batch, nodes, channels], got {u_batch.dim()}D"
        assert x_batch.dim() == 3, f"Coordinate data should be 3D [batch, nodes, coord_dim], got {x_batch.dim()}D"
        assert isinstance(encoder_graphs, list), "Encoder graphs should be a list"
        assert isinstance(decoder_graphs, list), "Decoder graphs should be a list"
        assert len(encoder_graphs) == u_batch.size(0), "Encoder graphs length should match batch size"
        assert len(decoder_graphs) == u_batch.size(0), "Decoder graphs length should match batch size"
    
    else:
        raise ValueError(f"Unknown coord_mode: {coord_mode}")
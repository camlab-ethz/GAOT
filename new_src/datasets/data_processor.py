"""
Data processing utilities for GAOT datasets.
Handles loading, preprocessing, normalization, and splitting of datasets.
"""
import os
import torch
import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ..core.trainer_utils import compute_data_stats, normalize_data
from ..utils.scaling import CoordinateScaler, rescale
from .data_utils import CustomDataset, collate_variable_batch


EPSILON = 1e-10


class DataProcessor:
    """
    Main data processing class that handles all dataset operations.
    Supports both fixed (fx) and variable (vx) coordinate modes.
    """
    
    def __init__(self, dataset_config, metadata, dtype: torch.dtype = torch.float32):
        self.dataset_config = dataset_config
        self.metadata = metadata
        self.dtype = dtype
        
        # Data statistics for normalization
        self.u_mean = None
        self.u_std = None
        self.c_mean = None
        self.c_std = None
        
        # Coordinate scaler
        self.coord_scaler = None
        
        # Special dataset handling
        self.poseidon_datasets = ["Poisson-Gauss", "CE-Gauss", "CE-RP", "CE-CRP", "CE-KH", "CE-RPUI",
                                  "NS-Gauss", "NS-PwC", "NS-SL", "NS-SVS", "NS-Sines"]
    
    def load_and_process_data(self) -> Tuple[Dict, bool]:
        """
        Load and process dataset from NetCDF file.
        
        Returns:
            tuple: (data_splits_dict, is_variable_coordinates)
        """
        print("Loading and preprocessing data...")
        
        # Load raw data
        raw_data = self._load_raw_data()

        # Determine coordinate mode
        is_variable_coords = self._determine_coordinate_mode(raw_data)
        
        # Split and normalize data
        data_splits = self._split_and_normalize_data(raw_data, is_variable_coords)
        
        print("Data loading and preprocessing complete.")
        return data_splits, is_variable_coords
    
    def _load_raw_data(self) -> Dict:
        """Load raw data from NetCDF file."""
        base_path = self.dataset_config.base_path
        dataset_name = self.dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        with xr.open_dataset(dataset_path) as ds:
            # Load u (solution) data
            u_array = ds[self.metadata.group_u].values
            
            # Load c (condition) data if available
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values
            else:
                c_array = None
            
            # Load x (coordinate) data
            x_array = self._load_coordinate_data(ds, u_array)
        
        return {
            'u': u_array,
            'c': c_array,
            'x': x_array
        }
    
    def _load_coordinate_data(self, ds: xr.Dataset, u_array: np.ndarray) -> np.ndarray:
        """Load coordinate data based on metadata configuration."""
        if self.metadata.group_x is not None:
            # Coordinates are explicitly provided in the dataset
            x_array = ds[self.metadata.group_x].values
            if not self.metadata.fix_x:
                # Variable coordinates - coordinates change across samples
                if x_array.shape[0] != u_array.shape[0]:
                    raise ValueError("Variable coordinates must have same number of samples as u_array")
                return x_array
            else:
                # Fixed coordinates - same coordinates for all samples
                return x_array
        else:
            # Generate coordinates from domain specification
            if self.metadata.domain_x is None:
                raise ValueError("Either group_x or domain_x must be specified in metadata")
            
            domain_x = self.metadata.domain_x
            nx, ny = u_array.shape[-2], u_array.shape[-1]
            
            x_lin = np.linspace(domain_x[0][0], domain_x[1][0], nx)
            y_lin = np.linspace(domain_x[0][1], domain_x[1][1], ny)
            xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
            x_array = np.stack((xv, yv), axis=-1)
            x_array = x_array.reshape(-1, 2)
            
            # For generated coordinates, add sample and timestep dimensions
            num_samples = u_array.shape[0]
            x_array = x_array[np.newaxis, np.newaxis, :, :]  # [1, 1, num_nodes, 2]
            x_array = np.repeat(x_array, num_samples, axis=0)  # [num_samples, 1, num_nodes, 2]
            
            return x_array
    
    def _determine_coordinate_mode(self, raw_data: Dict) -> bool:
        """
        Determine if coordinates are variable (vx) or fixed (fx).
        
        Returns:
            bool: True if variable coordinates, False if fixed
        """
        if self.metadata.group_x is not None:
            return not self.metadata.fix_x
        else:
            # Generated coordinates are typically fixed
            return False
    
    def _split_and_normalize_data(self, raw_data: Dict, is_variable_coords: bool) -> Dict:
        """Split data and apply normalization."""
        u_array = raw_data['u']
        c_array = raw_data['c']
        x_array = raw_data['x']
        
        # Handle dataset-specific preprocessing
        if self.dataset_config.name in self.poseidon_datasets and self.dataset_config.use_sparse:
            u_array = u_array[..., :9216, :]
            if c_array is not None:
                c_array = c_array[..., :9216, :]
            if x_array is not None:
                x_array = x_array[..., :9216, :]
        
        # Select active variables
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        
        # Validate shapes
        assert u_array.shape[1] == 1, "Expected num_timesteps to be 1 for static datasets"
        
        # Split data
        train_indices, val_indices, test_indices = self._get_split_indices(len(u_array))
        
        u_train = np.ascontiguousarray(u_array[train_indices])
        u_val = np.ascontiguousarray(u_array[val_indices])
        u_test = np.ascontiguousarray(u_array[test_indices])
        
        if c_array is not None:
            c_train = np.ascontiguousarray(c_array[train_indices])
            c_val = np.ascontiguousarray(c_array[val_indices])
            c_test = np.ascontiguousarray(c_array[test_indices])
        else:
            c_train = c_val = c_test = None

        if is_variable_coords:
            x_train = x_array[train_indices]
            x_val = x_array[val_indices]
            x_test = x_array[test_indices]
        else:
            # Fixed coordinates - same for all splits
            x_coord = x_array[0, 0] if x_array.ndim == 4 else x_array
            x_train = x_val = x_test = x_coord
        
        # Normalize data
        self._compute_and_apply_normalization(
            u_train, u_val, u_test, c_train, c_val, c_test
        )
        
        # Convert to tensors
        data_splits = self._convert_to_tensors(
            u_train, u_val, u_test,
            c_train, c_val, c_test,
            x_train, x_val, x_test,
            is_variable_coords
        )
        
        return data_splits
    
    def _get_split_indices(self, total_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get indices for train/val/test splits."""
        train_size = self.dataset_config.train_size
        val_size = self.dataset_config.val_size
        test_size = self.dataset_config.test_size

        assert train_size + val_size + test_size <= total_samples, \
            "Sum of train, val, and test sizes exceeds total samples"
        
        if self.dataset_config.rand_dataset:
            indices = np.random.permutation(total_samples)
        else:
            indices = np.arange(total_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[-test_size:]
        
        return train_indices, val_indices, test_indices
    
    def _compute_and_apply_normalization(self, u_train, u_val, u_test, 
                                        c_train, c_val, c_test):
        """Compute normalization statistics and apply to all splits."""
        print("Computing statistics and normalizing data")
        
        # Normalize u data
        u_train_flat = u_train.reshape(-1, u_train.shape[-1])
        u_mean = np.mean(u_train_flat, axis=0)
        u_std = np.std(u_train_flat, axis=0) + EPSILON
        
        self.u_mean = torch.tensor(u_mean, dtype=self.dtype)
        self.u_std = torch.tensor(u_std, dtype=self.dtype)
        
        u_train[:] = (u_train - u_mean) / u_std
        u_val[:] = (u_val - u_mean) / u_std
        u_test[:] = (u_test - u_mean) / u_std
        
        # Normalize c data if available
        if c_train is not None:
            c_train_flat = c_train.reshape(-1, c_train.shape[-1])
            c_mean = np.mean(c_train_flat, axis=0)
            c_std = np.std(c_train_flat, axis=0) + EPSILON
            
            self.c_mean = torch.tensor(c_mean, dtype=self.dtype)
            self.c_std = torch.tensor(c_std, dtype=self.dtype)
            
            c_train[:] = (c_train - c_mean) / c_std
            c_val[:] = (c_val - c_mean) / c_std
            c_test[:] = (c_test - c_mean) / c_std
        else:
            self.c_mean = None
            self.c_std = None
    
    def _convert_to_tensors(self, u_train, u_val, u_test, c_train, c_val, c_test,
                           x_train, x_val, x_test, is_variable_coords) -> Dict:
        """Convert numpy arrays to PyTorch tensors."""
        
        # Convert u data (squeeze timestep dimension)
        u_train = torch.tensor(u_train, dtype=self.dtype).squeeze(1)
        u_val = torch.tensor(u_val, dtype=self.dtype).squeeze(1)
        u_test = torch.tensor(u_test, dtype=self.dtype).squeeze(1)
        
        # Convert c data if available
        if c_train is not None:
            c_train = torch.tensor(c_train, dtype=self.dtype).squeeze(1)
            c_val = torch.tensor(c_val, dtype=self.dtype).squeeze(1)
            c_test = torch.tensor(c_test, dtype=self.dtype).squeeze(1)
        
        # Convert x data
        if is_variable_coords:
            x_train = torch.tensor(x_train, dtype=self.dtype).squeeze(1)
            x_val = torch.tensor(x_val, dtype=self.dtype).squeeze(1)
            x_test = torch.tensor(x_test, dtype=self.dtype).squeeze(1)
        else:
            x_coord = torch.tensor(x_train, dtype=self.dtype)
            x_train = x_val = x_test = x_coord
        
        return {
            "train": {"c": c_train, "u": u_train, "x": x_train},
            "val": {"c": c_val, "u": u_val, "x": x_val},
            "test": {"c": c_test, "u": u_test, "x": x_test},
        }
    
    def generate_latent_queries(self, token_size: Tuple[int, ...]) -> torch.Tensor:
        """Generate latent query points on a regular grid."""
        phy_domain = self.metadata.domain_x
        
        if len(token_size) == 2:
            # 2D case
            x_min, y_min = phy_domain[0]
            x_max, y_max = phy_domain[1]
            
            meshgrid = torch.meshgrid(
                torch.linspace(x_min, x_max, token_size[0], dtype=self.dtype),
                torch.linspace(y_min, y_max, token_size[1], dtype=self.dtype),
                indexing='ij'
            )
            latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1, 2)
        
        elif len(token_size) == 3:
            # 3D case
            x_min, y_min, z_min = phy_domain[0]
            x_max, y_max, z_max = phy_domain[1]
            
            meshgrid = torch.meshgrid(
                torch.linspace(x_min, x_max, token_size[0], dtype=self.dtype),
                torch.linspace(y_min, y_max, token_size[1], dtype=self.dtype),
                torch.linspace(z_min, z_max, token_size[2], dtype=self.dtype),
                indexing='ij'
            )
            latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1, 3)
        
        else:
            raise ValueError(f"Unsupported token_size dimensions: {len(token_size)}")
        
        # Initialize coordinate scaler if not already done
        if self.coord_scaler is None:
            self.coord_scaler = CoordinateScaler(
                target_range=(-1, 1),
                mode=self.dataset_config.coord_scaling
            )
        
        # Apply coordinate scaling
        latent_queries = self.coord_scaler(latent_queries)
        
        return latent_queries
    
    def create_data_loaders(self, data_splits: Dict, is_variable_coords: bool, 
                           latent_queries: Optional[torch.Tensor] = None,
                           encoder_graphs: Optional[list] = None,
                           decoder_graphs: Optional[list] = None) -> Dict[str, DataLoader]:
        """Create data loaders for train/val/test splits."""
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            # Skip train/val if not training
            if split in ['train', 'val'] and not getattr(self.dataset_config, 'train', True):
                loaders[split] = None
                continue
            
            c_data = data_splits[split]["c"]
            u_data = data_splits[split]["u"]
            x_data = data_splits[split]["x"]
            
            if is_variable_coords:
                # Variable coordinates - need graphs
                encoder_graphs_split = encoder_graphs[split] if encoder_graphs else None
                decoder_graphs_split = decoder_graphs[split] if decoder_graphs else None

                dataset = CustomDataset(
                    c_data, u_data, x_data,
                    encoder_graphs_split, decoder_graphs_split,
                    transform=self.coord_scaler
                )
                
                loader = DataLoader(
                    dataset,
                    batch_size=self.dataset_config.batch_size,
                    shuffle=self.dataset_config.shuffle if split == 'train' else False,
                    collate_fn=collate_variable_batch,
                    num_workers=self.dataset_config.num_workers,
                    pin_memory=True
                )
            else:
                # Fixed coordinates - simple tensor dataset
                if c_data is not None:
                    dataset = TensorDataset(c_data, u_data)
                else:
                    # Create dummy tensor for c_data when it's None
                    dummy_c = torch.empty(u_data.size(0), 0)
                    dataset = TensorDataset(dummy_c, u_data)
                
                loader = DataLoader(
                    dataset,
                    batch_size=self.dataset_config.batch_size,
                    shuffle=self.dataset_config.shuffle if split == 'train' else False,
                    num_workers=self.dataset_config.num_workers,
                    pin_memory=True
                )
            
            loaders[split] = loader
        
        return loaders
"""
Sequential data processing utilities for time-dependent GAOT datasets.
Extends the base DataProcessor for temporal data handling.
"""
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader

from .data_processor import DataProcessor, EPSILON
from ..core.trainer_utils import compute_data_stats, normalize_data
from ..core.trainer_utils import compute_sequential_stats

class SequentialDataProcessor(DataProcessor):
    """
    Specialized data processor for sequential (time-dependent) datasets.
    Supports both fixed (fx) and variable (vx) coordinate modes.
    """
    
    def __init__(self, dataset_config, metadata, dtype: torch.dtype = torch.float32):
        super().__init__(dataset_config, metadata, dtype)
        
        # Additional attributes for temporal data
        self.t_values = None
        self.stats = None
        
        # Time-related configurations
        self.max_time_diff = dataset_config.max_time_diff
        self.time_step = dataset_config.time_step
        self.stepper_mode = dataset_config.stepper_mode
        self.use_time_norm = dataset_config.use_time_norm
        self.use_metadata_stats = dataset_config.use_metadata_stats
        self.sample_rate = dataset_config.sample_rate
    
    def load_and_process_data(self) -> Tuple[Dict, bool]:
        """
        Load and process sequential dataset from NetCDF file.
        
        Returns:
            tuple: (data_splits_dict, is_variable_coordinates)
        """
        print("Loading and preprocessing sequential data...")
        
        # Load raw data with temporal dimension
        raw_data = self._load_raw_sequential_data()

        # Determine coordinate mode
        is_variable_coords = self._determine_coordinate_mode(raw_data)
        
        # Split and normalize data (preserving time dimension)
        data_splits = self._split_and_normalize_sequential_data(raw_data, is_variable_coords)
        
        print("Sequential data loading and preprocessing complete.")
        return data_splits, is_variable_coords
    
    def _load_raw_sequential_data(self) -> Dict:
        """Load raw sequential data from NetCDF file."""
        import xarray as xr
        import os
        
        base_path = self.dataset_config.base_path
        dataset_name = self.dataset_config.name
        dataset_path = os.path.join(base_path, f"{dataset_name}.nc")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        with xr.open_dataset(dataset_path) as ds:
            # Load u (solution) data - Shape: [num_samples, num_timesteps, num_nodes, num_channels]
            u_array = ds[self.metadata.group_u].values
            
            # Load c (condition) data if available
            if self.metadata.group_c is not None:
                c_array = ds[self.metadata.group_c].values
            else:
                c_array = None
            
            # Load x (coordinate) data
            x_array = self._load_sequential_coordinate_data(ds, u_array)
            
            # Load or generate time values
            if self.metadata.domain_t is not None:
                t_start, t_end = self.metadata.domain_t
                self.t_values = np.linspace(t_start, t_end, u_array.shape[1])
            else:
                raise ValueError("metadata.domain_t is None. Cannot compute actual time values.")
        
        # Handle special datasets (e.g., Poseidon sparse data)
        if dataset_name in self.poseidon_datasets and self.dataset_config.use_sparse:
            u_array = u_array[:, :, :9216, :]
            if c_array is not None:
                c_array = c_array[:, :, :9216, :]
            x_array = x_array[:, :, :9216, :]
        
        # Select active variables
        active_vars = self.metadata.active_variables
        u_array = u_array[..., active_vars]
        
        return {
            'u': u_array,  # [num_samples, num_timesteps, num_nodes, num_channels]
            'c': c_array,  # [num_samples, num_timesteps, num_nodes, num_c_channels] or None
            'x': x_array,  # [num_samples, num_timesteps, num_nodes, coord_dim] or [1, 1, num_nodes, coord_dim]
            't': self.t_values  # [num_timesteps]
        }
    
    def _load_sequential_coordinate_data(self, ds, u_array: np.ndarray) -> np.ndarray:
        """Load coordinate data for sequential datasets."""
        if self.metadata.group_x is not None:
            # Coordinates provided in dataset
            x_array = ds[self.metadata.group_x].values
            
            if self.metadata.fix_x:
                # Fixed coordinates - same for all samples and timesteps
                # Ensure shape is [1, 1, num_nodes, coord_dim]
                if x_array.ndim == 2:  # [num_nodes, coord_dim]
                    x_array = x_array[None, None, ...]
                elif x_array.ndim == 3:  # [1, num_nodes, coord_dim]
                    x_array = x_array[:, None, ...]
            else:
                # Variable coordinates - can change across samples/timesteps
                # Shape should be [num_samples, num_timesteps, num_nodes, coord_dim]
                if x_array.shape[0] != u_array.shape[0]:
                    raise ValueError("Variable coordinates must have same number of samples as u_array")
                    
        else:
            # Generate coordinates from domain (for structured grids)
            domain_x = self.metadata.domain_x
            # Note: For sequential data, we assume spatial grid structure from u_array
            if u_array.ndim == 4:  # [samples, time, nodes, channels]
                num_nodes = u_array.shape[2]
                # For structured grids, assume square grid
                grid_size = int(np.sqrt(num_nodes))
                if grid_size * grid_size != num_nodes:
                    raise ValueError(f"Cannot create square grid from {num_nodes} nodes")
                
                x_min, y_min = domain_x[0]
                x_max, y_max = domain_x[1]
                x_lin = np.linspace(x_min, x_max, grid_size)
                y_lin = np.linspace(y_min, y_max, grid_size)
                xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
                x_array = np.stack([xv, yv], axis=-1).reshape(-1, 2)
                x_array = x_array[None, None, ...]  # Add sample and time dimensions
            else:
                raise ValueError(f"Unexpected u_array shape: {u_array.shape}")
        
        return x_array
    
    def _split_and_normalize_sequential_data(self, raw_data: Dict, is_variable_coords: bool) -> Dict:
        """Split and normalize sequential data preserving temporal dimension."""
        u_array = raw_data['u']  # [num_samples, num_timesteps, num_nodes, num_channels]
        c_array = raw_data['c']  # [num_samples, num_timesteps, num_nodes, num_c_channels] or None
        x_array = raw_data['x']  # coordinate data
        t_values = raw_data['t']  # [num_timesteps]
        
        # Limit time steps based on max_time_diff
        if self.max_time_diff is not None:
            max_timesteps = self.max_time_diff + 1  # +1 because we need t=0 to t=max_time_diff
            u_array = u_array[:, :max_timesteps, :, :]
            if c_array is not None:
                c_array = c_array[:, :max_timesteps, :, :]
            if is_variable_coords and x_array.shape[1] > 1:
                x_array = x_array[:, :max_timesteps, :, :]
            t_values = t_values[:max_timesteps]
            self.t_values = t_values
        
        # Split data
        train_indices, val_indices, test_indices = self._get_split_indices(u_array.shape[0])

        u_train = np.ascontiguousarray(u_array[train_indices])
        u_val = np.ascontiguousarray(u_array[val_indices])
        u_test = np.ascontiguousarray(u_array[test_indices])
        
        if c_array is not None:
            c_train = np.ascontiguousarray(c_array[train_indices])
            c_val = np.ascontiguousarray(c_array[val_indices])
            c_test = np.ascontiguousarray(c_array[test_indices])
        else:
            c_train = c_val = c_test = None
        
        # Handle coordinate data
        if is_variable_coords:
            x_train = np.ascontiguousarray(x_array[train_indices])
            x_val = np.ascontiguousarray(x_array[val_indices])
            x_test = np.ascontiguousarray(x_array[test_indices])
        else:
            # Fixed coordinates - same for all samples
            x_coord = x_array[0, 0]  # [num_nodes, coord_dim]
            x_train = x_val = x_test = x_coord
        
        # Compute statistics for sequential data
        self.stats = self._compute_sequential_stats(u_train, c_train, t_values)
        
        # Convert stats to tensors with proper dtype
        for key, value in self.stats.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.stats[key][k] = torch.tensor(v, dtype=self.dtype)
        
        data_splits = self._convert_to_tensors(
            u_train, u_val, u_test,
            c_train, c_val, c_test,
            x_train, x_val, x_test,
            is_variable_coords
        )

        data_splits['train']['t'] = torch.tensor(t_values, dtype=self.dtype)
        data_splits['val']['t'] = torch.tensor(t_values, dtype=self.dtype)
        data_splits['test']['t'] = torch.tensor(t_values, dtype=self.dtype)
        
        return data_splits
    
    def _compute_sequential_stats(self, u_train: np.ndarray, c_train: Optional[np.ndarray], 
                                t_values: np.ndarray) -> Dict:
        """Compute statistics for sequential data including time features."""
        
        return compute_sequential_stats(
            u_data=u_train,
            c_data=c_train,
            t_values=t_values,
            metadata=self.metadata,
            max_time_diff=self.max_time_diff,
            time_step=self.time_step,
            sample_rate=self.sample_rate,
            use_metadata_stats=self.use_metadata_stats,
            use_time_norm=self.use_time_norm
        )
    
    def create_sequential_data_loaders(self, data_splits: Dict, is_variable_coords: bool,
                                     **kwargs) -> Dict[str, Optional[DataLoader]]:
        """Create data loaders for sequential data with temporal pairs."""
        from .data_utils import DynamicPairDataset, collate_sequential_batch
        
        # Extract data
        train_data = data_splits['train']
        val_data = data_splits['val']
        test_data = data_splits['test']
        
        loaders = {}
        # Create datasets
        if getattr(self.dataset_config, 'train', True):
            train_dataset = DynamicPairDataset(
                u_data=train_data['u'],
                c_data=train_data['c'],
                x_data=train_data['x'] if is_variable_coords else None,
                t_values=train_data['t'],
                metadata=self.metadata,
                max_time_diff=self.max_time_diff,
                stepper_mode=self.stepper_mode,
                stats=self.stats,
                use_time_norm=self.use_time_norm,
                is_variable_coords=is_variable_coords
            )
            
            val_dataset = DynamicPairDataset(
                u_data=val_data['u'],
                c_data=val_data['c'],
                x_data=val_data['x'] if is_variable_coords else None,
                t_values=val_data['t'],
                metadata=self.metadata,
                max_time_diff=self.max_time_diff,
                stepper_mode=self.stepper_mode,
                stats=self.stats,
                use_time_norm=self.use_time_norm,
                is_variable_coords=is_variable_coords
            )
            
            loaders['train'] = DataLoader(
                train_dataset,
                batch_size=self.dataset_config.batch_size,
                shuffle=self.dataset_config.shuffle,
                num_workers=self.dataset_config.num_workers,
                pin_memory=True,
                collate_fn=collate_sequential_batch
            )
            
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=self.dataset_config.batch_size,
                shuffle=False,
                num_workers=self.dataset_config.num_workers,
                pin_memory=True,
                collate_fn=collate_sequential_batch
            )
        else:
            loaders['train'] = None
            loaders['val'] = None

        # Test dataset
        test_dataset = DynamicPairDataset(
            u_data=test_data['u'],
            c_data=test_data['c'],
            x_data=test_data['x'] if is_variable_coords else None,
            t_values=test_data['t'],
            metadata=self.metadata,
            max_time_diff=self.max_time_diff,
            stepper_mode=self.stepper_mode,
            stats=self.stats,
            use_time_norm=self.use_time_norm,
            is_variable_coords=is_variable_coords
        )
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=self.dataset_config.batch_size,
            shuffle=False,
            num_workers=self.dataset_config.num_workers,
            pin_memory=True,
            collate_fn=collate_sequential_batch
        )
        
        return loaders
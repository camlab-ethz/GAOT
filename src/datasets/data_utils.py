"""
Data utility classes for GAOT datasets.
Custom dataset classes and data manipulation utilities.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable, List


class CustomDataset(Dataset):
    """
    Custom dataset for variable coordinate (vx) mode data.
    Handles data with pre-computed graphs for encoder and decoder.
    """
    
    def __init__(self, c_data: Optional[torch.Tensor], u_data: torch.Tensor, 
                 x_data: torch.Tensor, encoder_graphs: List, decoder_graphs: List,
                 transform: Optional[Callable] = None):
        """
        Initialize custom dataset.
        
        Args:
            c_data: Condition data tensor [n_samples, n_nodes, n_c_features] or None
            u_data: Solution data tensor [n_samples, n_nodes, n_u_features]  
            x_data: Coordinate data tensor [n_samples, n_nodes, coord_dim]
            encoder_graphs: List of encoder neighbor graphs for each sample
            decoder_graphs: List of decoder neighbor graphs for each sample
            transform: Optional transformation function for coordinates
        """
        self.c_data = c_data
        self.u_data = u_data
        self.x_data = x_data
        self.encoder_graphs = encoder_graphs
        self.decoder_graphs = decoder_graphs
        self.transform = transform
        
        # Validate data consistency
        n_samples = len(u_data)
        if c_data is not None and len(c_data) != n_samples:
            raise ValueError("c_data and u_data must have same number of samples")
        if len(x_data) != n_samples:
            raise ValueError("x_data and u_data must have same number of samples")
        if len(encoder_graphs) != n_samples:
            raise ValueError("encoder_graphs and u_data must have same number of samples")
        if len(decoder_graphs) != n_samples:
            raise ValueError("decoder_graphs and u_data must have same number of samples")
    
    def __len__(self):
        return len(self.u_data)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Returns:
            tuple: (c, u, x, encoder_graph, decoder_graph)
        """
        c = self.c_data[idx] if self.c_data is not None else torch.empty(0)
        u = self.u_data[idx]
        x = self.x_data[idx]
        
        # Apply coordinate transformation if specified
        if self.transform is not None:
            x = self.transform(x)
        
        encoder_graph = self.encoder_graphs[idx]
        decoder_graph = self.decoder_graphs[idx]
        
        return c, u, x, encoder_graph, decoder_graph


class DynamicPairDataset(Dataset):
    """
    Dataset for time-dependent data with dynamic time pairs.
    Used for sequential (time-dependent) training.
    Supports both fixed (fx) and variable (vx) coordinate modes.
    """
    
    def __init__(self, u_data: torch.Tensor, c_data: Optional[torch.Tensor], 
                 t_values: torch.Tensor, metadata, max_time_diff: int = 14, time_step: int = 2,
                 stepper_mode: str = "output", stats: Optional[dict] = None,
                 use_time_norm: bool = True, dataset_name: Optional[str] = None,
                 x_data: Optional[torch.Tensor] = None, is_variable_coords: bool = False):
        """
        Initialize dynamic pair dataset.
        
        Args:
            u_data: Solution data [n_samples, n_timesteps, n_nodes, n_vars]
            c_data: Condition data [n_samples, n_timesteps, n_nodes, n_c_vars] or None
            t_values: Time values [n_timesteps]
            metadata: Dataset metadata
            max_time_diff: Maximum time difference between input and output
            stepper_mode: Stepper mode ['output', 'residual', 'time_der']
            stats: Statistics dictionary
            use_time_norm: Whether to normalize time features
            dataset_name: Name of the dataset
            x_data: Coordinate data for variable coordinates mode
            is_variable_coords: Whether using variable coordinates mode
        """
        self.dataset_name = dataset_name
        self.u_data = u_data
        self.c_data = c_data
        self.x_data = x_data  # For variable coordinates
        self.t_values = t_values
        self.metadata = metadata
        self.stepper_mode = stepper_mode
        self.stats = stats
        self.use_time_norm = use_time_norm
        self.is_variable_coords = is_variable_coords

        self.num_samples, self.num_timesteps, self.num_nodes, self.num_vars = u_data.shape
        
        # Limit timesteps based on max_time_diff
        self.num_timesteps = min(self.num_timesteps-1, max_time_diff)
        self.t_values = self.t_values[:self.num_timesteps + 1]
        
        # Generate time pairs
        self._generate_time_pairs(self.num_timesteps, time_step)
    
    def _generate_time_pairs(self, num_timesteps: int, time_step: int):
        """Generate specific time pairs for training."""
        self.t_in_indices = []
        self.t_out_indices = []
        
        # Generate even lags from 2 to max_time_diff
        for lag in range(time_step, num_timesteps + 1, time_step):
            for i in range(0, num_timesteps - lag + 1, time_step):
                t_in_idx = i
                t_out_idx = i + lag
                self.t_in_indices.append(t_in_idx)
                self.t_out_indices.append(t_out_idx)
        
        self.t_in_indices = np.array(self.t_in_indices)
        self.t_out_indices = np.array(self.t_out_indices)
        
        self.time_diffs = self.t_values[self.t_out_indices] - self.t_values[self.t_in_indices]
        # Precompute normalized time features for all time pairs
        if self.use_time_norm and self.stats is not None:
            self.start_times = self.t_values[self.t_in_indices]
            self.start_times_norm = ((self.start_times - self.stats["start_time"]["mean"]) / 
                                   self.stats["start_time"]["std"])
            self.time_diffs_norm = ((self.time_diffs - self.stats["time_diffs"]["mean"]) / 
                                  self.stats["time_diffs"]["std"])
        else:
            self.start_times_norm = self.t_values[self.t_in_indices]
            self.time_diffs_norm = self.time_diffs
        
        # Prepare expanded time features
        self._prepare_time_features()
    
    def __len__(self):
        return self.num_samples * len(self.t_in_indices)
    
    def _prepare_time_features(self):
        """Prepare expanded time features for all time pairs."""
        self.start_time_expanded = self.start_times_norm.unsqueeze(1).expand(-1, self.num_nodes)
        self.time_diff_expanded = self.time_diffs_norm.unsqueeze(1).expand(-1, self.num_nodes)
        self.start_time_expanded = self.start_time_expanded[..., None]
        self.time_diff_expanded = self.time_diff_expanded[..., None]
    
    def __getitem__(self, idx):
        """
        Get a time pair sample.
        
        Returns:
            tuple: Input data tuple with time features
        """
        sample_idx = idx // len(self.t_in_indices)
        pair_idx = idx % len(self.t_in_indices)
        
        t_in_idx = self.t_in_indices[pair_idx]
        t_out_idx = self.t_out_indices[pair_idx]
        
        # Get input and output data
        u_in = self.u_data[sample_idx, t_in_idx]  # [num_nodes, num_vars]
        u_out = self.u_data[sample_idx, t_out_idx]
        
        # Normalize u data
        if self.stats is not None:
            u_in_norm = (u_in - self.stats["u"]["mean"]) / self.stats["u"]["std"]
        else:
            u_in_norm = u_in
        
        # Get condition data if available
        if self.c_data is not None:
            c_in = self.c_data[sample_idx, t_in_idx]
            if self.stats is not None and "c" in self.stats:
                c_in_norm = (c_in - self.stats["c"]["mean"]) / self.stats["c"]["std"]
            else:
                c_in_norm = c_in
        else:
            c_in_norm = None
        
        # Prepare input features
        input_features = [u_in_norm]
        if c_in_norm is not None:
            input_features.append(c_in_norm)
        
        # Add time features
        start_time_feat = self.start_time_expanded[pair_idx]  # [num_nodes, 1]
        time_diff_feat = self.time_diff_expanded[pair_idx]    # [num_nodes, 1]
        input_features.extend([start_time_feat, time_diff_feat])

        # Concatenate all features
        input_data = torch.cat(input_features, dim=-1)  # [num_nodes, total_features]
        
        # Compute target based on stepper mode
        if self.stepper_mode == "output":
            target = u_out
        elif self.stepper_mode == "residual":
            if self.stats is not None:
                res_mean = self.stats["res"]["mean"]
                res_std = self.stats["res"]["std"]
                target = (u_out - u_in - res_mean) / res_std
            else:
                target = u_out - u_in
        elif self.stepper_mode == "time_der":
            time_diff_actual = self.time_diffs[pair_idx]
            u_time_der = (u_out - u_in) / time_diff_actual
            if self.stats is not None:
                der_mean = self.stats["der"]["mean"]
                der_std = self.stats["der"]["std"]
                target = (u_time_der - der_mean) / der_std
            else:
                target = u_time_der
        else:
            raise ValueError(f"Unsupported stepper_mode: {self.stepper_mode}")
        
        # For variable coordinates, also return coordinate data
        if self.is_variable_coords and self.x_data is not None:
            x_coord = self.x_data[sample_idx, t_in_idx]
            return input_data, target, x_coord
        else:
            return input_data, target


class StaticDataset(Dataset):
    """
    Simple dataset for static (time-independent) data with fixed coordinates.
    """
    
    def __init__(self, c_data: Optional[torch.Tensor], u_data: torch.Tensor):
        """
        Initialize static dataset.
        
        Args:
            c_data: Condition data [n_samples, n_nodes, n_c_features] or None
            u_data: Solution data [n_samples, n_nodes, n_u_features]
        """
        self.c_data = c_data
        self.u_data = u_data
        
        if c_data is not None and len(c_data) != len(u_data):
            raise ValueError("c_data and u_data must have same number of samples")
    
    def __len__(self):
        return len(self.u_data)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Returns:
            tuple: (c, u)
        """
        c = self.c_data[idx] if self.c_data is not None else torch.empty(0)
        u = self.u_data[idx]
        return c, u


def collate_variable_batch(batch):
    """
    Custom collate function for variable-size batches.
    Handles padding and masking for irregular data.
    """
    # Separate different components
    c_list, u_list, x_list = [], [], []
    encoder_graphs_list, decoder_graphs_list = [], []
    
    for item in batch:
        c, u, x, encoder_graph, decoder_graph = item
        c_list.append(c)
        u_list.append(u)
        x_list.append(x)
        encoder_graphs_list.append(encoder_graph)
        decoder_graphs_list.append(decoder_graph)
    
    # Stack regular tensors
    c_batch = torch.stack(c_list) if c_list[0].numel() > 0 else None
    u_batch = torch.stack(u_list)
    x_batch = torch.stack(x_list)
    
    return c_batch, u_batch, x_batch, encoder_graphs_list, decoder_graphs_list


def collate_sequential_batch(batch):
    """
    Custom collate function for sequential data batches.
    Handles both fixed and variable coordinate modes.
    """
    if len(batch[0]) == 2:  # Fixed coordinates mode
        input_list, target_list = zip(*batch)
        inputs = torch.stack(input_list)
        targets = torch.stack(target_list)
        return inputs, targets
    elif len(batch[0]) == 3:  # Variable coordinates mode  
        input_list, target_list, coord_list = zip(*batch)
        inputs = torch.stack(input_list)
        targets = torch.stack(target_list)
        coords = torch.stack(coord_list)
        return inputs, targets, coords
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")


class TestDataset(Dataset):
    """
    Dataset for sequential testing with specific time indices.
    Used for autoregressive prediction evaluation.
    """
    
    def __init__(self, u_data: torch.Tensor, c_data: Optional[torch.Tensor],
                 t_values: torch.Tensor, metadata, time_indices: np.ndarray,
                 stats: dict, x_data: Optional[torch.Tensor] = None,
                 is_variable_coords: bool = False):
        """
        Initialize test dataset.
        
        Args:
            u_data: Solution data [n_samples, n_timesteps, n_nodes, n_vars]
            c_data: Condition data [n_samples, n_timesteps, n_nodes, n_c_vars] or None
            t_values: Time values [n_timesteps]
            metadata: Dataset metadata
            time_indices: Time indices to use [n_test_times]
            stats: Statistics dictionary
            x_data: Coordinate data for variable coords mode
            is_variable_coords: Whether using variable coordinates
        """
        self.u_data = u_data
        self.c_data = c_data
        self.x_data = x_data
        self.t_values = t_values
        self.metadata = metadata
        self.time_indices = time_indices
        self.stats = stats
        self.is_variable_coords = is_variable_coords
        
        self.num_samples = u_data.shape[0]
        self.num_nodes = u_data.shape[2]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get test sample at first time index.
        
        Returns:
            tuple: (input_features, target_sequence) or with coordinates
        """
        t_start_idx = self.time_indices[0]
        
        # Get initial state
        u_start = self.u_data[idx, t_start_idx]  # [num_nodes, num_vars]
        
        # Normalize input
        if self.stats is not None:
            u_start_norm = (u_start - self.stats["u"]["mean"]) / self.stats["u"]["std"]
        else:
            u_start_norm = u_start
        
        # Get condition data if available
        if self.c_data is not None:
            c_start = self.c_data[idx, t_start_idx]
            if self.stats is not None and "c" in self.stats:
                c_start_norm = (c_start - self.stats["c"]["mean"]) / self.stats["c"]["std"]
            else:
                c_start_norm = c_start
        else:
            c_start_norm = None
        
        # Prepare input features (at first timestep)
        input_features = [u_start_norm]
        if c_start_norm is not None:
            input_features.append(c_start_norm)
        
        # Add dummy time features (will be replaced during autoregressive prediction)
        dummy_time_feat = torch.zeros((self.num_nodes, 1), dtype=torch.float32)
        input_features.extend([dummy_time_feat, dummy_time_feat])
        
        input_data = torch.cat(input_features, dim=-1)
        
        # Get target sequence (excluding first timestep)
        target_sequence = self.u_data[idx, self.time_indices[1:]]  # [n_timesteps-1, num_nodes, num_vars]
        
        # For variable coordinates, also return coordinate data
        if self.is_variable_coords and self.x_data is not None:
            x_coord = self.x_data[idx, t_start_idx]
            return input_data, target_sequence, x_coord
        else:
            return input_data, target_sequence


def create_data_splits(data: torch.Tensor, train_ratio: float = 0.8, 
                      val_ratio: float = 0.1, shuffle: bool = True) -> dict:
    """
    Create train/validation/test splits from data.
    
    Args:
        data: Input data tensor
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        shuffle: Whether to shuffle before splitting
        
    Returns:
        dict: Dictionary with train/val/test tensors
    """
    n_samples = len(data)
    
    if shuffle:
        indices = torch.randperm(n_samples)
        data = data[indices]
    
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    return {
        'train': data[:train_end],
        'val': data[train_end:val_end], 
        'test': data[val_end:]
    }
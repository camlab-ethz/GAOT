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
    """
    
    def __init__(self, u_data: np.ndarray, c_data: Optional[np.ndarray], 
                 t_values: np.ndarray, metadata, max_time_diff: int = 14, time_step: int = 2,
                 stepper_mode: str = "output", stats: Optional[dict] = None,
                 use_time_norm: bool = True, dataset_name: Optional[str] = None):
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
        """
        self.dataset_name = dataset_name
        self.u_data = u_data
        self.c_data = c_data
        self.t_values = t_values
        self.metadata = metadata
        self.stepper_mode = stepper_mode
        self.stats = stats
        self.use_time_norm = use_time_norm
        
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
        for lag in range(2, num_timesteps + 1, time_step):
            for i in range(0, num_timesteps - lag + 1, time_step):
                t_in_idx = i
                t_out_idx = i + lag
                self.t_in_indices.append(t_in_idx)
                self.t_out_indices.append(t_out_idx)
        
        self.t_in_indices = np.array(self.t_in_indices)
        self.t_out_indices = np.array(self.t_out_indices)
        
        self.time_diffs = self.t_values[self.t_out_indices] - self.t_values[self.t_in_indices]
        
        # Normalize time differences if requested
        if self.use_time_norm and self.stats is not None:
            time_diff_mean = self.stats.get('time_diff_mean', 0.0)
            time_diff_std = self.stats.get('time_diff_std', 1.0)
            self.time_diffs_norm = (self.time_diffs - time_diff_mean) / time_diff_std
        else:
            self.time_diffs_norm = self.time_diffs
    
    def __len__(self):
        return self.num_samples * len(self.t_in_indices)
    
    def __getitem__(self, idx):
        """
        Get a time pair sample.
        
        Returns:
            tuple: Depends on stepper_mode
        """
        sample_idx = idx // len(self.t_in_indices)
        pair_idx = idx % len(self.t_in_indices)
        
        t_in_idx = self.t_in_indices[pair_idx]
        t_out_idx = self.t_out_indices[pair_idx]
        
        # Get input and output data
        u_in = torch.tensor(self.u_data[sample_idx, t_in_idx], dtype=torch.float32)
        u_out = torch.tensor(self.u_data[sample_idx, t_out_idx], dtype=torch.float32)
        
        # Get condition data if available
        if self.c_data is not None:
            c_in = torch.tensor(self.c_data[sample_idx, t_in_idx], dtype=torch.float32)
        else:
            c_in = torch.empty(0)
        
        # Get time information
        time_diff = torch.tensor(self.time_diffs_norm[pair_idx], dtype=torch.float32)
        
        # Return based on stepper mode
        if self.stepper_mode == "output":
            return u_in, u_out, c_in, time_diff
        elif self.stepper_mode == "residual":
            u_residual = u_out - u_in
            return u_in, u_residual, c_in, time_diff
        elif self.stepper_mode == "time_der":
            u_time_der = (u_out - u_in) / self.time_diffs[pair_idx]
            return u_in, u_time_der, c_in, time_diff
        else:
            raise ValueError(f"Unsupported stepper_mode: {self.stepper_mode}")


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
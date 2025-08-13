"""
Utility functions for trainers.
Common helper functions used across different trainer implementations.
"""
import os
import random
import torch
import numpy as np
from typing import Dict, List, Any, Optional


def manual_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_ckpt(path, **kwargs):
    """
        Save checkpoint to the path

        Usage:
        >>> save_ckpt("model/poisson_1000.pt", model=model, optimizer=optimizer, scheduler=scheduler)

        Parameters:
        -----------
            path: str
                path to save the checkpoint
            kwargs: dict
                key: str
                    name of the model
                value: StateFul torch object which has the .state_dict() method
                    save object
        
    """
    for k, v in kwargs.items():
        # Examine whether we need to wrap the model
        if isinstance(v, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            kwargs[k] = v.state_dict()  # save the wrapped model, includding the 'module.' prefix
        else:
            kwargs[k] = v.state_dict()
    torch.save(kwargs, path)


def load_ckpt(path, **kwargs):
    """
        Load checkpoint from the path

        Usage:
        >>> model, optimizer, scheduler = load_ckpt("model/poisson_1000.pt", model=model, optimizer=optimizer, scheduler=scheduler)

        Parameters:
        -----------
            path: str
                path to load the checkpoint
            kwargs: dict
                key: str
                    name of the model
                value: StateFul torch object which has the .state_dict() method
                    save object
        Returns:
        --------
            list of torch object
            [model, optimizer, scheduler]
    """
    ckpt = torch.load(path)

    for k, v in kwargs.items():
        state_dict = ckpt[k]
        model_keys = v.state_dict().keys()
        ckpt_keys = state_dict.keys()

        if all(key.startswith('module.') for key in ckpt_keys) and not any(key.startswith('module.') for key in model_keys):
            new_state_dict = {}
            for key in ckpt_keys:
                new_key = key.replace('module.', '', 1)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict
        elif not any(key.startswith('module.') for key in ckpt_keys) and all(key.startswith('module.') for key in model_keys):
            new_state_dict = {}
            for key in ckpt_keys:
                new_key = 'module.' + key
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

        v.load_state_dict(state_dict, strict=False)
    return [i for i in kwargs.values()]


def move_to_device(data, device: torch.device):
    """Recursively move all tensors in a nested structure to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data


def custom_collate_fn(batch):
    """
    Custom collate function for batches with variable graph structures.
    Used for variable coordinate (vx) mode datasets.
    """
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    coords = torch.stack([item[2] for item in batch])
    encoder_graphs = [item[3] for item in batch]
    decoder_graphs = [item[4] for item in batch]
    
    return inputs, labels, coords, encoder_graphs, decoder_graphs


def compute_data_stats(data: torch.Tensor, epsilon: float = 1e-10):
    """
    Compute mean and std statistics for data normalization.
    
    Args:
        data: Input data tensor
        epsilon: Small value to avoid division by zero
        
    Returns:
        tuple: (mean, std) tensors
    """
    data_flat = data.reshape(-1, data.shape[-1])
    mean = torch.mean(data_flat, dim=0)
    std = torch.std(data_flat, dim=0) + epsilon
    return mean, std


def normalize_data(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Normalize data using provided mean and std."""
    return (data - mean) / std


def denormalize_data(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Denormalize data using provided mean and std."""
    return data * std + mean


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially restore weights
            
        Returns:
            bool: True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
            
        return False


def create_directory_structure(path_config):
    """Create directory structure for output paths."""
    paths = [
        path_config.ckpt_path,
        path_config.loss_path,
        path_config.result_path,
        path_config.database_path
    ]
    
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_sequential_stats(u_data: np.ndarray, c_data: Optional[np.ndarray], 
                          t_values: np.ndarray, metadata, max_time_diff: int = 14, time_step: int = 2,
                          sample_rate: float = 1.0, use_metadata_stats: bool = False,
                          use_time_norm: bool = True) -> Dict:
    """
    Compute statistics for sequential data including time features.
    
    Args:
        u_data: Training solution data [n_samples, n_timesteps, n_nodes, n_vars]
        c_data: Training condition data [n_samples, n_timesteps, n_nodes, n_c_vars] or None
        t_values: Time values [n_timesteps]
        metadata: Dataset metadata
        max_time_diff: Maximum time difference for pairs
        sample_rate: Sample rate for statistics computation
        use_metadata_stats: Whether to use metadata-provided statistics
        use_time_norm: Whether to compute time normalization stats
        
    Returns:
        Dict: Statistics dictionary
    """
    
    EPSILON = 1e-10
    stats = {}
    
    # Compute u statistics
    if use_metadata_stats and hasattr(metadata, 'u_mean') and hasattr(metadata, 'u_std'):
        stats["u"] = {
            "mean": np.array(metadata.u_mean),
            "std": np.array(metadata.u_std)
        }
    else:
        u_flat = u_data.reshape(-1, u_data.shape[-1])
        u_mean = np.mean(u_flat, axis=0)
        u_std = np.std(u_flat, axis=0) + EPSILON
        stats["u"] = {"mean": u_mean, "std": u_std}
    
    # Compute c statistics if available
    if c_data is not None:
        if use_metadata_stats and hasattr(metadata, 'c_mean') and hasattr(metadata, 'c_std'):
            stats["c"] = {
                "mean": np.array(metadata.c_mean),
                "std": np.array(metadata.c_std)
            }
        else:
            c_flat = c_data.reshape(-1, c_data.shape[-1])
            c_mean = np.mean(c_flat, axis=0)
            c_std = np.std(c_flat, axis=0) + EPSILON
            stats["c"] = {"mean": c_mean, "std": c_std}
    
    # Compute time-related statistics
    if use_time_norm:
        # Generate time pairs for statistics
        t_in_indices, t_out_indices = [], []
        for lag in range(time_step, max_time_diff + 1, time_step):  # Even lags from time_step to max_time_diff
            for i in range(0, max_time_diff - lag + 1, time_step):
                t_in_indices.append(i)
                t_out_indices.append(i + lag)
        
        t_in_indices = np.array(t_in_indices)
        t_out_indices = np.array(t_out_indices)
        
        start_times = t_values[t_in_indices]
        time_diffs = t_values[t_out_indices] - t_values[t_in_indices]
        
        stats["start_time"] = {
            "mean": np.mean(start_times),
            "std": np.std(start_times) + EPSILON
        }
        stats["time_diffs"] = {
            "mean": np.mean(time_diffs),
            "std": np.std(time_diffs) + EPSILON
        }
    
    # Compute residual and derivative statistics for different stepper modes
    # Residual statistics
    residuals = []
    derivatives = []
    
    n_samples_subset = min(int(len(u_data) * sample_rate), len(u_data))
    u_subset = u_data[:n_samples_subset]
    for sample_idx in range(n_samples_subset):
        for t_idx in range(min(max_time_diff, u_subset.shape[1] - 1)):
            u_curr = u_subset[sample_idx, t_idx]
            u_next = u_subset[sample_idx, t_idx + 1]
            dt = t_values[t_idx + 1] - t_values[t_idx]
            
            residual = u_next - u_curr
            derivative = residual / dt
            
            residuals.append(residual)
            derivatives.append(derivative)
    
    if residuals:
        residuals = np.stack(residuals)
        residuals_flat = residuals.reshape(-1, residuals.shape[-1])
        res_mean = np.mean(residuals_flat, axis=0)
        res_std = np.std(residuals_flat, axis=0) + EPSILON
        stats["res"] = {"mean": res_mean, "std": res_std}
        
        derivatives = np.stack(derivatives)
        derivatives_flat = derivatives.reshape(-1, derivatives.shape[-1])
        der_mean = np.mean(derivatives_flat, axis=0)
        der_std = np.std(derivatives_flat, axis=0) + EPSILON
        stats["der"] = {"mean": der_mean, "std": der_std}
    
    return stats


def get_model_summary(model) -> Dict[str, Any]:
    """Get summary statistics of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }
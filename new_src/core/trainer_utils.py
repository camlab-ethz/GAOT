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


def save_ckpt(model, optimizer, epoch: int, loss: float, path: str):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_ckpt(model, optimizer, path: str, device: torch.device):
    """Load model checkpoint."""
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return None
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.6f})")
    return {'epoch': epoch, 'loss': loss}


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


def get_model_summary(model) -> Dict[str, Any]:
    """Get summary statistics of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }
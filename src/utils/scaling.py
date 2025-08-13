"""
Coordinate scaling utilities for GAOT.
Provides various scaling strategies for coordinate normalization.
"""
import torch
import numpy as np
from typing import Tuple, Union


def rescale(data: torch.Tensor, target_range: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    """
    Rescale data to target range.
    
    Args:
        data: Input tensor to rescale
        target_range: Target (min, max) range
        
    Returns:
        torch.Tensor: Rescaled data
    """
    data_min = torch.min(data, dim=0, keepdim=True)[0]
    data_max = torch.max(data, dim=0, keepdim=True)[0]
    
    # Avoid division by zero
    data_range = data_max - data_min
    data_range = torch.where(data_range == 0, torch.ones_like(data_range), data_range)
    
    # Normalize to [0, 1]
    normalized = (data - data_min) / data_range
    
    # Scale to target range
    target_min, target_max = target_range
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled


class CoordinateScaler:
    """
    Coordinate scaling utility with different scaling strategies.
    """
    
    def __init__(self, target_range: Tuple[float, float] = (-1, 1), 
                 mode: str = "per_dim_scaling"):
        """
        Initialize coordinate scaler.
        
        Args:
            target_range: Target (min, max) range for scaling
            mode: Scaling mode ['global_scaling', 'per_dim_scaling']
        """
        self.target_range = target_range
        self.mode = mode
        self.scale_params = None
        
    def fit(self, coords: torch.Tensor):
        """
        Fit scaling parameters to coordinate data.
        
        Args:
            coords: Coordinate tensor of shape [..., coord_dim]
        """
        if self.mode == "global_scaling":
            # Use global min/max across all dimensions
            global_min = torch.min(coords).item()
            global_max = torch.max(coords).item()
            self.scale_params = {
                'min': torch.tensor([global_min] * coords.shape[-1]),
                'max': torch.tensor([global_max] * coords.shape[-1])
            }
        
        elif self.mode == "per_dim_scaling":
            # Use per-dimension min/max
            coords_flat = coords.view(-1, coords.shape[-1])
            coord_min = torch.min(coords_flat, dim=0)[0]
            coord_max = torch.max(coords_flat, dim=0)[0]
            
            # Avoid division by zero
            coord_range = coord_max - coord_min
            coord_range = torch.where(coord_range == 0, torch.ones_like(coord_range), coord_range)
            
            self.scale_params = {
                'min': coord_min,
                'max': coord_max,
                'range': coord_range
            }
        
        else:
            raise ValueError(f"Unsupported scaling mode: {self.mode}")
    
    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Transform coordinates using fitted scaling parameters.
        
        Args:
            coords: Input coordinates
            
        Returns:
            torch.Tensor: Scaled coordinates
        """
        if self.scale_params is None:
            # Auto-fit if not already fitted
            self.fit(coords)
        original_shape = coords.shape
        coords_flat = coords.view(-1, original_shape[-1])
        
        if self.mode == "global_scaling":
            coord_min = self.scale_params['min'].to(coords.device)
            coord_max = self.scale_params['max'].to(coords.device)
            coord_range = coord_max - coord_min
            coord_range = torch.where(coord_range == 0, torch.ones_like(coord_range), coord_range)
            
        elif self.mode == "per_dim_scaling":
            coord_min = self.scale_params['min'].to(coords.device)
            coord_range = self.scale_params['range'].to(coords.device)
        
        # Normalize to [0, 1]
        normalized = (coords_flat - coord_min) / coord_range
        
        # Scale to target range
        target_min, target_max = self.target_range
        scaled = normalized * (target_max - target_min) + target_min
        
        return scaled.view(original_shape)
    
    def inverse_transform(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform coordinates back to original scale.
        
        Args:
            coords: Scaled coordinates
            
        Returns:
            torch.Tensor: Original scale coordinates
        """
        if self.scale_params is None:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        original_shape = coords.shape
        coords_flat = coords.view(-1, coords.shape[-1])
        
        # Scale back from target range to [0, 1]
        target_min, target_max = self.target_range
        normalized = (coords_flat - target_min) / (target_max - target_min)
        
        if self.mode == "global_scaling":
            coord_min = self.scale_params['min'].to(coords.device)
            coord_max = self.scale_params['max'].to(coords.device)
            coord_range = coord_max - coord_min
            coord_range = torch.where(coord_range == 0, torch.ones_like(coord_range), coord_range)
            
        elif self.mode == "per_dim_scaling":
            coord_min = self.scale_params['min'].to(coords.device)
            coord_range = self.scale_params['range'].to(coords.device)
        
        # Scale back to original range
        original = normalized * coord_range + coord_min
        
        return original.view(original_shape)
    
    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Transform coordinates (callable interface)."""
        return self.transform(coords)


class MinMaxScaler:
    """Simple min-max scaler for general data normalization."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.scale = None
    
    def fit(self, data: torch.Tensor):
        """Fit scaler to data."""
        self.data_min = torch.min(data, dim=0, keepdim=True)[0]
        self.data_max = torch.max(data, dim=0, keepdim=True)[0]
        
        data_range = self.data_max - self.data_min
        data_range = torch.where(data_range == 0, torch.ones_like(data_range), data_range)
        
        feature_min, feature_max = self.feature_range
        self.scale = (feature_max - feature_min) / data_range
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted parameters."""
        if self.scale is None:
            raise ValueError("Scaler must be fitted before transform")
        
        feature_min, _ = self.feature_range
        return (data - self.data_min.to(data.device)) * self.scale.to(data.device) + feature_min
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit and transform data in one step."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform data back to original scale."""
        if self.scale is None:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        feature_min, _ = self.feature_range
        return (data - feature_min) / self.scale.to(data.device) + self.data_min.to(data.device)


class StandardScaler:
    """Standard scaler (z-score normalization)."""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean = None
        self.std = None
    
    def fit(self, data: torch.Tensor):
        """Fit scaler to data."""
        self.mean = torch.mean(data, dim=0, keepdim=True)
        self.std = torch.std(data, dim=0, keepdim=True) + self.epsilon
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        
        return (data - self.mean.to(data.device)) / self.std.to(data.device)
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit and transform data in one step."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform data back to original scale."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        return data * self.std.to(data.device) + self.mean.to(data.device)


def normalize_coordinates(coords: torch.Tensor, method: str = "minmax", 
                         target_range: Tuple[float, float] = (-1, 1)) -> Tuple[torch.Tensor, object]:
    """
    Normalize coordinates using specified method.
    
    Args:
        coords: Input coordinates
        method: Normalization method ['minmax', 'standard']
        target_range: Target range for minmax scaling
        
    Returns:
        tuple: (normalized_coords, scaler_object)
    """
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=target_range)
        normalized = scaler.fit_transform(coords)
    elif method == "standard":
        scaler = StandardScaler()
        normalized = scaler.fit_transform(coords)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized, scaler
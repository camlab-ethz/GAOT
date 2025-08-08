"""
Metrics computation utilities for GAOT.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Union


def compute_relative_error(pred: torch.Tensor, target: torch.Tensor, 
                          epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute relative error between predictions and targets.
    
    Args:
        pred: Predicted values
        target: Target values
        epsilon: Small value to avoid division by zero
        
    Returns:
        torch.Tensor: Relative errors
    """
    target_norm = torch.norm(target, dim=-1, keepdim=True)
    error_norm = torch.norm(pred - target, dim=-1, keepdim=True)
    
    # Avoid division by zero
    target_norm = torch.clamp(target_norm, min=epsilon)
    
    relative_error = error_norm / target_norm
    return relative_error.squeeze(-1)


def compute_batch_errors(targets: torch.Tensor, predictions: torch.Tensor, 
                        metadata, error_type: str = "relative") -> torch.Tensor:
    """
    Compute errors for a batch of predictions.
    
    Args:
        targets: Target tensor [batch_size, ...]
        predictions: Prediction tensor [batch_size, ...]
        metadata: Dataset metadata
        error_type: Type of error to compute ["relative", "absolute", "mse"]
        
    Returns:
        torch.Tensor: Batch of error values [batch_size]
    """
    batch_size = targets.size(0)
    errors = []
    
    for i in range(batch_size):
        target_sample = targets[i]
        pred_sample = predictions[i]
        
        if error_type == "relative":
            error = compute_relative_error(pred_sample, target_sample)
        elif error_type == "absolute":
            error = torch.norm(pred_sample - target_sample, dim=-1)
        elif error_type == "mse":
            error = torch.mean((pred_sample - target_sample) ** 2, dim=-1)
        else:
            raise ValueError(f"Unsupported error type: {error_type}")
        
        # Take mean over spatial dimensions if needed
        if error.dim() > 0:
            error = torch.mean(error)
        
        errors.append(error)
    
    return torch.stack(errors)


def compute_final_metric(errors: torch.Tensor, metric_type: str = "mean") -> float:
    """
    Compute final metric from batch of errors.
    
    Args:
        errors: Tensor of error values
        metric_type: Type of final metric ["mean", "median", "max", "std"]
        
    Returns:
        float: Final metric value
    """
    if metric_type == "mean":
        return torch.mean(errors).item()
    elif metric_type == "median":
        return torch.median(errors).values.item()
    elif metric_type == "max":
        return torch.max(errors).item()
    elif metric_type == "std":
        return torch.std(errors).item()
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")


def compute_multiple_metrics(targets: torch.Tensor, predictions: torch.Tensor) -> Dict[str, float]:
    """
    Compute multiple error metrics.
    
    Args:
        targets: Target tensor
        predictions: Prediction tensor
        
    Returns:
        dict: Dictionary of metric names to values
    """
    metrics = {}
    
    # Mean squared error
    mse = torch.mean((predictions - targets) ** 2).item()
    metrics['mse'] = mse
    
    # Root mean squared error
    metrics['rmse'] = np.sqrt(mse)
    
    # Mean absolute error
    mae = torch.mean(torch.abs(predictions - targets)).item()
    metrics['mae'] = mae
    
    # Relative error
    rel_error = compute_relative_error(predictions, targets)
    metrics['relative_error_mean'] = torch.mean(rel_error).item()
    metrics['relative_error_std'] = torch.std(rel_error).item()
    metrics['relative_error_max'] = torch.max(rel_error).item()
    
    # R² score
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r2 = (1 - ss_res / ss_tot).item()
    metrics['r2'] = r2
    
    return metrics


class MetricTracker:
    """Utility class to track metrics during training."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_dict: Dict[str, float], count: int = 1):
        """Update tracked metrics."""
        for name, value in metric_dict.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += value * count
            self.counts[name] += count
    
    def compute_averages(self) -> Dict[str, float]:
        """Compute average values for all tracked metrics."""
        averages = {}
        for name in self.metrics:
            if self.counts[name] > 0:
                averages[name] = self.metrics[name] / self.counts[name]
            else:
                averages[name] = 0.0
        return averages
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.counts.clear()


def compute_field_statistics(field: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics for a field (useful for analysis).
    
    Args:
        field: Field tensor [..., spatial_dims]
        
    Returns:
        dict: Dictionary of field statistics
    """
    stats = {}
    
    field_flat = field.view(-1)
    
    stats['mean'] = torch.mean(field_flat).item()
    stats['std'] = torch.std(field_flat).item()
    stats['min'] = torch.min(field_flat).item()
    stats['max'] = torch.max(field_flat).item()
    stats['median'] = torch.median(field_flat).item()
    
    # Percentiles
    field_sorted = torch.sort(field_flat)[0]
    n = len(field_sorted)
    stats['p25'] = field_sorted[int(0.25 * n)].item()
    stats['p75'] = field_sorted[int(0.75 * n)].item()
    
    return stats


def compute_spectral_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute spectral metrics (requires 2D spatial fields).
    
    Args:
        pred: Predicted field [batch, height, width, channels]
        target: Target field [batch, height, width, channels]
        
    Returns:
        dict: Spectral metrics
    """
    if pred.dim() != 4 or target.dim() != 4:
        raise ValueError("Spectral metrics require 4D tensors [batch, H, W, channels]")
    
    metrics = {}
    
    batch_size = pred.size(0)
    for i in range(batch_size):
        pred_sample = pred[i, ..., 0]  # Take first channel
        target_sample = target[i, ..., 0]
        
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred_sample)
        target_fft = torch.fft.fft2(target_sample)
        
        # Power spectra
        pred_power = torch.abs(pred_fft) ** 2
        target_power = torch.abs(target_fft) ** 2
        
        # Spectral error
        spectral_error = torch.mean(torch.abs(pred_power - target_power)).item()
        
        if i == 0:
            metrics['spectral_error'] = spectral_error
        else:
            metrics['spectral_error'] += spectral_error
    
    # Average over batch
    metrics['spectral_error'] /= batch_size
    
    return metrics
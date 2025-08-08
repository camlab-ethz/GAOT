"""
Optimizer implementations for GAOT trainers.
"""
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizerArgsConfig:
    """Configuration for optimizer arguments."""
    lr: float = 1e-3                           # Learning rate
    weight_decay: float = 1e-4                 # Weight decay for regularization
    betas: tuple = (0.9, 0.999)               # Beta parameters for Adam optimizers
    eps: float = 1e-8                         # Epsilon for numerical stability
    amsgrad: bool = False                     # Whether to use AMSGrad variant


class AdamOptimizer:
    """Adam optimizer wrapper."""
    
    def __init__(self, parameters, config: OptimizerArgsConfig):
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    
    def step(self):
        """Perform one optimization step."""
        self.optimizer.step()
    
    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Return optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)


class AdamWOptimizer:
    """AdamW optimizer wrapper."""
    
    def __init__(self, parameters, config: OptimizerArgsConfig):
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    
    def step(self):
        """Perform one optimization step."""
        self.optimizer.step()
    
    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Return optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)
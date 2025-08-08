"""
Base trainer class for all GAOT trainers.
Provides common initialization, setup, and utilities.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from abc import ABC, abstractmethod
from typing import Optional

from .default_configs import SetUpConfig, ModelConfig, DatasetConfig, OptimizerConfig, PathConfig, merge_config
from .trainer_utils import manual_seed, load_ckpt, save_ckpt
from ..utils.optimizers import AdamOptimizer, AdamWOptimizer
from ..datasets.dataset import DATASET_METADATA


class BaseTrainer(ABC):
    """
    Base class for all trainers. Defines the core interface and common functionality.
    
    All trainers must implement:
    - init_dataset()
    - init_model() 
    - train_step()
    - validate()
    - test()
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration object containing all settings
        """
        # Store configuration
        self.config = config
        
        # Merge user config with defaults
        self.setup_config = merge_config(SetUpConfig, config.setup)
        self.model_config = merge_config(ModelConfig, config.model)
        self.dataset_config = merge_config(DatasetConfig, config.dataset)
        self.optimizer_config = merge_config(OptimizerConfig, config.optimizer)
        self.path_config = merge_config(PathConfig, config.path)
        
        # Load dataset metadata
        self.metadata = DATASET_METADATA[self.dataset_config.metaname]
        
        # Initialize distributed training if specified
        if self.setup_config.distributed:
            self._init_distributed_mode()
            torch.cuda.set_device(self.setup_config.local_rank)
            self.device = torch.device('cuda', self.setup_config.local_rank)
        else:
            self.device = torch.device(self.setup_config.device)
        
        # Set random seed
        manual_seed(self.setup_config.seed + self.setup_config.rank)
        
        # Set data type
        if self.setup_config.dtype in ["float", "torch.float32", "torch.FloatTensor"]:
            self.dtype = torch.float32
        elif self.setup_config.dtype in ["double", "torch.float64", "torch.DoubleTensor"]:
            self.dtype = torch.float64
        else:
            raise ValueError(f"Invalid dtype: {self.setup_config.dtype}")
        
        # Initialize loss function
        self.loss_fn = nn.MSELoss()
        
        # Initialize components (to be implemented by subclasses)
        self.init_dataset(self.dataset_config)
        self.init_model(self.model_config)
        self.init_optimizer(self.optimizer_config)
        
        # Print model statistics
        if self.setup_config.rank == 0:
            self._print_model_stats()
    
    def _init_distributed_mode(self):
        """Initialize distributed training mode."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.setup_config.rank = int(os.environ['RANK'])
            self.setup_config.world_size = int(os.environ['WORLD_SIZE'])
            self.setup_config.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            print('Not using distributed mode')
            self.setup_config.distributed = False
            self.setup_config.rank = 0
            self.setup_config.world_size = 1
            self.setup_config.local_rank = 0
            return

        dist.init_process_group(
            backend=self.setup_config.backend,
            init_method='env://',
            world_size=self.setup_config.world_size,
            rank=self.setup_config.rank
        )
        dist.barrier()
    
    def _print_model_stats(self):
        """Print model parameter statistics."""
        nparam = sum(
            [p.numel() * 2 if p.is_complex() else p.numel() for p in self.model.parameters()]
        )
        nbytes = sum(
            [p.numel() * 2 * p.element_size() if p.is_complex() else p.numel() * p.element_size() 
             for p in self.model.parameters()]
        )
        print(f"Number of parameters: {nparam}")
        self.config.datarow['nparams'] = nparam
        self.config.datarow['nbytes'] = nbytes
    
    @abstractmethod
    def init_dataset(self, dataset_config):
        """Initialize dataset and data loaders."""
        raise NotImplementedError("Subclasses must implement init_dataset()")
    
    @abstractmethod
    def init_model(self, model_config):
        """Initialize the model."""
        raise NotImplementedError("Subclasses must implement init_model()")
    
    def init_optimizer(self, optimizer_config):
        """Initialize the optimizer."""
        optimizer_map = {
            "adam": AdamOptimizer,
            "adamw": AdamWOptimizer
        }
        
        if optimizer_config.name not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")
        
        self.optimizer = optimizer_map[optimizer_config.name](
            self.model.parameters(), 
            optimizer_config.args
        )
    
    @abstractmethod
    def train_step(self, batch):
        """
        Perform one training step.
        
        Args:
            batch: Batch data from dataloader
            
        Returns:
            torch.Tensor: Loss value
        """
        raise NotImplementedError("Subclasses must implement train_step()")
    
    @abstractmethod
    def validate(self, loader):
        """
        Validate the model on validation set.
        
        Args:
            loader: Validation data loader
            
        Returns:
            float: Average validation loss
        """
        raise NotImplementedError("Subclasses must implement validate()")
    
    @abstractmethod
    def test(self):
        """Test the model and save results."""
        raise NotImplementedError("Subclasses must implement test()")
    
    def save_checkpoint(self, epoch: int, loss: float, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = self.path_config.ckpt_path
        
        save_ckpt(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            path=path
        )
    
    def load_checkpoint(self, path: Optional[str] = None):
        """Load model checkpoint."""
        if path is None:
            path = self.path_config.ckpt_path
        
        return load_ckpt(
            model=self.model,
            optimizer=self.optimizer,
            path=path,
            device=self.device
        )
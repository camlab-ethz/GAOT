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
from ..datasets.dataset import DATASET_METADATA
from ..utils.optimizers import AdamOptimizer, AdamWOptimizer


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
    
    def to(self, device):
        """Move model to device."""
        self.model.to(device)
    
    def type(self, dtype):
        """Set model data type."""
        self.model.type(dtype)

    def load_ckpt(self):
        """Load checkpoint from config path."""
        load_ckpt(self.path_config.ckpt_path, model=self.model)
        return self
    
    def save_ckpt(self):
        """Save checkpoint to config path."""
        os.makedirs(os.path.dirname(self.path_config.ckpt_path), exist_ok=True)
        save_ckpt(self.path_config.ckpt_path, model=self.model)
        return self

    def compute_test_errors(self):
        """Compute test errors (to be implemented by subclasses)."""
        raise NotImplementedError

    def fit(self, verbose=False):
        """
        Train the model with the original training loop.
        This is the complete training logic from the original codebase.
        """
        self.to(self.device)
        
        result = self.optimizer.optimize(self)
        self.config.datarow['training time'] = result['time']
        
        self.save_ckpt()

        if len(result['train']['loss']) == 0:
            self.test()
        else:
            kwargs = {
                "epochs": result['train']['epoch'],
                "losses": result['train']['loss']
            }
        
            if "valid" in result:
                kwargs['val_epochs'] = result['valid']['epoch']
                kwargs['val_losses'] = result['valid']['loss']
            
            if "best" in result:
                kwargs['best_epoch'] = result['best']['epoch']
                kwargs['best_loss'] = result['best']['loss']
            
            self.plot_losses(**kwargs)
            self.test()

    def plot_losses(self, epochs, losses, val_epochs=None, val_losses=None, 
                   best_epoch=None, best_loss=None):
        """Plot training and validation losses."""
        import matplotlib.pyplot as plt
        
        if val_losses is None:
            # plot only train loss
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, losses)
            ax.scatter([best_epoch], [best_loss], c='r', marker='o', label="best loss")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss vs Epoch')
            ax.legend()
            ax.set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax.set_yscale('log')
            np.savez(self.path_config.loss_path[:-4] + ".npz", epochs=epochs, losses=losses)
            plt.savefig(self.path_config.loss_path)
        else:
            # also plot valid loss
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            ax[0].plot(epochs, losses)
            ax[0].scatter([best_epoch], [best_loss], c='r', marker='o', label="best loss")
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].set_title('Loss vs Epoch')
            ax[0].legend()
            ax[0].set_xlim(left=0)
            if (np.array(losses) > 0).all():
                ax[0].set_yscale('log')

            ax[1].plot(val_epochs, val_losses)
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('relative error')
            ax[1].set_title('Loss vs relative error')
            ax[1].legend()
            ax[1].set_xlim(left=0)
            if (np.array(val_losses) > 0).all():
                ax[1].set_yscale('log')
            
            plt.savefig(self.path_config.loss_path)
            np.savez(self.path_config.loss_path[:-4] + ".npz", 
                    epochs=epochs, losses=losses, 
                    val_epochs=val_epochs, val_losses=val_losses)

    def plot_results(self):
        """Plot results (to be implemented by subclasses)."""
        raise NotImplementedError

    def variance_test(self):
        """Variance test (to be implemented by subclasses)."""
        raise NotImplementedError
"""
GAOT Trainer module.
"""
from .static_trainer import StaticTrainer
from .sequential_trainer import SequentialTrainer

__all__ = ['StaticTrainer', 'SequentialTrainer']
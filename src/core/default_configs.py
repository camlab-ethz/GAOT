"""
Default configuration classes for GAOT trainers.
Defines all configurable parameters with sensible defaults.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

from omegaconf import OmegaConf

from ..model.layers.attn import TransformerConfig
from ..model.layers.magno import MAGNOConfig
from ..utils.optimizers import OptimizerArgsConfig


def merge_config(default_config_class, user_config):
    """Merge user configuration with default configuration."""
    default_config_struct = OmegaConf.structured(default_config_class)
    merged_config = OmegaConf.merge(default_config_struct, user_config)
    return OmegaConf.to_object(merged_config)


@dataclass
class SetUpConfig:
    """Setup configuration for training environment."""
    seed: int = 42                                              # Random seed for reproducibility
    device: str = "cuda:0"                                      # Computation device (e.g., "cuda:0", "cpu")
    dtype: str = "torch.float32"                                # Data type for computation
    trainer_name: str = "static"                                # Type of trainer: ["static", "sequential"]
    train: bool = True                                          # Whether to run training phase
    test: bool = False                                          # Whether to run testing phase
    ckpt: bool = False                                          # Whether to load/save checkpoints
    
    # Distributed training parameters
    distributed: bool = False                                   # Enable distributed training
    world_size: int = 1                                         # Total number of processes
    rank: int = 0                                               # Rank of current process
    local_rank: int = 0                                         # Local rank of current process
    backend: str = "nccl"                                       # Backend for distributed training


@dataclass
class ModelArgsConfig:
    """Model arguments configuration."""
    magno: MAGNOConfig = field(default_factory=MAGNOConfig)                     # MAGNO encoder/decoder config
    transformer: TransformerConfig = field(default_factory=TransformerConfig)   # Transformer processor config


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gaot"                                             # Model name: "gaot"
    use_conditional_norm: bool = False                             # Time-conditional normalization
    latent_tokens_size: Tuple[int, int] = (64, 64)                 # Latent token dimensions (H,W) or (H,W,D)
    args: ModelArgsConfig = field(default_factory=ModelArgsConfig) # Model component configurations


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "CE-Gauss"                                      # Dataset name
    metaname: str = "compressible_flow/CE-Gauss"                # Dataset metadata identifier
    base_path: str = "/cluster/work/math/camlab-data/rigno-data/unstructured/"  # Base path to dataset
    train_size: int = 1024                                      # Training set size
    val_size: int = 128                                         # Validation set size
    test_size: int = 256                                        # Test set size
    coord_scaling: str = "per_dim_scaling"                      # Coordinate scaling: ["global_scaling", "per_dim_scaling"]
    batch_size: int = 64                                        # Batch size
    num_workers: int = 4                                        # Number of data loading workers
    shuffle: bool = True                                        # Whether to shuffle training data
    use_metadata_stats: bool = False                            # Use metadata statistics for normalization
    sample_rate: float = 0.1                                    # Sample rate for point clouds
    use_sparse: bool = False                                    # Use sparse representations (PDEGym datasets)
    rand_dataset: bool = False                                  # Randomize dataset sequence
    
    # Time-dependent dataset parameters
    max_time_diff: int = 14                                     # Maximum time difference for pairs
    time_step: int = 2                                          # Time step for sequence data
    use_time_norm: bool = True                                  # Normalize time features
    metric: str = "final_step"                                  # Evaluation metric: ["final_step", "all_step"]
    predict_mode: str = "all"                                   # Inference mode: ["all", "autoregressive", "direct", "star"]
    stepper_mode: str = "output"                                # Stepper mode: ["output", "residual", "time_der"]


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adamw"                                         # Optimizer name: ["adamw", "adam"]
    args: OptimizerArgsConfig = field(default_factory=OptimizerArgsConfig)  # Optimizer arguments


@dataclass
class PathConfig:
    """Path configuration for saving results."""
    ckpt_path: str = ".ckpt/test/test.pt"                       # Model checkpoint path
    loss_path: str = ".loss/test/test.png"                      # Loss curve plot path
    result_path: str = ".result/test/test.png"                  # Result visualization path
    database_path: str = ".database/test/test.csv"              # Experiment database path


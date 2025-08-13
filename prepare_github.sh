#!/bin/bash

# GitHub发布准备脚本
# 这个脚本将当前代码转换为GitHub发布格式

echo "🚀 准备GitHub发布版本..."

# 1. 创建GitHub发布分支
echo "📌 创建github-release分支..."
git checkout -b github-release 2>/dev/null || git checkout github-release

# 2. 删除不需要的文件夹
echo "🗑️  删除test和旧src文件夹..."
rm -rf test/
rm -rf src/

# 3. 重命名new_src为src
echo "📂 重命名new_src为src..."
mv new_src/ src/

# 4. 更新main.py的import路径
echo "🔧 更新main.py..."
# 备份原文件
cp main.py main.py.backup

# 创建新的main.py
cat > main.py << 'EOF'
import numpy as np
import torch
import pandas as pd
import os
import time
import argparse

import toml 
import json
from omegaconf import OmegaConf
from multiprocessing import Pool,Process
import subprocess
import platform
import torch.distributed as dist

# Import unified trainers only
from src.trainer.static_trainer import StaticTrainer
from src.trainer.sequential_trainer import SequentialTrainer

class FileParser:
    def __init__(self, filename):
        if filename.endswith(".toml"):
            with open(filename) as f:
                self.kwargs = OmegaConf.load(f)
        elif filename.endswith(".json"):
            with open(filename) as f:
                self.kwargs = OmegaConf.load(f)
        else:
            raise NotImplementedError(f"File type {filename} not supported, currently only toml and json are supported.")

    def convert_to_absolute_path(self):
        for path_key in ["base_path", "checkpoint_path", "figure_path", "result_path", "database_path"]:
            if hasattr(self.kwargs, "path") and hasattr(self.kwargs.path, path_key):
                if not os.path.isabs(self.kwargs.path[path_key]):
                    self.kwargs.path[path_key] = os.path.abspath(self.kwargs.path[path_key])

def prepare_arg(arg):
    """Prepare argument configuration with proper paths and defaults."""
    arg.convert_to_absolute_path()
    
    # Ensure required directories exist
    os.makedirs(arg.path.base_path, exist_ok=True)
    os.makedirs(arg.path.checkpoint_path, exist_ok=True)
    os.makedirs(arg.path.figure_path, exist_ok=True)
    os.makedirs(arg.path.result_path.rsplit('/', 1)[0], exist_ok=True)
    if hasattr(arg.path, 'database_path'):
        os.makedirs(arg.path.database_path.rsplit('/', 1)[0], exist_ok=True)
    
    # Initialize datarow for storing results
    if not hasattr(arg, 'datarow'):
        arg.datarow = {}
    
    # Initialize default error values
    arg.datarow['relative error (direct)'] = np.nan
    arg.datarow['relative error (auto2)'] = np.nan
    arg.datarow['relative error (auto4)'] = np.nan
    
    return arg

def run_arg(arg):
    arg = prepare_arg(arg)

    # Unified trainer mapping - only new trainers
    Trainer = {
        "static": StaticTrainer,        # Unified static trainer
        "sequential": SequentialTrainer, # Unified sequential trainer
    }[arg.setup["trainer_name"]]
    
    t = Trainer(arg)
    if arg.setup["train"]:
        if arg.setup["ckpt"]:
            t.load_ckpt()
        t.fit()
    if arg.setup["test"]:
        t.test()
        
    # Store results in database if configured
    if hasattr(arg.path, 'database_path'):
        try:
            df = pd.DataFrame([arg.datarow])
            if os.path.exists(arg.path.database_path):
                existing_df = pd.read_csv(arg.path.database_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(arg.path.database_path, index=False)
        except Exception as e:
            print(f"Warning: Could not save to database: {e}")
    
    return arg

def run_single_config(config_file, visible_devices="0", num_workers_per_device=1, debug=False):
    """Run a single configuration file."""
    print(f"🔥 Running config: {config_file}")
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
    
    if debug:
        # Disable multiprocessing in debug mode
        os.environ["GAOT_DEBUG"] = "1"
    
    arg = FileParser(config_file).kwargs
    
    try:
        result = run_arg(arg)
        print(f"✅ Completed: {config_file}")
        return result
    except Exception as e:
        print(f"❌ Failed: {config_file} - Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_folder_configs(folder_path, visible_devices="0", num_workers_per_device=1, debug=False):
    """Run all configuration files in a folder."""
    import glob
    
    config_files = glob.glob(os.path.join(folder_path, "*.json")) + glob.glob(os.path.join(folder_path, "*.toml"))
    
    if not config_files:
        print(f"No configuration files found in {folder_path}")
        return
    
    print(f"Found {len(config_files)} configuration files")
    
    for config_file in sorted(config_files):
        run_single_config(config_file, visible_devices, num_workers_per_device, debug)

def main():
    parser = argparse.ArgumentParser(description="GAOT: Geometry-Aware Operator Transformer")
    parser.add_argument("--config", type=str, help="Path to configuration file (.json or .toml)")
    parser.add_argument("--folder", type=str, help="Path to folder containing configuration files")
    parser.add_argument("--visible_devices", type=str, default="0", 
                       help="CUDA visible devices (e.g., '0' or '0,1,2,3')")
    parser.add_argument("--num_workers_per_device", type=int, default=1,
                       help="Number of worker processes per GPU device")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode (disables multiprocessing)")
    
    args = parser.parse_args()
    
    if args.config and args.folder:
        raise ValueError("Please specify either --config or --folder, not both")
    
    if not args.config and not args.folder:
        raise ValueError("Please specify either --config or --folder")
    
    if args.config:
        run_single_config(args.config, args.visible_devices, args.num_workers_per_device, args.debug)
    elif args.folder:
        run_folder_configs(args.folder, args.visible_devices, args.num_workers_per_device, args.debug)

if __name__ == "__main__":
    main()
EOF

echo "📝 更新README.md..."
# 创建新的README
cat > README.md << 'EOF'
# GAOT: Geometry-Aware Operator Transformer

A PyTorch-based neural operator for solving PDEs on arbitrary domains. GAOT combines Graph Neural Operators with Transformers to achieve both geometric awareness and global context modeling.

## Features

- **Unified Architecture**: Single model supporting both 2D/3D coordinates and fixed/variable coordinate modes
- **Flexible Training**: Supports both time-independent and time-dependent datasets
- **Multi-GPU Support**: Automatic GPU detection and distributed training
- **Rich Visualization**: Static plots and animated visualizations for results
- **Comprehensive Testing**: Full unit test coverage for reliable development

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gaot-private.git
cd gaot-private

# Create virtual environment (recommended)
python -m venv venv-gaot
source venv-gaot/bin/activate  # On Windows: venv-gaot\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```bash
# Train with a configuration file
python main.py --config config/examples/time_indep/poisson_gauss.json

# Run all configs in a folder
python main.py --folder config/examples/time_indep/

# Debug mode (disable multiprocessing)
python main.py --config config.json --debug

# Specify GPU devices
python main.py --config config.json --visible_devices 0,1
```

### Model Types

- **`static`**: Unified trainer for time-independent datasets (automatically detects fx/vx mode)
- **`sequential`**: Unified trainer for time-dependent datasets with autoregressive prediction

### Configuration

All experiments use JSON/TOML configuration files. Key settings:

```json
{
  "setup": {
    "trainer_name": "static",  // or "sequential"
    "train": true,
    "test": true
  },
  "model": {
    "name": "gaot"  // Unified GAOT model
  },
  "dataset": {
    "base_path": "path/to/your/data",
    "name": "your_dataset"
  }
}
```

## Architecture

### Core Components

- **Unified GAOT Model** (`src/model/gaot.py`): Supports both 2D/3D coordinates and fx/vx modes
- **Static Trainer** (`src/trainer/static_trainer.py`): For time-independent problems
- **Sequential Trainer** (`src/trainer/sequential_trainer.py`): For time-dependent problems with autoregressive prediction
- **Data Processing** (`src/datasets/`): Unified data loading and preprocessing pipeline
- **Visualization** (`src/utils/plotting.py`): Advanced plotting and animation capabilities

### Dataset Structure

```
your_dataset_directory/
├── time_indep/
│   ├── Poisson-Gauss.nc
│   ├── naca0012.nc
│   └── ...
└── time_dep/
    ├── ns_gauss.nc
    └── ...
```

## Advanced Usage

### Sequential Data Configuration

```json
{
  "dataset": {
    "max_time_diff": 14,
    "stepper_mode": "output",  // "output", "residual", "time_der"
    "predict_mode": "autoregressive",  // "autoregressive", "direct", "star", "all"
    "metric": "final_step",  // "final_step", "all_step"
    "use_time_norm": true
  }
}
```

### Multi-GPU Training

```bash
# Use multiple GPUs
python main.py --config config.json --visible_devices 0,1,2,3

# Control workers per device
python main.py --config config.json --num_workers_per_device 2
```

## Results

- Static plots are saved as PNG files
- Sequential data generates both static plots and animated GIFs
- All results include proper coordinate scaling and visualization

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gaot2024,
  title={GAOT: Geometry-Aware Operator Transformer for PDE Solving},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/gaot-private}
}
```
EOF

echo "✅ 准备完成！"
echo ""
echo "📋 下一步操作："
echo "1. 检查修改: git status"
echo "2. 提交更改: git add . && git commit -m 'Prepare for GitHub release'"
echo "3. 添加GitHub remote: git remote add github https://github.com/your-username/gaot-private.git"
echo "4. 推送到GitHub: git push github github-release:main"
echo ""
echo "⚠️  记住："
echo "- 配置文件中的trainer_name现在只支持 'static' 和 'sequential'"
echo "- 移除了所有legacy trainers (static_fx, static_vx, sequential_fx)"
echo "- test文件夹已被移除"
echo ""
echo "🔄 要恢复到开发版本: git checkout dev-eth"
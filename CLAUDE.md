# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GAOT (Geometry-Aware Operator Transformer) is a PyTorch-based neural operator for solving PDEs on arbitrary domains. It combines Graph Neural Operators with Transformers to achieve both geometric awareness and global context modeling.

## Development Commands

### Training and Inference
```bash
# Train a model with a configuration file
python main.py --config config/examples/time_indep/poisson_gauss.json

# Run inference (set setup.train: false, setup.test: true in config)
python main.py --config path/to/inference_config.json

# Run all configs in a folder
python main.py --folder config/examples/time_indep/

# Debug mode (disable multiprocessing)
python main.py --config config.json --debug

# Specify GPU devices
python main.py --config config.json --visible_devices 0 1

# Control parallel workers per device
python main.py --config config.json --num_works_per_device 5
```

### Environment Setup
```bash
# Activate existing environment (recommended)
source ~/Documents/Projects/venvs/neuralop/bin/activate

# Or create virtual environment
python -m venv venv-gaot
source venv-gaot/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all unit tests
source ~/Documents/Projects/venvs/neuralop/bin/activate
python -m pytest test/ -v

# Run specific test module
python -m pytest test/model/layers/utils/test_neighbor_search.py -v

# Run tests with coverage
python -m pytest test/ --cov=new_src --cov-report=html
```

## Architecture Overview

### Core Components

**Model Types:**
- `gaot`: Unified GAOT model supporting both 2D/3D coordinates and fx/vx modes
- Legacy: `goat2d_fx` and `goat2d_vx` (deprecated, use unified `gaot` instead)

**Trainer Types:**
- `static`: Unified trainer for time-independent datasets (automatically detects fx/vx mode)
- `static_fx`: Legacy - Time-independent datasets with fixed geometry
- `static_vx`: Legacy - Time-independent datasets with variable geometry  
- `sequential_fx`: Time-dependent datasets with fixed geometry

**Key Modules:**
- `new_src/core/`: Base trainer classes and configuration management
- `new_src/datasets/`: Unified data processing, graph building, and data utilities
- `new_src/trainer/`: Modular trainer implementations (StaticTrainer)
- `new_src/model/`: GAOT model and neural network layers
- `new_src/utils/`: Coordinate scaling, metrics, plotting utilities
- `test/`: Comprehensive unit test suite mirroring new_src structure
- `main.py`: Entry point with multi-GPU support and job orchestration

### Configuration System

All experiments use JSON/TOML configuration files located in `config/`. Key configuration sections:

- `setup.trainer_name`: Trainer type (`static` for unified trainer, legacy: `static_fx`, `static_vx`, `sequential_fx`)
- `model.name`: Model architecture (`gaot` for unified model, legacy: `goat2d_fx`, `goat2d_vx`)
- `dataset.base_path`: Path to NetCDF dataset files
- `dataset.name`: Dataset filename (without .nc extension)
- `path.*`: Output paths for checkpoints, plots, results, database

Default configurations are defined in `new_src/core/default_configs.py`.

### Dataset Structure

Datasets should be organized as:
```
your_base_dataset_directory/
├── time_indep/
│   ├── Poisson-Gauss.nc
│   ├── naca0012.nc
│   └── ...
└── time_dep/
    ├── ns_gauss.nc
    └── ...
```

## Development Patterns

### Adding New Models
1. Create model class in `new_src/model/` following `gaot.py` pattern
2. Add model import and initialization logic to `new_src/model/__init__.py`
3. Update supported models list in `init_model()` function

### Adding New Trainers
1. Inherit from `BaseTrainer` in `new_src/core/base_trainer.py`
2. Implement required methods: `init_dataset()`, `init_model()`, `train_step()`, `validate()`, `test()`
3. Add trainer to the mapping in `main.py:101-106`

### Configuration Management
- Default configurations use dataclasses in `new_src/core/default_configs.py`
- Use `merge_config()` to combine defaults with user configs
- All paths in configurations are converted to absolute paths during initialization

### Multi-GPU and Distributed Training
- Automatic GPU detection and job distribution on Linux systems  
- Process-based parallelism with configurable workers per device
- Distributed training support via PyTorch's distributed package

## Refactoring Progress

The codebase has been successfully refactored with new modular architecture in `new_src/` directory:
- **Goal**: ✅ Completed - Reduced hyperparameters, unified model variants, modular design, full 2D/3D support
- **Testing**: ✅ Comprehensive unit test suite in `test/` directory with complete coverage

### Completed Components

#### Core Architecture
- **Unified GAOT Model**: Combined `goat2d_fx` and `goat2d_vx` into single `gaot.py` supporting:
  - 2D and 3D coordinate spaces with automatic detection
  - Fixed coordinates (fx) and variable coordinates (vx) modes
  - Enhanced processor with 3D patch support
  - Comprehensive unit test coverage for all combinations

#### Trainer System 
- **Unified StaticTrainer**: Merged `StaticTrainer_FX` and `StaticTrainer_VX` into single modular trainer:
  - Automatic coordinate mode detection (fx vs vx)
  - Modular design with reusable components for future sequential trainers
  - Integrated with main.py as `trainer_name: "static"`

#### Dataset Infrastructure
- **DataProcessor**: Unified data loading, preprocessing, and normalization
  - Automatic coordinate mode detection
  - Support for both 2D and 3D coordinates
  - Normalized data pipeline with proper scaling
- **GraphBuilder**: Modular graph construction with neighbor search integration
  - CSR format graph data support
  - Caching capabilities for improved performance
  - Multiple neighbor search strategies
- **Data Utilities**: Custom dataset classes and collate functions
  - Support for variable batch sizes and CSR graph format
  - Comprehensive validation and error handling

#### Neural Network Components
- **Neighbor Search**: All strategies (`native`, `torch_cluster`, `grid`, `chunked`) verified to match reference implementation
- **Edge Drop Mechanism**: Extracted from AGNO to MAGNOEncoder/MAGNODecoder level with shared utility functions
- **MAGNO Layers**: Comprehensive unit tests for encoder and decoder covering 2D/3D, fx/vx modes

#### Testing Suite
- **Complete Unit Test Coverage**: 
  - `test/datasets/`: Data processing, graph building, integration tests with CSR format support
  - `test/model/`: GAOT model tests for all 2D/3D, fx/vx combinations
  - Mock dataset factory for reliable testing without external dependencies
  - Integration tests for complete data pipeline from NetCDF to DataLoaders

## Important Notes

### Migration Guide
- **Use new unified trainer**: Set `trainer_name: "static"` in config instead of `static_fx`/`static_vx`
- **Model architecture**: Use `model.name: "gaot"` for the unified model supporting 2D/3D and fx/vx modes
- **Testing**: Run comprehensive test suite with `python -m pytest test/ -v`

### Architecture Highlights
- **Modular Design**: Clean separation between core, datasets, trainers, model, and utils
- **Automatic Mode Detection**: No need to specify fx/vx mode - automatically detected from data
- **CSR Graph Format**: All graph operations use CSR (Compressed Sparse Row) format for efficiency
- **Comprehensive Testing**: Full unit test coverage with mock datasets for reliable CI/CD
- **2D/3D Support**: Unified codebase supporting both 2D and 3D coordinate spaces

### Legacy Components
- Original `src/` directory maintained for backward compatibility
- Legacy trainers (`static_fx`, `static_vx`) still available but deprecated
- All new development should use `new_src/` architecture
"""
Unified Static Trainer for GAOT.
"""
import torch
from typing import Optional

from ..core.base_trainer import BaseTrainer
from ..core.trainer_utils import move_to_device, denormalize_data
from ..datasets.data_processor import DataProcessor
from ..datasets.graph_builder import GraphBuilder
from ..model.gaot import GAOT
from ..utils.metrics import compute_batch_errors, compute_final_metric
from ..utils.plotting import plot_estimates


class StaticTrainer(BaseTrainer):
    """
    Unified trainer for static (time-independent) problems.
    Automatically handles both fixed and variable coordinate modes.
    """
    
    def __init__(self, config):
        # Initialize data processor
        self.data_processor = None
        self.graph_builder = None
        
        # Coordinate mode and data info
        self.coord_mode = None  # Will be determined from data
        self.coord_dim = None
        self.latent_tokens_coord = None
        self.coord = None  # For fx mode
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        super().__init__(config)
    
    def init_dataset(self, dataset_config):
        """Initialize dataset and data loaders."""
        print("Initializing dataset...")
        
        # Create data processor
        self.data_processor = DataProcessor(
            dataset_config=dataset_config,
            metadata=self.metadata,
            dtype=self.dtype
        )
        
        # Load and process data
        data_splits, is_variable_coords = self.data_processor.load_and_process_data()
        
        # Determine coordinate mode
        self.coord_mode = 'vx' if is_variable_coords else 'fx'
        print(f"Detected coordinate mode: {self.coord_mode}")
        
        # Generate latent queries
        latent_queries = self.data_processor.generate_latent_queries(
            self.model_config.latent_tokens_size
        )
        self.latent_tokens_coord = latent_queries
        
        # Get coordinate dimension
        coord_sample = (data_splits['train']['x'] if is_variable_coords 
                       else data_splits['train']['x'])
        self.coord_dim = coord_sample.shape[-1]
        
        # Store input/output channel information
        c_sample = data_splits['train']['c']
        u_sample = data_splits['train']['u']
        
        self.num_input_channels = c_sample.shape[-1] if c_sample is not None else 0
        self.num_output_channels = u_sample.shape[-1]
        
        if is_variable_coords:
            # Variable coordinates mode - need to build graphs
            self._init_variable_coords_mode(data_splits)
        else:
            # Fixed coordinates mode - simpler setup
            self._init_fixed_coords_mode(data_splits)
        
        print("Dataset initialization complete.")
    
    def _init_variable_coords_mode(self, data_splits):
        """Initialize for variable coordinates mode."""
        print("Setting up variable coordinates mode...")
        
        # Create graph builder
        neighbor_search_method = self.model_config.args.magno.neighbor_search_method
        self.graph_builder = GraphBuilder(neighbor_search_method=neighbor_search_method)
        
        # Get graph building parameters
        gno_radius = getattr(self.model_config.args.magno, 'radius', 0.033)
        scales = getattr(self.model_config.args.magno, 'scales', [1.0])
        
        # Build graphs for all splits
        all_graphs = self.graph_builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=self.latent_tokens_coord,
            gno_radius=gno_radius,
            scales=scales,
            build_train=self.setup_config.train
        )
        
        # Create data loaders with graphs
        loader_kwargs = {
            'encoder_graphs': {
                'train': all_graphs['train']['encoder'] if all_graphs['train'] else None,
                'val': all_graphs['val']['encoder'] if all_graphs['val'] else None,
                'test': all_graphs['test']['encoder']
            },
            'decoder_graphs': {
                'train': all_graphs['train']['decoder'] if all_graphs['train'] else None,
                'val': all_graphs['val']['decoder'] if all_graphs['val'] else None,
                'test': all_graphs['test']['decoder']
            }
        }
        
        loaders = self.data_processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=True,
            latent_queries=self.latent_tokens_coord,
            **loader_kwargs
        )
        
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
    
    def _init_fixed_coords_mode(self, data_splits):
        """Initialize for fixed coordinates mode."""
        print("Setting up fixed coordinates mode...")
        
        # Store fixed coordinates
        self.coord = self.data_processor.coord_scaler(data_splits['train']['x'])
        
        # Create simple data loaders
        loaders = self.data_processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=False
        )
        
        self.train_loader = loaders['train']
        self.val_loader = loaders['val']
        self.test_loader = loaders['test']
    
    def init_model(self, model_config):
        """Initialize the GAOT model."""
        # Update model config with coordinate dimension
        model_config.args.magno.coord_dim = self.coord_dim
        
        self.model = GAOT(
            input_size=self.num_input_channels,
            output_size=self.num_output_channels,
            config=model_config
        )
        
        print(f"Initialized {model_config.name} model with {self.coord_dim}D coordinates")
    
    def train_step(self, batch):
        """Perform one training step."""
        if self.coord_mode == 'fx':
            return self._train_step_fixed_coords(batch)
        else:
            return self._train_step_variable_coords(batch)
    
    def _train_step_fixed_coords(self, batch):
        """Training step for fixed coordinates mode."""
        x_batch, y_batch = batch
        
        # Handle case where c_data might be empty tensor
        if x_batch.numel() == 0:
            # No condition data, create None
            x_batch = None
        
        x_batch = x_batch.to(self.device) if x_batch is not None else None
        y_batch = y_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        coord = self.coord.to(self.device)
        
        pred = self.model(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coord,
            pndata=x_batch
        )
        
        return self.loss_fn(pred, y_batch)
    
    def _train_step_variable_coords(self, batch):
        """Training step for variable coordinates mode."""
        x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
        
        # Handle empty condition data
        if x_batch.numel() == 0:
            x_batch = None
        
        x_batch = x_batch.to(self.device) if x_batch is not None else None
        y_batch = y_batch.to(self.device)
        coord_batch = coord_batch.to(self.device)
        encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
        decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        
        pred = self.model(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coord_batch,
            pndata=x_batch,
            encoder_nbrs=encoder_graph_batch,
            decoder_nbrs=decoder_graph_batch
        )
        
        return self.loss_fn(pred, y_batch)
    
    def validate(self, loader):
        """Validate the model on validation set."""
        if loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                if self.coord_mode == 'fx':
                    loss = self._validate_fixed_coords(batch)
                else:
                    loss = self._validate_variable_coords(batch)
                
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_fixed_coords(self, batch):
        """Validation step for fixed coordinates."""
        x_batch, y_batch = batch
        
        if x_batch.numel() == 0:
            x_batch = None
        
        x_batch = x_batch.to(self.device) if x_batch is not None else None
        y_batch = y_batch.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        coord = self.coord.to(self.device)
        
        pred = self.model(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coord,
            pndata=x_batch
        )
        
        return self.loss_fn(pred, y_batch)
    
    def _validate_variable_coords(self, batch):
        """Validation step for variable coordinates."""
        x_batch, y_batch, coord_batch, encoder_graph_batch, decoder_graph_batch = batch
        
        if x_batch.numel() == 0:
            x_batch = None
        
        x_batch = x_batch.to(self.device) if x_batch is not None else None
        y_batch = y_batch.to(self.device)
        coord_batch = coord_batch.to(self.device)
        encoder_graph_batch = move_to_device(encoder_graph_batch, self.device)
        decoder_graph_batch = move_to_device(decoder_graph_batch, self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        
        pred = self.model(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coord_batch,
            pndata=x_batch,
            encoder_nbrs=encoder_graph_batch,
            decoder_nbrs=decoder_graph_batch
        )
        
        return self.loss_fn(pred, y_batch)
    
    def test(self):
        """Test the model and save results."""
        print("Starting model testing...")
        
        self.model.eval()
        self.model.to(self.device)
        
        all_relative_errors = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if self.coord_mode == 'fx':
                    pred, y_sample, x_sample, coord_used = self._test_step_fixed_coords(batch)
                else:
                    pred, y_sample, x_sample, coord_used = self._test_step_variable_coords(batch)
                
                # Denormalize predictions and targets
                pred_denorm = denormalize_data(pred, self.data_processor.u_mean.to(self.device), 
                                             self.data_processor.u_std.to(self.device))
                y_denorm = denormalize_data(y_sample, self.data_processor.u_mean.to(self.device), 
                                          self.data_processor.u_std.to(self.device))
                
                # Compute relative errors
                relative_errors = compute_batch_errors(y_denorm, pred_denorm, self.metadata)
                all_relative_errors.append(relative_errors)
        
        # Compute final metrics
        all_relative_errors = torch.cat(all_relative_errors, dim=0)
        final_metric = compute_final_metric(all_relative_errors)
        self.config.datarow["relative error (direct)"] = final_metric
        print(f"Relative error: {final_metric}")
        
        # Denormalize input for plotting
        if x_sample is not None and self.data_processor.c_mean is not None:
            x_sample_denorm = denormalize_data(x_sample, self.data_processor.c_mean.to(self.device),
                                             self.data_processor.c_std.to(self.device))
        else:
            x_sample_denorm = x_sample
        
        # Create and save plots
        fig = plot_estimates(
            u_inp=x_sample_denorm[-1].cpu().numpy() if x_sample_denorm is not None else None,
            u_gtr=y_denorm[-1].cpu().numpy(),
            u_prd=pred_denorm[-1].cpu().numpy(),
            x_inp=coord_used.cpu().numpy(),
            x_out=coord_used.cpu().numpy(),
            names=self.metadata.names.get('c', ['input']) if x_sample_denorm is not None else None,
            symmetric=self.metadata.signed['u']
        )
        
        fig.savefig(self.path_config.result_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"Plot saved to {self.path_config.result_path}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def _test_step_fixed_coords(self, batch):
        """Test step for fixed coordinates."""
        x_sample, y_sample = batch
        
        if x_sample.numel() == 0:
            x_sample = None
        
        x_sample = x_sample.to(self.device) if x_sample is not None else None
        y_sample = y_sample.to(self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        coord = self.coord.to(self.device)
        
        pred = self.model(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coord,
            pndata=x_sample
        )
        
        return pred, y_sample, x_sample, coord
    
    def _test_step_variable_coords(self, batch):
        """Test step for variable coordinates."""
        x_sample, y_sample, coord_sample, encoder_graph_sample, decoder_graph_sample = batch
        
        if x_sample.numel() == 0:
            x_sample = None
        
        x_sample = x_sample.to(self.device) if x_sample is not None else None
        y_sample = y_sample.to(self.device)
        coord_sample = coord_sample.to(self.device)
        encoder_graph_sample = move_to_device(encoder_graph_sample, self.device)
        decoder_graph_sample = move_to_device(decoder_graph_sample, self.device)
        latent_tokens_coord = self.latent_tokens_coord.to(self.device)
        
        pred = self.model(
            latent_tokens_coord=latent_tokens_coord,
            xcoord=coord_sample,
            pndata=x_sample,
            encoder_nbrs=encoder_graph_sample,
            decoder_nbrs=decoder_graph_sample
        )
        
        # Use the last sample's coordinates for plotting
        coord_for_plot = coord_sample[-1] if coord_sample.dim() > 2 else coord_sample
        
        return pred, y_sample, x_sample, coord_for_plot
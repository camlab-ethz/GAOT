import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from .layers.attn import Transformer
from .layers.magno import MAGNOEncoder, MAGNODecoder


class GAOT(nn.Module):
    """
    Geometry-Aware Operator Transformer (GAOT) for 2D/3D meshes with fixed or variable coordinates.
    Architecture: MAGNO Encoder + Vision Transformer + MAGNO Decoder
    
    Supports:
    - 2D and 3D coordinate spaces
    - Fixed coordinates (fx) and variable coordinates (vx) modes
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 config: Optional[dataclass] = None):
        nn.Module.__init__(self)
        
        # Validate parameters
        coord_dim = config.args.magno.coord_dim
        if coord_dim not in [2, 3]:
            raise ValueError(f"coord_dim must be 2 or 3, got {coord_dim}")
            
        # --- Define model parameters ---
        self.input_size = input_size
        self.output_size = output_size
        self.coord_dim = coord_dim
        self.node_latent_size = config.args.magno.lifting_channels 
        self.patch_size = config.args.transformer.patch_size
        
        # Get latent token dimensions
        latent_tokens_size = config.latent_tokens_size
        if coord_dim == 2:
            if len(latent_tokens_size) != 2:
                raise ValueError(f"For 2D, latent_tokens_size must have 2 dimensions, got {len(latent_tokens_size)}")
            self.H = latent_tokens_size[0]
            self.W = latent_tokens_size[1]
            self.D = None
        else:  # 3D
            if len(latent_tokens_size) != 3:
                raise ValueError(f"For 3D, latent_tokens_size must have 3 dimensions, got {len(latent_tokens_size)}")
            self.H = latent_tokens_size[0]
            self.W = latent_tokens_size[1] 
            self.D = latent_tokens_size[2]

        # Initialize encoder, processor, and decoder
        self.encoder = self.init_encoder(input_size, self.node_latent_size, config.args.magno)
        self.processor = self.init_processor(self.node_latent_size, config.args.transformer)
        self.decoder = self.init_decoder(output_size, self.node_latent_size, config.args.magno)
    
    def init_encoder(self, input_size, latent_size, config):
        return MAGNOEncoder(
            in_channels=input_size,
            out_channels=latent_size,
            config=config
        )
    
    def init_processor(self, node_latent_size, config):
        # Initialize the Vision Transformer processor
        if self.coord_dim == 2:
            patch_volume = self.patch_size * self.patch_size
        else:  # 3D
            patch_volume = self.patch_size * self.patch_size * self.patch_size
            
        self.patch_linear = nn.Linear(patch_volume * node_latent_size,
                                      patch_volume * node_latent_size)
    
        self.positional_embedding_name = config.positional_embedding
        self.positions = self._get_patch_positions()

        return Transformer(
            input_size=node_latent_size * patch_volume,
            output_size=node_latent_size * patch_volume,
            config=config
        )

    def init_decoder(self, output_size, latent_size, config):
        return MAGNODecoder(
            in_channels=latent_size,
            out_channels=output_size,
            config=config
        )

    def _get_patch_positions(self):
        """
        Generate positional embeddings for the patches.
        """
        P = self.patch_size
        
        if self.coord_dim == 2:
            num_patches_H = self.H // P
            num_patches_W = self.W // P
            positions = torch.stack(torch.meshgrid(
                torch.arange(num_patches_H, dtype=torch.float32),
                torch.arange(num_patches_W, dtype=torch.float32),
                indexing='ij'
            ), dim=-1).reshape(-1, 2)
        else:  # 3D
            num_patches_H = self.H // P
            num_patches_W = self.W // P
            num_patches_D = self.D // P
            positions = torch.stack(torch.meshgrid(
                torch.arange(num_patches_H, dtype=torch.float32),
                torch.arange(num_patches_W, dtype=torch.float32),
                torch.arange(num_patches_D, dtype=torch.float32),
                indexing='ij'
            ), dim=-1).reshape(-1, 3)

        return positions

    def _compute_absolute_embeddings(self, positions, embed_dim):
        """
        Compute absolute embeddings for the given positions.
        """
        num_pos_dims = positions.size(1)
        dim_touse = embed_dim // (2 * num_pos_dims)
        freq_seq = torch.arange(dim_touse, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / dim_touse))
        sinusoid_inp = positions[:, :, None] * inv_freq[None, None, :]
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(positions.size(0), -1)
        return pos_emb

    def encode(self, x_coord: torch.Tensor, 
               pndata: torch.Tensor, 
               latent_tokens_coord: torch.Tensor, 
               encoder_nbrs: list) -> torch.Tensor:
        
        encoded = self.encoder(
            x_coord=x_coord, 
            pndata=pndata,
            latent_tokens_coord=latent_tokens_coord,
            encoder_nbrs=encoder_nbrs)
        
        return encoded

    def process(self, rndata: Optional[torch.Tensor] = None,
                condition: Optional[float] = None
                ) -> torch.Tensor:
        """
        Process regional node data through Vision Transformer.
        
        Parameters
        ----------
        rndata : torch.Tensor
            Regional node data of shape [..., n_regional_nodes, node_latent_size]
        condition : Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            Processed regional node data of same shape
        """
        batch_size = rndata.shape[0]
        n_regional_nodes = rndata.shape[1]
        C = rndata.shape[2]
        P = self.patch_size
        
        if self.coord_dim == 2:
            H, W = self.H, self.W
            
            # Check input shape
            assert n_regional_nodes == H * W, \
                f"n_regional_nodes ({n_regional_nodes}) != H*W ({H}*{W})"
            assert H % P == 0 and W % P == 0, \
                f"H({H}) and W({W}) must be divisible by P({P})"

            # Reshape to 2D patches
            num_patches_H = H // P
            num_patches_W = W // P
            
            # Reshape to patches: [batch, H, W, C] -> [batch, num_patches, P*P*C]
            rndata = rndata.view(batch_size, H, W, C)
            rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, C)
            rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
            rndata = rndata.view(batch_size, num_patches_H * num_patches_W, P * P * C)
            
        else:  # 3D
            H, W, D = self.H, self.W, self.D
            
            # Check input shape
            assert n_regional_nodes == H * W * D, \
                f"n_regional_nodes ({n_regional_nodes}) != H*W*D ({H}*{W}*{D})"
            assert H % P == 0 and W % P == 0 and D % P == 0, \
                f"H({H}), W({W}), D({D}) must be divisible by P({P})"

            # Reshape to 3D patches
            num_patches_H = H // P
            num_patches_W = W // P
            num_patches_D = D // P
            
            # Reshape to patches: [batch, H*W*D, C] -> [batch, num_patches, P*P*P*C]
            rndata = rndata.view(batch_size, H, W, D, C)
            rndata = rndata.view(batch_size, num_patches_H, P, num_patches_W, P, num_patches_D, P, C)
            rndata = rndata.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            rndata = rndata.view(batch_size, num_patches_H * num_patches_W * num_patches_D, P * P * P * C)
        
        # Apply patch linear transformation
        rndata = self.patch_linear(rndata)
        pos = self.positions.to(rndata.device)
        
        # Apply positional encoding
        if self.positional_embedding_name == 'absolute':
            patch_volume = P ** self.coord_dim
            pos_emb = self._compute_absolute_embeddings(pos, patch_volume * self.node_latent_size)
            rndata = rndata + pos_emb
            relative_positions = None
        elif self.positional_embedding_name == 'rope':
            relative_positions = pos
        
        # Apply transformer processor
        rndata = self.processor(rndata, condition=condition, relative_positions=relative_positions)
        
        # Reshape back to original regional nodes format
        if self.coord_dim == 2:
            rndata = rndata.view(batch_size, num_patches_H, num_patches_W, P, P, C)
            rndata = rndata.permute(0, 1, 3, 2, 4, 5).contiguous()
            rndata = rndata.view(batch_size, H * W, C)
        else:  # 3D
            rndata = rndata.view(batch_size, num_patches_H, num_patches_W, num_patches_D, P, P, P, C)
            rndata = rndata.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
            rndata = rndata.view(batch_size, H * W * D, C)

        return rndata

    def decode(self, latent_tokens_coord: torch.Tensor, 
               rndata: torch.Tensor, 
               query_coord: torch.Tensor, 
               decoder_nbrs: list) -> torch.Tensor:
        
        decoded = self.decoder(
            latent_tokens_coord=latent_tokens_coord,
            rndata=rndata, 
            query_coord=query_coord,
            decoder_nbrs=decoder_nbrs)
        
        return decoded

    def forward(self,
                latent_tokens_coord: torch.Tensor,
                xcoord: torch.Tensor,
                pndata: torch.Tensor,
                query_coord: Optional[torch.Tensor] = None,
                encoder_nbrs: Optional[list] = None,
                decoder_nbrs: Optional[list] = None,
                condition: Optional[float] = None,
                ) -> torch.Tensor:
        """
        Forward pass for GAOT model.

        Parameters
        ----------
        latent_tokens_coord : torch.Tensor
            Regional node coordinates of shape [n_regional_nodes, coord_dim] (fx mode)
            or [batch_size, n_regional_nodes, coord_dim] (vx mode)
        xcoord : torch.Tensor
            Physical node coordinates of shape [n_physical_nodes, coord_dim] (fx mode) 
            or [batch_size, n_physical_nodes, coord_dim] (vx mode)
        pndata : torch.Tensor
            Physical node data of shape [batch_size, n_physical_nodes, input_size]
        query_coord : Optional[torch.Tensor]
            Query coordinates for output, defaults to xcoord
        encoder_nbrs : Optional[list]
            Precomputed neighbors for encoder
        decoder_nbrs : Optional[list] 
            Precomputed neighbors for decoder
        condition : Optional[float]
            Conditioning value for the model

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, n_query_nodes, output_size]
        """
        # Encode: Map physical nodes to regional nodes
        rndata = self.encode(
            x_coord=xcoord, 
            pndata=pndata,
            latent_tokens_coord=latent_tokens_coord,
            encoder_nbrs=encoder_nbrs)
        
        # Process: Apply Vision Transformer on regional nodes
        rndata = self.process(
            rndata=rndata, 
            condition=condition)

        # Decode: Map regional nodes back to query nodes
        if query_coord is None:
            query_coord = xcoord
        output = self.decode(
            latent_tokens_coord=latent_tokens_coord,
            rndata=rndata, 
            query_coord=query_coord,
            decoder_nbrs=decoder_nbrs)

        return output
    
    def autoregressive_predict(self,
                             x_batch: torch.Tensor,
                             time_indices: np.ndarray,
                             t_values: np.ndarray,
                             stats: Dict,
                             stepper_mode: str = "output",
                             latent_tokens_coord: Optional[torch.Tensor] = None,
                             fixed_coord: Optional[torch.Tensor] = None,
                             encoder_nbrs: Optional[List] = None,
                             decoder_nbrs: Optional[List] = None,
                             use_conditional_norm: bool = False) -> torch.Tensor:
        """
        Autoregressive prediction for sequential data.
        
        Args:
            x_batch: Initial input batch at time t=0 [batch_size, num_nodes, input_dim]
            time_indices: Array of time indices for prediction [num_timesteps]
            t_values: Actual time values corresponding to indices [num_timesteps_total]
            stats: Statistics dictionary containing normalization parameters
            stepper_mode: Prediction mode ['output', 'residual', 'time_der']
            latent_tokens_coord: Regional node coordinates for transformer
            fixed_coord: Fixed coordinates for fx mode [num_nodes, coord_dim]
            encoder_nbrs: Encoder neighbor graphs (for vx mode)
            decoder_nbrs: Decoder neighbor graphs (for vx mode)
            use_conditional_norm: Whether to use conditional normalization
            
        Returns:
            Predicted outputs over time [batch_size, num_timesteps-1, num_nodes, output_dim]
        """
        batch_size, num_nodes, input_dim = x_batch.shape
        num_timesteps = len(time_indices)
        predictions = []
        # Extract statistics for denormalization
        u_mean = stats["u"]["mean"].to(x_batch.device)
        u_std = stats["u"]["std"].to(x_batch.device)
        
        # Time statistics
        start_times_mean = stats["start_time"]["mean"]
        start_times_std = stats["start_time"]["std"]
        time_diffs_mean = stats["time_diffs"]["mean"]
        time_diffs_std = stats["time_diffs"]["std"]
        
        # Determine feature dimensions
        u_dim = stats["u"]["mean"].shape[0]
        c_dim = stats["c"]["mean"].shape[0] if "c" in stats else 0
        
        # Extract static condition features and initial state
        if c_dim > 0:
            c_features = x_batch[..., u_dim:u_dim+c_dim]  # Static condition features
        else:
            c_features = None
        
        current_u = x_batch[..., :u_dim]  # Initial state
        
        # Determine if we're using variable coordinates
        is_variable_coords = encoder_nbrs is not None and decoder_nbrs is not None
        
        # Autoregressive prediction loop
        for idx in range(1, num_timesteps):
            t_in_idx = time_indices[idx-1]
            t_out_idx = time_indices[idx]
            
            start_time = t_values[t_in_idx]
            time_diff = t_values[t_out_idx] - t_values[t_in_idx]
            
            # Normalize time features
            start_time_norm = (start_time - start_times_mean) / start_times_std
            time_diff_norm = (time_diff - time_diffs_mean) / time_diffs_std
            
            # Prepare time features (expanded to match num_nodes)
            start_time_expanded = torch.full((batch_size, num_nodes, 1), start_time_norm, 
                                           dtype=x_batch.dtype, device=x_batch.device)
            time_diff_expanded = torch.full((batch_size, num_nodes, 1), time_diff_norm,
                                          dtype=x_batch.dtype, device=x_batch.device)
            
            # Build input features
            input_features = [current_u]
            if c_features is not None:
                input_features.append(c_features)
            input_features.extend([start_time_expanded, time_diff_expanded])
            
            x_input = torch.cat(input_features, dim=-1)
            
            # Forward pass
            with torch.no_grad():
                if use_conditional_norm:
                    if is_variable_coords:
                        pred = self.forward(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=fixed_coord,  # Will be updated for vx mode
                            pndata=x_input[..., :-1],
                            condition=x_input[..., 0, -2:-1],
                            encoder_nbrs=encoder_nbrs,
                            decoder_nbrs=decoder_nbrs
                        )
                    else:
                        pred = self.forward(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=fixed_coord,
                            pndata=x_input[..., :-1],
                            condition=x_input[..., 0, -2:-1]
                        )
                else:
                    if is_variable_coords:
                        pred = self.forward(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=fixed_coord,  # Will be updated for vx mode
                            pndata=x_input,
                            encoder_nbrs=encoder_nbrs,
                            decoder_nbrs=decoder_nbrs
                        )
                    else:
                        pred = self.forward(
                            latent_tokens_coord=latent_tokens_coord,
                            xcoord=fixed_coord,
                            pndata=x_input
                        )
                
                # Process prediction based on stepper mode
                pred_denorm = self._process_autoregressive_prediction(
                    pred, current_u, u_mean, u_std, time_diff, stats, stepper_mode
                )
                predictions.append(pred_denorm)
                
                # Update current state for next iteration
                current_u = (pred_denorm - u_mean) / u_std
        
        return torch.stack(predictions, dim=1)  # [batch_size, num_timesteps-1, num_nodes, output_dim]
    
    def _process_autoregressive_prediction(self, pred: torch.Tensor, current_u: torch.Tensor, 
                                         u_mean: torch.Tensor, u_std: torch.Tensor, 
                                         time_diff: float, stats: Dict, stepper_mode: str) -> torch.Tensor:
        """
        Process prediction based on stepper mode for autoregressive prediction.
        
        Args:
            pred: Raw model prediction
            current_u: Current normalized state
            u_mean: Mean for denormalization
            u_std: Std for denormalization
            time_diff: Time difference for this step
            stats: Statistics dictionary
            stepper_mode: Prediction mode ['output', 'residual', 'time_der']
            
        Returns:
            Denormalized prediction
        """
        if stepper_mode == "output":
            pred_denorm = pred * u_std + u_mean
            
        elif stepper_mode == "residual":
            res_mean = stats["res"]["mean"].to(pred.device)
            res_std = stats["res"]["std"].to(pred.device)
            pred_denorm_res = pred * res_std + res_mean
            
            current_u_denorm = current_u * u_std + u_mean
            pred_denorm = current_u_denorm + pred_denorm_res
            
        elif stepper_mode == "time_der":
            der_mean = stats["der"]["mean"].to(pred.device)
            der_std = stats["der"]["std"].to(pred.device)
            pred_denorm_der = pred * der_std + der_mean
            
            current_u_denorm = current_u * u_std + u_mean
            time_diff_tensor = torch.tensor(time_diff, dtype=pred.dtype, device=pred.device)
            pred_denorm = current_u_denorm + time_diff_tensor * pred_denorm_der
            
        else:
            raise ValueError(f"Unsupported stepper_mode: {stepper_mode}")
        
        return pred_denorm
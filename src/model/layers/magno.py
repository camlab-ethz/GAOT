"""
Unified MAGNO (Multiscale Attentional Graph Neural Operator) implementation
Supporting both 2D and 3D coordinates with flexible batch processing modes.

This module is a flexible implementation that can handle:
- 2D and 3D coordinate spaces
- Fixed coordinates (fx mode) and variable coordinates (vx mode) 
- Efficient caching and neighbor computation
- Edge drop/sampling
"""

import torch
import torch.nn as nn
from typing import List, Optional, Literal, Union
from dataclasses import dataclass, field

from .mlp import ChannelMLP
from .utils.neighbor_search import NeighborSearch
from .utils.edge_drop import apply_edge_drop_csr
from .gemb import GeometricEmbedding, node_pos_encode
from .agno import AGNO

############
# MAGNO Config
############
@dataclass
class MAGNOConfig:
    """Simplified MAGNO Configuration with reduced parameters"""
    
    # --- Core Parameters ---
    coord_dim: int = 2                                      # Coordinate dimension (2 for 2D, 3 for 3D)
    radius: float = 0.033                                   # Radius for neighbor search
    hidden_size: int = 64                                   # Base hidden size for all MLPs
    mlp_layers: int = 3                                     # Number of MLP layers (consistent across all MLPs)
    lifting_channels: int = 32                              # Number of channels after Encoder

    
    # --- Multi-scale ---
    scales: List[float] = field(default_factory=lambda: [1.0])  # Multi-scale factors
    use_scale_weights: bool = False                         # Whether to use learnable scale weights
    
    # --- Attention and Embedding ---
    use_attention: bool = True                              # Enable attention mechanism  
    attention_type: str= 'cosine'                           # Attention type, support ['cosine', 'dot_product']
    use_geoembed: bool = True                              # Enable geometric embedding
    embedding_method: str = 'statistical'                   # Geometric embedding method, support ['statistical', 'pointnet']
    pooling: str = 'max'                                    # Pooling for pointnet embedding, support ['max', 'mean', 'sum']
    
    # --- Transform and Sampling ---
    transform_type: str = 'linear'                          # Transform type for both encoder and decoder, support ['linear', 'nonlinear']
    sampling_strategy: Optional[str] = None                 # Edge sampling strategy, support ['max_neighbors', 'ratio']
    max_neighbors: Optional[int] = None                     # Max neighbors for sampling
    sample_ratio: Optional[float] = None                    # Sample ratio for edge drop
    
    # --- Advanced ---
    node_embedding: bool = False                            # Use positional node embedding
    neighbor_search_method: str = 'auto'                   # Neighbor search method, support ['open3d', 'torch_cluster', 'grid', 'chunked', 'native']
    use_torch_scatter: bool = True                          # Use torch_scatter backend
    neighbor_strategy: str = 'radius'                      # Neighbor search strategy, support ['radius', 'knn']
    precompute_edges: bool = False                          # Whether edges are precomputed
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.coord_dim not in [2, 3]:
            raise ValueError(f"coord_dim must be 2 or 3, got {self.coord_dim}")
        if self.sampling_strategy == 'ratio' and (self.sample_ratio is None or not 0 < self.sample_ratio <= 1):
            raise ValueError("sample_ratio must be in (0, 1] when using 'ratio' sampling")
        if self.sampling_strategy == 'max_neighbors' and (self.max_neighbors is None or self.max_neighbors <= 0):
            raise ValueError("max_neighbors must be > 0 when using 'max_neighbors' sampling")

############
# Unified MAGNO Encoder
############
class MAGNOEncoder(nn.Module):
    """
    Unified MAGNO Encoder supporting both 2D/3D and fixed/variable coordinate modes.
    
    Coordinate Modes:
    - Fixed (fx): All batches share the same coordinate layout - more memory/compute efficient
    - Variable (vx): Each batch can have different coordinates - more flexible
    
    Dimension Support:  
    - 2D: Traditional x,y coordinates for 2D problems
    - 3D: Full x,y,z coordinates for 3D problems
    """
    
    def __init__(self, in_channels: int, out_channels: int, config: MAGNOConfig):
        super().__init__()
        
        # --- Store configuration ---
        self.config = config
        self.coord_dim = config.coord_dim
        self.scales = config.scales
        self.use_scale_weights = config.use_scale_weights
        self.precompute_edges = config.precompute_edges
        self.use_geoembed = config.use_geoembed
        self.node_embedding = config.node_embedding
        
        # --- Initialize neighbor search ---
        self.nb_search = NeighborSearch(method = config.neighbor_search_method)
        self.neighbor_cache = {}  # Unified caching for both fx and vx modes
        
        # --- Store edge drop parameters ---
        self.sampling_strategy = config.sampling_strategy
        self.max_neighbors = config.max_neighbors
        self.sample_ratio = config.sample_ratio
        
        # --- Determine kernel input dimension ---
        kernel_coord_dim = self._compute_kernel_coord_dim()
        kernel_input_dim = kernel_coord_dim * 2  # query + key coordinates
        
        if config.transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            kernel_input_dim += in_channels
        
        # --- Build MLP layer sizes ---
        mlp_sizes = [kernel_input_dim] + [config.hidden_size] * config.mlp_layers + [out_channels]
        
        # --- Initialize core modules ---
        self.agno = AGNO(
            channel_mlp_layers=mlp_sizes,
            transform_type=config.transform_type,
            use_attn=config.use_attention,
            attention_type=config.attention_type,
            coord_dim=kernel_coord_dim,
            use_torch_scatter=config.use_torch_scatter
        )
        
        self.lifting = ChannelMLP(
            in_channels=in_channels,
            hidden_channels=config.hidden_size,
            out_channels=out_channels,
            n_layers=1
        )
        
        # --- Optional geometric embedding ---
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=out_channels,
                method=config.embedding_method,
                pooling=config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                n_layers=1
            )
        
        # --- Optional scale weighting ---
        if self.use_scale_weights:
            self.scale_weighting = nn.Sequential(
                nn.Linear(kernel_coord_dim, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, len(self.scales))
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)
    
    def _compute_kernel_coord_dim(self) -> int:
        """Compute the effective coordinate dimension for kernel computation"""
        coord_dim = self.coord_dim
        if self.node_embedding:
            coord_dim = self.coord_dim * 4 * 2  # From node_pos_encode function
        return coord_dim
    
    def _detect_coordinate_mode(self, x_coord: torch.Tensor) -> Literal['fx', 'vx']:
        """Automatically detect coordinate mode based on input shape"""
        if x_coord.ndim == 2:
            return 'fx'  # Fixed coordinates: [num_nodes, coord_dim]
        elif x_coord.ndim == 3:
            return 'vx'  # Variable coordinates: [batch_size, num_nodes, coord_dim]
        else:
            raise ValueError(f"x_coord must be 2D or 3D tensor, got shape {x_coord.shape}")
    
    def _compute_neighbors(self, x_coord: torch.Tensor, latent_coord: torch.Tensor, 
                          mode: Literal['fx', 'vx']) -> List:
        """Compute neighbor lists with caching support"""
        cache_key = f"{mode}_{x_coord.shape}_{latent_coord.shape}_{tuple(self.scales)}"
        
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]
        
        neighbors_per_scale = []
        
        if mode == 'fx':
            # Fixed coordinates - compute once for all batches
            for scale in self.scales:
                scaled_radius = self.config.radius * scale
                neighbors = self.nb_search(
                    data=x_coord,
                    queries=latent_coord,
                    radius=scaled_radius
                )
                neighbors_per_scale.append(neighbors)
        
        else:  # mode == 'vx'
            # Variable coordinates - compute per batch
            batch_size = x_coord.shape[0]
            neighbors_per_batch = []
            
            for b in range(batch_size):
                neighbors_per_scale_batch = []
                for scale in self.scales:
                    scaled_radius = self.config.radius * scale
                    neighbors = self.nb_search(
                        data=x_coord[b],  # Current batch coordinates
                        queries=latent_coord,
                        radius=scaled_radius
                    )
                    neighbors_per_scale_batch.append(neighbors)
                neighbors_per_batch.append(neighbors_per_scale_batch)
            neighbors_per_scale = neighbors_per_batch
        
        # Cache the result
        self.neighbor_cache[cache_key] = neighbors_per_scale
        return neighbors_per_scale
    
    def forward(self, 
                x_coord: torch.Tensor, 
                pndata: torch.Tensor,
                latent_tokens_coord: torch.Tensor, 
                encoder_nbrs: Optional[Union[List, List[List]]] = None) -> torch.Tensor:
        """
        Unified forward pass supporting both coordinate modes and dimensions.
        
        Parameters
        ----------
        x_coord : torch.Tensor
            Physical node coordinates
            - fx mode: [num_nodes, coord_dim] - shared across batch
            - vx mode: [batch_size, num_nodes, coord_dim] - per batch
        pndata : torch.Tensor, shape [batch_size, num_nodes, in_channels]
            Physical node features
        latent_tokens_coord : torch.Tensor, shape [num_latent_nodes, coord_dim]
            Target latent grid coordinates
        encoder_nbrs : Optional neighbor lists
            Precomputed neighbors if precompute_edges=True
            
        Returns
        -------
        torch.Tensor, shape [batch_size, num_latent_nodes, out_channels]
            Encoded features on latent grid
        """
        
        # --- Auto-detect coordinate mode ---
        coord_mode = self._detect_coordinate_mode(x_coord)
        batch_size = pndata.shape[0]
        
        # --- Validate inputs based on detected mode ---
        if coord_mode == 'fx':
            if x_coord.shape[1] != self.coord_dim:
                raise ValueError(f"Expected x_coord shape [num_nodes, {self.coord_dim}], got {x_coord.shape}")
            num_nodes = x_coord.shape[0]
        else:  # vx mode
            if x_coord.shape[0] != batch_size or x_coord.shape[2] != self.coord_dim:
                raise ValueError(f"Expected x_coord shape [{batch_size}, num_nodes, {self.coord_dim}], got {x_coord.shape}")
            num_nodes = x_coord.shape[1]
        
        if pndata.shape[:2] != (batch_size, num_nodes):
            raise ValueError(f"pndata shape mismatch: expected [{batch_size}, {num_nodes}, in_channels], got {pndata.shape}")
        
        if latent_tokens_coord.shape[1] != self.coord_dim:
            raise ValueError(f"Expected latent_tokens_coord shape [num_latent, {self.coord_dim}], got {latent_tokens_coord.shape}")
        
        # --- Compute or use precomputed neighbors ---
        if self.precompute_edges:
            if encoder_nbrs is None:
                raise ValueError("encoder_nbrs required when precompute_edges=True")
            neighbors_per_scale = encoder_nbrs
        else:
            neighbors_per_scale = self._compute_neighbors(x_coord, latent_tokens_coord, coord_mode)
        
        # --- Lift input features ---
        pndata = pndata.permute(0, 2, 1)  # [batch, channels, nodes] for ChannelMLP
        pndata = self.lifting(pndata).permute(0, 2, 1)  # Back to [batch, nodes, channels]
        
        # --- Prepare scale weights if enabled ---
        if self.use_scale_weights:
            scale_weights = self.scale_weighting(latent_tokens_coord)  # [num_latent, num_scales]
            scale_weights = self.scale_weight_activation(scale_weights)
        
        # --- Process each scale ---
        if coord_mode == 'fx':
            encoded_scales = self._forward_fx_mode(
                x_coord, pndata, latent_tokens_coord, neighbors_per_scale
            )
        else:
            encoded_scales = self._forward_vx_mode(
                x_coord, pndata, latent_tokens_coord, neighbors_per_scale
            )
        
        # --- Combine scales ---
        if len(encoded_scales) == 1:
            encoded = encoded_scales[0]
        else:
            if self.use_scale_weights:
                # Weighted combination
                encoded = torch.zeros_like(encoded_scales[0])
                for i, enc in enumerate(encoded_scales):
                    weights = scale_weights[:, i:i+1].unsqueeze(0)  # [1, num_latent, 1]
                    encoded += weights * enc
            else:
                # Simple average
                encoded = torch.stack(encoded_scales, dim=0).mean(dim=0)
        
        return encoded
    
    def _forward_fx_mode(self, x_coord, pndata, latent_coord, neighbors_per_scale):
        """Forward pass for fixed coordinate mode"""
        batch_size = pndata.shape[0]
        encoded_scales = []
        
        for _, neighbors in enumerate(neighbors_per_scale):
            # Apply edge drop before passing to AGNO and geoembed
            neighbors_dropped = apply_edge_drop_csr(
                neighbors, 
                self.sampling_strategy, 
                self.max_neighbors, 
                self.sample_ratio, 
                self.training
            )
            
            # Prepare coordinates for kernel
            if self.node_embedding:
                phys_coord = node_pos_encode(x_coord)
                latent_coord_processed = node_pos_encode(latent_coord)
            else:
                phys_coord = x_coord
                latent_coord_processed = latent_coord
            
            # Apply AGNO for current scale
            encoded = self.agno(
                y=phys_coord,
                x=latent_coord_processed,
                f_y=pndata,
                neighbors=neighbors_dropped
            )
            
            # Apply geometric embedding if enabled
            if self.use_geoembed:
                geoembedding = self.geoembed(
                    input_geom=x_coord,
                    latent_queries=latent_coord,
                    spatial_nbrs=neighbors_dropped
                )
                # Expand to batch size and concatenate
                geoembedding = geoembedding.unsqueeze(0).expand(batch_size, -1, -1)
                encoded = torch.cat([encoded, geoembedding], dim=-1)
                # Apply recovery MLP
                encoded = encoded.permute(0, 2, 1)
                encoded = self.recovery(encoded).permute(0, 2, 1)
            
            encoded_scales.append(encoded)
        
        return encoded_scales
    
    def _forward_vx_mode(self, x_coord, pndata, latent_coord, neighbors_per_batch):
        """Forward pass for variable coordinate mode"""
        batch_size = x_coord.shape[0]
        encoded_scales = []
        
        for scale_idx in range(len(self.scales)):
            encoded_batch = []
            
            for b in range(batch_size):
                x_b = x_coord[b]  # [num_nodes, coord_dim]
                pndata_b = pndata[b:b+1]  # [1, num_nodes, in_channels]
                neighbors_b = neighbors_per_batch[b][scale_idx]
                
                # Apply edge drop before passing to AGNO and geoembed
                neighbors_b_dropped = apply_edge_drop_csr(
                    neighbors_b, 
                    self.sampling_strategy, 
                    self.max_neighbors, 
                    self.sample_ratio, 
                    self.training
                )
                
                # Prepare coordinates for kernel
                if self.node_embedding:
                    phys_coord = node_pos_encode(x_b)
                    latent_coord_processed = node_pos_encode(latent_coord)
                else:
                    phys_coord = x_b
                    latent_coord_processed = latent_coord
                
                # Apply AGNO for current batch and scale
                encoded_b = self.agno(
                    y=phys_coord,
                    x=latent_coord_processed,
                    f_y=pndata_b,
                    neighbors=neighbors_b_dropped
                )
                
                # Apply geometric embedding if enabled
                if self.use_geoembed:
                    geoembedding = self.geoembed(
                        input_geom=x_b,
                        latent_queries=latent_coord,
                        spatial_nbrs=neighbors_b_dropped
                    )
                    geoembedding = geoembedding.unsqueeze(0)  # Add batch dim
                    encoded_b = torch.cat([encoded_b, geoembedding], dim=-1)
                    # Apply recovery MLP
                    encoded_b = encoded_b.permute(0, 2, 1)
                    encoded_b = self.recovery(encoded_b).permute(0, 2, 1)
                
                encoded_batch.append(encoded_b)
            
            # Stack batch results
            encoded_scale = torch.cat(encoded_batch, dim=0)
            encoded_scales.append(encoded_scale)
        
        return encoded_scales

############
# Unified MAGNO Decoder
############
class MAGNODecoder(nn.Module):
    """
    Unified MAGNO Decoder supporting both 2D/3D and fixed/variable coordinate modes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, config: MAGNOConfig):
        super().__init__()
        
        # --- Store configuration ---
        self.config = config
        self.coord_dim = config.coord_dim
        self.scales = config.scales
        self.use_scale_weights = config.use_scale_weights
        self.precompute_edges = config.precompute_edges
        self.use_geoembed = config.use_geoembed
        self.node_embedding = config.node_embedding
        
        # --- Initialize neighbor search ---
        self.nb_search = NeighborSearch(method = config.neighbor_search_method)
        self.neighbor_cache = {}
        
        # --- Store edge drop parameters ---
        self.sampling_strategy = config.sampling_strategy
        self.max_neighbors = config.max_neighbors
        self.sample_ratio = config.sample_ratio
        
        # --- Determine kernel input dimension ---
        kernel_coord_dim = self._compute_kernel_coord_dim()
        kernel_input_dim = kernel_coord_dim * 2  # query + key coordinates
        
        if config.transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            kernel_input_dim += in_channels
        
        # --- Build MLP layer sizes ---
        mlp_sizes = [kernel_input_dim] + [config.hidden_size] * config.mlp_layers + [in_channels]
        
        # --- Initialize core modules ---
        self.agno = AGNO(
            channel_mlp_layers=mlp_sizes,
            transform_type=config.transform_type,
            use_attn=config.use_attention,
            attention_type=config.attention_type,
            coord_dim=kernel_coord_dim,
            use_torch_scatter=config.use_torch_scatter
        )
        
        self.projection = ChannelMLP(
            in_channels=in_channels,
            hidden_channels=config.hidden_size,
            out_channels=out_channels,
            n_layers=1
        )
        
        # --- Optional geometric embedding ---
        if self.use_geoembed:
            self.geoembed = GeometricEmbedding(
                input_dim=self.coord_dim,
                output_dim=in_channels,
                method=config.embedding_method,
                pooling=config.pooling
            )
            self.recovery = ChannelMLP(
                in_channels=2 * in_channels,
                out_channels=in_channels,
                n_layers=1
            )
        
        # --- Optional scale weighting ---
        if self.use_scale_weights:
            self.scale_weighting = nn.Sequential(
                nn.Linear(kernel_coord_dim, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, len(self.scales))
            )
            self.scale_weight_activation = nn.Softmax(dim=-1)
    
    def _compute_kernel_coord_dim(self) -> int:
        """Compute the effective coordinate dimension for kernel computation"""
        coord_dim = self.coord_dim
        if self.node_embedding:
            coord_dim = self.coord_dim * 4 * 2
        return coord_dim
    
    def _detect_coordinate_mode(self, query_coord: torch.Tensor) -> Literal['fx', 'vx']:
        """Automatically detect coordinate mode based on input shape"""
        if query_coord.ndim == 2:
            return 'fx'
        elif query_coord.ndim == 3:
            return 'vx'
        else:
            raise ValueError(f"query_coord must be 2D or 3D tensor, got shape {query_coord.shape}")
    
    def _compute_neighbors(self, latent_coord: torch.Tensor, query_coord: torch.Tensor,
                          mode: Literal['fx', 'vx']) -> List:
        """Compute neighbor lists with caching support"""
        cache_key = f"dec_{mode}_{latent_coord.shape}_{query_coord.shape}_{tuple(self.scales)}"
        
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]
        
        neighbors_per_scale = []
        
        if mode == 'fx':
            # Fixed coordinates
            for scale in self.scales:
                scaled_radius = self.config.radius * scale
                neighbors = self.nb_search(
                    data=latent_coord,
                    queries=query_coord,
                    radius=scaled_radius
                )
                neighbors_per_scale.append(neighbors)
        
        else:  # mode == 'vx'
            # Variable coordinates
            batch_size = query_coord.shape[0]
            neighbors_per_batch = []
            
            for b in range(batch_size):
                neighbors_per_scale_batch = []
                for scale in self.scales:
                    scaled_radius = self.config.radius * scale
                    neighbors = self.nb_search(
                        data=latent_coord,  # Same latent grid for all batches
                        queries=query_coord[b],  # Different query coords per batch
                        radius=scaled_radius
                    )
                    neighbors_per_scale_batch.append(neighbors)
                neighbors_per_batch.append(neighbors_per_scale_batch)
            neighbors_per_scale = neighbors_per_batch
        
        self.neighbor_cache[cache_key] = neighbors_per_scale
        return neighbors_per_scale
    
    def forward(self,
                latent_tokens_coord: torch.Tensor,
                rndata: torch.Tensor,
                query_coord: torch.Tensor,
                decoder_nbrs: Optional[Union[List, List[List]]] = None) -> torch.Tensor:
        """
        Unified forward pass supporting both coordinate modes and dimensions.
        
        Parameters
        ----------
        latent_tokens_coord : torch.Tensor, shape [num_latent_nodes, coord_dim]
            Latent grid coordinates (source)
        rndata : torch.Tensor, shape [batch_size, num_latent_nodes, in_channels]
            Regional/latent node features
        query_coord : torch.Tensor  
            Query coordinates (target)
            - fx mode: [num_query, coord_dim] - shared across batch
            - vx mode: [batch_size, num_query, coord_dim] - per batch
        decoder_nbrs : Optional neighbor lists
            Precomputed neighbors if precompute_edges=True
            
        Returns
        -------
        torch.Tensor
            Decoded features at query coordinates
            - fx mode: [batch_size, num_query, out_channels]
            - vx mode: [batch_size, num_query, out_channels]
        """
        
        # --- Auto-detect coordinate mode ---
        coord_mode = self._detect_coordinate_mode(query_coord)
        batch_size = rndata.shape[0]
        
        # --- Validate inputs ---
        if coord_mode == 'fx':
            if query_coord.shape[1] != self.coord_dim:
                raise ValueError(f"Expected query_coord shape [num_query, {self.coord_dim}], got {query_coord.shape}")
            num_query = query_coord.shape[0]
        else:  # vx mode
            if query_coord.shape[0] != batch_size or query_coord.shape[2] != self.coord_dim:
                raise ValueError(f"Expected query_coord shape [{batch_size}, num_query, {self.coord_dim}], got {query_coord.shape}")
            num_query = query_coord.shape[1]
        
        if latent_tokens_coord.shape[1] != self.coord_dim:
            raise ValueError(f"Expected latent_tokens_coord shape [num_latent, {self.coord_dim}], got {latent_tokens_coord.shape}")
        
        # --- Compute or use precomputed neighbors ---
        if self.precompute_edges:
            if decoder_nbrs is None:
                raise ValueError("decoder_nbrs required when precompute_edges=True")
            neighbors_per_scale = decoder_nbrs
        else:
            neighbors_per_scale = self._compute_neighbors(latent_tokens_coord, query_coord, coord_mode)
        
        # --- Prepare scale weights if enabled ---
        if self.use_scale_weights:
            if coord_mode == 'fx':
                scale_weights = self.scale_weighting(query_coord)  # [num_query, num_scales]
            else:
                # For vx mode, use the first batch's coordinates for scale weights
                scale_weights = self.scale_weighting(query_coord[0])  # [num_query, num_scales]
            scale_weights = self.scale_weight_activation(scale_weights)
        
        # --- Process each scale ---
        if coord_mode == 'fx':
            decoded_scales = self._forward_fx_mode(
                latent_tokens_coord, rndata, query_coord, neighbors_per_scale
            )
        else:
            decoded_scales = self._forward_vx_mode(
                latent_tokens_coord, rndata, query_coord, neighbors_per_scale
            )
        
        # --- Combine scales ---
        if len(decoded_scales) == 1:
            decoded = decoded_scales[0]
        else:
            if self.use_scale_weights:
                # Weighted combination
                decoded = torch.zeros_like(decoded_scales[0])
                for i, dec in enumerate(decoded_scales):
                    weights = scale_weights[:, i:i+1].unsqueeze(0)  # [1, num_query, 1]
                    decoded += weights * dec
            else:
                # Simple average
                decoded = torch.stack(decoded_scales, dim=0).mean(dim=0)
        
        # --- Apply final projection ---
        decoded = decoded.permute(0, 2, 1)  # [batch, channels, query] for ChannelMLP
        decoded = self.projection(decoded).permute(0, 2, 1)  # Back to [batch, query, channels]
        
        return decoded
    
    def _forward_fx_mode(self, latent_coord, rndata, query_coord, neighbors_per_scale):
        """Forward pass for fixed coordinate mode"""
        batch_size = rndata.shape[0]
        decoded_scales = []
        
        for _, neighbors in enumerate(neighbors_per_scale):
            # Apply edge drop before passing to AGNO and geoembed
            neighbors_dropped = apply_edge_drop_csr(
                neighbors, 
                self.sampling_strategy, 
                self.max_neighbors, 
                self.sample_ratio, 
                self.training
            )
            
            # Prepare coordinates for kernel
            if self.node_embedding:
                latent_coord_processed = node_pos_encode(latent_coord)
                query_coord_processed = node_pos_encode(query_coord)
            else:
                latent_coord_processed = latent_coord
                query_coord_processed = query_coord
            
            # Apply AGNO for current scale
            decoded = self.agno(
                y=latent_coord_processed,
                x=query_coord_processed,
                f_y=rndata,
                neighbors=neighbors_dropped
            )
            
            # Apply geometric embedding if enabled
            if self.use_geoembed:
                geoembedding = self.geoembed(
                    input_geom=latent_coord,
                    latent_queries=query_coord,
                    spatial_nbrs=neighbors_dropped
                )
                # Expand to batch size and concatenate
                geoembedding = geoembedding.unsqueeze(0).expand(batch_size, -1, -1)
                decoded = torch.cat([decoded, geoembedding], dim=-1)
                # Apply recovery MLP
                decoded = decoded.permute(0, 2, 1)
                decoded = self.recovery(decoded).permute(0, 2, 1)
            
            decoded_scales.append(decoded)
        
        return decoded_scales
    
    def _forward_vx_mode(self, latent_coord, rndata, query_coord, neighbors_per_batch):
        """Forward pass for variable coordinate mode"""
        batch_size = query_coord.shape[0]
        decoded_scales = []
        
        for scale_idx in range(len(self.scales)):
            decoded_batch = []
            
            for b in range(batch_size):
                query_b = query_coord[b]  # [num_query, coord_dim]
                rndata_b = rndata[b:b+1]  # [1, num_latent, in_channels]
                neighbors_b = neighbors_per_batch[b][scale_idx]
                
                # Apply edge drop before passing to AGNO and geoembed
                neighbors_b_dropped = apply_edge_drop_csr(
                    neighbors_b, 
                    self.sampling_strategy, 
                    self.max_neighbors, 
                    self.sample_ratio, 
                    self.training
                )
                
                # Prepare coordinates for kernel
                if self.node_embedding:
                    latent_coord_processed = node_pos_encode(latent_coord)
                    query_coord_processed = node_pos_encode(query_b)
                else:
                    latent_coord_processed = latent_coord
                    query_coord_processed = query_b
                
                # Apply AGNO for current batch and scale
                decoded_b = self.agno(
                    y=latent_coord_processed,
                    x=query_coord_processed,
                    f_y=rndata_b,
                    neighbors=neighbors_b_dropped
                )
                
                # Apply geometric embedding if enabled
                if self.use_geoembed:
                    geoembedding = self.geoembed(
                        input_geom=latent_coord,
                        latent_queries=query_b,
                        spatial_nbrs=neighbors_b_dropped
                    )
                    geoembedding = geoembedding.unsqueeze(0)  # Add batch dim
                    decoded_b = torch.cat([decoded_b, geoembedding], dim=-1)
                    # Apply recovery MLP
                    decoded_b = decoded_b.permute(0, 2, 1)
                    decoded_b = self.recovery(decoded_b).permute(0, 2, 1)
                
                decoded_batch.append(decoded_b)
            
            # Stack batch results
            decoded_scale = torch.cat(decoded_batch, dim=0)
            decoded_scales.append(decoded_scale)
        
        return decoded_scales



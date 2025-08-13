"""
This file contains the implementation of the Attentional Graph Neural Operator (AGNO) module.
It extends the traditional Graph Neural Operator (GNO) with attention mechanisms.

Key Features:
1. Attention Mechanism: Supports both cosine similarity and dot-product attention to weight neighbor contributions

Reference: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/integral_transform.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.segment_csr import segment_csr
from .mlp import LinearChannelMLP
from typing import Optional, Literal, Dict

############
# Attentional Graph Neural Operator (AGNO)
############
class AGNO(nn.Module):
    """Attentional Graph Neural Operator (AGNO)
    
    An enhanced Graph Neural Operator that combines traditional kernel integral transforms
    with attention mechanisms for improved performance and regularization.
    
    Computes attentionally-weighted integral transforms:
        (a) ∫_{A(x)} α(x,y) * k(x, y) dy
        (b) ∫_{A(x)} α(x,y) * k(x, y) * f(y) dy  
        (c) ∫_{A(x)} α(x,y) * k(x, y, f(y)) dy
        (d) ∫_{A(x)} α(x,y) * k(x, y, f(y)) * f(y) dy
    
    Where:
    - α(x,y) is the attention weight between query point x and neighbor y
    - A(x) is the (possibly sampled) neighborhood of x
    - k is a learnable kernel parametrized as an MLP
    - f is the input function defined on points y

    Key Enhancements over traditional GNO:
    Attention Weighting (α): 
       - Cosine similarity: α(x,y) = cos_sim(pos(x), pos(y))
       - Dot-product: α(x,y) = softmax(Q(x)·K(y) / √d)
       

    Parameters
    ----------
    channel_mlp : torch.nn.Module, optional
        Pre-initialized MLP for the kernel k.
    channel_mlp_layers : list, optional
        Layer sizes for the kernel MLP if channel_mlp is not provided.
    channel_mlp_non_linearity : callable, default F.gelu
        Non-linearity for the kernel MLP.
    transform_type : str, default 'linear'
        Type of integral transform ('linear', 'nonlinear', 'linear_kernelonly', 'nonlinear_kernelonly').
    use_attn : bool, default False
        Whether to use the attention mechanism for neighbor weighting.
    coord_dim : int, optional
        Coordinate dimension, required if use_attn is True.
    attention_type : Literal['cosine', 'dot_product'], default 'cosine'
        Type of attention mechanism:
        - 'cosine': Cosine similarity between coordinate embeddings
        - 'dot_product': Scaled dot-product attention with learnable projections
    use_torch_scatter : bool, default True
        Whether to use torch_scatter backend for segment_csr if available.
    """

    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type="linear",
        use_attn=None,
        attention_type='cosine',
        coord_dim=None,
        use_torch_scatter=True
    ):
        super().__init__()

        # --- Store configuration ---
        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter
        self.use_attn = use_attn
        self.attention_type = attention_type

        # --- Validate parameters ---
        if channel_mlp is None and channel_mlp_layers is None:
             raise ValueError("Either channel_mlp or channel_mlp_layers must be provided.")
        if self.transform_type not in ["linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"]:
            raise ValueError(f"Invalid transform_type: {transform_type}")
        if self.use_attn:
            if coord_dim is None:
                raise ValueError("coord_dim must be specified when use_attn is True")
            self.coord_dim = coord_dim # Store coord_dim only if use_attn is True
            if self.attention_type not in ['cosine', 'dot_product']:
                 raise ValueError(f"Invalid attention_type: {self.attention_type}")
        # Note: Edge drop validation removed - handled at encoder/decoder level

        # --- Initialize Modules ---
        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)
        else:
            self.channel_mlp = channel_mlp

        ## Initialize attention projection if needed
        if self.use_attn and self.attention_type == 'dot_product':
            attention_dim = 64 
            self.query_proj = nn.Linear(self.coord_dim, attention_dim)
            self.key_proj = nn.Linear(self.coord_dim, attention_dim)
            self.scaling_factor = 1.0 / (attention_dim ** 0.5)

    def _segment_softmax(self, attention_scores, splits):
        """
        Apply segment-wise softmax for attention weight normalization.
        
        Computes softmax over neighbors for each query node separately,
        ensuring attention weights sum to 1 within each neighborhood.

        Parameters
        ----------
        attention_scores : torch.Tensor, shape [num_neighbors]
            Raw attention scores between query nodes and their neighbors
        splits : torch.Tensor
            CSR row splits defining neighborhood boundaries

        Returns
        -------
        attention_weights : torch.Tensor, shape [num_neighbors]
            Normalized attention weights (sum to 1 within each neighborhood)
        """
        max_values = segment_csr(
            attention_scores, splits, reduce='max', use_scatter=self.use_torch_scatter
        )
        max_values_expanded = max_values.repeat_interleave(
            splits[1:] - splits[:-1], dim=0
        )
        attention_scores = attention_scores - max_values_expanded
        exp_scores = torch.exp(attention_scores)
        sum_exp = segment_csr(
            exp_scores, splits, reduce='sum', use_scatter=self.use_torch_scatter
        )
        sum_exp_expanded = sum_exp.repeat_interleave(
            splits[1:] - splits[:-1], dim=0
        )
        attention_weights = exp_scores / sum_exp_expanded
        return attention_weights

    def forward(self, 
                y: torch.Tensor, 
                neighbors: Dict[str, torch.Tensor], 
                x: Optional[torch.Tensor] = None, 
                f_y: Optional[torch.Tensor] = None, 
                weights: Optional[torch.Tensor] = None):
        """Compute attentional kernel integral transform with optional edge drop

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying the space to integrate over.
            If batched, these must remain constant over the whole batch.
        neighbors : dict
            Graph connectivity in CSR format. Must contain keys "neighbors_index"
            and "neighbors_row_splits." If batch > 1, neighbors must be constant
            across the entire batch.
        x : torch.Tensor of shape [m, d2], default None
            m query points of dimension d2. If None, x = y.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Input function defined on points y. If None, computes transform type (a).
        weights : torch.Tensor of shape [n,], default None
            Integration weights proportional to volume around each point.
            If None, uses 1/|A(x)| or attention-based weighting.

        Returns
        -------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function on query points x, with attention-weighted aggregation.
        """

        if x is None:
            x = y

        # Edge drop is now handled at encoder/decoder level
        neighbors_index = neighbors["neighbors_index"]
        neighbors_row_splits = neighbors["neighbors_row_splits"]
        num_query_nodes = neighbors_row_splits.shape[0] - 1

        # --- Gather features ---
        rep_features = y[neighbors_index]

        # --- Batching ---
        ## batching only matters if f_y (latent embedding) values are provided
        batched = False
        in_features = None
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors_index, :]
            elif f_y.ndim == 2:
                batched = False
                in_features = f_y[neighbors_index]
            else:
                raise ValueError(f"f_y has unexpected ndim: {f_y.ndim}")
        
        # --- Prepare 'self' features ---
        num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        # --- Attention Logic ---
        attention_weights = None
        if self.use_attn:
            query_coords = self_features[:, :self.coord_dim]
            key_coords = rep_features[:, :self.coord_dim]
            if self.attention_type == 'dot_product':
                query = self.query_proj(query_coords)  # [num_neighbors, attention_dim]
                key = self.key_proj(key_coords)        # [num_neighbors, attention_dim]
                attention_scores = torch.sum(query * key, dim=-1) * self.scaling_factor  # [num_neighbors] 
            elif self.attention_type == 'cosine':
                query_norm = F.normalize(query_coords, p=2, dim=-1)
                key_norm = F.normalize(key_coords, p=2, dim=-1)
                attention_scores = torch.sum(query_norm * key_norm, dim=-1)  # [num_neighbors]
            else:
                raise ValueError(f"Invalid attention_type: {self.attention_type}. Must be 'cosine' or 'dot_product'.")
            attention_weights = self._segment_softmax(attention_scores, neighbors_row_splits)
        else:
            attention_weights = None
        
        # --- Prepare input for the kernel MLP ---
        agg_features = torch.cat([rep_features, self_features], dim=-1)
        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if batched:
                # repeat agg features for every example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        # --- Apply Kernel MLP ---
        rep_features = self.channel_mlp(agg_features) # Compute kernel values k(x,y) or k(x,y,f)

        # --- Apply f_y multiplication ---
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features
        
        # --- Apply attention weights ---
        if self.use_attn:
            rep_features = rep_features * attention_weights.unsqueeze(-1)
        
        # --- Apply Integration Weights ---
        if weights is not None:
            assert weights.ndim == 1, "Weights must be of dimension 1 in all cases"
            nbr_weights = weights[neighbors_index]
            # repeat weights along batch dim if batched
            if batched:
                nbr_weights = nbr_weights.repeat(
                    [batch_size] + [1] * nbr_weights.ndim
                )
            rep_features = nbr_weights * rep_features
            reduction = "sum"
        else:
            reduction = "mean" if not self.use_attn else "sum"

        # --- Aggregate using segment_csr ---
        splits = neighbors_row_splits
        if batched:
            splits = splits.repeat([batch_size] + [1] * splits.ndim)

        out_features = segment_csr(rep_features, splits, reduce=reduction, use_scatter=self.use_torch_scatter)

        return out_features



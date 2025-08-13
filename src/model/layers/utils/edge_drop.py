"""
Edge drop utilities for neighbor sampling during training.
"""
import torch
from typing import Dict, Optional


def apply_edge_drop_csr(
    neighbors: Dict[str, torch.Tensor],
    sampling_strategy: Optional[str],
    max_neighbors: Optional[int] = None,
    sample_ratio: Optional[float] = None,
    training: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Apply edge drop (neighbor sampling) for training regularization.
    
    This implements two sampling strategies:
    1. Ratio sampling: Randomly keeps a fraction of edges across the entire graph
    2. Max neighbors: Limits the maximum number of neighbors per node
    
    Parameters
    ----------
    neighbors : Dict[str, torch.Tensor]
        Neighbors in CSR format with keys 'neighbors_index' and 'neighbors_row_splits'
    sampling_strategy : Optional[str]
        Sampling strategy to use ('ratio', 'max_neighbors', or None)
    max_neighbors : Optional[int]
        Maximum number of neighbors per node (for 'max_neighbors' strategy)
    sample_ratio : Optional[float]
        Fraction of edges to keep (for 'ratio' strategy)
    training : bool
        Whether model is in training mode
        
    Returns  
    -------
    Dict[str, torch.Tensor]
        Sampled neighbors in CSR format
    """
    if not training or sampling_strategy is None:
        return neighbors
        
    neighbors_index = neighbors["neighbors_index"]
    neighbors_row_splits = neighbors["neighbors_row_splits"]
    device = neighbors_index.device

    if neighbors_index.numel() == 0: 
        return neighbors  # No edges
        
    num_target_nodes = neighbors_row_splits.shape[0] - 1
    num_total_original_edges = neighbors_index.shape[0]

    # --- Strategy 1: Global Ratio Sampling ---
    if sampling_strategy == 'ratio':
        if sample_ratio is None or sample_ratio >= 1.0: 
            return neighbors
            
        num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        if (num_reps < 0).any() or torch.sum(num_reps) != num_total_original_edges:
            raise ValueError("Invalid CSR structure before ratio sampling.")
            
        target_node_indices = torch.arange(num_target_nodes, device=device).repeat_interleave(num_reps)
        keep_mask = torch.rand(num_total_original_edges, device=device) < sample_ratio
        sampled_neighbors_index = neighbors_index[keep_mask]
        sampled_target_node_indices = target_node_indices[keep_mask]
        new_num_reps = torch.bincount(sampled_target_node_indices, minlength=num_target_nodes)
        sampled_neighbors_row_splits = torch.zeros(num_target_nodes + 1, dtype=neighbors_row_splits.dtype, device=device)
        torch.cumsum(new_num_reps, dim=0, out=sampled_neighbors_row_splits[1:])
    
    elif sampling_strategy == 'max_neighbors':
        if max_neighbors is None: 
            return neighbors

        num_reps = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        if (num_reps < 0).any() or torch.sum(num_reps) != num_total_original_edges:
            raise ValueError("Invalid CSR structure before max_neighbors sampling.")

        needs_sampling_mask = num_reps > max_neighbors
        if not torch.any(needs_sampling_mask):
            return neighbors  # No node exceeds the limit
        
        keep_mask = torch.ones(num_total_original_edges, dtype=torch.bool, device=device)
        # Iterate over nodes that need sampling
        nodes_to_sample_idx = torch.where(needs_sampling_mask)[0]
        for i in nodes_to_sample_idx:
            start = neighbors_row_splits[i]
            end = neighbors_row_splits[i+1]
            num_node_neighbors = int(num_reps[i])
            perm = torch.randperm(num_node_neighbors, device=device)
            keep_local_indices = perm[:max_neighbors]
            local_keep_mask = torch.zeros(num_node_neighbors, dtype=torch.bool, device=device)
            local_keep_mask[keep_local_indices] = True
            keep_mask[start:end] = local_keep_mask
            
        sampled_neighbors_index = neighbors_index[keep_mask]
        max_n_tensor = torch.full_like(num_reps, max_neighbors)
        new_num_reps = torch.minimum(num_reps, max_n_tensor)
        sampled_neighbors_row_splits = torch.zeros(num_target_nodes + 1, dtype=neighbors_row_splits.dtype, device=device)
        torch.cumsum(new_num_reps, dim=0, out=sampled_neighbors_row_splits[1:])
    else:
        return neighbors  # No sampling needed
    
    return {
        "neighbors_index": sampled_neighbors_index,
        "neighbors_row_splits": sampled_neighbors_row_splits
    }
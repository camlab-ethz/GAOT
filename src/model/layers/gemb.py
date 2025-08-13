"""
This file contains the implementation of the geometric embedding (GEmb) module and the node position encoding (NPE) module.
"""
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum, scatter_max
import numpy as np

######################
# Node Pos Encoding
######################
def node_pos_encode(x:torch.Tensor,
                    freq: int = 4
                    )-> torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    Returns
    -------
    x_encoded: torch.Tensor
        2D tensor of shape [n_points, 2*freq]
    """
    assert x.ndim == 2, f"The x is expected to be 2D tensor, but got shape {x.shape}"

    device = x.device
    freqs = torch.arange(1, freq+1).to(device=device) # [freq]
    phi   = np.pi * (x + 1)
    x = freqs[None, :, None] * phi[:, None, :]        # [n_points, 1, dim] * [1, freq, 1] -> [n_points, freq, dim]
    x = torch.cat([x.sin(), x.cos()], dim=2)          # [n_points, freq, dim * 2]
    x = x.view(x.shape[0], -1)                        # [n_points, freq * 2 * dim]
    
    return x
    
######################
# Geometric Embedding
######################
class GeometricEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, method='statistical', pooling='max', **kwargs):
        super(GeometricEmbedding, self).__init__()
        # --- Initialize parameters ---
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method.lower()
        self.pooling = pooling.lower()
        self.kwargs = kwargs 

        if self.pooling not in ['max', 'mean']:
            raise ValueError(f"Unsupported pooling method: {self.pooling}. Supported methods: 'max', 'mean'.")


        if self.method == 'statistical':
            self.mlp = nn.Sequential(
                nn.Linear(self._get_stat_feature_dim(), 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
                nn.ReLU()
            )
           
        elif self.method == 'pointnet':
            self.pointnet_mlp = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.fc = nn.Sequential(
                nn.Linear(64, output_dim),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
    def _get_stat_feature_dim(self):
        """
        Returns:
            int: The number of features used in the statistical method.
        """
        num_features = 3 + 2 * self.input_dim
        return num_features
    
    def _compute_statistical_features(self, input_geom, latent_queries, spatial_nbrs):
            """
            Parameters:
                input_geom (torch.FloatTensor): The input geometry, shape: [num_nodes, num_dims]
                latent_queries (torch.FloatTensor): The latent queries, shape: [num_nodes, num_dims]
                spatial_nbrs (dict): {"neighbors_index": torch.LongTensor, "neighbors_row_splits": torch.LongTensor}             
                        neighbors_index: torch.Tensor with dtype=torch.int64
                            Index of each neighbor in data for every point
                            in queries. Neighbors are ordered in the same orderings
                            as the points in queries. Open3d and torch_cluster
                            implementations can differ by a permutation of the 
                            neighbors for every point.
                        neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                            The value at index j is the sum of the number of
                            neighbors up to query point j-1. First element is 0
                            and last element is the total number of neighbors.
            
            Returns:
                geo_features_normalized (torch.FloatTensor): The normalized geometric features, shape: [num_query_nodes, num_dims]
            """
            num_queries = latent_queries.shape[0]
            num_dims = latent_queries.shape[1]
            device = latent_queries.device

            neighbors_index = spatial_nbrs["neighbors_index"]               # Shape: [num_total_neighbors]
            neighbors_row_splits = spatial_nbrs['neighbors_row_splits']     # Shape: [num_queries + 1]

            num_neighbors_per_query = neighbors_row_splits[1:] - neighbors_row_splits[:-1]  # Shape: [num_queries]
            query_indices_per_neighbor = torch.repeat_interleave(torch.arange(num_queries, device=device), num_neighbors_per_query)
            # Shape: [num_total_neighbors]

            nbr_coords = input_geom[neighbors_index]                                # Shape: [num_total_neighbors, num_dims]
            query_coords_per_neighbor = latent_queries[query_indices_per_neighbor]  # Shape: [num_total_neighbors, num_dims]

            ## --- 1. compute distances ---
            distances = torch.norm(nbr_coords - query_coords_per_neighbor, dim=1)   # Shape: [num_total_neighbors]
            N_i = num_neighbors_per_query.float()
            has_neighbors = N_i > 0                                                 # Shape: [num_queries,]

            # --- 2. compute average distance of neighbors ---
            D_avg = scatter_mean(distances, query_indices_per_neighbor, dim=0, dim_size=num_queries)        # Shape: [num_queries,]

            # --- 3. compute variance of distances ---
            distances_squared = distances ** 2
            E_X2 = scatter_mean(distances_squared, query_indices_per_neighbor, dim=0, dim_size=num_queries)  # Shape: [num_queries,]
            E_X = D_avg
            E_X_squared = E_X ** 2                                                                           # Shape: [num_queries,]
            D_var = E_X2 - E_X_squared                                                                       # Shape: [num_queries,]
            D_var = torch.clamp(D_var, min=0.0)                                                              # Shape: [num_queries,]

            # --- 4. compute the difference between centroid of neighbors and latent query point ---
            nbr_centroid = scatter_mean(nbr_coords, query_indices_per_neighbor, dim=0, dim_size=num_queries) # Shape: [num_queries, num_dims]
            Delta = nbr_centroid - latent_queries 

            # --- 5. compute covariance matrix of neighbors ---
            nbr_coords_centered = nbr_coords - nbr_centroid[query_indices_per_neighbor]                      # Shape: [num_total_neighbors, num_dims]
            cov_components = nbr_coords_centered.unsqueeze(2) * nbr_coords_centered.unsqueeze(1)             # Shape: [num_total_neighbors, num_dims, num_dims]
            cov_sum = scatter_sum(cov_components, query_indices_per_neighbor, dim=0, dim_size=num_queries)   # Shape: [num_queries, num_dims, num_dims]
            N_i_clamped = N_i.clone()
            N_i_clamped[N_i_clamped == 0] = 1.0 
            cov_matrix = cov_sum / N_i_clamped.view(-1, 1, 1)                                                # Shape: [num_queries, num_dims, num_dims]

            # --- 6. compute PCA features ---
            PCA_features = torch.zeros(num_queries, num_dims, device=device)
            # For queries with neighbors, compute eigenvalues
            if has_neighbors.any():
                # covariance matrices
                cov_matrix_valid = cov_matrix[has_neighbors]                                                 # Shape: [num_valid_queries, num_dims, num_dims]
                eigenvalues = torch.linalg.eigvalsh(cov_matrix_valid)                                        # Shape: [num_valid_queries, num_dims]
                eigenvalues = eigenvalues.flip(dims=[1])
                PCA_features[has_neighbors] = eigenvalues

            # Stack all features
            N_i_tensor = N_i.unsqueeze(1)       # Shape: [num_queries, 1]
            D_avg_tensor = D_avg.unsqueeze(1)   # Shape: [num_queries, 1]
            D_var_tensor = D_var.unsqueeze(1)   # Shape: [num_queries, 1]
            geo_features = torch.cat([N_i_tensor, D_avg_tensor, D_var_tensor, Delta, PCA_features], dim=1) # Shape: [num_queries, num_features]

            geo_features[~has_neighbors] = 0.0

            # Feature normalization (Standardization)
            feature_mean = geo_features.mean(dim=0, keepdim=True)
            feature_std = geo_features.std(dim=0, keepdim=True)
            feature_std[feature_std < 1e-6] = 1.0  # Prevent division by near-zero std

            # Normalize features
            geo_features_normalized = (geo_features - feature_mean) / feature_std

            return geo_features_normalized

    def _compute_pointnet_features(self, input_geom, latent_queries, spatial_nbrs):
        """
        Use pointnet to compute geometric features for each query point.

        Parameters:
            input_geom (torch.FloatTensor): The input geometry, shape: [num_nodes, num_dims]
            latent_queries (torch.FloatTensor): The latent queries, shape: [num_nodes, num_dims]
            spatial_nbrs (dict): {"neighbors_index": torch.LongTensor, "neighbors_row_splits": torch.LongTensor}
                    neighbors_index: torch.Tensor with dtype=torch.int64
                        Index of each neighbor in data for every point
                        in queries. Neighbors are ordered in the same orderings
                        as the points in queries. Open3d and torch_cluster
                        implementations can differ by a permutation of the
                        neighbors for every point.
                    neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                        The value at index j is the sum of the number of
                        neighbors up to query point j-1. First element is 0
                        and last element is the total number of neighbors.
        Returns:
            geo_features (torch.FloatTensor): The geometric features, shape: [num_query_nodes, num_dims]
        """

        num_queries = latent_queries.shape[0]
        device = latent_queries.device

        neighbors_index = spatial_nbrs["neighbors_index"]  # [num_total_neighbors]
        neighbors_row_splits = spatial_nbrs['neighbors_row_splits']  # [num_queries + 1]

        num_neighbors_per_query = neighbors_row_splits[1:] - neighbors_row_splits[:-1]  # [num_queries]
        has_neighbors = num_neighbors_per_query > 0  # [num_queries]
        
        geo_features = torch.zeros(num_queries, self.output_dim, device=device)
    
        if has_neighbors.any():
            valid_query_indices = torch.nonzero(has_neighbors).squeeze(1)
            
            query_indices_per_neighbor = torch.repeat_interleave(valid_query_indices, num_neighbors_per_query[has_neighbors])
            nbr_coords = input_geom[neighbors_index]  # [num_total_neighbors, num_dims]
            query_coords_per_neighbor = latent_queries[query_indices_per_neighbor]  # [num_total_valid_neighbors, num_dims]

            nbr_coords_centered = nbr_coords - query_coords_per_neighbor  # [num_total_valid_neighbors, num_dims]
            nbr_features = self.pointnet_mlp(nbr_coords_centered)  # [num_total_valid_neighbors, feature_dim]
            
            if self.pooling == 'max':
                pooled_features, _ = scatter_max(nbr_features, query_indices_per_neighbor, dim=0, dim_size=num_queries)
            elif self.pooling == 'mean':
                pooled_features = scatter_mean(nbr_features, query_indices_per_neighbor, dim=0, dim_size=num_queries)
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling}")

            pointnet_features = pooled_features[valid_query_indices]  # [num_valid_queries, feature_dim]
            pointnet_features = self.fc(pointnet_features)  # [num_valid_queries, output_dim]

            geo_features[valid_query_indices] = pointnet_features

        return geo_features

    def forward(self, input_geom, latent_queries, spatial_nbrs):
        if self.method == 'statistical':
            geo_features_normalized = self._compute_statistical_features(input_geom, latent_queries, spatial_nbrs)
            return self.mlp(geo_features_normalized)
        elif self.method == 'pointnet':
            return self._compute_pointnet_features(input_geom, latent_queries, spatial_nbrs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

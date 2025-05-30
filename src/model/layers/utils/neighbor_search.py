import torch
from torch import nn

#############
# neighbor_search
#############

class NeighborSearch(nn.Module):
    """
    Neighborhood search between two arbitrary coordinate meshes.
    For each point `x` in `queries`, returns a set of the indices of all points `y` in `data` 
    within the ball of radius r `B_r(x)`

    Parameters
    ----------
    use_open3d : bool
        Whether to use open3d or native PyTorch implementation
        NOTE: open3d implementation requires 3d data
    """
    def __init__(self, use_open3d=True):
        super().__init__()
        if use_open3d: # slightly faster, works on GPU in 3d only
            from open3d.ml.torch.layers import FixedRadiusSearch
            self.search_fn = FixedRadiusSearch()
            self.use_open3d = use_open3d
        else: # slower fallback, works on GPU and CPU
            self.search_fn = native_neighbor_search
            self.use_open3d = False
              
    def forward(self, data, queries, radius):
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : torch.Tensor of shape [m, d]
            Points for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)
        
        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
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
        """
        return_dict = {}
        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict['neighbors_index'] = search_return.neighbors_index.long()
            return_dict['neighbors_row_splits'] = search_return.neighbors_row_splits.long()

        else:
            return_dict = self.search_fn(data, queries, radius)
        
        return return_dict

def native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: torch.Tensor):
    """
    Native PyTorch implementation of a neighborhood search
    between two arbitrary coordinate meshes.
     
    Parameters
    -----------

    data : torch.Tensor
        Vector of data points from which to find neighbors. Shape: (num_data, D)
    queries : torch.Tensor
        Centers of neighborhoods. Shape: (num_queries, D)
    radius : torch.Tensor or float
        Size of each neighborhood. If tensor, should be of shape (num_queries,)
    """

    # compute pairwise distances
    if isinstance(radius, torch.Tensor):
        if radius.dim() != 1 or radius.size(0) != queries.size(0):
            raise ValueError("If radius is a tensor, it must be one-dimensional and match the number of queries.")
        radius = radius.view(-1, 1) 
    else:
        radius = torch.tensor(radius, device=queries.device).view(1, 1)
    
    with torch.no_grad():
        dists = torch.cdist(queries, data).to(queries.device) # shaped num query points x num data points
        in_nbr = torch.where(dists <= radius, 1., 0.) # i,j is one if j is i's neighbor
        nbr_indices = in_nbr.nonzero()[:,1:].reshape(-1,) # only keep the column indices
        nbrhd_sizes = torch.cumsum(torch.sum(in_nbr, dim=1), dim=0) # num points in each neighborhood, summed cumulatively
        splits = torch.cat((torch.tensor([0.]).to(queries.device), nbrhd_sizes))
        nbr_dict = {}
        nbr_dict['neighbors_index'] = nbr_indices.long().to(queries.device)
        nbr_dict['neighbors_row_splits'] = splits.long()

        del dists, in_nbr, nbr_indices, nbrhd_sizes, splits
        torch.cuda.empty_cache()
    return nbr_dict


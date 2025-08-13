import torch
from torch import nn
import math
import warnings

try:
    import torch_cluster
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False

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
    method: str
        Search method to use:
        - 'open3d': Use open3d (3D only, requires open3d)
        - 'torch_cluster': Use torch_cluster (2D/3D, requires torch_cluster)
        - 'grid': Grid-based spatial partitioning (2D optimized)
        - 'chunked': Chunked processing to reduce memory usage
        - 'native': Native PyTorch implementation (2D/3D)
    grid_size : float, optional
        Grid cell size for grid-based method (default: radius)
    chunk_size : int, optional
        Chunk size for chunked method (default: 1000)
    """
    def __init__(self, method='auto', grid_size=None, chunk_size=1000):
        super().__init__()
        self.method = method
        self.grid_size = grid_size
        self.chunk_size = chunk_size

        if method == 'auto':
            if HAS_TORCH_CLUSTER:
                self.method = 'torch_cluster'
                print("Using torch_cluster for efficient neighbor search")
            else:
                self.method = 'grid'
                print("Using grid-based method for efficient neighbor search")
        
        if self.method == 'open3d':
            try:
                from open3d.ml.torch.layers import FixedRadiusSearch
                self.search_fn = FixedRadiusSearch()
                self.use_open3d = True
            except ImportError:
                warnings.warn("Open3D not available, falling back to native implementation")
                self.method = 'native'
                self.search_fn = _native_neighbor_search
                self.use_open3d = False
        elif self.method == 'torch_cluster':
            if not HAS_TORCH_CLUSTER:
                warnings.warn("torch_cluster not available, falling back to grid method")
                self.method = 'grid'

    def forward(self, data, queries, radius):
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors
        queries : torch.Tensor of shape [m, d]
            Points for which to find neighbors
        radius : float
            Radius of each ball: B(queries[j], radius)
        
        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: torch.Tensor with dtype=torch.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Different implementations
                    can differ by a permutation of the neighbors for every point.
                neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        if self.method == 'open3d':
            search_return = self.search_fn(data, queries, radius)
            return {
                'neighbors_index': search_return.neighbors_index.long(),
                'neighbors_row_splits': search_return.neighbors_row_splits.long()
            }
        elif self.method == 'torch_cluster':
            return _torch_cluster_neighbor_search(data, queries, radius)
        elif self.method == 'grid':
            grid_size = self.grid_size if self.grid_size is not None else radius
            return _grid_neighbor_search(data, queries, radius, grid_size)
        elif self.method == 'chunked':
            return _chunked_neighbor_search(data, queries, radius, self.chunk_size)
        else:  # native
            return _native_neighbor_search(data, queries, radius)

def _native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: torch.Tensor):
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


def _torch_cluster_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    """
    Efficient neighbor search using torch_cluster library
    
    Parameters
    ----------
    data : torch.Tensor of shape [n, d]
        Data points in search space
    queries : torch.Tensor of shape [m, d]
        Query points
    radius : float
        Search radius
    """
    if not HAS_TORCH_CLUSTER:
        raise ImportError("torch_cluster library is required for this method")
    
    from torch_cluster import radius as radius_neighbors
    
    row, col = radius_neighbors(data, queries, r=radius)
    
    # convert to CSR format
    num_queries = queries.size(0)
    neighbors_index = col.long()
    row_counts = torch.bincount(row, minlength=num_queries)

    neighbors_row_splits = torch.cat([
        torch.tensor([0], device=data.device, dtype=torch.long),
        torch.cumsum(row_counts, dim=0)
    ])
    
    return {
        'neighbors_index': neighbors_index,
        'neighbors_row_splits': neighbors_row_splits
    }


def _grid_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float, grid_size: float):
    """
    Grid-based spatial partitioning neighbor search (2D optimized)
    Divides space into grids and only searches between adjacent grids to greatly reduce computation
    
    Parameters
    ----------
    data : torch.Tensor of shape [n, d]
        Data points in search space
    queries : torch.Tensor of shape [m, d]
        Query points
    radius : float
        Search radius
    grid_size : float
        Size of each grid cell
    """
    device = data.device
    dim = data.size(1)
    
    if dim != 2:
        warnings.warn("Grid method is optimized for 2D data, falling back to chunked method")
        return _chunked_neighbor_search(data, queries, radius, 1000)
    
    # Determine boundaries
    all_points = torch.cat([data, queries], dim=0)
    min_coords = all_points.min(dim=0)[0] - radius
    max_coords = all_points.max(dim=0)[0] + radius
    
    # Calculate grid dimensions
    grid_dims = ((max_coords - min_coords) / grid_size).ceil().long()
    
    # Assign points to grid cells
    data_grid_coords = ((data - min_coords) / grid_size).floor().long()
    query_grid_coords = ((queries - min_coords) / grid_size).floor().long()
    
    # Create grid indices for data points
    data_grid_indices = data_grid_coords[:, 0] * grid_dims[1] + data_grid_coords[:, 1]
    
    # Create point lists for each grid cell
    max_grid_index = grid_dims[0] * grid_dims[1]
    grid_to_points = {}
    
    for i, grid_idx in enumerate(data_grid_indices):
        grid_idx = grid_idx.item()
        if grid_idx not in grid_to_points:
            grid_to_points[grid_idx] = []
        grid_to_points[grid_idx].append(i)
    
    # Search neighbors for each query point
    all_neighbors = []
    row_splits = [0]
    
    # Define neighbor grid offsets (including self)
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),  (0, 0),  (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for q_idx, query in enumerate(queries):
        query_grid = query_grid_coords[q_idx]
        query_neighbors = []
        
        # Search adjacent grids
        for dx, dy in neighbor_offsets:
            neighbor_grid = query_grid + torch.tensor([dx, dy], device=device)
            
            # Check if grid is within bounds
            if (neighbor_grid >= 0).all() and (neighbor_grid < grid_dims).all():
                grid_idx = (neighbor_grid[0] * grid_dims[1] + neighbor_grid[1]).item()
                
                if grid_idx in grid_to_points:
                    candidate_points = grid_to_points[grid_idx]
                    
                    # Exact distance check for candidate points
                    for point_idx in candidate_points:
                        dist = torch.norm(query - data[point_idx])
                        if dist <= radius:
                            query_neighbors.append(point_idx)
        
        all_neighbors.extend(query_neighbors)
        row_splits.append(len(all_neighbors))
    
    neighbors_index = torch.tensor(all_neighbors, device=device, dtype=torch.long)
    neighbors_row_splits = torch.tensor(row_splits, device=device, dtype=torch.long)
    
    return {
        'neighbors_index': neighbors_index,
        'neighbors_row_splits': neighbors_row_splits
    }


def _chunked_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float, chunk_size: int = 1000):
    """
    Chunked neighbor search to avoid creating large distance matrices and reduce memory usage
    
    Parameters
    ----------
    data : torch.Tensor of shape [n, d]
        Data points in search space
    queries : torch.Tensor of shape [m, d]
        Query points
    radius : float
        Search radius
    chunk_size : int
        Size of each chunk
    """
    device = data.device
    num_queries = queries.size(0)
    
    all_neighbors = []
    row_splits = [0]
    
    # Handle radius parameter
    if isinstance(radius, torch.Tensor):
        if radius.dim() != 1 or radius.size(0) != num_queries:
            raise ValueError("If radius is a tensor, it must be one-dimensional and match the number of queries.")
    else:
        radius = torch.tensor(radius, device=device)
    
    # Process query points in chunks
    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        chunk_queries = queries[start_idx:end_idx]
        
        if isinstance(radius, torch.Tensor) and radius.dim() == 1:
            chunk_radius = radius[start_idx:end_idx].view(-1, 1)
        else:
            chunk_radius = radius
        
        # Calculate distances between current chunk and all data points
        with torch.no_grad():
            dists = torch.cdist(chunk_queries, data)
            in_nbr = (dists <= chunk_radius).float()
            
            # Get neighbor indices
            chunk_neighbors = in_nbr.nonzero()
            if chunk_neighbors.size(0) > 0:
                neighbor_indices = chunk_neighbors[:, 1].tolist()
                all_neighbors.extend(neighbor_indices)
            
            # Calculate neighbor counts per query point and update row_splits
            neighbor_counts = in_nbr.sum(dim=1).long()
            for count in neighbor_counts:
                row_splits.append(row_splits[-1] + count.item())
    
    neighbors_index = torch.tensor(all_neighbors, device=device, dtype=torch.long)
    neighbors_row_splits = torch.tensor(row_splits, device=device, dtype=torch.long)
    
    return {
        'neighbors_index': neighbors_index,
        'neighbors_row_splits': neighbors_row_splits
    }




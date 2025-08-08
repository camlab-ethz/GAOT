import torch
import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from new_src.model.layers.utils.neighbor_search import (
    NeighborSearch,
    _native_neighbor_search,
    _torch_cluster_neighbor_search,
    _grid_neighbor_search,
    _chunked_neighbor_search,
    HAS_TORCH_CLUSTER
)


class TestNeighborSearch:
    """Test suite for neighbor search implementations"""
    
    @pytest.fixture
    def sample_data_2d(self):
        """Generate 2D test data"""
        torch.manual_seed(42)
        data = torch.rand(100, 2) * 10.0  # 100 points in [0, 10] x [0, 10]
        queries = torch.rand(20, 2) * 10.0  # 20 query points
        radius = 2.0
        return data, queries, radius
    
    @pytest.fixture
    def sample_data_3d(self):
        """Generate 3D test data"""
        torch.manual_seed(42)
        data = torch.rand(80, 3) * 8.0  # 80 points in [0, 8]^3
        queries = torch.rand(15, 3) * 8.0  # 15 query points
        radius = 1.5
        return data, queries, radius
    
    @pytest.fixture
    def dense_data_2d(self):
        """Generate dense 2D test data for edge cases"""
        torch.manual_seed(123)
        # Create a grid with some noise
        x = torch.linspace(0, 5, 10)
        y = torch.linspace(0, 5, 10)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        data = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        data += torch.randn_like(data) * 0.1  # Add small noise
        
        queries = torch.tensor([[2.5, 2.5], [0.0, 0.0], [5.0, 5.0]])
        radius = 1.0
        return data, queries, radius
    
    @pytest.fixture
    def variable_radius_data(self):
        """Generate test data with variable radius per query"""
        torch.manual_seed(456)
        data = torch.rand(50, 2) * 5.0
        queries = torch.rand(10, 2) * 5.0
        radius = torch.rand(10) * 2.0 + 0.5  # Variable radius between 0.5 and 2.5
        return data, queries, radius

    def _compare_neighbor_results(self, result1, result2, tolerance=1e-6):
        """Compare two neighbor search results"""
        # Check that row splits match exactly
        assert torch.allclose(result1['neighbors_row_splits'], result2['neighbors_row_splits']), \
            f"Row splits mismatch: {result1['neighbors_row_splits']} vs {result2['neighbors_row_splits']}"
        
        # Check that total number of neighbors match
        assert result1['neighbors_index'].size(0) == result2['neighbors_index'].size(0), \
            f"Number of neighbors mismatch: {result1['neighbors_index'].size(0)} vs {result2['neighbors_index'].size(0)}"
        
        # For each query point, check that the set of neighbors is the same
        splits1 = result1['neighbors_row_splits']
        splits2 = result2['neighbors_row_splits']
        
        for i in range(len(splits1) - 1):
            start1, end1 = splits1[i].item(), splits1[i+1].item()
            start2, end2 = splits2[i].item(), splits2[i+1].item()
            
            neighbors1 = set(result1['neighbors_index'][start1:end1].tolist())
            neighbors2 = set(result2['neighbors_index'][start2:end2].tolist())
            
            assert neighbors1 == neighbors2, \
                f"Neighbors mismatch for query {i}: {neighbors1} vs {neighbors2}"

    def test_native_neighbor_search_2d(self, sample_data_2d):
        """Test native neighbor search implementation"""
        data, queries, radius = sample_data_2d
        result = _native_neighbor_search(data, queries, radius)
        
        # Check output format
        assert 'neighbors_index' in result
        assert 'neighbors_row_splits' in result
        assert result['neighbors_index'].dtype == torch.long
        assert result['neighbors_row_splits'].dtype == torch.long
        assert result['neighbors_row_splits'].size(0) == queries.size(0) + 1
        assert result['neighbors_row_splits'][0] == 0
        
        # Verify neighbors are actually within radius
        splits = result['neighbors_row_splits']
        for i in range(queries.size(0)):
            start, end = splits[i].item(), splits[i+1].item()
            neighbor_indices = result['neighbors_index'][start:end]
            
            if neighbor_indices.size(0) > 0:
                neighbor_points = data[neighbor_indices]
                distances = torch.norm(queries[i].unsqueeze(0) - neighbor_points, dim=1)
                assert torch.all(distances <= radius + 1e-6), \
                    f"Found neighbor outside radius for query {i}: max_dist={distances.max()}, radius={radius}"

    def test_native_neighbor_search_3d(self, sample_data_3d):
        """Test native neighbor search implementation in 3D"""
        data, queries, radius = sample_data_3d
        result = _native_neighbor_search(data, queries, radius)
        
        # Check output format
        assert result['neighbors_index'].dtype == torch.long
        assert result['neighbors_row_splits'].dtype == torch.long
        assert result['neighbors_row_splits'].size(0) == queries.size(0) + 1

    def test_native_with_variable_radius(self, variable_radius_data):
        """Test native neighbor search with variable radius"""
        data, queries, radius = variable_radius_data
        result = _native_neighbor_search(data, queries, radius)
        
        # Verify neighbors are within their respective radius
        splits = result['neighbors_row_splits']
        for i in range(queries.size(0)):
            start, end = splits[i].item(), splits[i+1].item()
            neighbor_indices = result['neighbors_index'][start:end]
            
            if neighbor_indices.size(0) > 0:
                neighbor_points = data[neighbor_indices]
                distances = torch.norm(queries[i].unsqueeze(0) - neighbor_points, dim=1)
                assert torch.all(distances <= radius[i] + 1e-6), \
                    f"Found neighbor outside radius for query {i}: max_dist={distances.max()}, radius={radius[i]}"

    @pytest.mark.skipif(not HAS_TORCH_CLUSTER, reason="torch_cluster not available")
    def test_torch_cluster_vs_native(self, sample_data_2d):
        """Test torch_cluster implementation against native"""
        data, queries, radius = sample_data_2d
        
        native_result = _native_neighbor_search(data, queries, radius)
        cluster_result = _torch_cluster_neighbor_search(data, queries, radius)
        
        self._compare_neighbor_results(native_result, cluster_result)

    @pytest.mark.skipif(not HAS_TORCH_CLUSTER, reason="torch_cluster not available")
    def test_torch_cluster_vs_native_3d(self, sample_data_3d):
        """Test torch_cluster implementation against native in 3D"""
        data, queries, radius = sample_data_3d
        
        native_result = _native_neighbor_search(data, queries, radius)
        cluster_result = _torch_cluster_neighbor_search(data, queries, radius)
        
        self._compare_neighbor_results(native_result, cluster_result)

    def test_grid_vs_native_2d(self, sample_data_2d):
        """Test grid-based implementation against native"""
        data, queries, radius = sample_data_2d
        
        native_result = _native_neighbor_search(data, queries, radius)
        grid_result = _grid_neighbor_search(data, queries, radius, radius)
        
        self._compare_neighbor_results(native_result, grid_result)

    def test_grid_vs_native_dense(self, dense_data_2d):
        """Test grid-based implementation on dense data"""
        data, queries, radius = dense_data_2d
        
        native_result = _native_neighbor_search(data, queries, radius)
        grid_result = _grid_neighbor_search(data, queries, radius, radius)
        
        self._compare_neighbor_results(native_result, grid_result)

    def test_chunked_vs_native_2d(self, sample_data_2d):
        """Test chunked implementation against native"""
        data, queries, radius = sample_data_2d
        
        native_result = _native_neighbor_search(data, queries, radius)
        chunked_result = _chunked_neighbor_search(data, queries, radius, chunk_size=5)
        
        self._compare_neighbor_results(native_result, chunked_result)

    def test_chunked_vs_native_3d(self, sample_data_3d):
        """Test chunked implementation against native in 3D"""
        data, queries, radius = sample_data_3d
        
        native_result = _native_neighbor_search(data, queries, radius)
        chunked_result = _chunked_neighbor_search(data, queries, radius, chunk_size=3)
        
        self._compare_neighbor_results(native_result, chunked_result)

    def test_chunked_vs_native_variable_radius(self, variable_radius_data):
        """Test chunked implementation with variable radius"""
        data, queries, radius = variable_radius_data
        
        native_result = _native_neighbor_search(data, queries, radius)
        chunked_result = _chunked_neighbor_search(data, queries, radius, chunk_size=3)
        
        self._compare_neighbor_results(native_result, chunked_result)

    def test_neighbor_search_class_native(self, sample_data_2d):
        """Test NeighborSearch class with native method"""
        data, queries, radius = sample_data_2d
        
        search = NeighborSearch(method='native')
        result = search(data, queries, radius)
        native_result = _native_neighbor_search(data, queries, radius)
        
        self._compare_neighbor_results(result, native_result)

    def test_neighbor_search_class_grid(self, sample_data_2d):
        """Test NeighborSearch class with grid method"""
        data, queries, radius = sample_data_2d
        
        search = NeighborSearch(method='grid')
        result = search(data, queries, radius)
        native_result = _native_neighbor_search(data, queries, radius)
        
        self._compare_neighbor_results(result, native_result)

    def test_neighbor_search_class_chunked(self, sample_data_2d):
        """Test NeighborSearch class with chunked method"""
        data, queries, radius = sample_data_2d
        
        search = NeighborSearch(method='chunked', chunk_size=10)
        result = search(data, queries, radius)
        native_result = _native_neighbor_search(data, queries, radius)
        
        self._compare_neighbor_results(result, native_result)

    @pytest.mark.skipif(not HAS_TORCH_CLUSTER, reason="torch_cluster not available")
    def test_neighbor_search_class_torch_cluster(self, sample_data_2d):
        """Test NeighborSearch class with torch_cluster method"""
        data, queries, radius = sample_data_2d
        
        search = NeighborSearch(method='torch_cluster')
        result = search(data, queries, radius)
        native_result = _native_neighbor_search(data, queries, radius)
        
        self._compare_neighbor_results(result, native_result)

    def test_neighbor_search_auto_method(self, sample_data_2d):
        """Test NeighborSearch class with auto method selection"""
        data, queries, radius = sample_data_2d
        
        search = NeighborSearch(method='auto')
        result = search(data, queries, radius)
        native_result = _native_neighbor_search(data, queries, radius)
        
        self._compare_neighbor_results(result, native_result)

    def test_edge_case_no_neighbors(self):
        """Test case where no neighbors are found"""
        data = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        queries = torch.tensor([[5.0, 5.0]])
        radius = 0.1
        
        result = _native_neighbor_search(data, queries, radius)
        
        assert result['neighbors_index'].size(0) == 0
        assert torch.equal(result['neighbors_row_splits'], torch.tensor([0, 0]))

    def test_edge_case_all_neighbors(self):
        """Test case where all points are neighbors"""
        data = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        queries = torch.tensor([[0.1, 0.1]])
        radius = 1.0
        
        result = _native_neighbor_search(data, queries, radius)
        
        assert result['neighbors_index'].size(0) == 3
        assert torch.equal(result['neighbors_row_splits'], torch.tensor([0, 3]))

    def test_edge_case_single_point(self):
        """Test with single data and query point"""
        data = torch.tensor([[1.0, 1.0]])
        queries = torch.tensor([[1.5, 1.5]])
        radius = 1.0
        
        result = _native_neighbor_search(data, queries, radius)
        
        assert result['neighbors_index'].size(0) == 1
        assert result['neighbors_index'][0] == 0
        assert torch.equal(result['neighbors_row_splits'], torch.tensor([0, 1]))

    def test_different_devices(self, sample_data_2d):
        """Test neighbor search with different device configurations"""
        data, queries, radius = sample_data_2d
        
        # Test on CPU
        result_cpu = _native_neighbor_search(data, queries, radius)
        
        # Test on GPU if available
        if torch.cuda.is_available():
            data_gpu = data.cuda()
            queries_gpu = queries.cuda()
            result_gpu = _native_neighbor_search(data_gpu, queries_gpu, radius)
            
            # Compare results (move GPU results back to CPU)
            result_gpu_cpu = {
                'neighbors_index': result_gpu['neighbors_index'].cpu(),
                'neighbors_row_splits': result_gpu['neighbors_row_splits'].cpu()
            }
            self._compare_neighbor_results(result_cpu, result_gpu_cpu)

    def test_dtype_consistency(self, sample_data_2d):
        """Test that results maintain correct dtypes"""
        data, queries, radius = sample_data_2d
        
        # Test with float32
        result_f32 = _native_neighbor_search(data.float(), queries.float(), radius)
        assert result_f32['neighbors_index'].dtype == torch.long
        assert result_f32['neighbors_row_splits'].dtype == torch.long
        
        # Test with float64
        result_f64 = _native_neighbor_search(data.double(), queries.double(), radius)
        assert result_f64['neighbors_index'].dtype == torch.long
        assert result_f64['neighbors_row_splits'].dtype == torch.long

    def test_large_radius(self, sample_data_2d):
        """Test with very large radius (all points should be neighbors)"""
        data, queries, radius = sample_data_2d
        large_radius = 100.0
        
        result = _native_neighbor_search(data, queries, large_radius)
        
        # Each query should have all data points as neighbors
        splits = result['neighbors_row_splits']
        for i in range(queries.size(0)):
            start, end = splits[i].item(), splits[i+1].item()
            assert end - start == data.size(0), f"Query {i} should have {data.size(0)} neighbors, got {end - start}"

    def test_zero_radius(self, sample_data_2d):
        """Test with zero radius (should find no neighbors unless points coincide)"""
        data, queries, radius = sample_data_2d
        zero_radius = 0.0
        
        result = _native_neighbor_search(data, queries, zero_radius)
        
        # Should find very few or no neighbors
        assert result['neighbors_index'].size(0) <= queries.size(0)  # At most one per query
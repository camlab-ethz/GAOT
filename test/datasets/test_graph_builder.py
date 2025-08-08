"""
Unit tests for GraphBuilder module.
Tests graph building, neighbor computation, and caching functionality.
"""
import pytest
import torch
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from new_src.datasets.graph_builder import GraphBuilder, CachedGraphBuilder
from test.datasets.test_utils import assert_tensor_shape, MockDatasetFactory


class TestGraphBuilder:
    """Test suite for GraphBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = MockDatasetFactory()
        self.factory.setup()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.factory.cleanup()
    
    def test_initialization(self):
        """Test GraphBuilder initialization."""
        builder = GraphBuilder(neighbor_search_method="native")
        assert builder.nb_search is not None
    
    def test_initialization_with_different_methods(self):
        """Test initialization with different neighbor search methods."""
        methods = ["native", "torch_cluster", "grid", "chunked"]
        
        for method in methods:
            builder = GraphBuilder(neighbor_search_method=method)
            assert builder.nb_search is not None
    
    def _create_test_coordinates(self, n_samples: int = 3, n_nodes: int = 20, coord_dim: int = 2):
        """Create test coordinate data."""
        # Create coordinate data in [0, 1] range
        x_data = torch.rand(n_samples, n_nodes, coord_dim)
        return x_data
    
    def _create_latent_queries(self, n_latent: int = 16, coord_dim: int = 2):
        """Create test latent query coordinates."""
        # Create regular grid in [-1, 1] range
        if coord_dim == 2:
            side = int(np.sqrt(n_latent))
            x = torch.linspace(-1, 1, side)
            y = torch.linspace(-1, 1, side)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            latent_queries = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        else:  # 3D
            side = int(np.cbrt(n_latent))
            x = torch.linspace(-1, 1, side)
            y = torch.linspace(-1, 1, side)
            z = torch.linspace(-1, 1, side)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            latent_queries = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        
        return latent_queries
    
    def test_build_graphs_for_split_2d(self):
        """Test building graphs for 2D coordinates."""
        builder = GraphBuilder(neighbor_search_method="native")
        
        # Create test data
        n_samples, n_nodes = 3, 20
        x_data = self._create_test_coordinates(n_samples, n_nodes, coord_dim=2)
        latent_queries = self._create_latent_queries(n_latent=16, coord_dim=2)
        
        # Build graphs
        encoder_graphs, decoder_graphs = builder.build_graphs_for_split(
            x_data=x_data,
            latent_queries=latent_queries,
            gno_radius=0.2,
            scales=[1.0]
        )
        # Check output structure
        assert len(encoder_graphs) == n_samples, f"Expected {n_samples} encoder graphs, got {len(encoder_graphs)}"
        assert len(decoder_graphs) == n_samples, f"Expected {n_samples} decoder graphs, got {len(decoder_graphs)}"
        
        # Check individual sample structure
        for i in range(n_samples):
            enc_sample = encoder_graphs[i]
            dec_sample = decoder_graphs[i]
            
            assert isinstance(enc_sample, list), f"Encoder sample {i} should be list of scales"
            assert isinstance(dec_sample, list), f"Decoder sample {i} should be list of scales"
            assert len(enc_sample) == 1, f"Expected 1 scale, got {len(enc_sample)}"  # Single scale
            assert len(dec_sample) == 1, f"Expected 1 scale, got {len(dec_sample)}"
            
            # Check that neighbor data has correct CSR format
            enc_neighbors = enc_sample[0]  # First scale
            dec_neighbors = dec_sample[0]

            assert isinstance(enc_neighbors, dict), "Encoder neighbors should be CSR dict"
            assert isinstance(dec_neighbors, dict), "Decoder neighbors should be CSR dict"
            
            # Check CSR format for encoder
            assert 'neighbors_index' in enc_neighbors, "Encoder CSR should have neighbors_index"
            assert 'neighbors_row_splits' in enc_neighbors, "Encoder CSR should have neighbors_row_splits"
            assert isinstance(enc_neighbors['neighbors_index'], torch.Tensor), "neighbors_index should be tensor"
            assert isinstance(enc_neighbors['neighbors_row_splits'], torch.Tensor), "neighbors_row_splits should be tensor"
            
            # Check CSR format for decoder
            assert 'neighbors_index' in dec_neighbors, "Decoder CSR should have neighbors_index"
            assert 'neighbors_row_splits' in dec_neighbors, "Decoder CSR should have neighbors_row_splits"
            assert isinstance(dec_neighbors['neighbors_index'], torch.Tensor), "neighbors_index should be tensor"
            assert isinstance(dec_neighbors['neighbors_row_splits'], torch.Tensor), "neighbors_row_splits should be tensor"
    
    def test_build_graphs_for_split_3d(self):
        """Test building graphs for 3D coordinates."""
        builder = GraphBuilder(neighbor_search_method="native")
        
        # Create test data
        n_samples, n_nodes = 2, 15
        x_data = self._create_test_coordinates(n_samples, n_nodes, coord_dim=3)
        latent_queries = self._create_latent_queries(n_latent=8, coord_dim=3)  # 2x2x2
        
        # Build graphs
        encoder_graphs, decoder_graphs = builder.build_graphs_for_split(
            x_data=x_data,
            latent_queries=latent_queries,
            gno_radius=0.3,
            scales=[1.0, 2.0]  # Multiple scales
        )
        
        # Check output structure
        assert len(encoder_graphs) == n_samples
        assert len(decoder_graphs) == n_samples
        
        # Check multiple scales
        for i in range(n_samples):
            assert len(encoder_graphs[i]) == 2, "Should have 2 scales"
            assert len(decoder_graphs[i]) == 2, "Should have 2 scales"
    
    def test_build_graphs_different_input_shapes(self):
        """Test building graphs with different input coordinate shapes."""
        builder = GraphBuilder(neighbor_search_method="native")
        latent_queries = self._create_latent_queries(n_latent=9, coord_dim=2)
        
        # Test shape [n_samples, n_nodes, coord_dim]
        x_data_3d = torch.rand(2, 10, 2)
        encoder_graphs_3d, decoder_graphs_3d = builder.build_graphs_for_split(
            x_data=x_data_3d,
            latent_queries=latent_queries,
            gno_radius=0.2,
            scales=[1.0]
        )
        assert len(encoder_graphs_3d) == 2
        
        # Test shape [n_samples, 1, n_nodes, coord_dim]
        x_data_4d = torch.rand(2, 1, 10, 2)
        encoder_graphs_4d, decoder_graphs_4d = builder.build_graphs_for_split(
            x_data=x_data_4d,
            latent_queries=latent_queries,
            gno_radius=0.2,
            scales=[1.0]
        )
        assert len(encoder_graphs_4d) == 2
    
    def test_build_all_graphs(self):
        """Test building graphs for all data splits."""
        builder = GraphBuilder(neighbor_search_method="native")
        latent_queries = self._create_latent_queries(n_latent=16, coord_dim=2)
        
        # Create mock data splits
        data_splits = {
            'train': {'x': torch.rand(4, 15, 2)},
            'val': {'x': torch.rand(2, 15, 2)},
            'test': {'x': torch.rand(3, 15, 2)}
        }
        
        # Build all graphs
        all_graphs = builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=latent_queries,
            gno_radius=0.2,
            scales=[1.0],
            build_train=True
        )
        
        # Check structure
        expected_splits = ['train', 'val', 'test']
        for split in expected_splits:
            assert split in all_graphs, f"Missing split: {split}"
            assert 'encoder' in all_graphs[split], f"Missing encoder graphs for {split}"
            assert 'decoder' in all_graphs[split], f"Missing decoder graphs for {split}"
        
        # Check sizes
        assert len(all_graphs['train']['encoder']) == 4, "Train encoder graphs size mismatch"
        assert len(all_graphs['val']['encoder']) == 2, "Val encoder graphs size mismatch"  
        assert len(all_graphs['test']['encoder']) == 3, "Test encoder graphs size mismatch"
    
    def test_build_all_graphs_test_only(self):
        """Test building graphs only for test split."""
        builder = GraphBuilder(neighbor_search_method="native")
        latent_queries = self._create_latent_queries(n_latent=16, coord_dim=2)
        
        # Create mock data splits
        data_splits = {
            'train': {'x': torch.rand(4, 15, 2)},
            'val': {'x': torch.rand(2, 15, 2)},
            'test': {'x': torch.rand(3, 15, 2)}
        }
        
        # Build only test graphs
        all_graphs = builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=latent_queries,
            gno_radius=0.2,
            scales=[1.0],
            build_train=False
        )
        
        # Check that train/val are None but test exists
        assert all_graphs['train'] is None, "Train graphs should be None when build_train=False"
        assert all_graphs['val'] is None, "Val graphs should be None when build_train=False"
        assert all_graphs['test'] is not None, "Test graphs should exist"
        assert len(all_graphs['test']['encoder']) == 3, "Test encoder graphs size mismatch"
    
    def test_validate_graphs(self):
        """Test graph validation functionality."""
        builder = GraphBuilder(neighbor_search_method="native")
        
        # Create valid graphs with CSR format
        mock_csr_1 = {
            'neighbors_index': torch.zeros(5, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(6, dtype=torch.long)  # n_nodes + 1
        }
        mock_csr_2 = {
            'neighbors_index': torch.zeros(3, dtype=torch.long), 
            'neighbors_row_splits': torch.zeros(4, dtype=torch.long)
        }
        mock_csr_3 = {
            'neighbors_index': torch.zeros(7, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(8, dtype=torch.long)
        }
        mock_csr_4 = {
            'neighbors_index': torch.zeros(4, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(5, dtype=torch.long)
        }
        mock_csr_5 = {
            'neighbors_index': torch.zeros(6, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(7, dtype=torch.long)
        }
        mock_csr_6 = {
            'neighbors_index': torch.zeros(8, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(9, dtype=torch.long)
        }
        
        mock_graphs = {
            'train': {
                'encoder': [[mock_csr_1], [mock_csr_2]],
                'decoder': [[mock_csr_4], [mock_csr_5]]
            },
            'test': {
                'encoder': [[mock_csr_3]],
                'decoder': [[mock_csr_6]],
            }
        }
        
        expected_samples = {'train': 2, 'test': 1}
        
        # Should not raise any exception
        builder.validate_graphs(mock_graphs, expected_samples)
    
    def test_validate_graphs_size_mismatch(self):
        """Test graph validation with size mismatch."""
        builder = GraphBuilder(neighbor_search_method="native")
        
        # Create graphs with wrong sizes
        mock_csr_wrong = {
            'neighbors_index': torch.zeros(5, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(6, dtype=torch.long)
        }
        mock_csr_wrong_2 = {
            'neighbors_index': torch.zeros(4, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(5, dtype=torch.long)
        }
        
        mock_graphs = {
            'train': {
                'encoder': [[mock_csr_wrong]],  # Only 1 sample
                'decoder': [[mock_csr_wrong_2]]
            }
        }
        
        expected_samples = {'train': 2}  # Expect 2 samples
        
        # Should raise assertion error
        with pytest.raises(AssertionError, match="Encoder graphs for train"):
            builder.validate_graphs(mock_graphs, expected_samples)
    
    def test_error_handling_invalid_coordinates(self):
        """Test error handling for invalid coordinate shapes."""
        builder = GraphBuilder(neighbor_search_method="native")
        latent_queries = self._create_latent_queries(n_latent=9, coord_dim=2)
        
        # Invalid shape: 1D coordinates
        x_data_invalid = torch.rand(10)  # 1D instead of 3D or 4D
        
        with pytest.raises(ValueError, match="Unexpected coordinate shape"):
            builder.build_graphs_for_split(
                x_data=x_data_invalid,
                latent_queries=latent_queries,
                gno_radius=0.2,
                scales=[1.0]
            )


class TestCachedGraphBuilder:
    """Test suite for CachedGraphBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test CachedGraphBuilder initialization."""
        builder = CachedGraphBuilder(
            neighbor_search_method="native",
            cache_dir=self.temp_dir
        )
        assert builder.cache_dir == self.temp_dir
        assert builder.nb_search is not None
    
    def test_cache_path_generation(self):
        """Test cache path generation."""
        builder = CachedGraphBuilder(cache_dir=self.temp_dir)
        
        path = builder._get_cache_path("test_dataset", "train", "encoder")
        expected_path = os.path.join(self.temp_dir, "test_dataset_train_encoder_graphs.pt")
        
        assert path == expected_path
    
    def test_cache_path_no_cache_dir(self):
        """Test error when no cache directory is set."""
        builder = CachedGraphBuilder()  # No cache_dir
        
        with pytest.raises(ValueError, match="Cache directory not specified"):
            builder._get_cache_path("test", "train", "encoder")
    
    def test_save_and_load_graphs(self):
        """Test saving and loading graphs from cache."""
        builder = CachedGraphBuilder(cache_dir=self.temp_dir)
        
        # Create mock graphs with CSR format
        mock_csr_train_enc = {
            'neighbors_index': torch.zeros(5, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(6, dtype=torch.long)
        }
        mock_csr_train_dec = {
            'neighbors_index': torch.zeros(4, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(5, dtype=torch.long)
        }
        mock_csr_test_enc = {
            'neighbors_index': torch.zeros(3, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(4, dtype=torch.long)
        }
        mock_csr_test_dec = {
            'neighbors_index': torch.zeros(6, dtype=torch.long),
            'neighbors_row_splits': torch.zeros(7, dtype=torch.long)
        }
        
        mock_graphs = {
            'train': {
                'encoder': [[mock_csr_train_enc]],
                'decoder': [[mock_csr_train_dec]]
            },
            'test': {
                'encoder': [[mock_csr_test_enc]],
                'decoder': [[mock_csr_test_dec]]
            }
        }
        
        # Save graphs
        builder.save_graphs(mock_graphs, "test_dataset")
        
        # Check that files were created
        train_encoder_path = builder._get_cache_path("test_dataset", "train", "encoder")
        train_decoder_path = builder._get_cache_path("test_dataset", "train", "decoder")
        test_encoder_path = builder._get_cache_path("test_dataset", "test", "encoder")
        test_decoder_path = builder._get_cache_path("test_dataset", "test", "decoder")
        
        assert os.path.exists(train_encoder_path)
        assert os.path.exists(train_decoder_path)
        assert os.path.exists(test_encoder_path)
        assert os.path.exists(test_decoder_path)
        
        # Load graphs back
        loaded_graphs = builder.load_graphs("test_dataset", ['train', 'test'])
        
        # Check that loaded graphs match original
        assert loaded_graphs is not None
        assert 'train' in loaded_graphs
        assert 'test' in loaded_graphs
        assert len(loaded_graphs['train']['encoder']) == 1
        assert len(loaded_graphs['test']['decoder']) == 1
    
    def test_load_graphs_missing_files(self):
        """Test loading graphs when cache files are missing."""
        builder = CachedGraphBuilder(cache_dir=self.temp_dir)
        
        # Try to load non-existent graphs
        loaded_graphs = builder.load_graphs("nonexistent_dataset", ['train', 'test'])
        
        assert loaded_graphs is None
    
    def test_build_all_graphs_with_caching(self):
        """Test building graphs with caching enabled."""
        builder = CachedGraphBuilder(cache_dir=self.temp_dir)
        
        # Create mock data
        data_splits = {
            'train': {'x': torch.rand(2, 10, 2)},
            'test': {'x': torch.rand(1, 10, 2)}
        }
        latent_queries = torch.rand(9, 2)
        
        # First call - should build and cache graphs
        graphs1 = builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=latent_queries,
            gno_radius=0.2,
            scales=[1.0],
            dataset_name="cache_test",
            build_train=True,
            use_cache=True
        )
        
        # Check that graphs were built
        assert graphs1 is not None
        assert 'train' in graphs1 and 'test' in graphs1
        
        # Check that cache files exist
        train_encoder_path = builder._get_cache_path("cache_test", "train", "encoder")
        assert os.path.exists(train_encoder_path)
        
        # Second call - should load from cache
        # Mock the load_graphs method to return the cached graphs
        def mock_load_graphs(dataset_name, splits):
            if dataset_name == "cache_test":
                return graphs1  # Return the same graphs as first call
            return None
        
        with patch.object(builder, 'load_graphs', side_effect=mock_load_graphs):
            graphs2 = builder.build_all_graphs(
                data_splits=data_splits,
                latent_queries=latent_queries,
                gno_radius=0.2,
                scales=[1.0],
                dataset_name="cache_test",
                build_train=True,
                use_cache=True
            )
            
            # Graphs should be loaded from cache
            assert graphs2 == graphs1, "Should have loaded same graphs from cache"
        
        assert graphs2 is not None
    
    def test_build_all_graphs_cache_disabled(self):
        """Test building graphs with caching disabled."""
        builder = CachedGraphBuilder(cache_dir=self.temp_dir)
        
        data_splits = {
            'test': {'x': torch.rand(1, 10, 2)}
        }
        latent_queries = torch.rand(4, 2)
        
        # Build without caching
        graphs = builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=latent_queries,
            gno_radius=0.2,
            scales=[1.0],
            dataset_name="no_cache_test",
            build_train=False,
            use_cache=False
        )
        
        # Check that graphs were built
        assert graphs is not None
        assert graphs['test'] is not None
        
        # Check that no cache files were created
        test_encoder_path = builder._get_cache_path("no_cache_test", "test", "encoder")
        assert not os.path.exists(test_encoder_path)


if __name__ == "__main__":
    pytest.main([__file__])
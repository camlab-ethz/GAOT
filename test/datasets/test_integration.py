"""
Integration tests for complete data loading pipeline.
Tests end-to-end functionality from raw data to final data loaders.
"""
import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from new_src.datasets.data_processor import DataProcessor
from new_src.datasets.graph_builder import GraphBuilder
from test.datasets.test_utils import (
    MockDatasetFactory, get_test_config_fx, get_test_config_vx,
    validate_data_loader_output, assert_dataloader_properties
)


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = MockDatasetFactory()
        self.factory.setup()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.factory.cleanup()
    
    def test_complete_pipeline_fx_2d(self):
        """Test complete pipeline for fixed coordinates 2D data."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(
            coord_dim=2, n_samples=20, n_nodes=100
        )
        
        # Initialize data processor
        processor = DataProcessor(dataset_config, metadata, dtype=torch.float32)
        
        # Load and process data
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Should detect fixed coordinates
        assert not is_variable_coords, "Should detect fixed coordinates"
        
        # Generate latent queries
        latent_queries = processor.generate_latent_queries((8, 8))
        assert latent_queries.shape == (64, 2)
        
        # Create data loaders (no graphs needed for FX mode)
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords
        )
        
        # Validate all loaders
        assert_dataloader_properties(loaders['train'], expected_batch_size=2, expected_length=3)
        assert_dataloader_properties(loaders['val'], expected_batch_size=2, expected_length=1)
        assert_dataloader_properties(loaders['test'], expected_batch_size=2, expected_length=1)
        
        # Test data loader outputs
        validate_data_loader_output(loaders['train'], coord_mode='fx', has_condition=True)
        validate_data_loader_output(loaders['val'], coord_mode='fx', has_condition=True)
        validate_data_loader_output(loaders['test'], coord_mode='fx', has_condition=True)
        
        # Check that normalization was applied
        assert processor.u_mean is not None
        assert processor.u_std is not None
        assert processor.c_mean is not None
        assert processor.c_std is not None
    
    def test_complete_pipeline_vx_2d(self):
        """Test complete pipeline for variable coordinates 2D data."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_vx_dataset(
            coord_dim=2, n_samples=15, n_nodes=80
        )
        
        # Initialize components
        processor = DataProcessor(dataset_config, metadata, dtype=torch.float32)
        builder = GraphBuilder(neighbor_search_method="native")
        
        # Load and process data
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Should detect variable coordinates
        assert is_variable_coords, "Should detect variable coordinates"
        
        # Generate latent queries
        latent_queries = processor.generate_latent_queries((6, 6))
        assert latent_queries.shape == (36, 2)
        
        # Build graphs for all splits
        all_graphs = builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=latent_queries,
            gno_radius=0.15,
            scales=[1.0, 2.0],
            build_train=True
        )
        
        # Validate graph structure - check basic structure
        assert 'train' in all_graphs and all_graphs['train'] is not None
        assert 'val' in all_graphs and all_graphs['val'] is not None  
        assert 'test' in all_graphs and all_graphs['test'] is not None
        
        # Check that each split has encoder and decoder graphs
        for split in ['train', 'val', 'test']:
            split_graphs = all_graphs[split]
            assert 'encoder' in split_graphs
            assert 'decoder' in split_graphs
            assert isinstance(split_graphs['encoder'], list)
            assert isinstance(split_graphs['decoder'], list)
            
            # Each sample should have CSR format graphs
            for sample_enc, sample_dec in zip(split_graphs['encoder'], split_graphs['decoder']):
                assert isinstance(sample_enc, list)  # List of scales
                assert isinstance(sample_dec, list)
                if len(sample_enc) > 0 and len(sample_dec) > 0:
                    # Check first scale has CSR format
                    assert isinstance(sample_enc[0], dict), "Should be CSR dict"
                    assert isinstance(sample_dec[0], dict), "Should be CSR dict"
                    assert 'neighbors_index' in sample_enc[0]
                    assert 'neighbors_row_splits' in sample_enc[0]
        
        # Create data loaders with graphs
        encoder_graphs = {
            'train': all_graphs['train']['encoder'],
            'val': all_graphs['val']['encoder'],
            'test': all_graphs['test']['encoder']
        }
        decoder_graphs = {
            'train': all_graphs['train']['decoder'],
            'val': all_graphs['val']['decoder'],
            'test': all_graphs['test']['decoder']
        }
        
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords,
            encoder_graphs=encoder_graphs,
            decoder_graphs=decoder_graphs
        )
        
        # Validate loaders
        expected_train_length = dataset_config.train_size // dataset_config.batch_size
        expected_val_length = dataset_config.val_size // dataset_config.batch_size
        expected_test_length = dataset_config.test_size // dataset_config.batch_size
        
        assert_dataloader_properties(loaders['train'], dataset_config.batch_size, expected_train_length)
        assert_dataloader_properties(loaders['val'], dataset_config.batch_size, expected_val_length)
        assert_dataloader_properties(loaders['test'], dataset_config.batch_size, expected_test_length)
        
        # Test VX data loader outputs
        validate_data_loader_output(loaders['train'], coord_mode='vx', has_condition=True)
        validate_data_loader_output(loaders['val'], coord_mode='vx', has_condition=True)
        validate_data_loader_output(loaders['test'], coord_mode='vx', has_condition=True)
    
    def test_complete_pipeline_3d(self):
        """Test complete pipeline for 3D coordinates."""
        # Create 3D test dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(
            coord_dim=3, n_samples=12, n_nodes=64
        )
        
        # Initialize processor
        processor = DataProcessor(dataset_config, metadata, dtype=torch.float32)
        
        # Load and process data
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Check that 3D coordinates are handled correctly
        train_data = data_splits['train']
        x_train = train_data['x']
        assert x_train.shape[-1] == 3, f"Should have 3D coordinates, got {x_train.shape[-1]}D"
        
        # Generate 3D latent queries
        latent_queries = processor.generate_latent_queries((4, 4, 4))
        assert latent_queries.shape == (64, 3)
        
        # Create loaders
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords
        )
        
        # Validate basic functionality
        assert loaders['train'] is not None
        validate_data_loader_output(loaders['train'], coord_mode='fx', has_condition=True)
    
    def test_complete_pipeline_no_condition(self):
        """Test complete pipeline without condition data."""
        # Create dataset without condition variables
        _, dataset_config, metadata = self.factory.create_no_condition_dataset()
        
        # Initialize processor
        processor = DataProcessor(dataset_config, metadata, dtype=torch.float32)
        
        # Load and process data
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Check that condition data is None
        train_data = data_splits['train']
        assert train_data['c'] is None
        
        # Check that condition normalization stats are None
        assert processor.c_mean is None
        assert processor.c_std is None
        
        # Create loaders
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords
        )
        
        # Validate data loader with no condition data
        validate_data_loader_output(loaders['train'], coord_mode='fx', has_condition=False)
    
    def test_pipeline_with_different_batch_sizes(self):
        """Test pipeline with different batch sizes."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(n_samples=16)
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            dataset_config.batch_size = batch_size
            
            processor = DataProcessor(dataset_config, metadata)
            data_splits, is_variable_coords = processor.load_and_process_data()
            
            loaders = processor.create_data_loaders(
                data_splits=data_splits,
                is_variable_coords=is_variable_coords
            )
            
            # Check batch size
            assert loaders['train'].batch_size == batch_size
            
            # Test that batches have correct size
            sample_batch = next(iter(loaders['train']))
            if len(sample_batch) == 2:  # FX mode
                c_batch, u_batch = sample_batch
                expected_size = min(batch_size, dataset_config.train_size)
                assert u_batch.size(0) <= expected_size
    
    def test_pipeline_memory_efficiency(self):
        """Test pipeline memory efficiency with large-ish datasets."""
        # Create larger test dataset 
        _, dataset_config, metadata = self.factory.create_fx_dataset(
            n_samples=50, n_nodes=200
        )
        
        # Use smaller batch size for memory efficiency
        dataset_config.batch_size = 4
        dataset_config.num_workers = 0  # Avoid multiprocessing in tests
        
        processor = DataProcessor(dataset_config, metadata)
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Generate latent queries
        latent_queries = processor.generate_latent_queries((10, 10))
        
        # Create data loaders
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords
        )
        
        # Test iterating through entire dataset
        total_samples = 0
        for batch in loaders['train']:
            if len(batch) == 2:  # FX mode
                c_batch, u_batch = batch
                batch_size = u_batch.size(0)
                total_samples += batch_size
                
                # Check reasonable memory usage (tensors not too large)
                assert u_batch.numel() < 1e6, "Batch tensor too large for test"
        
        # Should have processed all training samples
        assert total_samples == dataset_config.train_size
    
    def test_pipeline_error_propagation(self):
        """Test that errors are properly propagated through pipeline."""
        # Test with invalid configuration
        _, dataset_config, metadata = get_test_config_fx()
        dataset_config.base_path = "/nonexistent/path"  # Invalid path
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Should propagate FileNotFoundError
        with pytest.raises(FileNotFoundError):
            processor.load_and_process_data()
    
    def test_pipeline_coordinate_scaling_consistency(self):
        """Test that coordinate scaling is consistent throughout pipeline."""
        # Create test dataset
        _, dataset_config, metadata = self.factory.create_vx_dataset(n_samples=10)
        
        processor = DataProcessor(dataset_config, metadata)
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Generate latent queries (this creates the coordinate scaler)
        latent_queries = processor.generate_latent_queries((4, 4))
        
        # Check that scaler was created
        assert processor.coord_scaler is not None
        
        # Check that latent queries are in expected range
        assert torch.all(latent_queries >= -1.1), "Latent queries should be in scaled range"
        assert torch.all(latent_queries <= 1.1), "Latent queries should be in scaled range"
        
        # Build graphs and create loaders
        builder = GraphBuilder(neighbor_search_method="native")
        all_graphs = builder.build_all_graphs(
            data_splits=data_splits,
            latent_queries=latent_queries,
            gno_radius=0.1,
            scales=[1.0],
            build_train=True
        )
        
        encoder_graphs = {split: graphs['encoder'] for split, graphs in all_graphs.items() if graphs}
        decoder_graphs = {split: graphs['decoder'] for split, graphs in all_graphs.items() if graphs}
        
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords,
            encoder_graphs=encoder_graphs,
            decoder_graphs=decoder_graphs
        )
        
        # Check that coordinate transformation is applied in data loader
        sample_batch = next(iter(loaders['train']))
        c_batch, u_batch, x_batch, enc_graphs, dec_graphs = sample_batch
        
        # Coordinates in data loader should be transformed
        # Note: exact range depends on transformation, but should be reasonable
        assert torch.isfinite(x_batch).all(), "All coordinates should be finite"
    
    def test_pipeline_data_integrity(self):
        """Test that data integrity is maintained throughout pipeline."""
        # Create test dataset with known pattern
        _, dataset_config, metadata = self.factory.create_fx_dataset(
            n_samples=10, n_nodes=50
        )
        dataset_config.rand_dataset = False  # Ensure deterministic ordering
        
        processor = DataProcessor(dataset_config, metadata)
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        # Check data split sizes
        assert data_splits['train']['u'].shape[0] == dataset_config.train_size
        assert data_splits['val']['u'].shape[0] == dataset_config.val_size
        assert data_splits['test']['u'].shape[0] == dataset_config.test_size
        
        # Check that total samples match
        total_processed = (data_splits['train']['u'].shape[0] + 
                          data_splits['val']['u'].shape[0] + 
                          data_splits['test']['u'].shape[0])
        expected_total = (dataset_config.train_size + 
                         dataset_config.val_size + 
                         dataset_config.test_size)
        assert total_processed == expected_total
        
        # Check data types
        for split in ['train', 'val', 'test']:
            u_data = data_splits[split]['u']
            assert u_data.dtype == torch.float32, f"{split} u_data should be float32"
            
            if data_splits[split]['c'] is not None:
                c_data = data_splits[split]['c']
                assert c_data.dtype == torch.float32, f"{split} c_data should be float32"


class TestPipelineRobustness:
    """Test pipeline robustness and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = MockDatasetFactory()
        self.factory.setup()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.factory.cleanup()
    
    def test_small_dataset_handling(self):
        """Test handling of very small datasets."""
        # Create minimal dataset
        _, dataset_config, metadata = self.factory.create_fx_dataset(n_samples=3, n_nodes=10)
        
        # Adjust config for small dataset
        dataset_config.train_size = 2
        dataset_config.val_size = 1
        dataset_config.test_size = 1  # This exceeds available samples, but should be handled
        dataset_config.batch_size = 1
        
        processor = DataProcessor(dataset_config, metadata)
        
        # Should handle gracefully or raise appropriate error
        try:
            data_splits, is_variable_coords = processor.load_and_process_data()
            # If it succeeds, check basic properties
            assert 'train' in data_splits
        except AssertionError as e:
            # Expected behavior when requesting more samples than available
            assert "exceeds total samples" in str(e)
    
    def test_single_sample_batches(self):
        """Test handling of single-sample batches."""
        _, dataset_config, metadata = self.factory.create_fx_dataset(n_samples=10)
        dataset_config.batch_size = 1  # Single sample batches
        
        processor = DataProcessor(dataset_config, metadata)
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords
        )
        
        # Test single sample batch
        sample_batch = next(iter(loaders['train']))
        if len(sample_batch) == 2:  # FX mode
            c_batch, u_batch = sample_batch
            assert u_batch.size(0) == 1, "Should have single sample in batch"
    
    def test_large_batch_handling(self):
        """Test handling when batch size exceeds available samples."""
        _, dataset_config, metadata = self.factory.create_fx_dataset(n_samples=10)
        
        # Set batch size larger than any split
        dataset_config.batch_size = 20
        dataset_config.train_size = 5
        dataset_config.val_size = 3
        dataset_config.test_size = 2
        
        processor = DataProcessor(dataset_config, metadata)
        data_splits, is_variable_coords = processor.load_and_process_data()
        
        loaders = processor.create_data_loaders(
            data_splits=data_splits,
            is_variable_coords=is_variable_coords
        )
        
        # Should handle gracefully - batch size should be effectively limited
        sample_batch = next(iter(loaders['train']))
        if len(sample_batch) == 2:  # FX mode
            c_batch, u_batch = sample_batch
            assert u_batch.size(0) <= dataset_config.train_size


if __name__ == "__main__":
    pytest.main([__file__])
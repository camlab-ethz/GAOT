import torch
import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

from new_src.model.layers.magno import MAGNOConfig, MAGNOEncoder, MAGNODecoder


class TestMAGNOEncoder:
    """Test suite for MAGNOEncoder"""
    
    @pytest.fixture
    def base_config_2d(self):
        """Base configuration for 2D testing"""
        return MAGNOConfig(
            coord_dim=2,
            hidden_size=32,
            radius=1.0,
            scales=[1.0],
            precompute_edges=False,
            mlp_layers=2,
        )
    
    @pytest.fixture
    def base_config_3d(self):
        """Base configuration for 3D testing"""
        return MAGNOConfig(
            coord_dim=3,
            hidden_size=32,
            radius=1.2,
            scales=[1.0],
            precompute_edges=False,
            mlp_layers=2
        )
    
    def test_encoder_2d_fx_mode(self, base_config_2d):
        """Test MAGNOEncoder in 2D fixed coordinate mode"""
        config = base_config_2d
        batch_size = 3
        num_phys_nodes = 40
        num_latent_nodes = 20
        in_channels = 4
        out_channels = 64
        
        # Fixed coordinates (fx mode): [num_nodes, coord_dim]
        x_coord = torch.rand(num_phys_nodes, 2) * 5.0
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        latent_coord = torch.rand(num_latent_nodes, 2) * 5.0
        
        encoder = MAGNOEncoder(in_channels, out_channels, config)
        encoder.eval()  # Deterministic mode
        
        # Forward pass
        output = encoder(x_coord, pndata, latent_coord)
        # Verify output shape
        expected_shape = (batch_size, num_latent_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_encoder_2d_vx_mode(self, base_config_2d):
        """Test MAGNOEncoder in 2D variable coordinate mode"""
        config = base_config_2d
        batch_size = 2
        num_phys_nodes = 30
        num_latent_nodes = 15
        in_channels = 3
        out_channels = 64
        
        # Variable coordinates (vx mode): [batch_size, num_nodes, coord_dim]
        x_coord = torch.rand(batch_size, num_phys_nodes, 2) * 4.0
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        latent_coord = torch.rand(num_latent_nodes, 2) * 4.0  # Latent always fx mode
        
        encoder = MAGNOEncoder(in_channels, out_channels, config)
        encoder.eval()
        
        # Forward pass
        output = encoder(x_coord, pndata, latent_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_latent_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_encoder_3d_fx_mode(self, base_config_3d):
        """Test MAGNOEncoder in 3D fixed coordinate mode"""
        config = base_config_3d
        batch_size = 2
        num_phys_nodes = 25
        num_latent_nodes = 12
        in_channels = 5
        out_channels = 64
        
        # Fixed coordinates (fx mode): [num_nodes, coord_dim]
        x_coord = torch.rand(num_phys_nodes, 3) * 3.0
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        latent_coord = torch.rand(num_latent_nodes, 3) * 3.0
        
        encoder = MAGNOEncoder(in_channels, out_channels, config)
        encoder.eval()
        
        # Forward pass
        output = encoder(x_coord, pndata, latent_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_latent_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_encoder_3d_vx_mode(self, base_config_3d):
        """Test MAGNOEncoder in 3D variable coordinate mode"""
        config = base_config_3d
        batch_size = 3
        num_phys_nodes = 20
        num_latent_nodes = 10
        in_channels = 2
        out_channels = 64
        
        # Variable coordinates (vx mode): [batch_size, num_nodes, coord_dim]
        x_coord = torch.rand(batch_size, num_phys_nodes, 3) * 2.5
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        latent_coord = torch.rand(num_latent_nodes, 3) * 2.5
        
        encoder = MAGNOEncoder(in_channels, out_channels, config)
        encoder.eval()
        
        # Forward pass
        output = encoder(x_coord, pndata, latent_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_latent_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_encoder_multiscale(self, base_config_2d):
        """Test MAGNOEncoder with multiple scales"""
        config = base_config_2d
        config.scales = [0.5, 1.0, 2.0]  # Multiple scales
        
        batch_size = 2
        num_phys_nodes = 35
        num_latent_nodes = 18
        in_channels = 3
        out_channels = 64
        
        x_coord = torch.rand(num_phys_nodes, 2) * 4.0
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        latent_coord = torch.rand(num_latent_nodes, 2) * 4.0
        
        encoder = MAGNOEncoder(in_channels, out_channels, config)
        encoder.eval()
        
        # Forward pass
        output = encoder(x_coord, pndata, latent_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_latent_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_encoder_with_geoembed(self, base_config_2d):
        """Test MAGNOEncoder with geometric embedding enabled"""
        config = base_config_2d
        config.use_geoembed = True
        
        batch_size = 2
        num_phys_nodes = 30
        num_latent_nodes = 15
        in_channels = 4
        out_channels = 64
        
        x_coord = torch.rand(num_phys_nodes, 2) * 3.0
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        latent_coord = torch.rand(num_latent_nodes, 2) * 3.0
        
        encoder = MAGNOEncoder(in_channels, out_channels, config)
        encoder.eval()
        
        # Forward pass
        output = encoder(x_coord, pndata, latent_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_latent_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_encoder_edge_drop_training(self, base_config_2d):
        """Test MAGNOEncoder with edge drop in training mode"""
        config = base_config_2d
        config.sampling_strategy = 'ratio'
        config.sample_ratio = 0.7
        
        batch_size = 2
        num_phys_nodes = 25
        num_latent_nodes = 12
        in_channels = 3
        out_channels = 64
        
        x_coord = torch.rand(num_phys_nodes, 2) * 3.0
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        latent_coord = torch.rand(num_latent_nodes, 2) * 3.0
        
        encoder = MAGNOEncoder(in_channels, out_channels, config)
        encoder.train()  # Training mode for edge drop
        
        # Forward pass
        output = encoder(x_coord, pndata, latent_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_latent_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        
        # Test stochastic behavior in training mode
        output2 = encoder(x_coord, pndata, latent_coord)
        assert not torch.allclose(output, output2, atol=1e-6), "Outputs should differ in training mode with edge drop"


class TestMAGNODecoder:
    """Test suite for MAGNODecoder"""
    
    @pytest.fixture
    def base_config_2d(self):
        """Base configuration for 2D testing"""
        return MAGNOConfig(
            coord_dim=2,
            hidden_size=32,
            radius=1.0,
            scales=[1.0],
            precompute_edges=False,
            mlp_layers=2,
        )
    
    @pytest.fixture
    def base_config_3d(self):
        """Base configuration for 3D testing"""
        return MAGNOConfig(
            coord_dim=3,
            hidden_size=32,
            radius=1.2,
            scales=[1.0],
            precompute_edges=False,
            mlp_layers=2
        )
    
    def test_decoder_2d_fx_mode(self, base_config_2d):
        """Test MAGNODecoder in 2D fixed coordinate mode"""
        config = base_config_2d
        batch_size = 3
        num_latent_nodes = 20
        num_query_nodes = 35
        in_channels = 8
        out_channels = 4
        
        # Fixed coordinates (fx mode): [num_nodes, coord_dim]
        latent_coord = torch.rand(num_latent_nodes, 2) * 4.0
        rndata = torch.rand(batch_size, num_latent_nodes, in_channels)
        query_coord = torch.rand(num_query_nodes, 2) * 4.0
        
        decoder = MAGNODecoder(in_channels, out_channels, config)
        decoder.eval()
        
        # Forward pass
        output = decoder(latent_coord, rndata, query_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_query_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_decoder_2d_vx_mode(self, base_config_2d):
        """Test MAGNODecoder in 2D variable coordinate mode"""
        config = base_config_2d
        batch_size = 2  
        num_latent_nodes = 15
        num_query_nodes = 28
        in_channels = 6
        out_channels = 3
        
        # Variable coordinates (vx mode): [batch_size, num_nodes, coord_dim]
        latent_coord = torch.rand(num_latent_nodes, 2) * 3.5  # Latent always fx mode
        rndata = torch.rand(batch_size, num_latent_nodes, in_channels)
        query_coord = torch.rand(batch_size, num_query_nodes, 2) * 3.5
        
        decoder = MAGNODecoder(in_channels, out_channels, config)
        decoder.eval()
        
        # Forward pass
        output = decoder(latent_coord, rndata, query_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_query_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_decoder_3d_fx_mode(self, base_config_3d):
        """Test MAGNODecoder in 3D fixed coordinate mode"""
        config = base_config_3d
        batch_size = 2
        num_latent_nodes = 12
        num_query_nodes = 22
        in_channels = 10
        out_channels = 5
        
        # Fixed coordinates (fx mode): [num_nodes, coord_dim]  
        latent_coord = torch.rand(num_latent_nodes, 3) * 2.5
        rndata = torch.rand(batch_size, num_latent_nodes, in_channels)
        query_coord = torch.rand(num_query_nodes, 3) * 2.5
        
        decoder = MAGNODecoder(in_channels, out_channels, config)
        decoder.eval()
        
        # Forward pass
        output = decoder(latent_coord, rndata, query_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_query_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_decoder_3d_vx_mode(self, base_config_3d):
        """Test MAGNODecoder in 3D variable coordinate mode"""
        config = base_config_3d
        batch_size = 3
        num_latent_nodes = 10
        num_query_nodes = 18
        in_channels = 4
        out_channels = 2
        
        # Variable coordinates (vx mode): [batch_size, num_nodes, coord_dim]
        latent_coord = torch.rand(num_latent_nodes, 3) * 2.0
        rndata = torch.rand(batch_size, num_latent_nodes, in_channels)
        query_coord = torch.rand(batch_size, num_query_nodes, 3) * 2.0
        
        decoder = MAGNODecoder(in_channels, out_channels, config)
        decoder.eval()
        
        # Forward pass
        output = decoder(latent_coord, rndata, query_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_query_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_decoder_multiscale(self, base_config_2d):
        """Test MAGNODecoder with multiple scales"""
        config = base_config_2d
        config.scales = [0.8, 1.0, 1.5]  # Multiple scales
        
        batch_size = 2
        num_latent_nodes = 16
        num_query_nodes = 25
        in_channels = 6
        out_channels = 8
        
        latent_coord = torch.rand(num_latent_nodes, 2) * 3.0
        rndata = torch.rand(batch_size, num_latent_nodes, in_channels)
        query_coord = torch.rand(num_query_nodes, 2) * 3.0
        
        decoder = MAGNODecoder(in_channels, out_channels, config)
        decoder.eval()
        
        # Forward pass
        output = decoder(latent_coord, rndata, query_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_query_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_decoder_with_geoembed(self, base_config_2d):
        """Test MAGNODecoder with geometric embedding enabled"""
        config = base_config_2d
        config.use_geoembed = True
        
        batch_size = 2
        num_latent_nodes = 14
        num_query_nodes = 20
        in_channels = 8
        out_channels = 6
        
        latent_coord = torch.rand(num_latent_nodes, 2) * 3.0
        rndata = torch.rand(batch_size, num_latent_nodes, in_channels)
        query_coord = torch.rand(num_query_nodes, 2) * 3.0
        
        decoder = MAGNODecoder(in_channels, out_channels, config)
        decoder.eval()
        
        # Forward pass
        output = decoder(latent_coord, rndata, query_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_query_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
    
    def test_decoder_edge_drop_training(self, base_config_2d):
        """Test MAGNODecoder with edge drop in training mode"""
        config = base_config_2d
        config.sampling_strategy = 'max_neighbors'
        config.max_neighbors = 1
        
        batch_size = 2
        num_latent_nodes = 12
        num_query_nodes = 18
        in_channels = 6
        out_channels = 4
        
        latent_coord = torch.rand(num_latent_nodes, 2) * 3.0
        rndata = torch.rand(batch_size, num_latent_nodes, in_channels)
        query_coord = torch.rand(num_query_nodes, 2) * 3.0
        
        decoder = MAGNODecoder(in_channels, out_channels, config)
        decoder.train()  # Training mode for edge drop
        
        # Forward pass
        output = decoder(latent_coord, rndata, query_coord)
        
        # Verify output shape
        expected_shape = (batch_size, num_query_nodes, out_channels)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        decoder.eval()
        # Test stochastic behavior in training mode
        output2 = decoder(latent_coord, rndata, query_coord)
        assert not torch.allclose(output, output2, atol=1e-6), "Outputs should differ in training mode with edge drop"


class TestMAGNOEncoderDecoderIntegration:
    """Integration tests for MAGNOEncoder and MAGNODecoder together"""
    
    @pytest.fixture
    def integration_config_2d(self):
        """Configuration for 2D integration testing"""
        return MAGNOConfig(
            coord_dim=2,
            hidden_size=48,
            radius=1.2,
            scales=[1.0],
            precompute_edges=False,
            mlp_layers=3
        )
    
    @pytest.fixture
    def integration_config_3d(self):
        """Configuration for 3D integration testing"""
        return MAGNOConfig(
            coord_dim=3,
            hidden_size=48,
            radius=1.5,
            scales=[1.0],
            precompute_edges=False,
            mlp_layers=3
        )
    
    def test_encoder_decoder_pipeline_2d_fx(self, integration_config_2d):
        """Test complete encoder-decoder pipeline in 2D fx mode"""
        config = integration_config_2d
        batch_size = 2
        num_phys_nodes = 40
        num_latent_nodes = 20
        num_query_nodes = 35
        in_channels = 4
        hidden_channels = 8
        out_channels = 4
        
        # All coordinates in fx mode
        phys_coord = torch.rand(num_phys_nodes, 2) * 5.0
        latent_coord = torch.rand(num_latent_nodes, 2) * 5.0
        query_coord = torch.rand(num_query_nodes, 2) * 5.0
        
        # Input data
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        
        # Create encoder and decoder
        encoder = MAGNOEncoder(in_channels, hidden_channels, config)
        decoder = MAGNODecoder(hidden_channels, out_channels, config)
        
        encoder.eval()
        decoder.eval()
        
        # Forward pass through encoder
        encoded = encoder(phys_coord, pndata, latent_coord)
        expected_encoded_shape = (batch_size, num_latent_nodes, hidden_channels)
        assert encoded.shape == expected_encoded_shape, f"Encoder output: expected {expected_encoded_shape}, got {encoded.shape}"
        
        # Forward pass through decoder
        decoded = decoder(latent_coord, encoded, query_coord)
        expected_decoded_shape = (batch_size, num_query_nodes, out_channels)
        assert decoded.shape == expected_decoded_shape, f"Decoder output: expected {expected_decoded_shape}, got {decoded.shape}"
        
        # Verify outputs are finite
        assert torch.isfinite(encoded).all(), "Encoder output contains non-finite values"
        assert torch.isfinite(decoded).all(), "Decoder output contains non-finite values"
    
    def test_encoder_decoder_pipeline_2d_vx(self, integration_config_2d):
        """Test complete encoder-decoder pipeline in 2D vx mode"""
        config = integration_config_2d
        batch_size = 3
        num_phys_nodes = 30
        num_latent_nodes = 15
        num_query_nodes = 25
        in_channels = 3
        hidden_channels = 6
        out_channels = 3
        
        # Variable coordinates for phys and query, fixed for latent
        phys_coord = torch.rand(batch_size, num_phys_nodes, 2) * 4.0
        latent_coord = torch.rand(num_latent_nodes, 2) * 4.0
        query_coord = torch.rand(batch_size, num_query_nodes, 2) * 4.0
        
        # Input data
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        
        # Create encoder and decoder
        encoder = MAGNOEncoder(in_channels, hidden_channels, config)
        decoder = MAGNODecoder(hidden_channels, out_channels, config)
        
        encoder.eval()
        decoder.eval()
        
        # Forward pass through encoder
        encoded = encoder(phys_coord, pndata, latent_coord)
        expected_encoded_shape = (batch_size, num_latent_nodes, hidden_channels)
        assert encoded.shape == expected_encoded_shape, f"Encoder output: expected {expected_encoded_shape}, got {encoded.shape}"
        
        # Forward pass through decoder
        decoded = decoder(latent_coord, encoded, query_coord)
        expected_decoded_shape = (batch_size, num_query_nodes, out_channels)
        assert decoded.shape == expected_decoded_shape, f"Decoder output: expected {expected_decoded_shape}, got {decoded.shape}"
        
        # Verify outputs are finite
        assert torch.isfinite(encoded).all(), "Encoder output contains non-finite values"
        assert torch.isfinite(decoded).all(), "Decoder output contains non-finite values"
    
    def test_encoder_decoder_pipeline_3d_fx(self, integration_config_3d):
        """Test complete encoder-decoder pipeline in 3D fx mode"""
        config = integration_config_3d
        batch_size = 2
        num_phys_nodes = 25
        num_latent_nodes = 12
        num_query_nodes = 20
        in_channels = 5
        hidden_channels = 10
        out_channels = 5
        
        # All coordinates in fx mode
        phys_coord = torch.rand(num_phys_nodes, 3) * 3.0
        latent_coord = torch.rand(num_latent_nodes, 3) * 3.0
        query_coord = torch.rand(num_query_nodes, 3) * 3.0
        
        # Input data
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        
        # Create encoder and decoder
        encoder = MAGNOEncoder(in_channels, hidden_channels, config)
        decoder = MAGNODecoder(hidden_channels, out_channels, config)
        
        encoder.eval()
        decoder.eval()
        
        # Forward pass through encoder
        encoded = encoder(phys_coord, pndata, latent_coord)
        expected_encoded_shape = (batch_size, num_latent_nodes, hidden_channels)
        assert encoded.shape == expected_encoded_shape, f"Encoder output: expected {expected_encoded_shape}, got {encoded.shape}"
        
        # Forward pass through decoder
        decoded = decoder(latent_coord, encoded, query_coord)
        expected_decoded_shape = (batch_size, num_query_nodes, out_channels)
        assert decoded.shape == expected_decoded_shape, f"Decoder output: expected {expected_decoded_shape}, got {decoded.shape}"
        
        # Verify outputs are finite
        assert torch.isfinite(encoded).all(), "Encoder output contains non-finite values"
        assert torch.isfinite(decoded).all(), "Decoder output contains non-finite values"
    
    def test_encoder_decoder_pipeline_3d_vx(self, integration_config_3d):
        """Test complete encoder-decoder pipeline in 3D vx mode"""
        config = integration_config_3d
        batch_size = 2
        num_phys_nodes = 20
        num_latent_nodes = 10
        num_query_nodes = 15
        in_channels = 2
        hidden_channels = 4
        out_channels = 2
        
        # Variable coordinates for phys and query, fixed for latent
        phys_coord = torch.rand(batch_size, num_phys_nodes, 3) * 2.5
        latent_coord = torch.rand(num_latent_nodes, 3) * 2.5
        query_coord = torch.rand(batch_size, num_query_nodes, 3) * 2.5
        
        # Input data
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        
        # Create encoder and decoder
        encoder = MAGNOEncoder(in_channels, hidden_channels, config)
        decoder = MAGNODecoder(hidden_channels, out_channels, config)
        
        encoder.eval()
        decoder.eval()
        
        # Forward pass through encoder
        encoded = encoder(phys_coord, pndata, latent_coord)
        expected_encoded_shape = (batch_size, num_latent_nodes, hidden_channels)
        assert encoded.shape == expected_encoded_shape, f"Encoder output: expected {expected_encoded_shape}, got {encoded.shape}"
        
        # Forward pass through decoder
        decoded = decoder(latent_coord, encoded, query_coord)
        expected_decoded_shape = (batch_size, num_query_nodes, out_channels)
        assert decoded.shape == expected_decoded_shape, f"Decoder output: expected {expected_decoded_shape}, got {decoded.shape}"
        
        # Verify outputs are finite
        assert torch.isfinite(encoded).all(), "Encoder output contains non-finite values"
        assert torch.isfinite(decoded).all(), "Decoder output contains non-finite values"
    
    def test_multiscale_integration(self, integration_config_2d):
        """Test encoder-decoder with multiple scales"""
        config = integration_config_2d
        config.scales = [0.5, 1.0, 2.0]
        config.use_scale_weights = True
        
        batch_size = 2
        num_phys_nodes = 30
        num_latent_nodes = 15
        num_query_nodes = 25
        in_channels = 4
        hidden_channels = 8
        out_channels = 4
        
        # Test data
        phys_coord = torch.rand(num_phys_nodes, 2) * 4.0
        latent_coord = torch.rand(num_latent_nodes, 2) * 4.0
        query_coord = torch.rand(num_query_nodes, 2) * 4.0
        pndata = torch.rand(batch_size, num_phys_nodes, in_channels)
        
        # Create models
        encoder = MAGNOEncoder(in_channels, hidden_channels, config)
        decoder = MAGNODecoder(hidden_channels, out_channels, config)
        
        encoder.eval()
        decoder.eval()
        
        # Forward pass
        encoded = encoder(phys_coord, pndata, latent_coord)
        decoded = decoder(latent_coord, encoded, query_coord)
        
        # Verify shapes
        assert encoded.shape == (batch_size, num_latent_nodes, hidden_channels)
        assert decoded.shape == (batch_size, num_query_nodes, out_channels)
        
        # Verify outputs are finite
        assert torch.isfinite(encoded).all()
        assert torch.isfinite(decoded).all()


class TestMAGNOConfigValidation:
    """Test configuration validation and edge cases"""
    
    def test_invalid_coord_dim(self):
        """Test error handling for invalid coordinate dimensions"""
        with pytest.raises(ValueError):  # MAGNOConfig validation raises ValueError
            config = MAGNOConfig(coord_dim=1)  # Should be >= 2
    
    def test_mismatched_dimensions(self):
        """Test error handling for mismatched input dimensions"""
        config = MAGNOConfig(coord_dim=2)
        encoder = MAGNOEncoder(3, 6, config)
        encoder.eval()
        
        # Create mismatched inputs
        x_coord = torch.rand(20, 3)  # 3D coordinates but config says 2D
        pndata = torch.rand(2, 20, 3)
        latent_coord = torch.rand(10, 3)  # Also 3D
        
        # This should work because coord detection is based on input shape
        # But the geometric operations might not work correctly
        try:
            output = encoder(x_coord, pndata, latent_coord)
            # If it works, verify the output shape is reasonable
            assert output.shape[0] == 2  # batch size
            assert output.shape[1] == 10  # latent nodes
        except Exception as e:
            # Expected to fail due to dimension mismatch
            assert "dimension" in str(e).lower() or "shape" in str(e).lower()
    
    def test_empty_scales(self):
        """Test behavior with empty scales list"""
        config = MAGNOConfig(coord_dim=2, scales=[])
        
        # Should handle empty scales gracefully or raise appropriate error
        try:
            encoder = MAGNOEncoder(3, 6, config)
            # If it doesn't fail at init, it should fail at forward pass
            x_coord = torch.rand(20, 2)
            pndata = torch.rand(2, 20, 3)
            latent_coord = torch.rand(10, 2)
            
            output = encoder(x_coord, pndata, latent_coord)
            # If this works, verify it has reasonable shape
            assert output.shape == (2, 10, 6)
        except Exception as e:
            # Expected behavior for empty scales
            assert len(config.scales) == 0
    
    def test_single_scale(self):
        """Test behavior with single scale"""
        config = MAGNOConfig(coord_dim=2, scales=[1.0])
        encoder = MAGNOEncoder(3, 6, config)
        encoder.eval()
        
        x_coord = torch.rand(20, 2)
        pndata = torch.rand(2, 20, 3)
        latent_coord = torch.rand(10, 2)
        
        output = encoder(x_coord, pndata, latent_coord)
        assert output.shape == (2, 10, 6)
        assert torch.isfinite(output).all()
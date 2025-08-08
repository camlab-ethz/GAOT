import torch
import pytest
from dataclasses import dataclass, field
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../new_src'))

from model.gaot import GAOT
from model.layers.attn import TransformerConfig
from model.layers.magno import MAGNOConfig


@dataclass
class Args:
    magno: MAGNOConfig = field(default_factory=MAGNOConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass
class GAOTConfig:
    args: Args = field(default_factory=Args)
    latent_tokens_size: List[int] = field(default_factory=lambda: [16, 16])


class TestGAOT:
    """Test suite for GAOT model"""
    
    def _get_config_2d(self):
        config = GAOTConfig()
        # Note: Using coord_dim to match MAGNOConfig
        config.args.magno.coord_dim = 2  
        config.latent_tokens_size = [16, 16]
        return config
    
    def _get_config_3d(self):
        config = GAOTConfig()
        # Note: Using coord_dim to match MAGNOConfig
        config.args.magno.coord_dim = 3
        config.latent_tokens_size = [8, 8, 8]
        return config
    
    def test_gaot_2d_fx_initialization(self):
        """Test GAOT 2D fixed coordinates initialization"""
        config = self._get_config_2d()
        
        model = GAOT(
            input_size=3,
            output_size=2,
            config=config
        )
        
        assert model.coord_dim == 2
        assert model.H == 16
        assert model.W == 16
        assert model.D is None
        assert model.input_size == 3
        assert model.output_size == 2
        
    def test_gaot_2d_vx_initialization(self):
        """Test GAOT 2D variable coordinates initialization"""
        config = self._get_config_2d()
        
        model = GAOT(
            input_size=4,
            output_size=3,
            config=config
        )
        
        assert model.coord_dim == 2
        assert model.H == 16
        assert model.W == 16
        assert model.D is None
        
    def test_gaot_3d_fx_initialization(self):
        """Test GAOT 3D fixed coordinates initialization"""
        config = self._get_config_3d()
        
        model = GAOT(
            input_size=3,
            output_size=2,
            config=config
        )
        
        assert model.coord_dim == 3
        assert model.H == 8
        assert model.W == 8
        assert model.D == 8
        
    def test_gaot_3d_vx_initialization(self):
        """Test GAOT 3D variable coordinates initialization"""
        config = self._get_config_3d()
        
        model = GAOT(
            input_size=4,
            output_size=3,
            config=config
        )
        
        assert model.coord_dim == 3
        assert model.H == 8
        assert model.W == 8
        assert model.D == 8
    
    def test_invalid_coord_dim(self):
        """Test that invalid coord_dim raises error"""
        config = self._get_config_2d()
        config.args.magno.coord_dim = 4  # Invalid
        
        with pytest.raises(ValueError, match="coord_dim must be 2 or 3"):
            GAOT(
                input_size=3,
                output_size=2,
                config=config
            )
    
    def test_invalid_latent_tokens_size_2d(self):
        """Test that invalid latent_tokens_size for 2D raises error"""
        config = GAOTConfig()
        config.args.magno.coord_dim = 2
        config.latent_tokens_size = [16, 16, 16]  # Should be 2D
        
        with pytest.raises(ValueError, match="For 2D, latent_tokens_size must have 2 dimensions"):
            GAOT(
                input_size=3,
                output_size=2,
                config=config
            )
    
    def test_invalid_latent_tokens_size_3d(self):
        """Test that invalid latent_tokens_size for 3D raises error"""
        config = GAOTConfig()
        config.args.magno.coord_dim = 3
        config.latent_tokens_size = [16, 16]  # Should be 3D
        
        with pytest.raises(ValueError, match="For 3D, latent_tokens_size must have 3 dimensions"):
            GAOT(
                input_size=3,
                output_size=2,
                config=config
            )
    
    def test_patch_positions_2d(self):
        """Test patch position generation for 2D"""
        config = self._get_config_2d()
        model = GAOT(input_size=3, output_size=2, config=config)
        
        # H=16, W=16, patch_size=8 -> 2x2 patches (default patch_size is 8)
        positions = model.positions
        assert positions.shape == (4, 2)  # 2*2 patches, 2D coordinates
        assert torch.all(positions >= 0)
        assert torch.all(positions[:, 0] < 2)  # H patches
        assert torch.all(positions[:, 1] < 2)  # W patches
    
    def test_patch_positions_3d(self):
        """Test patch position generation for 3D"""
        config = self._get_config_3d()
        model = GAOT(input_size=3, output_size=2, config=config)
        
        # H=8, W=8, D=8, patch_size=8 -> 1x1x1 patches (default patch_size is 8)
        positions = model.positions
        assert positions.shape == (1, 3)  # 1*1*1 patches, 3D coordinates
        assert torch.all(positions >= 0)
        assert torch.all(positions[:, 0] < 1)  # H patches
        assert torch.all(positions[:, 1] < 1)  # W patches
        assert torch.all(positions[:, 2] < 1)  # D patches
    
    def test_process_2d_shape(self):
        """Test process method for 2D maintains correct shapes"""
        config = self._get_config_2d()
        model = GAOT(input_size=3, output_size=2, config=config)
        
        batch_size = 2
        n_regional_nodes = 16 * 16  # H * W
        node_latent_size = 32  # Default lifting_channels
        
        # Input tensor
        rndata = torch.randn(batch_size, n_regional_nodes, node_latent_size)
        
        # Process
        output = model.process(rndata)
        
        # Check output shape
        assert output.shape == (batch_size, n_regional_nodes, node_latent_size)
    
    def test_process_3d_shape(self):
        """Test process method for 3D maintains correct shapes"""
        config = self._get_config_3d()
        model = GAOT(input_size=3, output_size=2, config=config)
        
        batch_size = 2
        n_regional_nodes = 8 * 8 * 8  # H * W * D
        node_latent_size = 32  # Default lifting_channels
        
        # Input tensor
        rndata = torch.randn(batch_size, n_regional_nodes, node_latent_size)
        
        # Process
        output = model.process(rndata)
        
        # Check output shape
        assert output.shape == (batch_size, n_regional_nodes, node_latent_size)
    
    def test_process_invalid_shape_2d(self):
        """Test process method with invalid regional nodes count for 2D"""
        config = self._get_config_2d()
        model = GAOT(input_size=3, output_size=2, config=config)
        
        batch_size = 2
        wrong_n_regional_nodes = 100  # Should be 16*16=256
        node_latent_size = 32
        
        rndata = torch.randn(batch_size, wrong_n_regional_nodes, node_latent_size)
        
        with pytest.raises(AssertionError, match="n_regional_nodes.*!=.*H.*W"):
            model.process(rndata)
    
    def test_process_invalid_shape_3d(self):
        """Test process method with invalid regional nodes count for 3D"""
        config = self._get_config_3d()
        model = GAOT(input_size=3, output_size=2, config=config)
        
        batch_size = 2
        wrong_n_regional_nodes = 100  # Should be 8*8*8=512
        node_latent_size = 32
        
        rndata = torch.randn(batch_size, wrong_n_regional_nodes, node_latent_size)
        
        with pytest.raises(AssertionError, match="n_regional_nodes.*!=.*H.*W.*D"):
            model.process(rndata)
    
    def test_absolute_embeddings_2d(self):
        """Test absolute embedding computation for 2D"""
        config = self._get_config_2d()
        config.args.transformer.positional_embedding = 'absolute'
        model = GAOT(input_size=3, output_size=2, config=config)
        
        positions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        embed_dim = 64
        
        embeddings = model._compute_absolute_embeddings(positions, embed_dim)
        
        assert embeddings.shape == (4, embed_dim)
        # Check that embeddings for different positions are different
        assert not torch.equal(embeddings[0], embeddings[1])
    
    def test_absolute_embeddings_3d(self):
        """Test absolute embedding computation for 3D"""
        config = self._get_config_3d()
        config.args.transformer.positional_embedding = 'absolute'
        model = GAOT(input_size=3, output_size=2, config=config)
        
        positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32)
        embed_dim = 64
        
        embeddings = model._compute_absolute_embeddings(positions, embed_dim)
        
        assert embeddings.shape == (4, embed_dim)
        # Check that embeddings for different positions are different
        assert not torch.equal(embeddings[0], embeddings[1])
    
    @pytest.mark.parametrize("coord_dim", [2, 3])
    def test_forward_shapes(self, coord_dim):
        """Test forward pass shapes for all combinations"""
        if coord_dim == 2:
            config = self._get_config_2d()
            H, W = 16, 16
            regional_nodes = H * W
        else:
            config = self._get_config_3d()
            H, W, D = 8, 8, 8
            regional_nodes = H * W * D
        
        model = GAOT(
            input_size=3,
            output_size=2,
            config=config
        )
        
        batch_size = 2
        n_physical_nodes = 100
        
        # Create coordinate tensors (testing fx mode pattern)
        if coord_dim == 2:
            xcoord_fx = torch.randn(n_physical_nodes, 2)
            latent_tokens_coord_fx = torch.randn(regional_nodes, 2)
        else:
            xcoord_fx = torch.randn(n_physical_nodes, 3)
            latent_tokens_coord_fx = torch.randn(regional_nodes, 3)
        
        pndata = torch.randn(batch_size, n_physical_nodes, 3)
        
        # Mock forward pass by testing individual components
        # Note: Full forward pass requires proper neighbor computation which is complex
        
        # Test encoding step (mock with regional data)
        rndata = torch.randn(batch_size, regional_nodes, config.args.magno.lifting_channels)
        
        # Test processing step
        processed = model.process(rndata)
        assert processed.shape == (batch_size, regional_nodes, config.args.magno.lifting_channels)
        
        # Note: Full forward testing would require proper neighbor computation
        # which is complex and tested in integration tests
    
    def test_model_device_movement(self):
        """Test that model can be moved to different devices"""
        config = self._get_config_2d()
        model = GAOT(input_size=3, output_size=2, config=config)
        
        # Check that model starts on CPU
        assert next(model.parameters()).device.type == 'cpu'
        
        # Test moving to CPU (should work)
        model = model.cpu()
        assert next(model.parameters()).device.type == 'cpu'
        
        # If CUDA is available, test GPU movement
        if torch.cuda.is_available():
            model = model.cuda()
            assert next(model.parameters()).device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__])
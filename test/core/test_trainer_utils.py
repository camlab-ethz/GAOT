"""
Unit tests for trainer utilities, focusing on sequential data processing.
"""
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from new_src.core.trainer_utils import compute_sequential_stats, move_to_device, denormalize_data


class TestComputeSequentialStats:
    """Test cases for compute_sequential_stats function."""
    
    def setup_method(self):
        """Set up test data."""
        # Create mock sequential data
        # Shape: [num_samples, num_timesteps, num_nodes, num_vars]
        self.num_samples = 8
        self.num_timesteps = 16
        self.num_nodes = 64
        self.u_vars = 2
        self.c_vars = 1
        
        # Create synthetic u_train data
        np.random.seed(42)
        self.u_train = np.random.randn(self.num_samples, self.num_timesteps, self.num_nodes, self.u_vars)
        
        # Create synthetic c_train data
        self.c_train = np.random.randn(self.num_samples, self.num_timesteps, self.num_nodes, self.c_vars)
        
        # Create time values
        self.t_values = np.linspace(0.0, 1.0, self.num_timesteps)
        
        # Create mock metadata
        self.metadata = MagicMock()
        self.metadata.u_mean = np.array([0.1, 0.2])
        self.metadata.u_std = np.array([1.1, 1.2])
        self.metadata.c_mean = np.array([0.05])
        self.metadata.c_std = np.array([0.8])
    
    def test_compute_sequential_stats_basic(self):
        """Test basic functionality of compute_sequential_stats."""
        stats = compute_sequential_stats(
            u_data=self.u_train,
            c_data=self.c_train,
            t_values=self.t_values,
            metadata=self.metadata,
            max_time_diff=14,
            sample_rate=1.0,
            use_metadata_stats=False,
            use_time_norm=True
        )
        
        # Check that all required keys are present
        required_keys = ['u', 'c', 'start_time', 'time_diffs', 'res', 'der']
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
        
        # Check u statistics
        assert 'mean' in stats['u'] and 'std' in stats['u']
        assert stats['u']['mean'].shape == (self.u_vars,)
        assert stats['u']['std'].shape == (self.u_vars,)
        assert np.all(stats['u']['std'] > 0), "Standard deviation should be positive"
        
        # Check c statistics
        assert 'mean' in stats['c'] and 'std' in stats['c']
        assert stats['c']['mean'].shape == (self.c_vars,)
        assert stats['c']['std'].shape == (self.c_vars,)
        
        # Check time statistics
        assert isinstance(stats['start_time']['mean'], (float, np.floating))
        assert isinstance(stats['start_time']['std'], (float, np.floating))
        assert isinstance(stats['time_diffs']['mean'], (float, np.floating))
        assert isinstance(stats['time_diffs']['std'], (float, np.floating))
        assert stats['start_time']['std'] > 0
        assert stats['time_diffs']['std'] > 0

        # Check residual and derivative statistics
        assert stats['res']['mean'].shape == (self.u_vars,)
        assert stats['res']['std'].shape == (self.u_vars,)
        assert stats['der']['mean'].shape == (self.u_vars,)
        assert stats['der']['std'].shape == (self.u_vars,)
    
    def test_compute_sequential_stats_with_metadata_stats(self):
        """Test using metadata-provided statistics."""
        stats = compute_sequential_stats(
            u_data=self.u_train,
            c_data=self.c_train,
            t_values=self.t_values,
            metadata=self.metadata,
            max_time_diff=14,
            use_metadata_stats=True,
            use_time_norm=True
        )
        # Should use metadata stats for u and c
        np.testing.assert_array_equal(stats['u']['mean'], self.metadata.u_mean)
        np.testing.assert_array_equal(stats['u']['std'], self.metadata.u_std)
        np.testing.assert_array_equal(stats['c']['mean'], self.metadata.c_mean)
        np.testing.assert_array_equal(stats['c']['std'], self.metadata.c_std)
    
    def test_compute_sequential_stats_no_c_data(self):
        """Test with no condition data."""
        stats = compute_sequential_stats(
            u_data=self.u_train,
            c_data=None,
            t_values=self.t_values,
            metadata=self.metadata,
            max_time_diff=14,
            use_time_norm=True
        )
        
        # Should not have c statistics
        assert 'c' not in stats
        
        # Should still have u and time statistics
        assert 'u' in stats
        assert 'start_time' in stats
        assert 'time_diffs' in stats
    
    def test_compute_sequential_stats_no_time_norm(self):
        """Test without time normalization."""
        stats = compute_sequential_stats(
            u_data=self.u_train,
            c_data=self.c_train,
            t_values=self.t_values,
            metadata=self.metadata,
            max_time_diff=14,
            use_time_norm=False
        )
        
        # Should not have time statistics
        assert 'start_time' not in stats
        assert 'time_diffs' not in stats
        
        # Should still have u, c, res, der statistics
        assert 'u' in stats
        assert 'c' in stats
        assert 'res' in stats
        assert 'der' in stats
    
    def test_compute_sequential_stats_reduced_sample_rate(self):
        """Test with reduced sample rate for statistics computation."""
        stats = compute_sequential_stats(
            u_data=self.u_train,
            c_data=self.c_train,
            t_values=self.t_values,
            metadata=self.metadata,
            max_time_diff=14,
            sample_rate=0.5,  # Use only half the samples
            use_time_norm=True
        )
        
        # Should still compute all statistics
        assert 'u' in stats
        assert 'c' in stats
        assert 'res' in stats
        assert 'der' in stats
        
        # Statistics should be reasonable (not exact due to sampling)
        assert stats['u']['mean'].shape == (self.u_vars,)
        assert stats['res']['std'].shape == (self.u_vars,)
    
    def test_compute_sequential_stats_time_pairs_generation(self):
        """Test that time pairs are generated correctly."""
        max_time_diff = 6
        stats = compute_sequential_stats(
            u_data=self.u_train,
            c_data=self.c_train,
            t_values=self.t_values,
            metadata=self.metadata,
            max_time_diff=max_time_diff,
            use_time_norm=True
        )
        
        # Time differences should be within expected range
        assert stats['time_diffs']['mean'] > 0
        
        # Check that start times are reasonable
        assert 0 <= stats['start_time']['mean'] <= 1.0
    
    def test_epsilon_handling(self):
        """Test that epsilon is properly added to prevent division by zero."""
        # Create data with zero variance in one dimension
        u_constant = np.zeros_like(self.u_train)
        u_constant[..., 0] = 1.0  # First dimension is constant
        u_constant[..., 1] = self.u_train[..., 1]  # Second dimension varies
        
        stats = compute_sequential_stats(
            u_data=u_constant,
            c_data=self.c_train,
            t_values=self.t_values,
            metadata=self.metadata,
            max_time_diff=14,
            use_time_norm=True
        )
        
        # All standard deviations should be positive (due to epsilon)
        assert np.all(stats['u']['std'] > 0)
        assert np.all(stats['c']['std'] > 0)
        assert np.all(stats['res']['std'] > 0)
        assert np.all(stats['der']['std'] > 0)


class TestTrainerUtilsOther:
    """Test other trainer utility functions."""
    
    def test_move_to_device_tensor(self):
        """Test moving tensor to device."""
        device = torch.device('cpu')
        tensor = torch.randn(3, 4)
        result = move_to_device(tensor, device)
        assert result.device == device
        assert torch.equal(tensor, result)
    
    def test_move_to_device_dict(self):
        """Test moving dictionary of tensors to device."""
        device = torch.device('cpu')
        data = {
            'a': torch.randn(2, 3),
            'b': torch.randn(4, 5),
            'c': 'not_a_tensor'  # Should remain unchanged
        }
        
        result = move_to_device(data, device)
        
        assert isinstance(result, dict)
        assert result['a'].device == device
        assert result['b'].device == device
        assert result['c'] == 'not_a_tensor'
    
    def test_move_to_device_list(self):
        """Test moving list of tensors to device."""
        device = torch.device('cpu')
        data = [torch.randn(2, 3), torch.randn(4, 5), 'not_a_tensor']
        
        result = move_to_device(data, device)
        
        assert isinstance(result, list)
        assert result[0].device == device
        assert result[1].device == device
        assert result[2] == 'not_a_tensor'
    
    def test_move_to_device_nested(self):
        """Test moving nested structure to device."""
        device = torch.device('cpu')
        data = {
            'level1': {
                'tensor': torch.randn(2, 2),
                'list': [torch.randn(3, 3), torch.randn(1, 1)]
            },
            'simple': torch.randn(4, 4)
        }
        
        result = move_to_device(data, device)
        
        assert result['level1']['tensor'].device == device
        assert result['level1']['list'][0].device == device
        assert result['level1']['list'][1].device == device
        assert result['simple'].device == device
    
    def test_denormalize_data(self):
        """Test data denormalization."""
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([0.5, 1.5])
        normalized_data = torch.tensor([[0.0, 0.0], [2.0, -1.0]])  # [batch, features]
        
        expected = torch.tensor([[1.0, 2.0], [2.0, 0.5]])  # data * std + mean
        result = denormalize_data(normalized_data, mean, std)
        
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    
    def test_denormalize_data_broadcast(self):
        """Test denormalization with broadcasting."""
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([0.5, 1.5])
        # Shape: [batch, nodes, features]
        normalized_data = torch.zeros(2, 3, 2)
        
        result = denormalize_data(normalized_data, mean, std)
        
        # Should broadcast mean across batch and node dimensions
        expected = mean.expand(2, 3, 2)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__])
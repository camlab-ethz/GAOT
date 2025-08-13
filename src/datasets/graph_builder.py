"""
Graph building utilities for variable coordinate datasets.
Handles neighbor computation and graph construction for VX mode.
"""
import time
import torch
from typing import List, Tuple, Optional

from ..model.layers.utils.neighbor_search import NeighborSearch
from ..utils.scaling import rescale


class GraphBuilder:
    """
    Builds encoder and decoder graphs for variable coordinate datasets.
    Handles neighbor computation with multiple radius scales.
    """
    
    def __init__(self, neighbor_search_method: str = "auto"):
        """
        Initialize graph builder.
        
        Args:
            neighbor_search_method: Method for neighbor search
        """
        self.nb_search = NeighborSearch(neighbor_search_method)
    
    def build_graphs_for_split(self, x_data: torch.Tensor, latent_queries: torch.Tensor,
                              gno_radius: float, scales: List[float]) -> Tuple[List, List]:
        """
        Build encoder and decoder graphs for a data split.
        
        Args:
            x_data: Coordinate data [n_samples, n_nodes, coord_dim] or [n_samples, 1, n_nodes, coord_dim]
            latent_queries: Latent query coordinates [n_latent, coord_dim]
            gno_radius: Base radius for neighbor search
            scales: List of scale factors for multi-scale graphs
            
        Returns:
            tuple: (encoder_graphs_list, decoder_graphs_list)
        """
        print(f"Building graphs for {len(x_data)} samples...")
        start_time = time.time()
        
        encoder_graphs = []
        decoder_graphs = []
        
        for i, x_sample in enumerate(x_data):
            # Handle different input shapes
            if x_sample.dim() == 3 and x_sample.shape[0] == 1:
                # Shape: [1, n_nodes, coord_dim] -> [n_nodes, coord_dim]
                x_coord = x_sample[0]
            elif x_sample.dim() == 2:
                # Shape: [n_nodes, coord_dim]
                x_coord = x_sample
            else:
                raise ValueError(f"Unexpected coordinate shape: {x_sample.shape}")
            
            # Rescale coordinates to [-1, 1] range
            x_coord_scaled = rescale(x_coord, (-1, 1))
            
            # Build encoder graphs (physical -> latent)
            encoder_nbrs_sample = []
            for scale in scales:
                scaled_radius = gno_radius * scale
                with torch.no_grad():
                    nbrs = self.nb_search(x_coord_scaled, latent_queries, scaled_radius)
                encoder_nbrs_sample.append(nbrs)
            encoder_graphs.append(encoder_nbrs_sample)
            
            # Build decoder graphs (latent -> physical)
            decoder_nbrs_sample = []
            for scale in scales:
                scaled_radius = gno_radius * scale
                with torch.no_grad():
                    nbrs = self.nb_search(latent_queries, x_coord_scaled, scaled_radius)
                decoder_nbrs_sample.append(nbrs)
            decoder_graphs.append(decoder_nbrs_sample)
            
            if (i + 1) % 100 == 0 or i == len(x_data) - 1:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{len(x_data)} samples ({elapsed:.2f}s)")
        
        total_time = time.time() - start_time
        print(f"Graph building completed in {total_time:.2f}s")
        
        return encoder_graphs, decoder_graphs
    
    def build_all_graphs(self, data_splits: dict, latent_queries: torch.Tensor,
                        gno_radius: float, scales: List[float],
                        build_train: bool = True) -> dict:
        """
        Build graphs for all data splits.
        
        Args:
            data_splits: Dictionary with train/val/test splits
            latent_queries: Latent query coordinates
            gno_radius: Base radius for neighbor search
            scales: Scale factors for multi-scale graphs
            build_train: Whether to build train/val graphs (skip if testing only)
            
        Returns:
            dict: Dictionary with encoder/decoder graphs for each split
        """
        all_graphs = {}
        
        # Always build test graphs
        if 'test' in data_splits:
            print("Building test graphs...")
            encoder_test, decoder_test = self.build_graphs_for_split(
                data_splits['test']['x'], latent_queries, gno_radius, scales
            )
            all_graphs['test'] = {
                'encoder': encoder_test,
                'decoder': decoder_test
            }
        
        # Build train/val graphs if requested
        if build_train:
            if 'train' in data_splits:
                print("Building train graphs...")
                encoder_train, decoder_train = self.build_graphs_for_split(
                    data_splits['train']['x'], latent_queries, gno_radius, scales
                )
                all_graphs['train'] = {
                    'encoder': encoder_train,
                    'decoder': decoder_train
                }
            
            if 'val' in data_splits:
                print("Building val graphs...")
                encoder_val, decoder_val = self.build_graphs_for_split(
                    data_splits['val']['x'], latent_queries, gno_radius, scales
                )
                all_graphs['val'] = {
                    'encoder': encoder_val,
                    'decoder': decoder_val
                }
        else:
            print("Skipping train/val graph building (testing mode)")
            all_graphs['train'] = None
            all_graphs['val'] = None
        
        return all_graphs
    
    def validate_graphs(self, graphs: dict, expected_samples: dict):
        """
        Validate that graphs have correct structure and sizes.
        
        Args:
            graphs: Graph dictionary
            expected_samples: Expected number of samples per split
        """
        for split_name, split_graphs in graphs.items():
            if split_graphs is None:
                continue
                
            encoder_graphs = split_graphs['encoder']
            decoder_graphs = split_graphs['decoder']
            expected_count = expected_samples.get(split_name, 0)
            
            assert len(encoder_graphs) == expected_count, \
                f"Encoder graphs for {split_name}: expected {expected_count}, got {len(encoder_graphs)}"
            assert len(decoder_graphs) == expected_count, \
                f"Decoder graphs for {split_name}: expected {expected_count}, got {len(decoder_graphs)}"
            
            # Validate individual samples
            for i, (enc_sample, dec_sample) in enumerate(zip(encoder_graphs, decoder_graphs)):
                assert isinstance(enc_sample, list), f"Encoder sample {i} should be list of scales"
                assert isinstance(dec_sample, list), f"Decoder sample {i} should be list of scales"
                assert len(enc_sample) == len(dec_sample), \
                    f"Encoder and decoder should have same number of scales for sample {i}"
        
        print("Graph validation passed")


class CachedGraphBuilder(GraphBuilder):
    """
    Graph builder with caching capabilities.
    Can save and load pre-computed graphs to avoid recomputation.
    """
    
    def __init__(self, neighbor_search_method: str = "auto", cache_dir: Optional[str] = None):
        super().__init__(neighbor_search_method)
        self.cache_dir = cache_dir
    
    def _get_cache_path(self, dataset_name: str, split_name: str, graph_type: str) -> str:
        """Get cache file path for graphs."""
        if self.cache_dir is None:
            raise ValueError("Cache directory not specified")
        
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{dataset_name}_{split_name}_{graph_type}_graphs.pt")
    
    def save_graphs(self, graphs: dict, dataset_name: str):
        """Save graphs to cache."""
        if self.cache_dir is None:
            print("No cache directory specified, skipping graph save")
            return
        
        for split_name, split_graphs in graphs.items():
            if split_graphs is None:
                continue
            
            # Save encoder graphs
            encoder_path = self._get_cache_path(dataset_name, split_name, "encoder")
            torch.save(split_graphs['encoder'], encoder_path)
            
            # Save decoder graphs
            decoder_path = self._get_cache_path(dataset_name, split_name, "decoder")
            torch.save(split_graphs['decoder'], decoder_path)
        
        print(f"Graphs saved to cache directory: {self.cache_dir}")
    
    def load_graphs(self, dataset_name: str, splits: List[str]) -> Optional[dict]:
        """Load graphs from cache."""
        if self.cache_dir is None:
            return None
        
        try:
            all_graphs = {}
            for split_name in splits:
                encoder_path = self._get_cache_path(dataset_name, split_name, "encoder")
                decoder_path = self._get_cache_path(dataset_name, split_name, "decoder")
                
                import os
                if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
                    print(f"Cache files not found for split: {split_name}")
                    return None
                
                encoder_graphs = torch.load(encoder_path)
                decoder_graphs = torch.load(decoder_path)
                
                all_graphs[split_name] = {
                    'encoder': encoder_graphs,
                    'decoder': decoder_graphs
                }
            
            print(f"Graphs loaded from cache directory: {self.cache_dir}")
            return all_graphs
            
        except Exception as e:
            print(f"Failed to load graphs from cache: {e}")
            return None
    
    def build_all_graphs(self, data_splits: dict, latent_queries: torch.Tensor,
                        gno_radius: float, scales: List[float],
                        dataset_name: str = "dataset", build_train: bool = True,
                        use_cache: bool = True) -> dict:
        """
        Build graphs with caching support.
        
        Args:
            data_splits: Data splits dictionary
            latent_queries: Latent query coordinates
            gno_radius: Base radius
            scales: Scale factors
            dataset_name: Name for cache files
            build_train: Whether to build train/val graphs
            use_cache: Whether to use cached graphs
            
        Returns:
            dict: Graph dictionary
        """
        # Try to load from cache first
        if use_cache and self.cache_dir is not None:
            cache_splits = ['test']
            if build_train:
                cache_splits.extend(['train', 'val'])
            
            cached_graphs = self.load_graphs(dataset_name, cache_splits)
            if cached_graphs is not None:
                return cached_graphs
        
        # Build graphs if not cached
        graphs = super().build_all_graphs(
            data_splits, latent_queries, gno_radius, scales, build_train
        )
        
        # Save to cache
        if use_cache:
            self.save_graphs(graphs, dataset_name)
        
        return graphs
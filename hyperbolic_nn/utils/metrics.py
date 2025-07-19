"""
Metrics and evaluation utilities for hyperbolic neural networks.

This module provides specialized metrics for evaluating hyperbolic neural
networks, including hyperbolic-specific distance measures and quality metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, List
import logging
import numpy as np

from ..core.math_ops import hyperbolic_distance, project_to_poincare_ball
from ..core.manifolds import PoincareManifold


logger = logging.getLogger(__name__)


class HyperbolicMetrics:
    """
    Collection of metrics for hyperbolic neural networks.
    
    This class provides various metrics specifically designed for evaluating
    the performance and quality of hyperbolic neural networks.
    
    Args:
        curvature: Curvature parameter for hyperbolic space
        manifold: Hyperbolic manifold (optional)
        
    Example:
        >>> metrics = HyperbolicMetrics(curvature=1.0)
        >>> embeddings = torch.randn(100, 64) * 0.1
        >>> distances = metrics.pairwise_distances(embeddings)
        >>> stats = metrics.embedding_statistics(embeddings)
    """
    
    def __init__(
        self,
        curvature: float = 1.0,
        manifold: Optional[PoincareManifold] = None
    ):
        self.curvature = curvature
        self.manifold = manifold or PoincareManifold(curvature=curvature)
        
        logger.info(f"Initialized HyperbolicMetrics with curvature={curvature}")
    
    def pairwise_distances(
        self,
        embeddings: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute pairwise hyperbolic distances between embeddings.
        
        Args:
            embeddings: Tensor of embeddings, shape (N, dim)
            batch_size: Optional batch size for memory-efficient computation
            
        Returns:
            Pairwise distance matrix, shape (N, N)
        """
        N = embeddings.size(0)
        
        if batch_size is None or N <= batch_size:
            # Compute all distances at once
            embeddings_expanded_1 = embeddings.unsqueeze(1)  # (N, 1, dim)
            embeddings_expanded_2 = embeddings.unsqueeze(0)  # (1, N, dim)
            
            # Broadcast and compute distances
            embeddings_1 = embeddings_expanded_1.expand(N, N, -1).reshape(-1, embeddings.size(-1))
            embeddings_2 = embeddings_expanded_2.expand(N, N, -1).reshape(-1, embeddings.size(-1))
            
            distances = hyperbolic_distance(embeddings_1, embeddings_2, c=self.curvature)
            distances = distances.view(N, N)
        else:
            # Compute distances in batches for memory efficiency
            distances = torch.zeros(N, N, device=embeddings.device, dtype=embeddings.dtype)
            
            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                for j in range(0, N, batch_size):
                    end_j = min(j + batch_size, N)
                    
                    batch_i = embeddings[i:end_i]  # (batch_i_size, dim)
                    batch_j = embeddings[j:end_j]  # (batch_j_size, dim)
                    
                    # Expand for pairwise computation
                    batch_i_exp = batch_i.unsqueeze(1)  # (batch_i_size, 1, dim)
                    batch_j_exp = batch_j.unsqueeze(0)  # (1, batch_j_size, dim)
                    
                    batch_i_flat = batch_i_exp.expand(-1, batch_j.size(0), -1).reshape(-1, embeddings.size(-1))
                    batch_j_flat = batch_j_exp.expand(batch_i.size(0), -1, -1).reshape(-1, embeddings.size(-1))
                    
                    batch_distances = hyperbolic_distance(batch_i_flat, batch_j_flat, c=self.curvature)
                    batch_distances = batch_distances.view(end_i - i, end_j - j)
                    
                    distances[i:end_i, j:end_j] = batch_distances
        
        return distances
    
    def embedding_statistics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics about embeddings in hyperbolic space.
        
        Args:
            embeddings: Tensor of embeddings, shape (N, dim)
            
        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            # Basic statistics
            norms = torch.norm(embeddings, dim=-1)
            
            # Distances from origin
            origin = torch.zeros_like(embeddings)
            distances_from_origin = hyperbolic_distance(embeddings, origin, c=self.curvature)
            
            # Pairwise distances (sample for efficiency if too many embeddings)
            if embeddings.size(0) > 1000:
                # Sample subset for pairwise distance computation
                indices = torch.randperm(embeddings.size(0))[:1000]
                sample_embeddings = embeddings[indices]
                pairwise_dists = self.pairwise_distances(sample_embeddings)
            else:
                pairwise_dists = self.pairwise_distances(embeddings)
            
            # Remove diagonal (self-distances)
            mask = ~torch.eye(pairwise_dists.size(0), dtype=torch.bool, device=embeddings.device)
            pairwise_dists_no_diag = pairwise_dists[mask]
            
            stats = {
                'num_embeddings': embeddings.size(0),
                'embedding_dim': embeddings.size(1),
                'mean_norm': norms.mean().item(),
                'std_norm': norms.std().item(),
                'max_norm': norms.max().item(),
                'min_norm': norms.min().item(),
                'mean_distance_from_origin': distances_from_origin.mean().item(),
                'std_distance_from_origin': distances_from_origin.std().item(),
                'max_distance_from_origin': distances_from_origin.max().item(),
                'mean_pairwise_distance': pairwise_dists_no_diag.mean().item(),
                'std_pairwise_distance': pairwise_dists_no_diag.std().item(),
                'max_pairwise_distance': pairwise_dists_no_diag.max().item(),
                'min_pairwise_distance': pairwise_dists_no_diag.min().item(),
            }
        
        return stats
    
    def hierarchical_distortion(
        self,
        embeddings: torch.Tensor,
        true_distances: torch.Tensor,
        sample_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute hierarchical distortion metrics.
        
        Measures how well the hyperbolic embeddings preserve hierarchical
        relationships compared to true distances.
        
        Args:
            embeddings: Hyperbolic embeddings, shape (N, dim)
            true_distances: True hierarchical distances, shape (N, N)
            sample_size: Optional sample size for efficiency
            
        Returns:
            Dictionary of distortion metrics
        """
        N = embeddings.size(0)
        
        # Sample if dataset is too large
        if sample_size is not None and N > sample_size:
            indices = torch.randperm(N)[:sample_size]
            embeddings = embeddings[indices]
            true_distances = true_distances[indices][:, indices]
            N = sample_size
        
        # Compute hyperbolic distances
        hyp_distances = self.pairwise_distances(embeddings)
        
        # Remove diagonal
        mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
        hyp_dists_flat = hyp_distances[mask]
        true_dists_flat = true_distances[mask]
        
        # Compute distortion metrics
        with torch.no_grad():
            # Mean absolute error
            mae = torch.abs(hyp_dists_flat - true_dists_flat).mean().item()
            
            # Mean squared error
            mse = torch.pow(hyp_dists_flat - true_dists_flat, 2).mean().item()
            
            # Relative error
            relative_error = (torch.abs(hyp_dists_flat - true_dists_flat) / 
                            (true_dists_flat + 1e-8)).mean().item()
            
            # Correlation
            correlation = torch.corrcoef(torch.stack([hyp_dists_flat, true_dists_flat]))[0, 1].item()
            
            # Rank correlation (Spearman)
            hyp_ranks = torch.argsort(torch.argsort(hyp_dists_flat)).float()
            true_ranks = torch.argsort(torch.argsort(true_dists_flat)).float()
            rank_correlation = torch.corrcoef(torch.stack([hyp_ranks, true_ranks]))[0, 1].item()
        
        distortion_metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'relative_error': relative_error,
            'correlation': correlation,
            'rank_correlation': rank_correlation,
        }
        
        return distortion_metrics
    
    def embedding_quality_score(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute overall quality score for embeddings.
        
        Args:
            embeddings: Hyperbolic embeddings, shape (N, dim)
            labels: Optional labels for supervised quality metrics
            
        Returns:
            Dictionary of quality metrics
        """
        stats = self.embedding_statistics(embeddings)
        
        quality_metrics = {
            'norm_stability': 1.0 / (1.0 + stats['std_norm']),  # Lower std is better
            'distance_spread': stats['std_pairwise_distance'],  # Good spread
            'origin_concentration': 1.0 / (1.0 + stats['mean_distance_from_origin']),  # Closer to origin
        }
        
        if labels is not None:
            # Compute supervised quality metrics
            supervised_metrics = self._compute_supervised_metrics(embeddings, labels)
            quality_metrics.update(supervised_metrics)
        
        # Compute overall quality score
        quality_metrics['overall_score'] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics
    
    def _compute_supervised_metrics(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute supervised quality metrics using labels.
        
        Args:
            embeddings: Hyperbolic embeddings
            labels: Class labels
            
        Returns:
            Dictionary of supervised metrics
        """
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        
        # Compute within-class and between-class distances
        within_class_distances = []
        between_class_distances = []
        
        for i, label_i in enumerate(unique_labels):
            mask_i = labels == label_i
            embeddings_i = embeddings[mask_i]
            
            if embeddings_i.size(0) > 1:
                # Within-class distances
                within_dists = self.pairwise_distances(embeddings_i)
                mask_within = ~torch.eye(within_dists.size(0), dtype=torch.bool, device=embeddings.device)
                within_class_distances.extend(within_dists[mask_within].tolist())
            
            # Between-class distances
            for j, label_j in enumerate(unique_labels):
                if i < j:  # Avoid double counting
                    mask_j = labels == label_j
                    embeddings_j = embeddings[mask_j]
                    
                    # Compute distances between classes
                    for emb_i in embeddings_i:
                        for emb_j in embeddings_j:
                            dist = hyperbolic_distance(
                                emb_i.unsqueeze(0), 
                                emb_j.unsqueeze(0), 
                                c=self.curvature
                            ).item()
                            between_class_distances.append(dist)
        
        within_class_distances = torch.tensor(within_class_distances)
        between_class_distances = torch.tensor(between_class_distances)
        
        # Compute supervised metrics
        supervised_metrics = {
            'within_class_mean': within_class_distances.mean().item() if len(within_class_distances) > 0 else 0.0,
            'between_class_mean': between_class_distances.mean().item() if len(between_class_distances) > 0 else 0.0,
        }
        
        # Separation score (higher is better)
        if len(within_class_distances) > 0 and len(between_class_distances) > 0:
            separation_score = (between_class_distances.mean() / 
                              (within_class_distances.mean() + 1e-8)).item()
            supervised_metrics['separation_score'] = separation_score
        
        return supervised_metrics
    
    def tree_distortion(
        self,
        embeddings: torch.Tensor,
        tree_edges: torch.Tensor,
        tree_distances: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute tree distortion for hierarchical data.
        
        Measures how well hyperbolic embeddings preserve tree structure.
        
        Args:
            embeddings: Hyperbolic embeddings, shape (N, dim)
            tree_edges: Tree edges, shape (num_edges, 2)
            tree_distances: Optional precomputed tree distances
            
        Returns:
            Dictionary of tree distortion metrics
        """
        N = embeddings.size(0)
        
        if tree_distances is None:
            # Compute shortest path distances in tree
            tree_distances = self._compute_tree_distances(N, tree_edges)
        
        # Compute hyperbolic distances
        hyp_distances = self.pairwise_distances(embeddings)
        
        # Compute distortion metrics
        distortion_metrics = self.hierarchical_distortion(
            embeddings, tree_distances
        )
        
        # Add tree-specific metrics
        with torch.no_grad():
            # Edge length preservation
            edge_distortions = []
            for edge in tree_edges:
                i, j = edge[0].item(), edge[1].item()
                hyp_dist = hyp_distances[i, j].item()
                tree_dist = tree_distances[i, j].item()
                if tree_dist > 0:
                    edge_distortion = abs(hyp_dist - tree_dist) / tree_dist
                    edge_distortions.append(edge_distortion)
            
            if edge_distortions:
                distortion_metrics['edge_distortion_mean'] = np.mean(edge_distortions)
                distortion_metrics['edge_distortion_std'] = np.std(edge_distortions)
        
        return distortion_metrics
    
    def _compute_tree_distances(self, n_nodes: int, edges: torch.Tensor) -> torch.Tensor:
        """
        Compute shortest path distances in a tree.
        
        Args:
            n_nodes: Number of nodes
            edges: Tree edges, shape (num_edges, 2)
            
        Returns:
            Distance matrix
        """
        # Use Floyd-Warshall algorithm
        distances = torch.full((n_nodes, n_nodes), float('inf'))
        
        # Initialize diagonal
        for i in range(n_nodes):
            distances[i, i] = 0
        
        # Initialize edges
        for edge in edges:
            i, j = edge[0].item(), edge[1].item()
            distances[i, j] = 1.0
            distances[j, i] = 1.0
        
        # Floyd-Warshall
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])
        
        return distances
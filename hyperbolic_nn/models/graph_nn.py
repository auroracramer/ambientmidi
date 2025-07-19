"""
Hyperbolic Graph Neural Network.

This module implements a graph neural network that operates in hyperbolic space,
particularly suited for hierarchical graph data and tree structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

from ..core.base import HyperbolicModule
from ..layers.linear import HyperbolicLinear
from ..layers.activation import HyperbolicDropout, HyperbolicLayerNorm
from ..core.math_ops import (
    logarithmic_map, 
    exponential_map,
    hyperbolic_centroid,
    mobius_add
)


logger = logging.getLogger(__name__)


class HyperbolicGraphNN(HyperbolicModule):
    """
    Hyperbolic Graph Neural Network.
    
    This GNN operates in hyperbolic space and is particularly effective
    for hierarchical graph structures where the hyperbolic geometry
    naturally captures tree-like relationships.
    
    Args:
        input_dim: Input node feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of GNN layers
        aggregation: Aggregation method ('mean', 'sum', 'max', or 'hyperbolic_centroid')
        dropout: Dropout probability
        use_layer_norm: Whether to use layer normalization
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> gnn = HyperbolicGraphNN(
        ...     input_dim=64, hidden_dim=128, output_dim=10, num_layers=3
        ... )
        >>> # Node features: (num_nodes, input_dim)
        >>> # Edge indices: (2, num_edges) 
        >>> node_features = torch.randn(100, 64) * 0.1
        >>> edge_index = torch.randint(0, 100, (2, 200))
        >>> output = gnn(node_features, edge_index)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        aggregation: str = 'hyperbolic_centroid',
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.dropout_prob = dropout
        self.use_layer_norm = use_layer_norm
        
        # Input projection
        self.input_proj = HyperbolicLinear(
            input_dim, hidden_dim,
            manifold=self.manifold,
            curvature=self.c.item()
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        
        for i in range(num_layers):
            layer = HyperbolicGNNLayer(
                hidden_dim,
                hidden_dim,
                aggregation=aggregation,
                manifold=self.manifold,
                curvature=self.c.item()
            )
            self.gnn_layers.append(layer)
            
            if use_layer_norm:
                ln = HyperbolicLayerNorm(
                    hidden_dim,
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
                self.layer_norms.append(ln)
            
            if dropout > 0:
                drop = HyperbolicDropout(
                    p=dropout,
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
                self.dropouts.append(drop)
        
        # Output projection
        self.output_proj = HyperbolicLinear(
            hidden_dim, output_dim,
            manifold=self.manifold,
            curvature=self.c.item()
        )
        
        logger.info(f"Initialized HyperbolicGraphNN({input_dim} -> {hidden_dim} -> {output_dim})")
    
    def hyperbolic_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the graph neural network.
        
        Args:
            x: Node features, shape (num_nodes, input_dim)
            edge_index: Edge indices, shape (2, num_edges)
            batch: Batch assignment for each node (for batched graphs)
            
        Returns:
            Updated node features in hyperbolic space
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            # GNN layer
            x = gnn_layer(x, edge_index)
            
            # Layer normalization
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            
            # Activation (in tangent space)
            origin = torch.zeros_like(x)
            x_tangent = logarithmic_map(origin, x, c=self.c.item())
            x_tangent = F.relu(x_tangent)
            x = exponential_map(x_tangent, origin, c=self.c.item())
            x = self.project_to_manifold(x)
            
            # Dropout
            if self.dropouts is not None:
                x = self.dropouts[i](x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x
    
    def extra_repr(self) -> str:
        """String representation of the GNN."""
        return (f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'output_dim={self.output_dim}, num_layers={self.num_layers}, '
                f'aggregation={self.aggregation}, dropout={self.dropout_prob}, '
                f'curvature={self.c.item():.3f}')


class HyperbolicGNNLayer(HyperbolicModule):
    """
    Single layer of hyperbolic graph neural network.
    
    This layer implements message passing in hyperbolic space where
    node features are aggregated from neighbors using hyperbolic operations.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        aggregation: Aggregation method
        use_self_loops: Whether to include self-loops
        **kwargs: Additional arguments passed to HyperbolicModule
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        aggregation: str = 'hyperbolic_centroid',
        use_self_loops: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.use_self_loops = use_self_loops
        
        # Message transformation
        self.message_proj = HyperbolicLinear(
            input_dim, output_dim,
            manifold=self.manifold,
            curvature=self.c.item()
        )
        
        # Update transformation (for combining self and neighbor features)
        self.update_proj = HyperbolicLinear(
            output_dim, output_dim,
            manifold=self.manifold,
            curvature=self.c.item()
        )
        
        logger.debug(f"Initialized HyperbolicGNNLayer({input_dim} -> {output_dim})")
    
    def hyperbolic_forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of GNN layer.
        
        Args:
            x: Node features, shape (num_nodes, input_dim)
            edge_index: Edge indices, shape (2, num_edges)
            
        Returns:
            Updated node features, shape (num_nodes, output_dim)
        """
        num_nodes = x.size(0)
        
        # Add self-loops if specified
        if self.use_self_loops:
            self_loops = torch.arange(num_nodes, device=x.device).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
        
        # Transform node features for messages
        messages = self.message_proj(x)
        
        # Aggregate messages from neighbors
        aggregated = self._aggregate_messages(messages, edge_index, num_nodes)
        
        # Update node features
        updated = self.update_proj(aggregated)
        
        return updated
    
    def _aggregate_messages(
        self,
        messages: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Aggregate messages from neighbors.
        
        Args:
            messages: Transformed node features for messages
            edge_index: Edge indices
            num_nodes: Number of nodes
            
        Returns:
            Aggregated features for each node
        """
        source, target = edge_index[0], edge_index[1]
        
        if self.aggregation == 'hyperbolic_centroid':
            # Use hyperbolic centroid for aggregation
            aggregated = torch.zeros_like(messages)
            
            for node in range(num_nodes):
                # Find neighbors of the current node
                neighbor_mask = target == node
                if neighbor_mask.sum() == 0:
                    # No neighbors, keep original features
                    aggregated[node] = messages[node]
                    continue
                
                neighbor_indices = source[neighbor_mask]
                neighbor_features = messages[neighbor_indices]  # (num_neighbors, feature_dim)
                
                if neighbor_features.size(0) == 1:
                    # Single neighbor
                    aggregated[node] = neighbor_features[0]
                else:
                    # Multiple neighbors - compute hyperbolic centroid
                    # Uniform weights for simplicity
                    weights = torch.ones(neighbor_features.size(0), device=messages.device)
                    weights = weights / weights.sum()
                    
                    centroid = hyperbolic_centroid(
                        neighbor_features.unsqueeze(0),  # (1, num_neighbors, feature_dim)
                        weights.unsqueeze(0),  # (1, num_neighbors)
                        c=self.c.item()
                    ).squeeze(0)  # (feature_dim,)
                    
                    aggregated[node] = centroid
        
        elif self.aggregation == 'mean':
            # Mean aggregation in tangent space
            aggregated = torch.zeros_like(messages)
            
            for node in range(num_nodes):
                neighbor_mask = target == node
                if neighbor_mask.sum() == 0:
                    aggregated[node] = messages[node]
                    continue
                
                neighbor_indices = source[neighbor_mask]
                neighbor_features = messages[neighbor_indices]
                
                # Map to tangent space, compute mean, map back
                origin = torch.zeros_like(neighbor_features)
                neighbor_tangent = logarithmic_map(origin, neighbor_features, c=self.c.item())
                mean_tangent = neighbor_tangent.mean(dim=0)
                
                origin_single = torch.zeros_like(mean_tangent)
                mean_hyp = exponential_map(
                    mean_tangent.unsqueeze(0),
                    origin_single.unsqueeze(0),
                    c=self.c.item()
                ).squeeze(0)
                
                aggregated[node] = self.project_to_manifold(mean_hyp)
        
        elif self.aggregation == 'sum':
            # Sum aggregation using Möbius addition
            aggregated = torch.zeros_like(messages)
            
            for node in range(num_nodes):
                neighbor_mask = target == node
                if neighbor_mask.sum() == 0:
                    aggregated[node] = messages[node]
                    continue
                
                neighbor_indices = source[neighbor_mask]
                neighbor_features = messages[neighbor_indices]
                
                # Sequential Möbius addition
                result = neighbor_features[0]
                for i in range(1, neighbor_features.size(0)):
                    result = mobius_add(result, neighbor_features[i], c=self.c.item())
                
                aggregated[node] = result
        
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")
        
        return aggregated
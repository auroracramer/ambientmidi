"""
Hyperbolic linear (fully connected) layers.

This module implements linear layers that operate in hyperbolic space,
including various approaches to hyperbolic linear transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math
import logging

from ..core.base import HyperbolicModule
from ..core.math_ops import (
    exponential_map,
    logarithmic_map, 
    project_to_poincare_ball,
    mobius_add,
    hyperbolic_linear_transform
)


logger = logging.getLogger(__name__)


class HyperbolicLinear(HyperbolicModule):
    """
    Hyperbolic linear (fully connected) layer.
    
    This layer performs a linear transformation in hyperbolic space by:
    1. Mapping input from hyperbolic space to tangent space at origin
    2. Applying standard linear transformation in tangent space
    3. Mapping result back to hyperbolic space via exponential map
    
    This approach preserves the hyperbolic geometry while allowing for
    efficient computation using standard linear algebra operations.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features  
        bias: Whether to use bias term (default: True)
        dropout: Dropout probability (default: 0.0)
        use_bias_in_tangent: Whether to apply bias in tangent space (default: True)
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> layer = HyperbolicLinear(128, 64, curvature=1.0)
        >>> x = torch.randn(32, 128) * 0.1  # Small norm for stability
        >>> x = layer.project_to_manifold(x)  # Ensure on manifold
        >>> y = layer(x)  # Shape: (32, 64)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
        use_bias_in_tangent: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.dropout = dropout
        self.use_bias_in_tangent = use_bias_in_tangent
        
        # Linear transformation in tangent space
        self.linear = nn.Linear(in_features, out_features, bias=bias and use_bias_in_tangent)
        
        # Additional bias in hyperbolic space (if not using tangent space bias)
        if bias and not use_bias_in_tangent:
            self.hyperbolic_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('hyperbolic_bias', None)
        
        # Dropout layer
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(f"Initialized HyperbolicLinear({in_features} -> {out_features}, "
                   f"bias={bias}, dropout={dropout})")
    
    def _init_parameters(self):
        """Initialize parameters appropriately for hyperbolic space."""
        # Initialize linear layer with smaller weights for stability
        with torch.no_grad():
            # Xavier/Glorot initialization scaled down for hyperbolic space
            std = math.sqrt(2.0 / (self.in_features + self.out_features)) * 0.5
            nn.init.normal_(self.linear.weight, mean=0.0, std=std)
            
            if self.linear.bias is not None:
                nn.init.normal_(self.linear.bias, mean=0.0, std=std * 0.1)
            
            # Initialize hyperbolic bias if present
            if self.hyperbolic_bias is not None:
                # Initialize bias as small points near origin
                nn.init.normal_(self.hyperbolic_bias, mean=0.0, std=0.01)
                self.hyperbolic_bias.data = self.project_to_manifold(self.hyperbolic_bias.data)
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space, shape (..., in_features)
            
        Returns:
            Output tensor in hyperbolic space, shape (..., out_features)
        """
        # Apply dropout to input if specified
        if self.dropout_layer is not None and self.training:
            x = self.dropout_layer(x)
        
        # Method 1: Use the dedicated hyperbolic linear transform function
        if self.use_bias_in_tangent:
            # Apply linear transformation with bias in tangent space
            output = hyperbolic_linear_transform(
                x, 
                self.linear.weight, 
                self.linear.bias,
                c=self.c.item()
            )
        else:
            # Apply linear transformation without bias, then add hyperbolic bias
            output = hyperbolic_linear_transform(
                x,
                self.linear.weight,
                bias=None,
                c=self.c.item()
            )
            
            # Add hyperbolic bias if present
            if self.hyperbolic_bias is not None:
                # Möbius addition to add bias in hyperbolic space
                bias_expanded = self.hyperbolic_bias.unsqueeze(0).expand_as(output)
                output = mobius_add(output, bias_expanded, c=self.c.item())
        
        # Ensure output is on manifold
        output = self.project_to_manifold(output)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation of layer parameters."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.use_bias}, dropout={self.dropout}, '
                f'curvature={self.c.item():.3f}')


class HyperbolicMLP(HyperbolicModule):
    """
    Multi-layer perceptron (MLP) in hyperbolic space.
    
    This module stacks multiple HyperbolicLinear layers with activation functions
    to create a deep network operating entirely in hyperbolic space.
    
    Args:
        layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        activation: Activation function to use between layers
        dropout: Dropout probability for hidden layers
        use_batch_norm: Whether to use batch normalization (experimental)
        final_activation: Whether to apply activation to final layer
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> mlp = HyperbolicMLP([784, 256, 128, 10], dropout=0.1)
        >>> x = torch.randn(32, 784) * 0.1
        >>> x = mlp.project_to_manifold(x)
        >>> y = mlp(x)  # Shape: (32, 10)
    """
    
    def __init__(
        self,
        layer_sizes: list,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        final_activation: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements")
        
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.final_activation = final_activation
        
        # Build layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(len(layer_sizes) - 1):
            # Add linear layer
            layer = HyperbolicLinear(
                layer_sizes[i],
                layer_sizes[i + 1],
                bias=True,
                dropout=0.0,  # We'll handle dropout separately
                manifold=self.manifold,
                curvature=self.c.item()
            )
            self.layers.append(layer)
            
            # Add activation (skip for last layer unless specified)
            if i < len(layer_sizes) - 2 or final_activation:
                if activation == 'relu':
                    self.activations.append(nn.ReLU())
                elif activation == 'tanh':
                    self.activations.append(nn.Tanh())
                elif activation == 'sigmoid':
                    self.activations.append(nn.Sigmoid())
                elif activation == 'gelu':
                    self.activations.append(nn.GELU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
            else:
                self.activations.append(nn.Identity())
            
            # Add batch norm if specified (experimental)
            if use_batch_norm and (i < len(layer_sizes) - 2 or final_activation):
                # Note: Batch normalization in hyperbolic space is non-trivial
                # This is a simplified implementation
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            elif use_batch_norm:
                self.batch_norms.append(nn.Identity())
        
        # Dropout layer
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
        logger.info(f"Initialized HyperbolicMLP with layers {layer_sizes}")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Output tensor in hyperbolic space
        """
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            # Apply linear transformation in hyperbolic space
            x = layer(x)
            
            # For intermediate layers, we may need to map to Euclidean space for activation
            if i < len(self.layers) - 1 or self.final_activation:
                if not isinstance(activation, nn.Identity):
                    # Map to tangent space at origin for activation
                    origin = torch.zeros_like(x)
                    x_tangent = logarithmic_map(origin, x, c=self.c.item())
                    
                    # Apply activation in tangent space
                    x_tangent = activation(x_tangent)
                    
                    # Apply batch norm if specified
                    if self.batch_norms is not None:
                        if not isinstance(self.batch_norms[i], nn.Identity):
                            x_tangent = self.batch_norms[i](x_tangent)
                    
                    # Apply dropout
                    if self.dropout_layer is not None and self.training:
                        x_tangent = self.dropout_layer(x_tangent)
                    
                    # Map back to hyperbolic space
                    x = exponential_map(x_tangent, origin, c=self.c.item())
                    x = self.project_to_manifold(x)
        
        return x
    
    def extra_repr(self) -> str:
        """String representation of the MLP."""
        return (f'layers={self.layer_sizes}, activation={self.activation_name}, '
                f'dropout={self.dropout}, curvature={self.c.item():.3f}')


class HyperbolicBilinear(HyperbolicModule):
    """
    Hyperbolic bilinear layer for computing interactions between two inputs.
    
    This layer computes a bilinear transformation in hyperbolic space, useful
    for tasks requiring interaction modeling between two sets of features.
    
    Args:
        in1_features: Number of features in first input
        in2_features: Number of features in second input  
        out_features: Number of output features
        bias: Whether to use bias term
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> layer = HyperbolicBilinear(64, 64, 128)
        >>> x1 = torch.randn(32, 64) * 0.1
        >>> x2 = torch.randn(32, 64) * 0.1
        >>> x1, x2 = layer.project_to_manifold(x1), layer.project_to_manifold(x2)
        >>> y = layer(x1, x2)  # Shape: (32, 128)
    """
    
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Bilinear transformation in tangent space
        self.bilinear = nn.Bilinear(in1_features, in2_features, out_features, bias=bias)
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(f"Initialized HyperbolicBilinear({in1_features}, {in2_features} -> {out_features})")
    
    def _init_parameters(self):
        """Initialize parameters for hyperbolic space."""
        with torch.no_grad():
            # Scale down initialization for stability
            std = math.sqrt(1.0 / (self.in1_features + self.in2_features)) * 0.5
            nn.init.normal_(self.bilinear.weight, mean=0.0, std=std)
            
            if self.bilinear.bias is not None:
                nn.init.normal_(self.bilinear.bias, mean=0.0, std=std * 0.1)
    
    def hyperbolic_forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bilinear transformation.
        
        Args:
            x1: First input tensor in hyperbolic space
            x2: Second input tensor in hyperbolic space
            
        Returns:
            Output tensor in hyperbolic space
        """
        # Map both inputs to tangent space at origin
        origin1 = torch.zeros_like(x1)
        origin2 = torch.zeros_like(x2)
        
        x1_tangent = logarithmic_map(origin1, x1, c=self.c.item())
        x2_tangent = logarithmic_map(origin2, x2, c=self.c.item())
        
        # Apply bilinear transformation in tangent space
        output_tangent = self.bilinear(x1_tangent, x2_tangent)
        
        # Map back to hyperbolic space
        origin_out = torch.zeros_like(output_tangent)
        output = exponential_map(output_tangent, origin_out, c=self.c.item())
        
        # Ensure output is on manifold
        output = self.project_to_manifold(output)
        
        return output
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with manifold projection.
        
        Args:
            x1, x2: Input tensors
            
        Returns:
            Output tensor in hyperbolic space
        """
        # Ensure inputs are on manifold
        x1 = self.project_to_manifold(x1)
        x2 = self.project_to_manifold(x2)
        
        return self.hyperbolic_forward(x1, x2)
    
    def extra_repr(self) -> str:
        """String representation of layer parameters."""
        return (f'in1_features={self.in1_features}, in2_features={self.in2_features}, '
                f'out_features={self.out_features}, bias={self.use_bias}, '
                f'curvature={self.c.item():.3f}')
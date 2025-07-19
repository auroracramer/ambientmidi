"""
Hyperbolic activation functions.

This module implements activation functions that are suitable for use
in hyperbolic neural networks, preserving the hyperbolic geometry
while providing non-linear transformations.
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
    conformal_factor
)


logger = logging.getLogger(__name__)


class HyperbolicActivation(HyperbolicModule):
    """
    Hyperbolic activation function that applies activations in tangent space.
    
    This is a general wrapper that allows applying standard activation functions
    to hyperbolic data by:
    1. Mapping to tangent space at origin
    2. Applying the activation function
    3. Mapping back to hyperbolic space
    
    Args:
        activation: Activation function name or callable
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> activation = HyperbolicActivation('relu')
        >>> x = torch.randn(32, 64) * 0.1
        >>> x = activation.project_to_manifold(x)
        >>> y = activation(x)
    """
    
    def __init__(self, activation: Union[str, nn.Module] = 'relu', **kwargs):
        super().__init__(**kwargs)
        
        self.activation_name = activation if isinstance(activation, str) else str(activation)
        
        # Create activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = nn.ReLU()
            elif activation.lower() == 'tanh':
                self.activation = nn.Tanh()
            elif activation.lower() == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation.lower() == 'gelu':
                self.activation = nn.GELU()
            elif activation.lower() == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif activation.lower() == 'elu':
                self.activation = nn.ELU()
            elif activation.lower() == 'selu':
                self.activation = nn.SELU()
            elif activation.lower() == 'swish' or activation.lower() == 'silu':
                self.activation = nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
        else:
            self.activation = activation
        
        logger.info(f"Initialized HyperbolicActivation with {self.activation_name}")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply activation function in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Activated tensor in hyperbolic space
        """
        # Map to tangent space at origin
        origin = torch.zeros_like(x)
        x_tangent = logarithmic_map(origin, x, c=self.c.item())
        
        # Apply activation in tangent space
        y_tangent = self.activation(x_tangent)
        
        # Map back to hyperbolic space
        y = exponential_map(y_tangent, origin, c=self.c.item())
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y
    
    def extra_repr(self) -> str:
        """String representation of the activation."""
        return f'activation={self.activation_name}, curvature={self.c.item():.3f}'


class HyperbolicReLU(HyperbolicModule):
    """
    Hyperbolic ReLU activation function.
    
    A specialized implementation of ReLU for hyperbolic space that can be
    more efficient than the general HyperbolicActivation wrapper.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initialized HyperbolicReLU")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU in hyperbolic space."""
        # Map to tangent space at origin
        origin = torch.zeros_like(x)
        x_tangent = logarithmic_map(origin, x, c=self.c.item())
        
        # Apply ReLU in tangent space
        y_tangent = F.relu(x_tangent)
        
        # Map back to hyperbolic space
        y = exponential_map(y_tangent, origin, c=self.c.item())
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y


class HyperbolicGELU(HyperbolicModule):
    """
    Hyperbolic GELU activation function.
    
    GELU (Gaussian Error Linear Unit) adapted for hyperbolic space.
    This is particularly useful in transformer architectures.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initialized HyperbolicGELU")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU in hyperbolic space."""
        # Map to tangent space at origin
        origin = torch.zeros_like(x)
        x_tangent = logarithmic_map(origin, x, c=self.c.item())
        
        # Apply GELU in tangent space
        y_tangent = F.gelu(x_tangent)
        
        # Map back to hyperbolic space
        y = exponential_map(y_tangent, origin, c=self.c.item())
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y


class HyperbolicSoftmax(HyperbolicModule):
    """
    Hyperbolic softmax function.
    
    This implements a softmax-like operation in hyperbolic space using
    the hyperbolic distance to the origin as the "logit" value.
    
    Args:
        dim: Dimension along which softmax is computed
        temperature: Temperature parameter for softmax (default: 1.0)
        **kwargs: Additional arguments passed to HyperbolicModule
    """
    
    def __init__(self, dim: int = -1, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.dim = dim
        self.temperature = temperature
        
        logger.info(f"Initialized HyperbolicSoftmax(dim={dim}, temperature={temperature})")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hyperbolic softmax.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Softmax probabilities
        """
        # Compute distances from origin
        origin = torch.zeros_like(x[..., :1])  # Just need right shape
        origin = origin.expand_as(x)
        
        distances = self.distance(x, origin)  # Shape: (..., )
        
        # Apply softmax to negative distances (closer points have higher probability)
        logits = -distances / self.temperature
        
        # Standard softmax
        probabilities = F.softmax(logits, dim=self.dim)
        
        return probabilities
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, temperature={self.temperature}, curvature={self.c.item():.3f}'


class HyperbolicLayerNorm(HyperbolicModule):
    """
    Layer normalization adapted for hyperbolic space.
    
    This implementation normalizes in the tangent space at the origin,
    which preserves the hyperbolic geometry while providing the benefits
    of normalization.
    
    Args:
        normalized_shape: Input shape for normalization
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn affine parameters
        **kwargs: Additional arguments passed to HyperbolicModule
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, tuple],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        logger.info(f"Initialized HyperbolicLayerNorm({normalized_shape})")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Normalized tensor in hyperbolic space
        """
        # Map to tangent space at origin
        origin = torch.zeros_like(x)
        x_tangent = logarithmic_map(origin, x, c=self.c.item())
        
        # Apply standard layer normalization in tangent space
        y_tangent = F.layer_norm(
            x_tangent,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        )
        
        # Map back to hyperbolic space
        y = exponential_map(y_tangent, origin, c=self.c.item())
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y
    
    def extra_repr(self) -> str:
        return (f'normalized_shape={self.normalized_shape}, eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}, '
                f'curvature={self.c.item():.3f}')


class HyperbolicDropout(HyperbolicModule):
    """
    Dropout adapted for hyperbolic space.
    
    This implementation applies dropout in the tangent space to preserve
    the hyperbolic geometry while providing regularization.
    
    Args:
        p: Dropout probability
        inplace: Whether to do dropout in-place
        **kwargs: Additional arguments passed to HyperbolicModule
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False, **kwargs):
        super().__init__(**kwargs)
        
        self.p = p
        self.inplace = inplace
        self.dropout = nn.Dropout(p=p, inplace=False)  # Never inplace for hyperbolic
        
        logger.info(f"Initialized HyperbolicDropout(p={p})")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Tensor with dropout applied in hyperbolic space
        """
        if not self.training:
            return x
        
        # Map to tangent space at origin
        origin = torch.zeros_like(x)
        x_tangent = logarithmic_map(origin, x, c=self.c.item())
        
        # Apply dropout in tangent space
        y_tangent = self.dropout(x_tangent)
        
        # Map back to hyperbolic space
        y = exponential_map(y_tangent, origin, c=self.c.item())
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y
    
    def extra_repr(self) -> str:
        return f'p={self.p}, curvature={self.c.item():.3f}'


class HyperbolicAttentionActivation(HyperbolicModule):
    """
    Specialized activation function for attention mechanisms in hyperbolic space.
    
    This combines a scaled exponential operation with projection to maintain
    numerical stability while preserving the hyperbolic structure.
    
    Args:
        scale: Scaling factor for the activation
        **kwargs: Additional arguments passed to HyperbolicModule
    """
    
    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.scale = scale
        
        logger.info(f"Initialized HyperbolicAttentionActivation(scale={scale})")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-specific activation in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Activated tensor in hyperbolic space
        """
        # Get the conformal factor (local scaling)
        lambda_x = conformal_factor(x, c=self.c.item())
        
        # Scale based on position in hyperbolic space
        scaling = torch.tanh(self.scale * lambda_x)
        
        # Apply scaling via Möbius scalar multiplication
        # This is a simplified version - in practice might use more complex operations
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        
        # Avoid division by zero
        eps = 1e-15
        direction = x / (norm_x + eps)
        
        # Scale the norm and reconstruct
        scaled_norm = norm_x * scaling
        y = direction * scaled_norm
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y
    
    def extra_repr(self) -> str:
        return f'scale={self.scale}, curvature={self.c.item():.3f}'
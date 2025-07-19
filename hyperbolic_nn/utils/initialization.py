"""
Initialization utilities for hyperbolic neural networks.

This module provides specialized initialization schemes that are
appropriate for parameters in hyperbolic neural networks.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Union
import logging

from ..core.manifolds import PoincareManifold
from ..core.math_ops import project_to_poincare_ball


logger = logging.getLogger(__name__)


def hyperbolic_init(
    tensor: torch.Tensor,
    manifold: Optional[PoincareManifold] = None,
    curvature: float = 1.0,
    method: str = 'uniform',
    scale: float = 1e-3
) -> torch.Tensor:
    """
    Initialize tensor for hyperbolic neural networks.
    
    This function initializes tensors with values appropriate for
    hyperbolic space, ensuring they lie within the Poincaré ball
    and have appropriate magnitudes for stable training.
    
    Args:
        tensor: Tensor to initialize
        manifold: Hyperbolic manifold (if None, creates PoincareManifold)
        curvature: Curvature parameter
        method: Initialization method ('uniform', 'normal', 'xavier', 'kaiming')
        scale: Scaling factor for initialization
        
    Returns:
        Initialized tensor
        
    Example:
        >>> weight = torch.empty(128, 64)
        >>> initialized_weight = hyperbolic_init(weight, method='xavier', scale=1e-3)
    """
    if manifold is None:
        manifold = PoincareManifold(curvature=curvature)
    
    with torch.no_grad():
        if method == 'uniform':
            # Uniform initialization in a small ball around origin
            nn.init.uniform_(tensor, -scale, scale)
        
        elif method == 'normal':
            # Normal initialization with small variance
            nn.init.normal_(tensor, mean=0.0, std=scale)
        
        elif method == 'xavier':
            # Xavier/Glorot initialization scaled for hyperbolic space
            fan_in, fan_out = _get_fan_in_out(tensor)
            std = math.sqrt(2.0 / (fan_in + fan_out)) * scale
            nn.init.normal_(tensor, mean=0.0, std=std)
        
        elif method == 'kaiming':
            # Kaiming/He initialization scaled for hyperbolic space
            fan_in, _ = _get_fan_in_out(tensor)
            std = math.sqrt(2.0 / fan_in) * scale
            nn.init.normal_(tensor, mean=0.0, std=std)
        
        else:
            raise ValueError(f"Unsupported initialization method: {method}")
        
        # Project to hyperbolic manifold
        tensor.data = manifold.project(tensor.data)
    
    logger.debug(f"Initialized tensor with shape {tensor.shape} using {method} method")
    return tensor


def _get_fan_in_out(tensor: torch.Tensor) -> tuple:
    """
    Get fan-in and fan-out for a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tuple of (fan_in, fan_out)
    """
    if tensor.dim() < 2:
        raise ValueError("Tensor must have at least 2 dimensions")
    
    if tensor.dim() == 2:
        # Standard linear layer
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    elif tensor.dim() == 4:
        # Convolutional layer (out_channels, in_channels, kernel_height, kernel_width)
        kernel_size = tensor.size(2) * tensor.size(3)
        fan_in = tensor.size(1) * kernel_size
        fan_out = tensor.size(0) * kernel_size
    else:
        # General case
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
        for i in range(2, tensor.dim()):
            fan_in *= tensor.size(i)
            fan_out *= tensor.size(i)
    
    return fan_in, fan_out


def init_hyperbolic_module(
    module: nn.Module,
    curvature: float = 1.0,
    method: str = 'xavier',
    scale: float = 1e-3
) -> nn.Module:
    """
    Initialize all parameters in a module for hyperbolic neural networks.
    
    Args:
        module: PyTorch module to initialize
        curvature: Curvature parameter
        method: Initialization method
        scale: Scaling factor
        
    Returns:
        Initialized module
        
    Example:
        >>> model = HyperbolicLinear(128, 64)
        >>> model = init_hyperbolic_module(model, method='xavier')
    """
    manifold = PoincareManifold(curvature=curvature)
    
    for name, param in module.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # Initialize weight parameters
            hyperbolic_init(param, manifold, curvature, method, scale)
        elif 'bias' in name:
            # Initialize bias parameters with smaller values
            hyperbolic_init(param, manifold, curvature, method, scale * 0.1)
        elif 'hyperbolic' in name.lower():
            # Initialize hyperbolic-specific parameters
            hyperbolic_init(param, manifold, curvature, method, scale)
    
    logger.info(f"Initialized module {module.__class__.__name__} with {method} method")
    return module


class HyperbolicInitializer:
    """
    Class for consistent hyperbolic initialization across a model.
    
    This class provides a convenient way to apply consistent initialization
    to all layers in a hyperbolic neural network.
    
    Args:
        curvature: Curvature parameter
        method: Initialization method
        scale: Scaling factor
        
    Example:
        >>> initializer = HyperbolicInitializer(method='xavier', scale=1e-3)
        >>> model = HyperbolicMLP([784, 256, 128, 10])
        >>> model = initializer.init_model(model)
    """
    
    def __init__(
        self,
        curvature: float = 1.0,
        method: str = 'xavier',
        scale: float = 1e-3
    ):
        self.curvature = curvature
        self.method = method
        self.scale = scale
        self.manifold = PoincareManifold(curvature=curvature)
        
        logger.info(f"Created HyperbolicInitializer with {method} method, scale={scale}")
    
    def init_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Initialize a single tensor."""
        return hyperbolic_init(
            tensor, 
            self.manifold, 
            self.curvature, 
            self.method, 
            self.scale
        )
    
    def init_module(self, module: nn.Module) -> nn.Module:
        """Initialize all parameters in a module."""
        return init_hyperbolic_module(
            module,
            self.curvature,
            self.method,
            self.scale
        )
    
    def init_model(self, model: nn.Module) -> nn.Module:
        """
        Initialize all parameters in a model recursively.
        
        Args:
            model: PyTorch model to initialize
            
        Returns:
            Initialized model
        """
        def init_recursive(module):
            for child in module.children():
                if len(list(child.children())) > 0:
                    # Recursively initialize child modules
                    init_recursive(child)
                else:
                    # Initialize leaf modules
                    self.init_module(child)
        
        init_recursive(model)
        logger.info(f"Initialized entire model {model.__class__.__name__}")
        return model


def init_hyperbolic_embedding(
    embedding: nn.Embedding,
    curvature: float = 1.0,
    scale: float = 1e-3
) -> nn.Embedding:
    """
    Initialize embedding layer for hyperbolic space.
    
    Args:
        embedding: Embedding layer to initialize
        curvature: Curvature parameter
        scale: Scaling factor
        
    Returns:
        Initialized embedding layer
    """
    manifold = PoincareManifold(curvature=curvature)
    
    with torch.no_grad():
        # Initialize with small normal distribution
        nn.init.normal_(embedding.weight, mean=0.0, std=scale)
        
        # Project to hyperbolic manifold
        embedding.weight.data = manifold.project(embedding.weight.data)
        
        # Set padding embedding to origin if specified
        if embedding.padding_idx is not None:
            embedding.weight.data[embedding.padding_idx].fill_(0)
    
    logger.debug(f"Initialized embedding layer with {embedding.num_embeddings} embeddings")
    return embedding


def get_hyperbolic_lr_scheduler(
    optimizer,
    mode: str = 'reduce_on_plateau',
    factor: float = 0.5,
    patience: int = 10,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler appropriate for hyperbolic neural networks.
    
    Hyperbolic neural networks often benefit from different learning rate
    schedules due to the geometry of the space.
    
    Args:
        optimizer: PyTorch optimizer
        mode: Scheduler mode ('reduce_on_plateau', 'exponential', 'cosine')
        factor: Factor by which to reduce learning rate
        patience: Patience for plateau detection
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        Learning rate scheduler
    """
    if mode == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True,
            **kwargs
        )
    elif mode == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            **kwargs
        )
    elif mode == 'cosine':
        T_max = kwargs.get('T_max', 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported scheduler mode: {mode}")
    
    logger.info(f"Created {mode} learning rate scheduler")
    return scheduler
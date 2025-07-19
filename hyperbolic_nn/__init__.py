"""
Hyperbolic Neural Networks for PyTorch Lightning

This module provides implementations of various hyperbolic neural network layers
including fully connected, convolutional, and transformer layers, all built
on top of PyTorch Lightning for scalable training and inference.

The hyperbolic layers operate in hyperbolic space (Poincaré ball model) which
has been shown to be particularly effective for hierarchical data representation
and tasks involving tree-like or graph structures.
"""

from .core.base import HyperbolicModule
from .core.manifolds import PoincareManifold, HyperbolicManifold
from .core.math_ops import (
    mobius_add,
    mobius_mul,
    exponential_map,
    logarithmic_map,
    hyperbolic_distance,
    parallel_transport
)

from .layers.linear import HyperbolicLinear, HyperbolicMLP
from .layers.conv import HyperbolicConv1d, HyperbolicConv2d
from .layers.transformer import (
    HyperbolicSelfAttention,
    HyperbolicTransformerBlock,
    HyperbolicTransformer
)
from .layers.activation import HyperbolicActivation, HyperbolicLayerNorm

from .models.classifier import HyperbolicClassifier
from .models.autoencoder import HyperbolicAutoEncoder
from .models.graph_nn import HyperbolicGraphNN

from .utils.initialization import hyperbolic_init
from .utils.metrics import HyperbolicMetrics
from .utils.visualization import plot_poincare_embeddings

__version__ = "1.0.0"
__author__ = "Hyperbolic Neural Networks Team"

__all__ = [
    # Core components
    "HyperbolicModule",
    "PoincareManifold",
    "HyperbolicManifold",
    
    # Mathematical operations
    "mobius_add",
    "mobius_mul", 
    "exponential_map",
    "logarithmic_map",
    "hyperbolic_distance",
    "parallel_transport",
    
    # Layer implementations
    "HyperbolicLinear",
    "HyperbolicMLP",
    "HyperbolicConv1d",
    "HyperbolicConv2d",
    "HyperbolicSelfAttention",
    "HyperbolicTransformerBlock", 
    "HyperbolicTransformer",
    "HyperbolicActivation",
    "HyperbolicLayerNorm",
    
    # Models
    "HyperbolicClassifier",
    "HyperbolicAutoEncoder",
    "HyperbolicGraphNN",
    
    # Utilities
    "hyperbolic_init",
    "HyperbolicMetrics",
    "plot_poincare_embeddings",
]
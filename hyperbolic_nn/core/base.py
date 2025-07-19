"""
Base classes for hyperbolic neural network modules.

This module provides the fundamental base class that all hyperbolic neural network
components inherit from, including common functionality for working with hyperbolic
geometry and PyTorch Lightning integration.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Dict, Any, Union, Tuple
from abc import ABC, abstractmethod
import logging

from .manifolds import PoincareManifold


logger = logging.getLogger(__name__)


class HyperbolicModule(pl.LightningModule, ABC):
    """
    Base class for all hyperbolic neural network modules.
    
    This class extends PyTorch Lightning's LightningModule to provide
    common functionality for hyperbolic neural networks, including:
    - Manifold management
    - Curvature parameter handling
    - Common hyperbolic operations
    - Training utilities specific to hyperbolic geometry
    
    Args:
        manifold: The hyperbolic manifold to operate on. Defaults to Poincaré ball.
        curvature: The curvature parameter (negative for hyperbolic space).
        learning_rate: Learning rate for optimization.
        weight_decay: Weight decay for regularization.
        clip_norm: Gradient clipping norm value.
        
    Attributes:
        manifold: The hyperbolic manifold instance
        c: Curvature parameter (positive value, represents -curvature)
        
    Example:
        >>> class MyHyperbolicLayer(HyperbolicModule):
        ...     def __init__(self, in_features, out_features, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self.linear = nn.Linear(in_features, out_features)
        ...     
        ...     def hyperbolic_forward(self, x):
        ...         # Implement hyperbolic forward pass
        ...         return self.manifold.exponential_map(self.linear(x))
    """
    
    def __init__(
        self,
        manifold: Optional[PoincareManifold] = None,
        curvature: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_norm: Optional[float] = 1.0,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters for Lightning
        self.save_hyperparameters()
        
        # Initialize manifold
        self.manifold = manifold or PoincareManifold(curvature=curvature)
        self.c = torch.tensor(curvature, dtype=torch.float32)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        
        logger.info(f"Initialized {self.__class__.__name__} with curvature={curvature}")
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers for hyperbolic neural networks.
        
        Uses Riemannian Adam optimizer when available, falls back to
        standard Adam with appropriate parameter handling for hyperbolic layers.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration.
        """
        try:
            # Try to use Riemannian optimizers if available
            from geoopt.optim import RiemannianAdam
            
            # Separate parameters for Riemannian and Euclidean optimization
            riemannian_params = []
            euclidean_params = []
            
            for name, param in self.named_parameters():
                if hasattr(param, 'manifold') or 'hyperbolic' in name.lower():
                    riemannian_params.append(param)
                else:
                    euclidean_params.append(param)
            
            optimizers = []
            
            if riemannian_params:
                riemannian_opt = RiemannianAdam(
                    riemannian_params,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
                optimizers.append(riemannian_opt)
            
            if euclidean_params:
                euclidean_opt = torch.optim.Adam(
                    euclidean_params,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
                optimizers.append(euclidean_opt)
            
            if len(optimizers) == 1:
                return optimizers[0]
            else:
                return optimizers
                
        except ImportError:
            # Fall back to standard Adam optimizer
            logger.warning("geoopt not available, using standard Adam optimizer")
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
    
    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """Apply gradient clipping before optimizer step."""
        if self.clip_norm is not None:
            # Clip gradients to prevent exploding gradients in hyperbolic space
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=self.clip_norm
            )
    
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project tensor to the hyperbolic manifold.
        
        Args:
            x: Input tensor to project
            
        Returns:
            Tensor projected onto the hyperbolic manifold
        """
        return self.manifold.project(x)
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between two points.
        
        Args:
            x, y: Points in hyperbolic space
            
        Returns:
            Hyperbolic distances between x and y
        """
        return self.manifold.distance(x, y)
    
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from tangent space to manifold.
        
        Args:
            x: Base point on manifold
            v: Tangent vector at x
            
        Returns:
            Point on manifold obtained by moving from x in direction v
        """
        return self.manifold.exponential_map(v, x)
    
    def log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map from manifold to tangent space.
        
        Args:
            x: Base point on manifold
            y: Target point on manifold
            
        Returns:
            Tangent vector at x pointing towards y
        """
        return self.manifold.logarithmic_map(x, y)
    
    @abstractmethod
    def hyperbolic_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Abstract method for hyperbolic forward pass.
        
        Subclasses must implement this method to define their specific
        hyperbolic computation.
        
        Args:
            x: Input tensor in hyperbolic space
            *args, **kwargs: Additional arguments specific to the layer
            
        Returns:
            Output tensor in hyperbolic space
        """
        pass
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Standard forward pass that ensures input is on manifold.
        
        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments
            
        Returns:
            Output of hyperbolic_forward
        """
        # Ensure input is on the manifold
        x = self.project_to_manifold(x)
        return self.hyperbolic_forward(x, *args, **kwargs)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Default training step for hyperbolic modules.
        
        Can be overridden for specific training procedures.
        
        Args:
            batch: Training batch (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Default validation step for hyperbolic modules.
        
        Args:
            batch: Validation batch (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for hyperbolic predictions.
        
        Default implementation uses hyperbolic distance.
        Override for specific loss functions.
        
        Args:
            predictions: Model predictions in hyperbolic space
            targets: Target values in hyperbolic space
            
        Returns:
            Computed loss
        """
        # Default to using hyperbolic distance as loss
        distances = self.distance(predictions, targets)
        return distances.mean()
    
    def get_embeddings_stats(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics about embeddings in hyperbolic space.
        
        Args:
            embeddings: Tensor of embeddings in hyperbolic space
            
        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            norms = torch.norm(embeddings, dim=-1)
            distances_from_origin = self.distance(
                embeddings, 
                torch.zeros_like(embeddings)
            )
            
            stats = {
                'mean_norm': norms.mean().item(),
                'std_norm': norms.std().item(),
                'max_norm': norms.max().item(),
                'mean_distance_from_origin': distances_from_origin.mean().item(),
                'std_distance_from_origin': distances_from_origin.std().item(),
            }
            
        return stats
    
    def __repr__(self) -> str:
        """String representation of the module."""
        return (f"{self.__class__.__name__}("
                f"curvature={self.c.item():.3f}, "
                f"manifold={self.manifold.__class__.__name__})")


class HyperbolicEmbedding(HyperbolicModule):
    """
    Hyperbolic embedding layer that maps discrete tokens to hyperbolic space.
    
    This layer learns embeddings directly in hyperbolic space, which can be
    particularly effective for hierarchical or tree-structured data.
    
    Args:
        num_embeddings: Size of the dictionary of embeddings
        embedding_dim: Size of each embedding vector
        padding_idx: If given, pads the output with the embedding vector at padding_idx
        max_norm: If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm
        scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of the words in the mini-batch
        sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> embedding = HyperbolicEmbedding(1000, 64, curvature=1.0)
        >>> tokens = torch.randint(0, 1000, (32, 10))  # batch_size=32, seq_len=10
        >>> embeddings = embedding(tokens)  # Shape: (32, 10, 64)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        # Initialize embedding table
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=None,  # We'll handle normalization manually
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        )
        
        # Initialize embeddings in hyperbolic space
        self._init_hyperbolic_embeddings()
        
        logger.info(f"Initialized HyperbolicEmbedding with {num_embeddings} embeddings of dimension {embedding_dim}")
    
    def _init_hyperbolic_embeddings(self):
        """Initialize embeddings appropriate for hyperbolic space."""
        with torch.no_grad():
            # Initialize with small norm in Euclidean space, then project
            nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
            
            # Project to hyperbolic manifold
            self.embedding.weight.data = self.project_to_manifold(self.embedding.weight.data)
            
            # Set padding embedding to origin if specified
            if self.padding_idx is not None:
                self.embedding.weight.data[self.padding_idx].fill_(0)
    
    def hyperbolic_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for hyperbolic embeddings.
        
        Args:
            input: LongTensor containing indices into the embedding table
            
        Returns:
            Embedded vectors in hyperbolic space
        """
        # Get embeddings
        embeddings = self.embedding(input)
        
        # Ensure embeddings are on manifold
        embeddings = self.project_to_manifold(embeddings)
        
        # Apply max norm constraint in hyperbolic space if specified
        if self.max_norm is not None:
            with torch.no_grad():
                norms = torch.norm(embeddings, dim=-1, keepdim=True)
                mask = norms > self.max_norm
                if mask.any():
                    # Scale down embeddings that exceed max_norm
                    scale_factor = self.max_norm / norms
                    embeddings = torch.where(mask, embeddings * scale_factor, embeddings)
                    embeddings = self.project_to_manifold(embeddings)
        
        return embeddings
"""
Hyperbolic neural network classifier.

This module implements a classifier built using hyperbolic neural network
layers, demonstrating how to create complete models for classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

from ..core.base import HyperbolicModule, HyperbolicEmbedding
from ..layers.linear import HyperbolicLinear, HyperbolicMLP
from ..layers.activation import HyperbolicDropout, HyperbolicLayerNorm
from ..core.math_ops import logarithmic_map, exponential_map


logger = logging.getLogger(__name__)


class HyperbolicClassifier(HyperbolicModule):
    """
    Hyperbolic neural network classifier.
    
    This classifier operates entirely in hyperbolic space and can be used
    for various classification tasks. It includes optional embedding layers
    for discrete inputs and flexible architecture configuration.
    
    Args:
        input_dim: Input feature dimension (for continuous inputs)
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        vocab_size: Vocabulary size for embedding (None for continuous inputs)
        embedding_dim: Embedding dimension (used if vocab_size is provided)
        dropout: Dropout probability
        use_layer_norm: Whether to use layer normalization
        final_activation: Activation for final layer (None for logits)
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> # For continuous inputs
        >>> classifier = HyperbolicClassifier(
        ...     input_dim=784, num_classes=10, hidden_dims=[256, 128]
        ... )
        >>> x = torch.randn(32, 784) * 0.1
        >>> logits = classifier(x)  # Shape: (32, 10)
        >>>
        >>> # For discrete inputs (with embedding)
        >>> classifier = HyperbolicClassifier(
        ...     vocab_size=10000, embedding_dim=128, num_classes=5,
        ...     hidden_dims=[256, 128]
        ... )
        >>> tokens = torch.randint(0, 10000, (32, 50))
        >>> logits = classifier(tokens)  # Shape: (32, 5)
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        num_classes: int = 10,
        hidden_dims: list = [256, 128],
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        final_activation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Determine actual input dimension
        if vocab_size is not None:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be provided when vocab_size is given")
            actual_input_dim = embedding_dim
            self.use_embedding = True
        else:
            if input_dim is None:
                raise ValueError("input_dim must be provided when vocab_size is None")
            actual_input_dim = input_dim
            self.use_embedding = False
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout
        self.use_layer_norm = use_layer_norm
        
        # Embedding layer for discrete inputs
        if self.use_embedding:
            self.embedding = HyperbolicEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                manifold=self.manifold,
                curvature=self.c.item()
            )
        else:
            self.embedding = None
        
        # Input projection (if needed)
        if self.use_embedding:
            # For sequence inputs, we need to aggregate embeddings
            self.input_projection = None  # Will aggregate in forward pass
        else:
            self.input_projection = None  # Direct continuous input
        
        # Build the main network
        layer_dims = [actual_input_dim] + hidden_dims + [num_classes]
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        
        for i in range(len(layer_dims) - 1):
            # Add main layer
            layer = HyperbolicLinear(
                layer_dims[i],
                layer_dims[i + 1],
                bias=True,
                manifold=self.manifold,
                curvature=self.c.item()
            )
            self.layers.append(layer)
            
            # Add layer norm (except for output layer)
            if use_layer_norm and i < len(layer_dims) - 2:
                ln = HyperbolicLayerNorm(
                    layer_dims[i + 1],
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
                self.layer_norms.append(ln)
            elif use_layer_norm:
                self.layer_norms.append(nn.Identity())
            
            # Add dropout (except for output layer)
            if dropout > 0 and i < len(layer_dims) - 2:
                drop = HyperbolicDropout(
                    p=dropout,
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
                self.dropouts.append(drop)
            elif dropout > 0:
                self.dropouts.append(nn.Identity())
        
        # Final activation
        self.final_activation = final_activation
        
        logger.info(f"Initialized HyperbolicClassifier with {len(layer_dims)-1} layers")
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input tensor (continuous features or token indices)
            
        Returns:
            Class logits (in Euclidean space for standard loss computation)
        """
        # Handle embedding for discrete inputs
        if self.use_embedding:
            if x.dim() == 1:
                # Single sequence: (seq_len,)
                x = x.unsqueeze(0)  # (1, seq_len)
            
            # Embed tokens: (batch_size, seq_len, embedding_dim)
            x = self.embedding(x)
            
            # Aggregate embeddings (mean pooling in hyperbolic space)
            # This is a simplified aggregation - could use attention or other methods
            batch_size, seq_len, embed_dim = x.shape
            x_flat = x.view(batch_size * seq_len, embed_dim)
            
            # Compute centroid for each sequence
            x_aggregated = []
            for i in range(batch_size):
                seq_embeddings = x[i]  # (seq_len, embed_dim)
                # Simple mean in tangent space (could use hyperbolic centroid)
                origin = torch.zeros_like(seq_embeddings)
                seq_tangent = logarithmic_map(origin, seq_embeddings, c=self.c.item())
                mean_tangent = seq_tangent.mean(dim=0)
                mean_hyp = exponential_map(
                    mean_tangent.unsqueeze(0),
                    torch.zeros_like(mean_tangent.unsqueeze(0)),
                    c=self.c.item()
                ).squeeze(0)
                x_aggregated.append(mean_hyp)
            
            x = torch.stack(x_aggregated, dim=0)  # (batch_size, embed_dim)
        
        # Pass through network layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply layer norm (except for last layer)
            if self.layer_norms is not None and i < len(self.layers) - 1:
                if not isinstance(self.layer_norms[i], nn.Identity):
                    x = self.layer_norms[i](x)
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                # Apply activation in tangent space
                origin = torch.zeros_like(x)
                x_tangent = logarithmic_map(origin, x, c=self.c.item())
                x_tangent = F.relu(x_tangent)  # Using ReLU as default
                x = exponential_map(x_tangent, origin, c=self.c.item())
                x = self.project_to_manifold(x)
            
            # Apply dropout (except for last layer)
            if self.dropouts is not None and i < len(self.layers) - 1:
                if not isinstance(self.dropouts[i], nn.Identity):
                    x = self.dropouts[i](x)
        
        # Final output processing
        if self.final_activation is not None:
            # Apply final activation in tangent space
            origin = torch.zeros_like(x)
            x_tangent = logarithmic_map(origin, x, c=self.c.item())
            
            if self.final_activation.lower() == 'relu':
                x_tangent = F.relu(x_tangent)
            elif self.final_activation.lower() == 'tanh':
                x_tangent = F.tanh(x_tangent)
            elif self.final_activation.lower() == 'sigmoid':
                x_tangent = F.sigmoid(x_tangent)
            
            x = exponential_map(x_tangent, origin, c=self.c.item())
            x = self.project_to_manifold(x)
        
        # Convert to Euclidean space for loss computation
        # Map final hyperbolic features to tangent space for classification
        origin = torch.zeros_like(x)
        logits = logarithmic_map(origin, x, c=self.c.item())
        
        return logits
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            predictions: Model predictions (logits in Euclidean space)
            targets: Target class indices
            
        Returns:
            Cross-entropy loss
        """
        return F.cross_entropy(predictions, targets)
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def extra_repr(self) -> str:
        """String representation of the classifier."""
        return (f'input_dim={self.input_dim}, num_classes={self.num_classes}, '
                f'hidden_dims={self.hidden_dims}, vocab_size={self.vocab_size}, '
                f'embedding_dim={self.embedding_dim}, dropout={self.dropout_prob}, '
                f'curvature={self.c.item():.3f}')
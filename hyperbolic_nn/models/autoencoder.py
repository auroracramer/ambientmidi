"""
Hyperbolic autoencoder model.

This module implements an autoencoder that operates in hyperbolic space,
useful for dimensionality reduction and representation learning of
hierarchical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

from ..core.base import HyperbolicModule
from ..layers.linear import HyperbolicLinear
from ..layers.activation import HyperbolicDropout, HyperbolicLayerNorm
from ..core.math_ops import logarithmic_map, exponential_map


logger = logging.getLogger(__name__)


class HyperbolicAutoEncoder(HyperbolicModule):
    """
    Hyperbolic autoencoder for representation learning.
    
    This autoencoder operates entirely in hyperbolic space and can learn
    hierarchical representations that are naturally suited to tree-like
    or graph-structured data.
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension
        encoder_dims: List of encoder hidden layer dimensions
        decoder_dims: List of decoder hidden layer dimensions (None to mirror encoder)
        dropout: Dropout probability
        use_layer_norm: Whether to use layer normalization
        tie_weights: Whether to tie encoder and decoder weights
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> autoencoder = HyperbolicAutoEncoder(
        ...     input_dim=784, latent_dim=64, encoder_dims=[256, 128]
        ... )
        >>> x = torch.randn(32, 784) * 0.1
        >>> x = autoencoder.project_to_manifold(x)
        >>> reconstructed, encoded = autoencoder(x)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_dims: list = [256, 128],
        decoder_dims: Optional[list] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        tie_weights: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims or list(reversed(encoder_dims))
        self.dropout_prob = dropout
        self.use_layer_norm = use_layer_norm
        self.tie_weights = tie_weights
        
        # Build encoder
        encoder_layer_dims = [input_dim] + encoder_dims + [latent_dim]
        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList() if use_layer_norm else None
        self.encoder_dropouts = nn.ModuleList() if dropout > 0 else None
        
        for i in range(len(encoder_layer_dims) - 1):
            layer = HyperbolicLinear(
                encoder_layer_dims[i],
                encoder_layer_dims[i + 1],
                bias=True,
                manifold=self.manifold,
                curvature=self.c.item()
            )
            self.encoder_layers.append(layer)
            
            # Add layer norm (except for output layer)
            if use_layer_norm and i < len(encoder_layer_dims) - 2:
                ln = HyperbolicLayerNorm(
                    encoder_layer_dims[i + 1],
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
                self.encoder_norms.append(ln)
            elif use_layer_norm:
                self.encoder_norms.append(nn.Identity())
            
            # Add dropout (except for output layer)
            if dropout > 0 and i < len(encoder_layer_dims) - 2:
                drop = HyperbolicDropout(
                    p=dropout,
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
                self.encoder_dropouts.append(drop)
            elif dropout > 0:
                self.encoder_dropouts.append(nn.Identity())
        
        # Build decoder
        decoder_layer_dims = [latent_dim] + self.decoder_dims + [input_dim]
        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList() if use_layer_norm else None
        self.decoder_dropouts = nn.ModuleList() if dropout > 0 else None
        
        for i in range(len(decoder_layer_dims) - 1):
            if tie_weights and i == len(decoder_layer_dims) - 2:
                # Tie weights with the first encoder layer (transposed)
                layer = TiedHyperbolicLinear(
                    self.encoder_layers[0],
                    decoder_layer_dims[i],
                    decoder_layer_dims[i + 1]
                )
            else:
                layer = HyperbolicLinear(
                    decoder_layer_dims[i],
                    decoder_layer_dims[i + 1],
                    bias=True,
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
            self.decoder_layers.append(layer)
            
            # Add layer norm (except for output layer)
            if use_layer_norm and i < len(decoder_layer_dims) - 2:
                ln = HyperbolicLayerNorm(
                    decoder_layer_dims[i + 1],
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
                self.decoder_norms.append(ln)
            elif use_layer_norm:
                self.decoder_norms.append(nn.Identity())
            
            # Add dropout (except for output layer)
            if dropout > 0 and i < len(decoder_layer_dims) - 2:
                drop = HyperbolicDropout(
                    p=dropout,
                    manifold=self.manifold,
                    curvature=self.c.item()
                )
                self.decoder_dropouts.append(drop)
            elif dropout > 0:
                self.decoder_dropouts.append(nn.Identity())
        
        logger.info(f"Initialized HyperbolicAutoEncoder({input_dim} -> {latent_dim} -> {input_dim})")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Latent representation in hyperbolic space
        """
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            
            # Apply layer norm (except for last layer)
            if self.encoder_norms is not None and i < len(self.encoder_layers) - 1:
                if not isinstance(self.encoder_norms[i], nn.Identity):
                    x = self.encoder_norms[i](x)
            
            # Apply activation (except for last layer)
            if i < len(self.encoder_layers) - 1:
                origin = torch.zeros_like(x)
                x_tangent = logarithmic_map(origin, x, c=self.c.item())
                x_tangent = F.relu(x_tangent)
                x = exponential_map(x_tangent, origin, c=self.c.item())
                x = self.project_to_manifold(x)
            
            # Apply dropout (except for last layer)
            if self.encoder_dropouts is not None and i < len(self.encoder_layers) - 1:
                if not isinstance(self.encoder_dropouts[i], nn.Identity):
                    x = self.encoder_dropouts[i](x)
        
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent representation in hyperbolic space
            
        Returns:
            Reconstructed output in hyperbolic space
        """
        x = z
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            
            # Apply layer norm (except for last layer)
            if self.decoder_norms is not None and i < len(self.decoder_layers) - 1:
                if not isinstance(self.decoder_norms[i], nn.Identity):
                    x = self.decoder_norms[i](x)
            
            # Apply activation (except for last layer)
            if i < len(self.decoder_layers) - 1:
                origin = torch.zeros_like(x)
                x_tangent = logarithmic_map(origin, x, c=self.c.item())
                x_tangent = F.relu(x_tangent)
                x = exponential_map(x_tangent, origin, c=self.c.item())
                x = self.project_to_manifold(x)
            
            # Apply dropout (except for last layer)
            if self.decoder_dropouts is not None and i < len(self.decoder_layers) - 1:
                if not isinstance(self.decoder_dropouts[i], nn.Identity):
                    x = self.decoder_dropouts[i](x)
        
        return x
    
    def hyperbolic_forward(
        self, 
        x: torch.Tensor, 
        return_latent: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor in hyperbolic space
            return_latent: Whether to return latent representation
            
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        # Encode
        latent = self.encode(x)
        
        # Decode
        reconstructed = self.decode(latent)
        
        if return_latent:
            return reconstructed, latent
        else:
            return reconstructed, None
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructed: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute autoencoder loss.
        
        Args:
            x: Original input
            reconstructed: Reconstructed output
            latent: Latent representation (for regularization)
            beta: Weight for latent regularization
            
        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss using hyperbolic distance
        reconstruction_loss = self.distance(x, reconstructed).mean()
        
        losses = {'reconstruction_loss': reconstruction_loss}
        
        # Latent regularization (encourage points near origin)
        if latent is not None and beta > 0:
            origin = torch.zeros_like(latent)
            latent_distances = self.distance(latent, origin)
            latent_reg = latent_distances.mean()
            losses['latent_reg'] = beta * latent_reg
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = batch  # Ignore labels for unsupervised learning
        else:
            x = batch
        
        reconstructed, latent = self(x, return_latent=True)
        losses = self.compute_loss(x, reconstructed, latent)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = batch
        else:
            x = batch
        
        reconstructed, latent = self(x, return_latent=True)
        losses = self.compute_loss(x, reconstructed, latent)
        
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True, prog_bar=True)
        
        return losses['total_loss']
    
    def extra_repr(self) -> str:
        """String representation of the autoencoder."""
        return (f'input_dim={self.input_dim}, latent_dim={self.latent_dim}, '
                f'encoder_dims={self.encoder_dims}, decoder_dims={self.decoder_dims}, '
                f'dropout={self.dropout_prob}, tie_weights={self.tie_weights}, '
                f'curvature={self.c.item():.3f}')


class TiedHyperbolicLinear(nn.Module):
    """
    Tied hyperbolic linear layer for weight sharing in autoencoders.
    
    This layer shares weights with another HyperbolicLinear layer,
    typically used in the decoder to tie weights with the encoder.
    """
    
    def __init__(
        self,
        tied_layer: HyperbolicLinear,
        in_features: int,
        out_features: int
    ):
        super().__init__()
        
        self.tied_layer = tied_layer
        self.in_features = in_features
        self.out_features = out_features
        
        # Create bias parameter if the tied layer has bias
        if tied_layer.linear.bias is not None:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using tied weights."""
        # Use transposed weights from the tied layer
        weight = self.tied_layer.linear.weight.t()
        
        # Apply linear transformation in tangent space
        origin = torch.zeros_like(x)
        x_tangent = logarithmic_map(origin, x, c=self.tied_layer.c.item())
        
        # Linear transformation with tied weights
        y_tangent = F.linear(x_tangent, weight, self.bias)
        
        # Map back to hyperbolic space
        origin_out = torch.zeros_like(y_tangent)
        y = exponential_map(y_tangent, origin_out, c=self.tied_layer.c.item())
        
        # Project to manifold
        y = self.tied_layer.project_to_manifold(y)
        
        return y
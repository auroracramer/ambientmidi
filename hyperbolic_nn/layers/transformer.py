"""
Hyperbolic transformer layers.

This module implements transformer architectures adapted for hyperbolic space,
including self-attention mechanisms, transformer blocks, and complete models.
The implementation preserves hyperbolic geometry while providing the
representational power of transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import logging

from ..core.base import HyperbolicModule
from ..core.math_ops import (
    exponential_map,
    logarithmic_map,
    project_to_poincare_ball,
    mobius_add,
    hyperbolic_distance,
    hyperbolic_centroid
)
from .linear import HyperbolicLinear
from .activation import HyperbolicLayerNorm, HyperbolicDropout, HyperbolicGELU, HyperbolicActivation


logger = logging.getLogger(__name__)


class HyperbolicSelfAttention(HyperbolicModule):
    """
    Hyperbolic self-attention mechanism.
    
    This implements self-attention in hyperbolic space where attention weights
    are computed using hyperbolic distances rather than dot products, and
    the weighted aggregation is performed using hyperbolic centroid computation.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
        use_hyperbolic_distance: Whether to use hyperbolic distance for attention
        temperature: Temperature for attention softmax
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> attention = HyperbolicSelfAttention(512, num_heads=8)
        >>> x = torch.randn(32, 100, 512) * 0.1  # (batch, seq_len, embed_dim)
        >>> x = attention.project_to_manifold(x)
        >>> y = attention(x)  # Shape: (32, 100, 512)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        use_hyperbolic_distance: bool = True,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_prob = dropout
        self.use_hyperbolic_distance = use_hyperbolic_distance
        self.temperature = temperature
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Query, key, value projections
        self.q_proj = HyperbolicLinear(
            embed_dim, embed_dim, bias=bias,
            manifold=self.manifold, curvature=self.c.item()
        )
        self.k_proj = HyperbolicLinear(
            embed_dim, embed_dim, bias=bias,
            manifold=self.manifold, curvature=self.c.item()
        )
        self.v_proj = HyperbolicLinear(
            embed_dim, embed_dim, bias=bias,
            manifold=self.manifold, curvature=self.c.item()
        )
        
        # Output projection
        self.out_proj = HyperbolicLinear(
            embed_dim, embed_dim, bias=bias,
            manifold=self.manifold, curvature=self.c.item()
        )
        
        # Dropout
        if dropout > 0:
            self.dropout = HyperbolicDropout(p=dropout, manifold=self.manifold, curvature=self.c.item())
        else:
            self.dropout = None
        
        logger.info(f"Initialized HyperbolicSelfAttention(embed_dim={embed_dim}, "
                   f"num_heads={num_heads}, use_hyperbolic_distance={use_hyperbolic_distance})")
    
    def hyperbolic_forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for hyperbolic self-attention.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor in hyperbolic space, optionally with attention weights
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to query, key, value
        q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, embed_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        if self.use_hyperbolic_distance:
            # Use hyperbolic distance for attention
            attn_scores = self._compute_hyperbolic_attention(q, k)
        else:
            # Use standard dot-product attention in tangent space
            attn_scores = self._compute_euclidean_attention(q, k)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores / self.temperature, dim=-1)
        
        # Apply dropout to attention weights
        if self.dropout is not None and self.training:
            # Note: applying standard dropout to attention weights (not hyperbolic)
            attn_weights = F.dropout(attn_weights, p=self.dropout_prob, training=self.training)
        
        # Apply attention to values
        if self.use_hyperbolic_distance:
            attn_output = self._apply_hyperbolic_attention(attn_weights, v)
        else:
            attn_output = self._apply_euclidean_attention(attn_weights, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, embed_dim
        )
        
        # Final output projection
        output = self.out_proj(attn_output)
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output
    
    def _compute_hyperbolic_attention(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute attention scores using hyperbolic distances."""
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_k = k.shape[2]
        
        # Expand for pairwise distance computation
        q_expanded = q.unsqueeze(3)  # (batch, heads, seq_q, 1, head_dim)
        k_expanded = k.unsqueeze(2)  # (batch, heads, 1, seq_k, head_dim)
        
        # Broadcast to compute all pairwise distances
        q_broadcast = q_expanded.expand(-1, -1, -1, seq_len_k, -1)
        k_broadcast = k_expanded.expand(-1, -1, seq_len_q, -1, -1)
        
        # Flatten for hyperbolic distance computation
        q_flat = q_broadcast.reshape(-1, head_dim)
        k_flat = k_broadcast.reshape(-1, head_dim)
        
        # Compute hyperbolic distances
        distances = hyperbolic_distance(q_flat, k_flat, c=self.c.item())
        
        # Reshape back to attention score format
        distances = distances.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Convert distances to similarity scores (negative distance)
        attn_scores = -distances * self.scale
        
        return attn_scores
    
    def _compute_euclidean_attention(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute attention scores using dot product in tangent space."""
        # Map to tangent space for dot product computation
        origin_q = torch.zeros_like(q)
        origin_k = torch.zeros_like(k)
        
        q_tangent = logarithmic_map(origin_q, q, c=self.c.item())
        k_tangent = logarithmic_map(origin_k, k, c=self.c.item())
        
        # Compute scaled dot-product attention in tangent space
        attn_scores = torch.matmul(q_tangent, k_tangent.transpose(-2, -1)) * self.scale
        
        return attn_scores
    
    def _apply_hyperbolic_attention(self, attn_weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply attention weights to values using hyperbolic centroid."""
        batch_size, num_heads, seq_len_q, seq_len_v = attn_weights.shape
        head_dim = v.shape[-1]
        
        # Initialize output
        attn_output = torch.zeros_like(v[:, :, :seq_len_q, :])
        
        # For each query position, compute weighted centroid of values
        for i in range(seq_len_q):
            # Get attention weights for query i: (batch, heads, seq_v)
            weights_i = attn_weights[:, :, i, :]  # (batch, heads, seq_v)
            
            # Get values: (batch, heads, seq_v, head_dim)
            values_i = v  # (batch, heads, seq_v, head_dim)
            
            # Reshape for centroid computation
            weights_flat = weights_i.reshape(-1, seq_len_v)  # (batch*heads, seq_v)
            values_flat = values_i.reshape(-1, seq_len_v, head_dim)  # (batch*heads, seq_v, head_dim)
            
            # Compute hyperbolic centroid for each batch*head
            centroids = []
            for j in range(weights_flat.shape[0]):
                centroid = hyperbolic_centroid(
                    values_flat[j:j+1],  # (1, seq_v, head_dim)
                    weights_flat[j:j+1],  # (1, seq_v)
                    c=self.c.item()
                )
                centroids.append(centroid)
            
            centroids = torch.stack(centroids, dim=0)  # (batch*heads, head_dim)
            centroids = centroids.reshape(batch_size, num_heads, head_dim)
            
            attn_output[:, :, i, :] = centroids
        
        return attn_output
    
    def _apply_euclidean_attention(self, attn_weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply attention weights to values using standard weighted sum in tangent space."""
        # Map values to tangent space
        origin_v = torch.zeros_like(v)
        v_tangent = logarithmic_map(origin_v, v, c=self.c.item())
        
        # Apply attention weights in tangent space
        attn_output_tangent = torch.matmul(attn_weights, v_tangent)
        
        # Map back to hyperbolic space
        origin_out = torch.zeros_like(attn_output_tangent)
        attn_output = exponential_map(attn_output_tangent, origin_out, c=self.c.item())
        
        # Ensure output is on manifold
        attn_output = self.project_to_manifold(attn_output)
        
        return attn_output
    
    def extra_repr(self) -> str:
        return (f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, '
                f'dropout={self.dropout_prob}, use_hyperbolic_distance={self.use_hyperbolic_distance}, '
                f'curvature={self.c.item():.3f}')


class HyperbolicTransformerBlock(HyperbolicModule):
    """
    Hyperbolic transformer block.
    
    This implements a complete transformer block with hyperbolic self-attention,
    layer normalization, and feed-forward network, all adapted for hyperbolic space.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward network hidden dimension
        dropout: Dropout probability
        activation: Activation function for feed-forward network
        use_hyperbolic_distance: Whether to use hyperbolic distance in attention
        **kwargs: Additional arguments passed to HyperbolicModule
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_hyperbolic_distance: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if ff_dim is None:
            ff_dim = embed_dim * 4
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_prob = dropout
        
        # Self-attention layer
        self.self_attn = HyperbolicSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_hyperbolic_distance=use_hyperbolic_distance,
            manifold=self.manifold,
            curvature=self.c.item()
        )
        
        # Layer normalization
        self.norm1 = HyperbolicLayerNorm(
            embed_dim,
            manifold=self.manifold,
            curvature=self.c.item()
        )
        self.norm2 = HyperbolicLayerNorm(
            embed_dim,
            manifold=self.manifold,
            curvature=self.c.item()
        )
        
        # Feed-forward network
        self.ff = nn.ModuleList([
            HyperbolicLinear(
                embed_dim, ff_dim,
                manifold=self.manifold,
                curvature=self.c.item()
            ),
            HyperbolicGELU(manifold=self.manifold, curvature=self.c.item())
            if activation.lower() == 'gelu' else
            HyperbolicActivation(activation, manifold=self.manifold, curvature=self.c.item()),
            HyperbolicLinear(
                ff_dim, embed_dim,
                manifold=self.manifold,
                curvature=self.c.item()
            )
        ])
        
        # Dropout
        if dropout > 0:
            self.dropout = HyperbolicDropout(
                p=dropout,
                manifold=self.manifold,
                curvature=self.c.item()
            )
        else:
            self.dropout = None
        
        logger.info(f"Initialized HyperbolicTransformerBlock(embed_dim={embed_dim}, "
                   f"num_heads={num_heads}, ff_dim={ff_dim})")
    
    def hyperbolic_forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor in hyperbolic space
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor in hyperbolic space
        """
        # Self-attention with residual connection
        attn_output = self.self_attn(x, attention_mask=attention_mask)
        
        # Apply dropout to attention output
        if self.dropout is not None:
            attn_output = self.dropout(attn_output)
        
        # Residual connection in hyperbolic space (Möbius addition)
        x = mobius_add(x, attn_output, c=self.c.item())
        
        # Layer normalization
        x = self.norm1(x)
        
        # Feed-forward network
        ff_output = x
        for layer in self.ff:
            ff_output = layer(ff_output)
        
        # Apply dropout to feed-forward output
        if self.dropout is not None:
            ff_output = self.dropout(ff_output)
        
        # Residual connection in hyperbolic space
        x = mobius_add(x, ff_output, c=self.c.item())
        
        # Layer normalization
        x = self.norm2(x)
        
        return x
    
    def extra_repr(self) -> str:
        return (f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, '
                f'ff_dim={self.ff_dim}, dropout={self.dropout_prob}, '
                f'curvature={self.c.item():.3f}')


class HyperbolicTransformer(HyperbolicModule):
    """
    Complete hyperbolic transformer model.
    
    This implements a full transformer architecture operating in hyperbolic space,
    with optional positional encoding and flexible output configurations.
    
    Args:
        embed_dim: Embedding dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ff_dim: Feed-forward network hidden dimension
        max_seq_len: Maximum sequence length for positional encoding
        dropout: Dropout probability
        use_pos_encoding: Whether to use positional encoding
        use_hyperbolic_distance: Whether to use hyperbolic distance in attention
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> transformer = HyperbolicTransformer(
        ...     embed_dim=512, num_layers=6, num_heads=8, max_seq_len=1000
        ... )
        >>> x = torch.randn(32, 100, 512) * 0.1
        >>> x = transformer.project_to_manifold(x)
        >>> y = transformer(x)  # Shape: (32, 100, 512)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        max_seq_len: int = 1000,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
        use_hyperbolic_distance: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if ff_dim is None:
            ff_dim = embed_dim * 4
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout_prob = dropout
        self.use_pos_encoding = use_pos_encoding
        
        # Positional encoding (in tangent space)
        if use_pos_encoding:
            self.pos_encoding = self._create_positional_encoding()
        else:
            self.register_buffer('pos_encoding', None)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HyperbolicTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_hyperbolic_distance=use_hyperbolic_distance,
                manifold=self.manifold,
                curvature=self.c.item()
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = HyperbolicLayerNorm(
            embed_dim,
            manifold=self.manifold,
            curvature=self.c.item()
        )
        
        # Input dropout
        if dropout > 0:
            self.input_dropout = HyperbolicDropout(
                p=dropout,
                manifold=self.manifold,
                curvature=self.c.item()
            )
        else:
            self.input_dropout = None
        
        logger.info(f"Initialized HyperbolicTransformer(embed_dim={embed_dim}, "
                   f"num_layers={num_layers}, num_heads={num_heads})")
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() *
                            (-math.log(10000.0) / self.embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Scale down for hyperbolic space
        pe = pe * 0.1
        
        return pe
    
    def hyperbolic_forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Forward pass through hyperbolic transformer.
        
        Args:
            x: Input tensor in hyperbolic space, shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask
            return_all_layers: Whether to return outputs from all layers
            
        Returns:
            Output tensor(s) in hyperbolic space
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Add positional encoding if enabled
        if self.use_pos_encoding and self.pos_encoding is not None:
            # Get positional encoding for current sequence length
            pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
            pos_enc = pos_enc.to(x.device)
            
            # Map positional encoding to hyperbolic space
            pos_enc_hyp = exponential_map(
                pos_enc,
                torch.zeros_like(pos_enc),
                c=self.c.item()
            )
            pos_enc_hyp = self.project_to_manifold(pos_enc_hyp)
            
            # Add positional encoding using Möbius addition
            x = mobius_add(x, pos_enc_hyp, c=self.c.item())
        
        # Apply input dropout
        if self.input_dropout is not None:
            x = self.input_dropout(x)
        
        # Pass through transformer blocks
        all_layer_outputs = []
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
            if return_all_layers:
                all_layer_outputs.append(x)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        if return_all_layers:
            return x, all_layer_outputs
        else:
            return x
    
    def extra_repr(self) -> str:
        return (f'embed_dim={self.embed_dim}, num_layers={self.num_layers}, '
                f'num_heads={self.num_heads}, ff_dim={self.ff_dim}, '
                f'max_seq_len={self.max_seq_len}, dropout={self.dropout_prob}, '
                f'curvature={self.c.item():.3f}')
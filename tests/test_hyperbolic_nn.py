"""
Comprehensive unit tests for hyperbolic neural networks.

This test suite covers all major components of the hyperbolic neural network
implementation including core mathematical operations, layers, models,
and utilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional

# Import hyperbolic neural network components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hyperbolic_nn.core.manifolds import PoincareManifold, HyperboloidManifold
from hyperbolic_nn.core.math_ops import (
    mobius_add, mobius_mul, exponential_map, logarithmic_map,
    hyperbolic_distance, project_to_poincare_ball
)
from hyperbolic_nn.core.base import HyperbolicEmbedding
from hyperbolic_nn.layers.linear import HyperbolicLinear, HyperbolicMLP
from hyperbolic_nn.layers.conv import HyperbolicConv1d, HyperbolicConv2d
from hyperbolic_nn.layers.activation import (
    HyperbolicActivation, HyperbolicLayerNorm, HyperbolicDropout
)
from hyperbolic_nn.layers.transformer import (
    HyperbolicSelfAttention, HyperbolicTransformerBlock, HyperbolicTransformer
)
from hyperbolic_nn.models.classifier import HyperbolicClassifier
from hyperbolic_nn.models.autoencoder import HyperbolicAutoEncoder
from hyperbolic_nn.models.graph_nn import HyperbolicGraphNN
from hyperbolic_nn.utils.initialization import hyperbolic_init, HyperbolicInitializer
from hyperbolic_nn.utils.metrics import HyperbolicMetrics


class TestPoincareManifold:
    """Test Poincaré manifold operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.curvature = 1.0
        self.manifold = PoincareManifold(curvature=self.curvature)
        self.device = torch.device('cpu')
        
    def test_initialization(self):
        """Test manifold initialization."""
        assert self.manifold.c == self.curvature
        assert self.manifold.eps == 1e-15
        
    def test_projection(self):
        """Test projection to Poincaré ball."""
        # Test points inside the ball
        x_inside = torch.tensor([[0.1, 0.2], [0.3, -0.1]])
        projected = self.manifold.project(x_inside)
        
        # Points inside should remain unchanged
        torch.testing.assert_close(projected, x_inside, atol=1e-6, rtol=1e-6)
        
        # Test points outside the ball
        x_outside = torch.tensor([[2.0, 1.0], [-1.5, 0.8]])
        projected = self.manifold.project(x_outside)
        
        # Points should be projected inside
        norms = torch.norm(projected, dim=1)
        assert torch.all(norms < self.manifold.max_norm)
        
    def test_distance(self):
        """Test hyperbolic distance computation."""
        x = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
        y = torch.tensor([[0.0, 0.0], [0.0, 0.1]])
        
        distances = self.manifold.distance(x, y)
        
        # Distance from point to itself should be zero
        assert torch.abs(distances[0]) < 1e-6
        
        # Distance should be positive and symmetric
        assert distances[1] > 0
        distances_reverse = self.manifold.distance(y, x)
        torch.testing.assert_close(distances, distances_reverse, atol=1e-6, rtol=1e-6)
        
    def test_exponential_logarithmic_map(self):
        """Test exponential and logarithmic maps are inverses."""
        x = torch.tensor([[0.1, 0.2], [0.0, 0.0]])
        v = torch.tensor([[0.01, 0.02], [0.05, -0.03]])
        
        # Test exp(log_x(y)) = y
        y = self.manifold.exponential_map(v, x)
        v_recovered = self.manifold.logarithmic_map(x, y)
        
        torch.testing.assert_close(v, v_recovered, atol=1e-5, rtol=1e-5)
        
    def test_mobius_operations(self):
        """Test Möbius addition properties."""
        x = torch.tensor([[0.1, 0.2]])
        y = torch.tensor([[0.2, -0.1]])
        z = torch.tensor([[0.05, 0.15]])
        
        # Test commutativity: x ⊕ y = y ⊕ x
        xy = self.manifold.mobius_add(x, y)
        yx = self.manifold.mobius_add(y, x)
        torch.testing.assert_close(xy, yx, atol=1e-6, rtol=1e-6)
        
        # Test identity: x ⊕ 0 = x
        zero = torch.zeros_like(x)
        x_plus_zero = self.manifold.mobius_add(x, zero)
        torch.testing.assert_close(x, x_plus_zero, atol=1e-6, rtol=1e-6)


class TestMathOperations:
    """Test core mathematical operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.curvature = 1.0
        
    def test_mobius_add(self):
        """Test Möbius addition function."""
        x = torch.tensor([[0.1, 0.2]])
        y = torch.tensor([[0.2, -0.1]])
        
        result = mobius_add(x, y, c=self.curvature)
        
        # Result should be in Poincaré ball
        norm = torch.norm(result, dim=1)
        max_norm = (1.0 - 1e-3) / math.sqrt(self.curvature)
        assert torch.all(norm < max_norm)
        
    def test_hyperbolic_distance(self):
        """Test hyperbolic distance function."""
        x = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
        y = torch.tensor([[0.0, 0.0], [0.0, 0.1]])
        
        distances = hyperbolic_distance(x, y, c=self.curvature)
        
        # Distance from point to itself should be zero
        assert torch.abs(distances[0]) < 1e-6
        assert distances[1] > 0
        
    def test_exponential_logarithmic_map(self):
        """Test exponential and logarithmic map functions."""
        x = torch.tensor([[0.1, 0.2]])
        v = torch.tensor([[0.01, 0.02]])
        
        y = exponential_map(v, x, c=self.curvature)
        v_recovered = logarithmic_map(x, y, c=self.curvature)
        
        torch.testing.assert_close(v, v_recovered, atol=1e-5, rtol=1e-5)
        
    def test_projection(self):
        """Test projection function."""
        x_outside = torch.tensor([[2.0, 1.0]])
        projected = project_to_poincare_ball(x_outside, c=self.curvature)
        
        norm = torch.norm(projected, dim=1)
        max_norm = (1.0 - 1e-3) / math.sqrt(self.curvature)
        assert torch.all(norm < max_norm)


class TestHyperbolicLayers:
    """Test hyperbolic neural network layers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.curvature = 1.0
        self.batch_size = 4
        self.input_dim = 8
        self.output_dim = 6
        
    def test_hyperbolic_linear(self):
        """Test hyperbolic linear layer."""
        layer = HyperbolicLinear(
            self.input_dim, self.output_dim,
            curvature=self.curvature
        )
        
        # Test forward pass
        x = torch.randn(self.batch_size, self.input_dim) * 0.1
        x = layer.project_to_manifold(x)
        
        y = layer(x)
        
        # Check output shape
        assert y.shape == (self.batch_size, self.output_dim)
        
        # Check output is on manifold
        norms = torch.norm(y, dim=1)
        max_norm = (1.0 - 1e-3) / math.sqrt(self.curvature)
        assert torch.all(norms < max_norm)
        
    def test_hyperbolic_mlp(self):
        """Test hyperbolic MLP."""
        layer_sizes = [self.input_dim, 16, 12, self.output_dim]
        mlp = HyperbolicMLP(
            layer_sizes,
            curvature=self.curvature,
            dropout=0.1
        )
        
        x = torch.randn(self.batch_size, self.input_dim) * 0.1
        x = mlp.project_to_manifold(x)
        
        y = mlp(x)
        
        assert y.shape == (self.batch_size, self.output_dim)
        
    def test_hyperbolic_conv1d(self):
        """Test hyperbolic 1D convolution."""
        in_channels, out_channels = 4, 8
        length = 20
        
        conv = HyperbolicConv1d(
            in_channels, out_channels,
            kernel_size=3, padding=1,
            curvature=self.curvature
        )
        
        x = torch.randn(self.batch_size, in_channels, length) * 0.1
        x = conv.project_to_manifold(x)
        
        y = conv(x)
        
        assert y.shape == (self.batch_size, out_channels, length)
        
    def test_hyperbolic_conv2d(self):
        """Test hyperbolic 2D convolution."""
        in_channels, out_channels = 3, 16
        height, width = 12, 12
        
        conv = HyperbolicConv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1,
            curvature=self.curvature
        )
        
        x = torch.randn(self.batch_size, in_channels, height, width) * 0.1
        x = conv.project_to_manifold(x)
        
        y = conv(x)
        
        assert y.shape == (self.batch_size, out_channels, height, width)
        
    def test_hyperbolic_activation(self):
        """Test hyperbolic activation functions."""
        activation = HyperbolicActivation('relu', curvature=self.curvature)
        
        x = torch.randn(self.batch_size, self.input_dim) * 0.1
        x = activation.project_to_manifold(x)
        
        y = activation(x)
        
        assert y.shape == x.shape
        
    def test_hyperbolic_layer_norm(self):
        """Test hyperbolic layer normalization."""
        layer_norm = HyperbolicLayerNorm(
            self.input_dim,
            curvature=self.curvature
        )
        
        x = torch.randn(self.batch_size, self.input_dim) * 0.1
        x = layer_norm.project_to_manifold(x)
        
        y = layer_norm(x)
        
        assert y.shape == x.shape
        
    def test_hyperbolic_dropout(self):
        """Test hyperbolic dropout."""
        dropout = HyperbolicDropout(p=0.5, curvature=self.curvature)
        
        x = torch.randn(self.batch_size, self.input_dim) * 0.1
        x = dropout.project_to_manifold(x)
        
        # Test training mode
        dropout.train()
        y_train = dropout(x)
        
        # Test eval mode
        dropout.eval()
        y_eval = dropout(x)
        
        assert y_train.shape == x.shape
        assert y_eval.shape == x.shape
        
        # In eval mode, output should be identical to input
        torch.testing.assert_close(y_eval, x, atol=1e-6, rtol=1e-6)


class TestHyperbolicTransformer:
    """Test hyperbolic transformer components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.curvature = 1.0
        self.batch_size = 2
        self.seq_len = 8
        self.embed_dim = 16
        self.num_heads = 4
        
    def test_hyperbolic_self_attention(self):
        """Test hyperbolic self-attention."""
        attention = HyperbolicSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            curvature=self.curvature
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim) * 0.1
        x = attention.project_to_manifold(x)
        
        y = attention(x)
        
        assert y.shape == (self.batch_size, self.seq_len, self.embed_dim)
        
        # Test with attention weights
        y, weights = attention(x, return_attention_weights=True)
        assert weights.shape == (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        
    def test_hyperbolic_transformer_block(self):
        """Test hyperbolic transformer block."""
        block = HyperbolicTransformerBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            curvature=self.curvature
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim) * 0.1
        x = block.project_to_manifold(x)
        
        y = block(x)
        
        assert y.shape == (self.batch_size, self.seq_len, self.embed_dim)
        
    def test_hyperbolic_transformer(self):
        """Test full hyperbolic transformer."""
        transformer = HyperbolicTransformer(
            embed_dim=self.embed_dim,
            num_layers=2,
            num_heads=self.num_heads,
            max_seq_len=self.seq_len,
            curvature=self.curvature
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim) * 0.1
        x = transformer.project_to_manifold(x)
        
        y = transformer(x)
        
        assert y.shape == (self.batch_size, self.seq_len, self.embed_dim)


class TestHyperbolicModels:
    """Test complete hyperbolic models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.curvature = 1.0
        self.batch_size = 4
        
    def test_hyperbolic_classifier(self):
        """Test hyperbolic classifier."""
        classifier = HyperbolicClassifier(
            input_dim=64,
            num_classes=5,
            hidden_dims=[32, 16],
            curvature=self.curvature
        )
        
        x = torch.randn(self.batch_size, 64) * 0.1
        logits = classifier(x)
        
        assert logits.shape == (self.batch_size, 5)
        
        # Test with discrete inputs
        classifier_discrete = HyperbolicClassifier(
            vocab_size=1000,
            embedding_dim=32,
            num_classes=3,
            hidden_dims=[16],
            curvature=self.curvature
        )
        
        tokens = torch.randint(0, 1000, (self.batch_size, 10))
        logits = classifier_discrete(tokens)
        
        assert logits.shape == (self.batch_size, 3)
        
    def test_hyperbolic_autoencoder(self):
        """Test hyperbolic autoencoder."""
        autoencoder = HyperbolicAutoEncoder(
            input_dim=32,
            latent_dim=8,
            encoder_dims=[16],
            curvature=self.curvature
        )
        
        x = torch.randn(self.batch_size, 32) * 0.1
        x = autoencoder.project_to_manifold(x)
        
        reconstructed, latent = autoencoder(x, return_latent=True)
        
        assert reconstructed.shape == (self.batch_size, 32)
        assert latent.shape == (self.batch_size, 8)
        
        # Test loss computation
        losses = autoencoder.compute_loss(x, reconstructed, latent)
        assert 'reconstruction_loss' in losses
        assert 'total_loss' in losses
        
    def test_hyperbolic_graph_nn(self):
        """Test hyperbolic graph neural network."""
        gnn = HyperbolicGraphNN(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            num_layers=2,
            curvature=self.curvature
        )
        
        num_nodes = 20
        num_edges = 40
        
        node_features = torch.randn(num_nodes, 16) * 0.1
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        node_features = gnn.project_to_manifold(node_features)
        output = gnn(node_features, edge_index)
        
        assert output.shape == (num_nodes, 8)


class TestHyperbolicEmbedding:
    """Test hyperbolic embedding layer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.curvature = 1.0
        self.vocab_size = 100
        self.embedding_dim = 16
        self.batch_size = 4
        self.seq_len = 10
        
    def test_hyperbolic_embedding(self):
        """Test hyperbolic embedding layer."""
        embedding = HyperbolicEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            curvature=self.curvature
        )
        
        # Test forward pass
        tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        embeddings = embedding(tokens)
        
        assert embeddings.shape == (self.batch_size, self.seq_len, self.embedding_dim)
        
        # Check embeddings are on manifold
        norms = torch.norm(embeddings, dim=-1)
        max_norm = (1.0 - 1e-3) / math.sqrt(self.curvature)
        assert torch.all(norms < max_norm)
        
    def test_embedding_with_padding(self):
        """Test embedding with padding index."""
        padding_idx = 0
        embedding = HyperbolicEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
            curvature=self.curvature
        )
        
        # Check padding embedding is at origin
        padding_embedding = embedding.embedding.weight[padding_idx]
        assert torch.allclose(padding_embedding, torch.zeros_like(padding_embedding))


class TestUtilities:
    """Test utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.curvature = 1.0
        
    def test_hyperbolic_init(self):
        """Test hyperbolic initialization."""
        tensor = torch.empty(10, 8)
        initialized = hyperbolic_init(
            tensor,
            curvature=self.curvature,
            method='xavier'
        )
        
        # Check tensor is on manifold
        norms = torch.norm(initialized, dim=1)
        max_norm = (1.0 - 1e-3) / math.sqrt(self.curvature)
        assert torch.all(norms < max_norm)
        
    def test_hyperbolic_initializer(self):
        """Test hyperbolic initializer class."""
        initializer = HyperbolicInitializer(
            curvature=self.curvature,
            method='normal',
            scale=1e-3
        )
        
        layer = HyperbolicLinear(16, 8, curvature=self.curvature)
        initialized_layer = initializer.init_module(layer)
        
        # Check layer weights are initialized properly
        norms = torch.norm(initialized_layer.linear.weight, dim=1)
        max_norm = (1.0 - 1e-3) / math.sqrt(self.curvature)
        assert torch.all(norms < max_norm)
        
    def test_hyperbolic_metrics(self):
        """Test hyperbolic metrics."""
        metrics = HyperbolicMetrics(curvature=self.curvature)
        
        embeddings = torch.randn(50, 8) * 0.1
        embeddings = project_to_poincare_ball(embeddings, c=self.curvature)
        
        # Test statistics computation
        stats = metrics.embedding_statistics(embeddings)
        
        assert 'num_embeddings' in stats
        assert 'embedding_dim' in stats
        assert 'mean_norm' in stats
        assert 'mean_distance_from_origin' in stats
        assert stats['num_embeddings'] == 50
        assert stats['embedding_dim'] == 8
        
        # Test pairwise distances
        distances = metrics.pairwise_distances(embeddings[:10])
        assert distances.shape == (10, 10)
        
        # Check distance matrix properties
        assert torch.allclose(torch.diag(distances), torch.zeros(10), atol=1e-6)
        assert torch.allclose(distances, distances.t(), atol=1e-6)


class TestNumericalStability:
    """Test numerical stability of operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.curvature = 1.0
        
    def test_extreme_values(self):
        """Test operations with extreme values."""
        manifold = PoincareManifold(curvature=self.curvature)
        
        # Test with points very close to boundary
        x = torch.tensor([[0.999, 0.0]]) / math.sqrt(self.curvature)
        projected = manifold.project(x)
        
        # Should be projected to valid range
        norm = torch.norm(projected, dim=1)
        assert torch.all(norm < manifold.max_norm)
        
        # Test distance computation doesn't explode
        y = torch.tensor([[0.0, 0.0]])
        distance = manifold.distance(projected, y)
        assert torch.isfinite(distance).all()
        
    def test_zero_vectors(self):
        """Test operations with zero vectors."""
        manifold = PoincareManifold(curvature=self.curvature)
        
        x = torch.zeros(1, 2)
        v = torch.zeros(1, 2)
        
        # Exponential map of zero vector should return original point
        result = manifold.exponential_map(v, x)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)
        
        # Logarithmic map of same point should return zero vector
        log_result = manifold.logarithmic_map(x, x)
        torch.testing.assert_close(log_result, v, atol=1e-6, rtol=1e-6)
        
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        layer = HyperbolicLinear(4, 2, curvature=self.curvature)
        
        x = torch.randn(2, 4) * 0.1
        x = layer.project_to_manifold(x)
        x.requires_grad_(True)
        
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        for param in layer.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_training_loop(self):
        """Test a simple training loop."""
        # Create model
        model = HyperbolicClassifier(
            input_dim=10,
            num_classes=3,
            hidden_dims=[8],
            curvature=1.0
        )
        
        # Create data
        batch_size = 4
        x = torch.randn(batch_size, 10) * 0.1
        y = torch.randint(0, 3, (batch_size,))
        
        # Training step
        model.train()
        logits = model(x)
        loss = model.compute_loss(logits, y)
        
        # Check loss is finite
        assert torch.isfinite(loss)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()
                
    def test_embedding_to_classifier_pipeline(self):
        """Test pipeline from embeddings to classification."""
        # Create embedding layer
        vocab_size = 100
        embedding_dim = 16
        embedding = HyperbolicEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            curvature=1.0
        )
        
        # Create classifier
        classifier = HyperbolicClassifier(
            input_dim=embedding_dim,
            num_classes=5,
            hidden_dims=[8],
            curvature=1.0
        )
        
        # Test pipeline
        batch_size = 4
        seq_len = 6
        
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        embeddings = embedding(tokens)
        
        # Average embeddings for classification
        avg_embeddings = embeddings.mean(dim=1)
        logits = classifier(avg_embeddings)
        
        assert logits.shape == (batch_size, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
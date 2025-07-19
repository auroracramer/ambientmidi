"""
Comprehensive demonstration of hyperbolic neural networks.

This script demonstrates the various components and capabilities of the
hyperbolic neural network library, including:
- Basic hyperbolic operations
- Individual layer usage
- Complete model training
- Visualization of embeddings
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import hyperbolic neural network components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hyperbolic_nn import (
    # Core components
    PoincareManifold, HyperbolicModule,
    mobius_add, hyperbolic_distance, exponential_map, logarithmic_map,
    
    # Layers
    HyperbolicLinear, HyperbolicMLP, HyperbolicConv2d,
    HyperbolicSelfAttention, HyperbolicTransformer,
    HyperbolicActivation, HyperbolicLayerNorm,
    
    # Models
    HyperbolicClassifier, HyperbolicAutoEncoder, HyperbolicGraphNN,
    
    # Utilities
    hyperbolic_init, HyperbolicMetrics, plot_poincare_embeddings
)


def demo_basic_operations():
    """Demonstrate basic hyperbolic operations."""
    print("\n" + "="*50)
    print("DEMO: Basic Hyperbolic Operations")
    print("="*50)
    
    # Create a Poincaré manifold
    curvature = 1.0
    manifold = PoincareManifold(curvature=curvature)
    
    # Create some points in hyperbolic space
    x = torch.tensor([[0.1, 0.2], [0.3, -0.1], [0.0, 0.0]])
    y = torch.tensor([[0.2, -0.1], [0.1, 0.3], [0.05, 0.15]])
    
    print(f"Points x: {x}")
    print(f"Points y: {y}")
    
    # Compute hyperbolic distances
    distances = manifold.distance(x, y)
    print(f"Hyperbolic distances: {distances}")
    
    # Möbius addition
    mobius_sum = manifold.mobius_add(x, y)
    print(f"Möbius addition (x ⊕ y): {mobius_sum}")
    
    # Exponential and logarithmic maps
    v = torch.tensor([[0.01, 0.02], [0.03, -0.01], [0.0, 0.0]])
    exp_result = manifold.exponential_map(v, x)
    log_result = manifold.logarithmic_map(x, exp_result)
    
    print(f"Tangent vectors v: {v}")
    print(f"Exponential map exp_x(v): {exp_result}")
    print(f"Logarithmic map log_x(exp_x(v)): {log_result}")
    print(f"Reconstruction error: {torch.norm(v - log_result, dim=1)}")


def demo_linear_layers():
    """Demonstrate hyperbolic linear layers."""
    print("\n" + "="*50)
    print("DEMO: Hyperbolic Linear Layers")
    print("="*50)
    
    # Single linear layer
    layer = HyperbolicLinear(in_features=8, out_features=4, curvature=1.0)
    
    # Create input data
    batch_size = 5
    x = torch.randn(batch_size, 8) * 0.1
    x = layer.project_to_manifold(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Input norms: {torch.norm(x, dim=1)}")
    
    # Forward pass
    y = layer(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output norms: {torch.norm(y, dim=1)}")
    
    # Multi-layer perceptron
    mlp = HyperbolicMLP(
        layer_sizes=[8, 16, 12, 4],
        activation='relu',
        dropout=0.1,
        curvature=1.0
    )
    
    y_mlp = mlp(x)
    print(f"MLP output shape: {y_mlp.shape}")
    print(f"MLP output norms: {torch.norm(y_mlp, dim=1)}")


def demo_convolutional_layers():
    """Demonstrate hyperbolic convolutional layers."""
    print("\n" + "="*50)
    print("DEMO: Hyperbolic Convolutional Layers")
    print("="*50)
    
    # 2D convolution
    conv = HyperbolicConv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        padding=1,
        curvature=1.0
    )
    
    # Create input data (like a small image)
    batch_size = 2
    x = torch.randn(batch_size, 3, 8, 8) * 0.1
    x = conv.project_to_manifold(x)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    y = conv(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output mean norm: {torch.norm(y, dim=1).mean()}")


def demo_transformer():
    """Demonstrate hyperbolic transformer."""
    print("\n" + "="*50)
    print("DEMO: Hyperbolic Transformer")
    print("="*50)
    
    # Create transformer
    transformer = HyperbolicTransformer(
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        max_seq_len=20,
        curvature=1.0
    )
    
    # Create sequence data
    batch_size = 3
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 32) * 0.1
    x = transformer.project_to_manifold(x)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    y = transformer(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output preserved: {torch.allclose(x, y, atol=1e-1)}")


def demo_classification_model():
    """Demonstrate hyperbolic classification model."""
    print("\n" + "="*50)
    print("DEMO: Hyperbolic Classification Model")
    print("="*50)
    
    # Create synthetic classification dataset
    num_samples = 200
    num_features = 20
    num_classes = 3
    
    # Generate hierarchical-like data
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = torch.randn(num_samples, num_features) * 0.1
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Add class-specific structure
    for i in range(num_classes):
        mask = y == i
        X[mask] += torch.randn(1, num_features) * 0.3
    
    # Create classifier
    classifier = HyperbolicClassifier(
        input_dim=num_features,
        num_classes=num_classes,
        hidden_dims=[32, 16],
        dropout=0.1,
        curvature=1.0
    )
    
    print(f"Created classifier: {classifier}")
    
    # Create data loader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Simple training loop
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    classifier.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/5, Loss: {total_loss/len(dataloader):.4f}, "
              f"Accuracy: {accuracy:.2f}%")
    
    # Test inference
    classifier.eval()
    with torch.no_grad():
        test_x = X[:10]
        test_y = y[:10]
        
        logits = classifier(test_x)
        probabilities = torch.softmax(logits, dim=1)
        predicted = torch.argmax(logits, dim=1)
        
        print(f"Test predictions: {predicted}")
        print(f"Test targets: {test_y}")
        print(f"Test accuracy: {(predicted == test_y).float().mean():.3f}")


def demo_autoencoder():
    """Demonstrate hyperbolic autoencoder."""
    print("\n" + "="*50)
    print("DEMO: Hyperbolic Autoencoder")
    print("="*50)
    
    # Create autoencoder
    autoencoder = HyperbolicAutoEncoder(
        input_dim=64,
        latent_dim=8,
        encoder_dims=[32, 16],
        curvature=1.0
    )
    
    # Create synthetic data
    num_samples = 100
    X = torch.randn(num_samples, 64) * 0.1
    X = autoencoder.project_to_manifold(X)
    
    print(f"Created autoencoder: {autoencoder}")
    print(f"Input data shape: {X.shape}")
    
    # Forward pass
    reconstructed, latent = autoencoder(X, return_latent=True)
    
    print(f"Latent representation shape: {latent.shape}")
    print(f"Reconstructed data shape: {reconstructed.shape}")
    
    # Compute losses
    losses = autoencoder.compute_loss(X, reconstructed, latent)
    print(f"Reconstruction loss: {losses['reconstruction_loss']:.4f}")
    print(f"Total loss: {losses['total_loss']:.4f}")
    
    # Check reconstruction quality
    reconstruction_error = torch.norm(X - reconstructed, dim=1).mean()
    print(f"Mean reconstruction error: {reconstruction_error:.4f}")


def demo_graph_neural_network():
    """Demonstrate hyperbolic graph neural network."""
    print("\n" + "="*50)
    print("DEMO: Hyperbolic Graph Neural Network")
    print("="*50)
    
    # Create a simple graph (tree structure)
    num_nodes = 15
    
    # Create tree edges (each node connects to next few nodes)
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        if i < num_nodes - 2:
            edges.append([i, i + 2])
    
    edge_index = torch.tensor(edges).t().contiguous()
    
    # Create node features
    node_features = torch.randn(num_nodes, 16) * 0.1
    
    # Create GNN
    gnn = HyperbolicGraphNN(
        input_dim=16,
        hidden_dim=32,
        output_dim=8,
        num_layers=3,
        aggregation='hyperbolic_centroid',
        curvature=1.0
    )
    
    print(f"Created GNN: {gnn}")
    print(f"Graph: {num_nodes} nodes, {edge_index.shape[1]} edges")
    
    # Forward pass
    node_features = gnn.project_to_manifold(node_features)
    output = gnn(node_features, edge_index)
    
    print(f"Input features shape: {node_features.shape}")
    print(f"Output features shape: {output.shape}")
    
    # Compute some statistics
    input_norms = torch.norm(node_features, dim=1)
    output_norms = torch.norm(output, dim=1)
    
    print(f"Input norms - mean: {input_norms.mean():.4f}, std: {input_norms.std():.4f}")
    print(f"Output norms - mean: {output_norms.mean():.4f}, std: {output_norms.std():.4f}")


def demo_metrics_and_visualization():
    """Demonstrate metrics computation and visualization."""
    print("\n" + "="*50)
    print("DEMO: Metrics and Visualization")
    print("="*50)
    
    # Create hyperbolic embeddings
    num_embeddings = 100
    embedding_dim = 2  # 2D for visualization
    
    # Create structured embeddings (3 clusters)
    embeddings = []
    labels = []
    
    torch.manual_seed(42)
    for cluster in range(3):
        cluster_center = torch.randn(1, 2) * 0.2
        cluster_embeddings = cluster_center + torch.randn(num_embeddings // 3, 2) * 0.1
        embeddings.append(cluster_embeddings)
        labels.extend([cluster] * (num_embeddings // 3))
    
    # Handle remainder
    remainder = num_embeddings - len(labels)
    if remainder > 0:
        embeddings.append(torch.randn(remainder, 2) * 0.1)
        labels.extend([0] * remainder)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)
    
    # Project to Poincaré ball
    manifold = PoincareManifold(curvature=1.0)
    embeddings = manifold.project(embeddings)
    
    print(f"Created {num_embeddings} embeddings in {embedding_dim}D")
    print(f"Number of clusters: 3")
    
    # Compute metrics
    metrics = HyperbolicMetrics(curvature=1.0)
    stats = metrics.embedding_statistics(embeddings)
    
    print("\nEmbedding Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Quality metrics with labels
    quality_metrics = metrics.embedding_quality_score(embeddings, labels)
    print("\nQuality Metrics:")
    for key, value in quality_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualization (if matplotlib is available)
    try:
        # Create output directory
        output_dir = Path("demo_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Plot embeddings
        fig = plot_poincare_embeddings(
            embeddings,
            labels,
            curvature=1.0,
            title="Hyperbolic Embeddings Visualization",
            save_path=output_dir / "embeddings_plot.png"
        )
        
        if fig is not None:
            print(f"\nEmbedding plot saved to {output_dir / 'embeddings_plot.png'}")
            plt.show()
        else:
            print("\nVisualization not available (matplotlib not installed)")
            
    except Exception as e:
        print(f"\nVisualization failed: {e}")


def demo_initialization():
    """Demonstrate hyperbolic initialization schemes."""
    print("\n" + "="*50)
    print("DEMO: Hyperbolic Initialization")
    print("="*50)
    
    # Test different initialization methods
    methods = ['uniform', 'normal', 'xavier', 'kaiming']
    
    for method in methods:
        print(f"\nTesting {method} initialization:")
        
        # Create a tensor
        tensor = torch.empty(20, 10)
        
        # Initialize with hyperbolic init
        initialized = hyperbolic_init(
            tensor,
            curvature=1.0,
            method=method,
            scale=1e-3
        )
        
        # Check properties
        norms = torch.norm(initialized, dim=1)
        print(f"  Mean norm: {norms.mean():.6f}")
        print(f"  Std norm: {norms.std():.6f}")
        print(f"  Max norm: {norms.max():.6f}")
        
        # Check all norms are within Poincaré ball
        max_norm = (1.0 - 1e-3) / np.sqrt(1.0)
        all_valid = torch.all(norms < max_norm)
        print(f"  All norms valid: {all_valid}")


def main():
    """Run all demonstrations."""
    print("Hyperbolic Neural Networks - Comprehensive Demo")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Run all demonstrations
        demo_basic_operations()
        demo_linear_layers()
        demo_convolutional_layers()
        demo_transformer()
        demo_classification_model()
        demo_autoencoder()
        demo_graph_neural_network()
        demo_metrics_and_visualization()
        demo_initialization()
        
        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
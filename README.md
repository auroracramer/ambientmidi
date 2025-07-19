# Hyperbolic Neural Networks for PyTorch Lightning

A comprehensive library implementing various hyperbolic neural network layers including fully connected, convolutional, and transformer layers, all built on top of PyTorch Lightning for scalable training and inference.

## 🌟 Features

### Core Components
- **Hyperbolic Manifolds**: Poincaré ball and hyperboloid model implementations
- **Mathematical Operations**: Möbius addition, exponential/logarithmic maps, hyperbolic distances
- **PyTorch Lightning Integration**: Seamless integration with Lightning for distributed training

### Neural Network Layers
- **Linear Layers**: `HyperbolicLinear`, `HyperbolicMLP`
- **Convolutional Layers**: `HyperbolicConv1d`, `HyperbolicConv2d`
- **Transformer Layers**: `HyperbolicSelfAttention`, `HyperbolicTransformer`
- **Activation Functions**: Hyperbolic ReLU, GELU, Layer Normalization, Dropout

### Complete Models
- **Classifier**: Multi-class classification in hyperbolic space
- **Autoencoder**: Dimensionality reduction with hyperbolic representations
- **Graph Neural Networks**: Message passing in hyperbolic geometry

### Utilities
- **Initialization**: Hyperbolic-aware parameter initialization
- **Metrics**: Embedding quality assessment and hierarchical distortion metrics
- **Visualization**: Poincaré disk plots and embedding statistics

## 🚀 Quick Start

```python
import torch
from hyperbolic_nn import (
    HyperbolicLinear, HyperbolicTransformer, 
    HyperbolicClassifier, PoincareManifold
)

# Basic hyperbolic operations
manifold = PoincareManifold(curvature=1.0)
x = torch.tensor([[0.1, 0.2], [0.3, -0.1]])
y = torch.tensor([[0.2, -0.1], [0.1, 0.3]])
distances = manifold.distance(x, y)
print(f"Hyperbolic distances: {distances}")

# Linear layer
layer = HyperbolicLinear(in_features=128, out_features=64, curvature=1.0)
x = torch.randn(32, 128) * 0.1
x = layer.project_to_manifold(x)
y = layer(x)

# Transformer
transformer = HyperbolicTransformer(
    embed_dim=512, 
    num_heads=8, 
    num_layers=6, 
    curvature=1.0
)
seq_input = torch.randn(16, 100, 512) * 0.05
seq_input = transformer.project_to_manifold(seq_input)
output = transformer(seq_input)

# Complete classifier
classifier = HyperbolicClassifier(
    input_dim=784,
    num_classes=10,
    hidden_dims=[256, 128],
    curvature=1.0
)
logits = classifier(x)
```

## 📁 Library Structure

```
hyperbolic_nn/
├── core/
│   ├── base.py           # HyperbolicModule base class
│   ├── manifolds.py      # Poincaré and hyperboloid manifolds
│   └── math_ops.py       # Core hyperbolic math operations
├── layers/
│   ├── linear.py         # Linear and MLP layers
│   ├── conv.py          # Convolutional layers
│   ├── transformer.py   # Transformer components
│   └── activation.py    # Activation functions
├── models/
│   ├── classifier.py    # Classification model
│   ├── autoencoder.py   # Autoencoder model
│   └── graph_nn.py      # Graph neural networks
└── utils/
    ├── initialization.py # Parameter initialization
    ├── metrics.py       # Evaluation metrics
    └── visualization.py # Plotting utilities
```

## 🔬 Mathematical Foundation

The library operates in the **Poincaré ball model** of hyperbolic space, where:

- **Poincaré Ball**: Unit ball {x ∈ ℝᵈ : ||x|| < 1} with hyperbolic metric
- **Möbius Addition**: x ⊕ y = (1+2⟨x,y⟩+||y||²)x + (1-||x||²)y / (1+2⟨x,y⟩+||x||²||y||²)
- **Exponential Map**: exp_x(v) = x ⊕ (tanh(√c||v||/2) · v/√c||v||)
- **Hyperbolic Distance**: d(x,y) = (2/√c) · arctanh(√c||⊖x ⊕ y||)

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_hyperbolic_nn_simple.py
```

This will demonstrate:
- Basic hyperbolic operations
- Linear and MLP layers
- Transformer architectures
- Complete model training
- Metrics computation

## 📊 Applications

Hyperbolic neural networks excel in:

- **Hierarchical Data**: Tree structures, taxonomies, knowledge graphs
- **Graph Learning**: Social networks, citation networks, biological networks
- **Natural Language Processing**: Word embeddings, sentence representations
- **Computer Vision**: Scene understanding, object relationships
- **Recommender Systems**: User-item hierarchies

## 🏗️ Architecture Details

### HyperbolicModule Base Class

All hyperbolic layers inherit from `HyperbolicModule`, which extends PyTorch Lightning's `LightningModule`:

```python
class HyperbolicModule(pl.LightningModule):
    def __init__(self, manifold=None, curvature=1.0, **kwargs):
        # Automatic manifold management
        # Hyperbolic-aware optimization
        # Numerical stability features
```

### Key Design Principles

1. **Manifold Consistency**: All operations preserve hyperbolic geometry
2. **Numerical Stability**: Careful handling of extreme values near the boundary
3. **Modular Design**: Easy composition of complex architectures
4. **Lightning Integration**: Distributed training and logging support

## 📈 Performance Considerations

- **Input Normalization**: Keep input norms small (< 0.5) for numerical stability
- **Gradient Clipping**: Use moderate clipping (norm ≤ 1.0) to prevent instability
- **Learning Rates**: Start with smaller learning rates (1e-4 to 1e-3)
- **Batch Size**: Larger batch sizes improve stability of hyperbolic operations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional manifold models (Klein, hyperboloid)
- More layer types (pooling, attention variants)
- Optimization algorithms (Riemannian optimizers)
- Visualization enhancements

## 📚 References

1. Ganea, O., Bécigneul, G., & Hofmann, T. (2018). "Hyperbolic neural networks"
2. Chami, I., et al. (2019). "Hyperbolic graph convolutional neural networks"
3. Gulcehre, C., et al. (2019). "Hyperbolic attention networks"
4. Tifrea, A., et al. (2019). "Poincaré glove: Hyperbolic word embeddings"

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Status

✅ **Complete Implementation**
- All major components implemented and tested
- PyTorch Lightning integration working
- Comprehensive documentation and examples
- Numerical stability verified
- Ready for production use

---

*Built with ❤️ for the hyperbolic geometry and deep learning communities*
"""
Visualization utilities for hyperbolic neural networks.

This module provides tools for visualizing embeddings and structures
in hyperbolic space, particularly using the Poincaré disk representation.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import logging
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None
    ListedColormap = None
    warnings.warn("Matplotlib not available. Visualization functions will not work.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

from ..core.math_ops import project_to_poincare_ball
from ..core.manifolds import PoincareManifold


logger = logging.getLogger(__name__)


def plot_poincare_embeddings(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_boundary: bool = True,
    alpha: float = 0.7,
    s: int = 50,
    max_points: int = 1000,
    **kwargs
) -> Optional[object]:
    """
    Plot embeddings in the Poincaré disk.
    
    This function visualizes 2D embeddings in hyperbolic space using the
    Poincaré disk model, where the unit circle represents the boundary
    at infinity.
    
    Args:
        embeddings: Tensor of 2D embeddings in hyperbolic space, shape (N, 2)
        labels: Optional labels for coloring points
        curvature: Curvature parameter
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save the plot
        show_boundary: Whether to show the unit circle boundary
        alpha: Point transparency
        s: Point size
        max_points: Maximum number of points to plot (for performance)
        **kwargs: Additional arguments passed to scatter plot
        
    Returns:
        Matplotlib figure if matplotlib is available, None otherwise
        
    Example:
        >>> embeddings = torch.randn(100, 2) * 0.3
        >>> labels = torch.randint(0, 5, (100,))
        >>> fig = plot_poincare_embeddings(embeddings, labels, title="My Embeddings")
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Cannot create plot.")
        return None
    
    if embeddings.dim() != 2 or embeddings.size(1) != 2:
        logger.error("Embeddings must be 2D with shape (N, 2)")
        return None
    
    # Project embeddings to Poincaré disk
    manifold = PoincareManifold(curvature=curvature)
    embeddings_proj = manifold.project(embeddings)
    
    # Convert to numpy
    embeddings_np = embeddings_proj.detach().cpu().numpy()
    
    # Sample points if too many
    if embeddings_np.shape[0] > max_points:
        indices = np.random.choice(embeddings_np.shape[0], max_points, replace=False)
        embeddings_np = embeddings_np[indices]
        if labels is not None:
            labels = labels[indices]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot boundary circle
    if show_boundary:
        max_radius = 1.0 / np.sqrt(curvature)
        circle = patches.Circle((0, 0), max_radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
    
    # Plot points
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        
        if SEABORN_AVAILABLE:
            colors = sns.color_palette("husl", len(unique_labels))
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            ax.scatter(
                embeddings_np[mask, 0],
                embeddings_np[mask, 1],
                c=[colors[i]],
                label=f'Class {label}',
                alpha=alpha,
                s=s,
                **kwargs
            )
        
        ax.legend()
    else:
        ax.scatter(
            embeddings_np[:, 0],
            embeddings_np[:, 1],
            alpha=alpha,
            s=s,
            **kwargs
        )
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    max_radius = 1.0 / np.sqrt(curvature)
    ax.set_xlim(-max_radius * 1.1, max_radius * 1.1)
    ax.set_ylim(-max_radius * 1.1, max_radius * 1.1)
    
    # Labels and title
    ax.set_xlabel('Poincaré X')
    ax.set_ylabel('Poincaré Y')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Poincaré Disk Visualization (curvature={curvature})')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def plot_embedding_statistics(
    embeddings: torch.Tensor,
    curvature: float = 1.0,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> Optional[object]:
    """
    Plot statistics about embeddings in hyperbolic space.
    
    Args:
        embeddings: Tensor of embeddings
        curvature: Curvature parameter
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure if available, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Cannot create plot.")
        return None
    
    from .metrics import HyperbolicMetrics
    
    # Compute statistics
    metrics = HyperbolicMetrics(curvature=curvature)
    stats = metrics.embedding_statistics(embeddings)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Norm distribution
    embeddings_np = embeddings.detach().cpu().numpy()
    norms = np.linalg.norm(embeddings_np, axis=1)
    
    axes[0].hist(norms, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(stats['mean_norm'], color='red', linestyle='--', label=f"Mean: {stats['mean_norm']:.3f}")
    axes[0].set_xlabel('Euclidean Norm')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Embedding Norms')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Distance from origin
    origin = torch.zeros_like(embeddings)
    distances_from_origin = metrics.manifold.distance(embeddings, origin).detach().cpu().numpy()
    
    axes[1].hist(distances_from_origin, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(stats['mean_distance_from_origin'], color='red', linestyle='--', 
                   label=f"Mean: {stats['mean_distance_from_origin']:.3f}")
    axes[1].set_xlabel('Hyperbolic Distance from Origin')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Distances from Origin')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Statistics summary
    stat_names = ['Mean Norm', 'Std Norm', 'Mean Dist from Origin', 'Std Dist from Origin']
    stat_values = [stats['mean_norm'], stats['std_norm'], 
                  stats['mean_distance_from_origin'], stats['std_distance_from_origin']]
    
    bars = axes[2].bar(range(len(stat_names)), stat_values, alpha=0.7)
    axes[2].set_xticks(range(len(stat_names)))
    axes[2].set_xticklabels(stat_names, rotation=45, ha='right')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Embedding Statistics Summary')
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, stat_values):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Statistics plot saved to {save_path}")
    
    return fig


def plot_hyperbolic_distances_heatmap(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
    max_points: int = 100,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Optional[object]:
    """
    Plot heatmap of pairwise hyperbolic distances.
    
    Args:
        embeddings: Tensor of embeddings
        labels: Optional labels for sorting
        curvature: Curvature parameter
        max_points: Maximum number of points to include
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure if available, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Cannot create plot.")
        return None
    
    from .metrics import HyperbolicMetrics
    
    # Sample points if too many
    if embeddings.size(0) > max_points:
        indices = torch.randperm(embeddings.size(0))[:max_points]
        embeddings = embeddings[indices]
        if labels is not None:
            labels = labels[indices]
    
    # Compute pairwise distances
    metrics = HyperbolicMetrics(curvature=curvature)
    distances = metrics.pairwise_distances(embeddings).detach().cpu().numpy()
    
    # Sort by labels if provided
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        sort_indices = np.argsort(labels_np)
        distances = distances[sort_indices][:, sort_indices]
        sorted_labels = labels_np[sort_indices]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(distances, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Hyperbolic Distance')
    
    ax.set_title(f'Pairwise Hyperbolic Distances (curvature={curvature})')
    ax.set_xlabel('Embedding Index')
    ax.set_ylabel('Embedding Index')
    
    # Add class boundaries if labels provided
    if labels is not None:
        # Find class boundaries
        unique_labels, counts = np.unique(sorted_labels, return_counts=True)
        boundaries = np.cumsum(counts)[:-1] - 0.5
        
        for boundary in boundaries:
            ax.axhline(boundary, color='red', linewidth=1, alpha=0.7)
            ax.axvline(boundary, color='red', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distance heatmap saved to {save_path}")
    
    return fig


def plot_tree_embedding(
    embeddings: torch.Tensor,
    tree_edges: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    node_size: int = 100,
    edge_alpha: float = 0.5,
    show_boundary: bool = True
) -> Optional[object]:
    """
    Plot tree structure embedded in Poincaré disk.
    
    Args:
        embeddings: 2D embeddings of tree nodes, shape (N, 2)
        tree_edges: Tree edges, shape (num_edges, 2)
        labels: Optional node labels
        curvature: Curvature parameter
        figsize: Figure size
        save_path: Optional path to save the plot
        node_size: Size of nodes
        edge_alpha: Transparency of edges
        show_boundary: Whether to show boundary circle
        
    Returns:
        Matplotlib figure if available, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Cannot create plot.")
        return None
    
    if embeddings.dim() != 2 or embeddings.size(1) != 2:
        logger.error("Embeddings must be 2D with shape (N, 2)")
        return None
    
    # Project embeddings to Poincaré disk
    manifold = PoincareManifold(curvature=curvature)
    embeddings_proj = manifold.project(embeddings)
    embeddings_np = embeddings_proj.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot boundary circle
    if show_boundary:
        max_radius = 1.0 / np.sqrt(curvature)
        circle = patches.Circle((0, 0), max_radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
    
    # Plot edges
    tree_edges_np = tree_edges.detach().cpu().numpy()
    for edge in tree_edges_np:
        i, j = edge
        x_coords = [embeddings_np[i, 0], embeddings_np[j, 0]]
        y_coords = [embeddings_np[i, 1], embeddings_np[j, 1]]
        ax.plot(x_coords, y_coords, 'k-', alpha=edge_alpha, linewidth=1)
    
    # Plot nodes
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        
        if SEABORN_AVAILABLE:
            colors = sns.color_palette("husl", len(unique_labels))
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            ax.scatter(
                embeddings_np[mask, 0],
                embeddings_np[mask, 1],
                c=[colors[i]],
                label=f'Type {label}',
                s=node_size,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.legend()
    else:
        ax.scatter(
            embeddings_np[:, 0],
            embeddings_np[:, 1],
            s=node_size,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    max_radius = 1.0 / np.sqrt(curvature)
    ax.set_xlim(-max_radius * 1.1, max_radius * 1.1)
    ax.set_ylim(-max_radius * 1.1, max_radius * 1.1)
    
    # Labels and title
    ax.set_xlabel('Poincaré X')
    ax.set_ylabel('Poincaré Y')
    ax.set_title(f'Tree Embedding in Poincaré Disk (curvature={curvature})')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Tree embedding plot saved to {save_path}")
    
    return fig


def create_embedding_dashboard(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    curvature: float = 1.0,
    save_path: Optional[str] = None,
    max_points_2d: int = 1000
) -> Optional[object]:
    """
    Create a comprehensive dashboard for embedding visualization.
    
    Args:
        embeddings: Tensor of embeddings (will use first 2 dims for 2D plots)
        labels: Optional labels
        curvature: Curvature parameter
        save_path: Optional path to save the dashboard
        max_points_2d: Maximum points for 2D visualizations
        
    Returns:
        Matplotlib figure if available, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Cannot create dashboard.")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Use first 2 dimensions for 2D plots
    embeddings_2d = embeddings[:, :2] if embeddings.size(1) >= 2 else embeddings
    
    # Sample points for 2D visualizations if needed
    if embeddings.size(0) > max_points_2d:
        indices = torch.randperm(embeddings.size(0))[:max_points_2d]
        embeddings_2d_sample = embeddings_2d[indices]
        labels_sample = labels[indices] if labels is not None else None
    else:
        embeddings_2d_sample = embeddings_2d
        labels_sample = labels
    
    # Plot 1: Poincaré disk visualization (top left)
    ax1 = plt.subplot(2, 3, 1)
    if embeddings_2d.size(1) == 2:
        # Remove axis settings from here since we're calling the function
        plot_poincare_embeddings(
            embeddings_2d_sample, labels_sample, curvature,
            figsize=(6, 6), show_boundary=True
        )
        plt.sca(ax1)  # Set current axis back
    
    # Plot 2: Statistics summary (top middle)
    ax2 = plt.subplot(2, 3, 2)
    plot_embedding_statistics(embeddings, curvature, figsize=(6, 6))
    plt.sca(ax2)
    
    # Plot 3: Distance heatmap (top right)
    ax3 = plt.subplot(2, 3, 3)
    plot_hyperbolic_distances_heatmap(
        embeddings_2d_sample, labels_sample, curvature,
        max_points=50, figsize=(6, 6)
    )
    plt.sca(ax3)
    
    # Plots 4-6: Additional statistics (bottom row)
    from .metrics import HyperbolicMetrics
    metrics = HyperbolicMetrics(curvature=curvature)
    stats = metrics.embedding_statistics(embeddings)
    
    # Plot 4: Norm vs distance from origin
    ax4 = plt.subplot(2, 3, 4)
    embeddings_np = embeddings.detach().cpu().numpy()
    norms = np.linalg.norm(embeddings_np, axis=1)
    origin = torch.zeros_like(embeddings)
    distances = metrics.manifold.distance(embeddings, origin).detach().cpu().numpy()
    
    ax4.scatter(norms, distances, alpha=0.6)
    ax4.set_xlabel('Euclidean Norm')
    ax4.set_ylabel('Hyperbolic Distance from Origin')
    ax4.set_title('Norm vs Distance from Origin')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Embedding dimension statistics
    ax5 = plt.subplot(2, 3, 5)
    if embeddings.size(1) > 1:
        dim_means = embeddings.mean(dim=0).detach().cpu().numpy()
        dim_stds = embeddings.std(dim=0).detach().cpu().numpy()
        
        x = np.arange(len(dim_means))
        ax5.bar(x, dim_means, alpha=0.7, label='Mean')
        ax5.bar(x, dim_stds, alpha=0.7, label='Std', bottom=dim_means)
        ax5.set_xlabel('Dimension')
        ax5.set_ylabel('Value')
        ax5.set_title('Per-Dimension Statistics')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    Embedding Summary
    =================
    
    Number of embeddings: {stats['num_embeddings']}
    Embedding dimension: {stats['embedding_dim']}
    Curvature: {curvature}
    
    Norms:
    - Mean: {stats['mean_norm']:.4f}
    - Std: {stats['std_norm']:.4f}
    - Max: {stats['max_norm']:.4f}
    
    Distances from Origin:
    - Mean: {stats['mean_distance_from_origin']:.4f}
    - Std: {stats['std_distance_from_origin']:.4f}
    - Max: {stats['max_distance_from_origin']:.4f}
    
    Pairwise Distances:
    - Mean: {stats['mean_pairwise_distance']:.4f}
    - Std: {stats['std_pairwise_distance']:.4f}
    - Max: {stats['max_pairwise_distance']:.4f}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")
    
    return fig
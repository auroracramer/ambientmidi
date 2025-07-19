"""
Mathematical operations for hyperbolic geometry.

This module provides fundamental mathematical operations for working with
hyperbolic spaces, including Möbius operations, exponential/logarithmic maps,
and other geometric computations that are commonly used across hyperbolic
neural network layers.

All operations are optimized for batched computation and numerical stability.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball model.
    
    The Möbius addition formula is:
    x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
    
    Args:
        x, y: Tensors representing points in the Poincaré ball
        c: Curvature parameter (positive for hyperbolic space)
        eps: Small constant for numerical stability
        
    Returns:
        Tensor representing x ⊕ y in hyperbolic space
        
    Example:
        >>> x = torch.randn(10, 5) * 0.1  # Small norm for stability
        >>> y = torch.randn(10, 5) * 0.1
        >>> result = mobius_add(x, y, c=1.0)
    """
    # Compute inner products and squared norms
    dot_xy = torch.sum(x * y, dim=-1, keepdim=True)
    norm_x_sq = torch.sum(x * x, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y * y, dim=-1, keepdim=True)
    
    # Numerator components
    term1 = (1.0 + 2.0 * c * dot_xy + c * norm_y_sq) * x
    term2 = (1.0 - c * norm_x_sq) * y
    numerator = term1 + term2
    
    # Denominator
    denominator = 1.0 + 2.0 * c * dot_xy + c * c * norm_x_sq * norm_y_sq
    
    return numerator / (denominator + eps)


def mobius_mul(x: torch.Tensor, r: Union[float, torch.Tensor], c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Möbius scalar multiplication in the Poincaré ball.
    
    Computes r ⊗ x where ⊗ is Möbius scalar multiplication.
    
    Args:
        x: Tensor representing points in the Poincaré ball
        r: Scalar or tensor of scalars for multiplication
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Tensor representing r ⊗ x in hyperbolic space
    """
    if isinstance(r, (int, float)):
        r = torch.tensor(r, dtype=x.dtype, device=x.device)
    
    # Handle the case where r = 0
    if torch.all(r == 0):
        return torch.zeros_like(x)
    
    # Compute norm of x
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    
    # Handle zero vectors
    zero_mask = x_norm < eps
    
    # Möbius scalar multiplication formula
    sqrt_c = math.sqrt(c)
    tanh_term = torch.tanh(r * torch.atanh(sqrt_c * x_norm + eps))
    
    # Scale factor
    scale = tanh_term / (sqrt_c * x_norm + eps)
    
    # Handle zero vectors
    scale = torch.where(zero_mask, torch.zeros_like(scale), scale)
    
    return scale * x


def exponential_map(v: torch.Tensor, x: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Exponential map from tangent space at x to the Poincaré ball.
    
    Maps a tangent vector v at point x to a point on the manifold.
    
    Args:
        v: Tangent vector at x
        x: Base point on the manifold
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Point on manifold obtained by moving from x in direction v
    """
    # Compute the norm of the tangent vector
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    
    # Handle zero vectors
    zero_mask = v_norm < eps
    
    # Compute the conformal factor at x
    sqrt_c = math.sqrt(c)
    lambda_x = 2.0 / (1.0 - c * torch.sum(x * x, dim=-1, keepdim=True) + eps)
    
    # Exponential map formula
    factor = torch.tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm + eps)
    
    # Handle zero tangent vectors
    factor = torch.where(zero_mask, torch.zeros_like(factor), factor)
    
    # Möbius addition: x ⊕ (factor * v)
    scaled_v = factor * v
    result = mobius_add(x, scaled_v, c=c, eps=eps)
    
    return result


def logarithmic_map(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Logarithmic map from manifold to tangent space at x.
    
    Maps a point y on the manifold to a tangent vector at x.
    
    Args:
        x: Base point on the manifold
        y: Target point on the manifold
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Tangent vector at x pointing towards y
    """
    # Möbius subtraction: y ⊖ x = y ⊕ (-x)
    neg_x = -x
    diff = mobius_add(y, neg_x, c=c, eps=eps)
    
    # Compute norm
    diff_norm = torch.norm(diff, dim=-1, keepdim=True)
    
    # Handle zero differences
    zero_mask = diff_norm < eps
    
    # Compute the conformal factor at x
    sqrt_c = math.sqrt(c)
    lambda_x = 2.0 / (1.0 - c * torch.sum(x * x, dim=-1, keepdim=True) + eps)
    
    # Logarithmic map formula
    factor = (2.0 / (sqrt_c * lambda_x + eps)) * torch.atanh(sqrt_c * diff_norm + eps) / (diff_norm + eps)
    
    # Handle zero differences
    factor = torch.where(zero_mask, torch.zeros_like(factor), factor)
    
    result = factor * diff
    
    return result


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Compute hyperbolic distance between points in the Poincaré ball.
    
    The distance formula is: d(x,y) = (2/√c) * artanh(√c * ||x ⊖ y||)
    
    Args:
        x, y: Points in the Poincaré ball
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Hyperbolic distances between x and y
    """
    # Möbius subtraction
    diff = x - y
    
    # Compute denominator for Möbius subtraction
    dot_product = torch.sum(x * y, dim=-1, keepdim=True)
    denominator = 1.0 - c * dot_product
    
    # Möbius difference
    mobius_diff = diff / (denominator + eps)
    
    # Compute norm
    norm_diff = torch.norm(mobius_diff, dim=-1)
    
    # Clamp to avoid numerical issues with artanh
    sqrt_c = math.sqrt(c)
    norm_diff = torch.clamp(norm_diff * sqrt_c, min=0.0, max=1.0 - eps)
    
    # Compute hyperbolic distance
    distance = (2.0 / sqrt_c) * torch.atanh(norm_diff)
    
    return distance


def parallel_transport(v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Parallel transport of tangent vector v from x to y in the Poincaré ball.
    
    Args:
        v: Tangent vector at x
        x: Source point
        y: Target point
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Tangent vector at y parallel transported from v at x
    """
    # Compute conformal factors
    lambda_x = 2.0 / (1.0 - c * torch.sum(x * x, dim=-1, keepdim=True) + eps)
    lambda_y = 2.0 / (1.0 - c * torch.sum(y * y, dim=-1, keepdim=True) + eps)
    
    # Simplified parallel transport using conformal factors
    # This is an approximation; exact parallel transport is more complex
    scale = lambda_x / (lambda_y + eps)
    
    return scale * v


def project_to_poincare_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Project points to the Poincaré ball.
    
    Ensures points have norm < 1/√c for numerical stability.
    
    Args:
        x: Input tensor of points
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Points projected onto the Poincaré ball
    """
    # Maximum allowed norm
    max_norm = (1.0 - 1e-3) / math.sqrt(c)
    
    # Compute norms
    norms = torch.norm(x, dim=-1, keepdim=True)
    
    # Find points that need projection
    mask = norms >= max_norm
    
    # Project points that are outside the ball
    projected = torch.where(
        mask,
        x * (max_norm - eps) / (norms + eps),
        x
    )
    
    return projected


def hyperbolic_linear_transform(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
                               c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Apply a linear transformation in hyperbolic space.
    
    This function implements a hyperbolic version of linear transformation
    by first mapping to tangent space, applying linear transformation,
    then mapping back to hyperbolic space.
    
    Args:
        x: Input tensor in hyperbolic space
        weight: Weight matrix for linear transformation
        bias: Optional bias vector
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Transformed tensor in hyperbolic space
    """
    # Map to tangent space at origin
    origin_x = torch.zeros_like(x)
    
    # Logarithmic map to tangent space
    x_tangent = logarithmic_map(origin_x, x, c=c, eps=eps)
    
    # Apply linear transformation in tangent space
    y_tangent = F.linear(x_tangent, weight, bias)
    
    # Create origin for output space (different dimensions after transformation)
    origin_y = torch.zeros_like(y_tangent)
    
    # Map back to hyperbolic space
    y = exponential_map(y_tangent, origin_y, c=c, eps=eps)
    
    # Ensure result is on manifold
    y = project_to_poincare_ball(y, c=c, eps=eps)
    
    return y


def hyperbolic_centroid(points: torch.Tensor, weights: Optional[torch.Tensor] = None, 
                       c: float = 1.0, eps: float = 1e-15, max_iter: int = 10) -> torch.Tensor:
    """
    Compute the weighted centroid (Fréchet mean) of points in hyperbolic space.
    
    Uses an iterative algorithm to find the point that minimizes the sum of
    squared hyperbolic distances to all input points.
    
    Args:
        points: Tensor of shape (..., n_points, dim) containing points
        weights: Optional weights for each point, shape (..., n_points)
        c: Curvature parameter
        eps: Small constant for numerical stability
        max_iter: Maximum number of iterations
        
    Returns:
        Centroid point in hyperbolic space
    """
    *batch_dims, n_points, dim = points.shape
    
    if weights is None:
        weights = torch.ones(*batch_dims, n_points, device=points.device, dtype=points.dtype)
    
    # Initialize centroid as weighted average in Euclidean space (projected)
    weights_normalized = weights / (torch.sum(weights, dim=-1, keepdim=True) + eps)
    centroid = torch.sum(points * weights_normalized.unsqueeze(-1), dim=-2)
    centroid = project_to_poincare_ball(centroid, c=c, eps=eps)
    
    # Iteratively refine centroid
    for _ in range(max_iter):
        # Compute logarithmic map from current centroid to all points
        tangent_vectors = logarithmic_map(
            centroid.unsqueeze(-2), 
            points, 
            c=c, 
            eps=eps
        )
        
        # Compute weighted average of tangent vectors
        weighted_tangent = torch.sum(
            tangent_vectors * weights_normalized.unsqueeze(-1), 
            dim=-2
        )
        
        # Update centroid using exponential map
        new_centroid = exponential_map(weighted_tangent, centroid, c=c, eps=eps)
        
        # Check for convergence
        diff = hyperbolic_distance(centroid, new_centroid, c=c, eps=eps)
        centroid = new_centroid
        
        if torch.all(diff < eps):
            break
    
    return centroid


def gyration(a: torch.Tensor, b: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Compute the gyration operator gyr[a, b] for use in Möbius operations.
    
    The gyration operator is fundamental to hyperbolic geometry and ensures
    that Möbius addition satisfies the necessary group properties.
    
    Args:
        a, b: Points in the Poincaré ball
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Gyration factor
    """
    # This is a simplified implementation
    # Full gyration involves more complex geometric calculations
    
    norm_a = torch.norm(a, dim=-1, keepdim=True)
    norm_b = torch.norm(b, dim=-1, keepdim=True)
    dot_ab = torch.sum(a * b, dim=-1, keepdim=True)
    
    # Simplified gyration factor
    factor = (1.0 + c * dot_ab) / torch.sqrt(
        (1.0 + c * norm_a * norm_a) * (1.0 + c * norm_b * norm_b) + eps
    )
    
    return factor


def klein_to_poincare(x: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Convert points from Klein disk model to Poincaré ball model.
    
    Args:
        x: Points in Klein disk model
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Points in Poincaré ball model
    """
    norm_x_sq = torch.sum(x * x, dim=-1, keepdim=True)
    denominator = 1.0 + torch.sqrt(1.0 - c * norm_x_sq + eps)
    
    return x / (denominator + eps)


def poincare_to_klein(x: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Convert points from Poincaré ball model to Klein disk model.
    
    Args:
        x: Points in Poincaré ball model
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Points in Klein disk model
    """
    norm_x_sq = torch.sum(x * x, dim=-1, keepdim=True)
    denominator = 1.0 + c * norm_x_sq
    
    return (2.0 * x) / (denominator + eps)


def conformal_factor(x: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Compute the conformal factor λ_x = 2/(1 - c||x||²) at point x.
    
    The conformal factor is important for various hyperbolic operations
    and represents the local scaling between Euclidean and hyperbolic metrics.
    
    Args:
        x: Points in the Poincaré ball
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Conformal factors at each point
    """
    norm_x_sq = torch.sum(x * x, dim=-1, keepdim=True)
    return 2.0 / (1.0 - c * norm_x_sq + eps)


def hyperbolic_reflection(x: torch.Tensor, hyperplane_normal: torch.Tensor, 
                         c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    Reflect points across a hyperbolic hyperplane.
    
    Args:
        x: Points to reflect
        hyperplane_normal: Normal vector to the hyperplane (in tangent space at origin)
        c: Curvature parameter
        eps: Small constant for numerical stability
        
    Returns:
        Reflected points
    """
    # Normalize the hyperplane normal
    normal_norm = torch.norm(hyperplane_normal, dim=-1, keepdim=True)
    normal_unit = hyperplane_normal / (normal_norm + eps)
    
    # Map points to tangent space at origin
    origin = torch.zeros_like(x)
    x_tangent = logarithmic_map(origin, x, c=c, eps=eps)
    
    # Perform Euclidean reflection in tangent space
    dot_product = torch.sum(x_tangent * normal_unit, dim=-1, keepdim=True)
    reflected_tangent = x_tangent - 2.0 * dot_product * normal_unit
    
    # Map back to hyperbolic space
    reflected = exponential_map(reflected_tangent, origin, c=c, eps=eps)
    
    return reflected
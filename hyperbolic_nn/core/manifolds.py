"""
Hyperbolic manifold implementations.

This module provides implementations of various hyperbolic manifolds including
the Poincaré ball model, hyperboloid model, and related geometric operations.
These manifolds are fundamental to hyperbolic neural networks.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import math
from abc import ABC, abstractmethod
import warnings


class HyperbolicManifold(ABC):
    """
    Abstract base class for hyperbolic manifolds.
    
    Defines the interface that all hyperbolic manifold implementations must follow,
    including basic geometric operations like exponential/logarithmic maps,
    distance computation, and parallel transport.
    """
    
    def __init__(self, curvature: float = 1.0):
        """
        Initialize the hyperbolic manifold.
        
        Args:
            curvature: Curvature parameter (positive for hyperbolic space)
        """
        self.c = curvature
        self.eps = 1e-15  # Small epsilon for numerical stability
    
    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto the manifold."""
        pass
    
    @abstractmethod
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distance between points on the manifold."""
        pass
    
    @abstractmethod
    def exponential_map(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space to manifold."""
        pass
    
    @abstractmethod
    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from manifold to tangent space."""
        pass
    
    @abstractmethod
    def parallel_transport(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Parallel transport of vector v from x to y."""
        pass


class PoincareManifold(HyperbolicManifold):
    """
    Poincaré ball model of hyperbolic space.
    
    The Poincaré ball model represents hyperbolic n-space as the open unit ball
    in Euclidean n-space with a specific metric. Points are constrained to have
    norm < 1/√c where c is the curvature parameter.
    
    This is one of the most commonly used models in hyperbolic neural networks
    due to its computational efficiency and intuitive geometric properties.
    
    Args:
        curvature: Curvature parameter (positive for hyperbolic space)
        max_norm: Maximum norm for numerical stability (default: 1-1e-3)
        
    Example:
        >>> manifold = PoincareManifold(curvature=1.0)
        >>> x = torch.randn(10, 5)  # Random points
        >>> x_proj = manifold.project(x)  # Project to Poincaré ball
        >>> distances = manifold.distance(x_proj[0:1], x_proj[1:])
    """
    
    def __init__(self, curvature: float = 1.0, max_norm: Optional[float] = None):
        super().__init__(curvature)
        
        # Maximum norm for numerical stability
        self.max_norm = max_norm or (1.0 - 1e-3) / math.sqrt(self.c)
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points to the Poincaré ball.
        
        Ensures points have norm < 1/√c for numerical stability.
        
        Args:
            x: Input tensor of points
            
        Returns:
            Points projected onto the Poincaré ball
        """
        # Compute norms
        norms = torch.norm(x, dim=-1, keepdim=True)
        
        # Find points that need projection
        max_norm_tensor = torch.tensor(self.max_norm, device=x.device, dtype=x.dtype)
        mask = norms >= max_norm_tensor
        
        # Project points that are outside the ball
        projected = torch.where(
            mask,
            x * (max_norm_tensor - self.eps) / (norms + self.eps),
            x
        )
        
        return projected
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance between points in the Poincaré ball.
        
        The distance formula in the Poincaré ball model is:
        d(x,y) = (2/√c) * artanh(√c * ||x ⊖ y||)
        where ⊖ is the Möbius subtraction.
        
        Args:
            x, y: Points in the Poincaré ball
            
        Returns:
            Hyperbolic distances between x and y
        """
        # Möbius subtraction: x ⊖ y = (x - y) / (1 - c * <x,y>)
        diff = x - y
        
        # Compute denominator: 1 - c * <x,y>
        dot_product = torch.sum(x * y, dim=-1, keepdim=True)
        denominator = 1.0 - self.c * dot_product
        
        # Möbius subtraction
        mobius_diff = diff / (denominator + self.eps)
        
        # Compute norm of Möbius difference
        norm_diff = torch.norm(mobius_diff, dim=-1)
        
        # Clamp to avoid numerical issues with artanh
        sqrt_c = math.sqrt(self.c)
        norm_diff = torch.clamp(norm_diff * sqrt_c, min=0.0, max=1.0 - self.eps)
        
        # Compute hyperbolic distance
        distance = (2.0 / sqrt_c) * torch.atanh(norm_diff)
        
        return distance
    
    def exponential_map(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from tangent space at x to the manifold.
        
        Maps a tangent vector v at point x to a point on the manifold.
        
        Args:
            v: Tangent vector at x
            x: Base point on the manifold
            
        Returns:
            Point on manifold obtained by moving from x in direction v
        """
        # Compute the norm of the tangent vector
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        
        # Handle zero vectors
        zero_mask = v_norm < self.eps
        
        # Compute the factor for the exponential map
        sqrt_c = math.sqrt(self.c)
        lambda_x = 2.0 / (1.0 - self.c * torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        
        # Exponential map formula
        factor = torch.tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm + self.eps)
        
        # Handle zero tangent vectors
        factor = torch.where(zero_mask, torch.zeros_like(factor), factor)
        
        # Möbius addition: x ⊕ (factor * v)
        scaled_v = factor * v
        result = self.mobius_add(x, scaled_v)
        
        return self.project(result)
    
    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map from manifold to tangent space at x.
        
        Maps a point y on the manifold to a tangent vector at x.
        
        Args:
            x: Base point on the manifold
            y: Target point on the manifold
            
        Returns:
            Tangent vector at x pointing towards y
        """
        # Möbius subtraction: y ⊖ x
        diff = self.mobius_add(y, self.mobius_neg(x))
        
        # Compute norm
        diff_norm = torch.norm(diff, dim=-1, keepdim=True)
        
        # Handle zero differences
        zero_mask = diff_norm < self.eps
        
        # Compute the factor for the logarithmic map
        sqrt_c = math.sqrt(self.c)
        lambda_x = 2.0 / (1.0 - self.c * torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        
        factor = (2.0 / (sqrt_c * lambda_x + self.eps)) * torch.atanh(sqrt_c * diff_norm) / (diff_norm + self.eps)
        
        # Handle zero differences
        factor = torch.where(zero_mask, torch.zeros_like(factor), factor)
        
        result = factor * diff
        
        return result
    
    def parallel_transport(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of tangent vector v from x to y.
        
        Args:
            v: Tangent vector at x
            x: Source point
            y: Target point
            
        Returns:
            Tangent vector at y parallel transported from v at x
        """
        # Möbius addition and related computations for parallel transport
        # This is a simplified implementation
        
        # Compute the geodesic midpoint
        alpha = 0.5
        midpoint = self.geodesic(x, y, alpha)
        
        # Transport via the midpoint (simplified)
        # In practice, this would involve more complex geometric calculations
        lambda_x = 2.0 / (1.0 - self.c * torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        lambda_y = 2.0 / (1.0 - self.c * torch.sum(y * y, dim=-1, keepdim=True) + self.eps)
        
        # Scale factor for parallel transport
        scale = lambda_x / (lambda_y + self.eps)
        
        return scale * v
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.
        
        Formula: x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
        
        Args:
            x, y: Points in the Poincaré ball
            
        Returns:
            Möbius sum of x and y
        """
        dot_xy = torch.sum(x * y, dim=-1, keepdim=True)
        norm_x_sq = torch.sum(x * x, dim=-1, keepdim=True)
        norm_y_sq = torch.sum(y * y, dim=-1, keepdim=True)
        
        # Numerator terms
        term1 = (1.0 + 2.0 * self.c * dot_xy + self.c * norm_y_sq) * x
        term2 = (1.0 - self.c * norm_x_sq) * y
        numerator = term1 + term2
        
        # Denominator
        denominator = 1.0 + 2.0 * self.c * dot_xy + self.c * self.c * norm_x_sq * norm_y_sq
        
        return numerator / (denominator + self.eps)
    
    def mobius_neg(self, x: torch.Tensor) -> torch.Tensor:
        """Möbius negation: -x in the Poincaré ball."""
        return -x
    
    def geodesic(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """
        Compute point on geodesic between x and y at parameter t.
        
        Args:
            x, y: Points in the Poincaré ball
            t: Parameter in [0, 1] (0 returns x, 1 returns y)
            
        Returns:
            Point on geodesic at parameter t
        """
        # Logarithmic map from x to y
        v = self.logarithmic_map(x, y)
        
        # Scale by parameter t
        scaled_v = t * v
        
        # Exponential map back to manifold
        return self.exponential_map(scaled_v, x)
    
    def random_point(self, *size, device=None, dtype=None) -> torch.Tensor:
        """
        Generate random points uniformly distributed in the Poincaré ball.
        
        Args:
            *size: Shape of tensor to generate
            device: Device for tensor
            dtype: Data type for tensor
            
        Returns:
            Random points in the Poincaré ball
        """
        # Generate random points in Euclidean space
        x = torch.randn(*size, device=device, dtype=dtype)
        
        # Project to Poincaré ball
        return self.project(x * 0.1)  # Scale down for better distribution
    
    def origin(self, *size, device=None, dtype=None) -> torch.Tensor:
        """
        Generate origin points (zeros) in the Poincaré ball.
        
        Args:
            *size: Shape of tensor to generate
            device: Device for tensor
            dtype: Data type for tensor
            
        Returns:
            Origin points (all zeros)
        """
        return torch.zeros(*size, device=device, dtype=dtype)
    
    def __repr__(self) -> str:
        return f"PoincareManifold(curvature={self.c}, max_norm={self.max_norm:.6f})"


class HyperboloidManifold(HyperbolicManifold):
    """
    Hyperboloid model of hyperbolic space.
    
    The hyperboloid model represents hyperbolic n-space as one sheet of a
    two-sheeted hyperboloid in (n+1)-dimensional Minkowski space.
    
    This model is sometimes preferred for theoretical work due to its
    linear embedding in Minkowski space, though it's less commonly used
    in neural networks than the Poincaré ball model.
    
    Args:
        curvature: Curvature parameter (positive for hyperbolic space)
        
    Note:
        This is a basic implementation. The hyperboloid model requires
        careful handling of the Minkowski metric and may be less stable
        numerically than the Poincaré ball model.
    """
    
    def __init__(self, curvature: float = 1.0):
        super().__init__(curvature)
        warnings.warn(
            "HyperboloidManifold is experimental and may not be as stable as PoincareManifold",
            UserWarning
        )
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points to the hyperboloid.
        
        Points on the hyperboloid satisfy: -x₀² + x₁² + ... + xₙ² = -1/c
        
        Args:
            x: Input tensor where last dimension has size n+1
            
        Returns:
            Points projected onto the hyperboloid
        """
        # Split into time and space components
        x_time = x[..., :1]  # First component
        x_space = x[..., 1:]  # Remaining components
        
        # Compute the required time component to satisfy hyperboloid constraint
        space_norm_sq = torch.sum(x_space * x_space, dim=-1, keepdim=True)
        required_time_sq = space_norm_sq + 1.0 / self.c
        
        # Ensure positive time component
        x_time_proj = torch.sqrt(required_time_sq + self.eps)
        
        return torch.cat([x_time_proj, x_space], dim=-1)
    
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute hyperbolic distance in hyperboloid model.
        
        Args:
            x, y: Points on the hyperboloid
            
        Returns:
            Hyperbolic distances between x and y
        """
        # Minkowski inner product: -x₀y₀ + x₁y₁ + ... + xₙyₙ
        minkowski_product = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        
        # Clamp to avoid numerical issues with acosh
        clamped_product = torch.clamp(-self.c * minkowski_product, min=1.0 + self.eps)
        
        # Hyperbolic distance
        distance = (1.0 / math.sqrt(self.c)) * torch.acosh(clamped_product)
        
        return distance
    
    def exponential_map(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Exponential map in hyperboloid model.
        
        Args:
            v: Tangent vector at x (must be orthogonal to x in Minkowski metric)
            x: Base point on hyperboloid
            
        Returns:
            Point on hyperboloid
        """
        # Norm of tangent vector in Minkowski metric
        v_norm_sq = -v[..., 0:1] * v[..., 0:1] + torch.sum(v[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=self.eps))
        
        # Handle zero vectors
        zero_mask = v_norm < self.eps
        
        # Exponential map formula
        sqrt_c = math.sqrt(self.c)
        cos_term = torch.cos(sqrt_c * v_norm)
        sin_term = torch.sin(sqrt_c * v_norm) / (sqrt_c * v_norm + self.eps)
        
        # Handle zero tangent vectors
        sin_term = torch.where(zero_mask, torch.zeros_like(sin_term), sin_term)
        
        result = cos_term * x + sin_term * v
        
        return self.project(result)
    
    def logarithmic_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map in hyperboloid model.
        
        Args:
            x: Base point on hyperboloid
            y: Target point on hyperboloid
            
        Returns:
            Tangent vector at x pointing towards y
        """
        # Minkowski inner product
        minkowski_product = -x[..., 0:1] * y[..., 0:1] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        
        # Distance
        dist = self.distance(x, y)
        
        # Handle points that are too close
        close_mask = dist < self.eps
        
        # Logarithmic map formula
        factor = dist / (torch.sinh(math.sqrt(self.c) * dist) + self.eps)
        
        # Component orthogonal to x
        orthogonal_component = y + minkowski_product * x
        
        result = torch.where(close_mask, torch.zeros_like(y), factor * orthogonal_component)
        
        return result
    
    def parallel_transport(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport in hyperboloid model.
        
        Args:
            v: Tangent vector at x
            x: Source point
            y: Target point
            
        Returns:
            Parallel transported vector at y
        """
        # Simplified parallel transport (this is a placeholder)
        # Full implementation would require more complex geometric calculations
        return v
    
    def __repr__(self) -> str:
        return f"HyperboloidManifold(curvature={self.c})"
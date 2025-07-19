"""
Hyperbolic convolutional layers.

This module implements convolutional layers that operate in hyperbolic space,
including 1D and 2D convolutions with proper hyperbolic geometry handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional
import math
import logging

from ..core.base import HyperbolicModule
from ..core.math_ops import (
    exponential_map,
    logarithmic_map,
    project_to_poincare_ball,
    hyperbolic_linear_transform
)


logger = logging.getLogger(__name__)


class HyperbolicConv1d(HyperbolicModule):
    """
    1D hyperbolic convolutional layer.
    
    This layer performs 1D convolution in hyperbolic space by:
    1. Mapping input from hyperbolic space to tangent space
    2. Applying standard convolution in tangent space
    3. Mapping result back to hyperbolic space
    
    This preserves the hyperbolic geometry while leveraging efficient
    convolution implementations.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections (default: 1)
        bias: Whether to use bias (default: True)
        padding_mode: Padding mode (default: 'zeros')
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> conv = HyperbolicConv1d(16, 32, kernel_size=3, padding=1)
        >>> x = torch.randn(8, 16, 100) * 0.1  # (batch, channels, length)
        >>> x = conv.project_to_manifold(x)
        >>> y = conv(x)  # Shape: (8, 32, 100)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        
        # Standard 1D convolution for tangent space operations
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(f"Initialized HyperbolicConv1d({in_channels} -> {out_channels}, "
                   f"kernel_size={kernel_size})")
    
    def _init_parameters(self):
        """Initialize parameters appropriately for hyperbolic space."""
        with torch.no_grad():
            # Scale down initialization for hyperbolic stability
            fan_in = self.in_channels
            if isinstance(self.kernel_size, (list, tuple)):
                fan_in *= self.kernel_size[0]
            else:
                fan_in *= self.kernel_size
            
            std = math.sqrt(2.0 / fan_in) * 0.5  # Scale down for hyperbolic space
            nn.init.normal_(self.conv1d.weight, mean=0.0, std=std)
            
            if self.conv1d.bias is not None:
                nn.init.normal_(self.conv1d.bias, mean=0.0, std=std * 0.1)
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space, shape (batch, channels, length)
            
        Returns:
            Output tensor in hyperbolic space
        """
        batch_size, channels, length = x.shape
        
        # Reshape to apply log/exp maps point-wise
        x_flat = x.view(batch_size * channels, length, 1)  # Treat each position as a point
        
        # Map to tangent space at origin for each spatial position
        origin = torch.zeros_like(x_flat)
        x_tangent_flat = logarithmic_map(origin, x_flat, c=self.c.item())
        
        # Reshape back to convolution format
        x_tangent = x_tangent_flat.view(batch_size, channels, length)
        
        # Apply convolution in tangent space
        y_tangent = self.conv1d(x_tangent)
        
        # Get output dimensions
        batch_size_out, channels_out, length_out = y_tangent.shape
        
        # Reshape for mapping back to hyperbolic space
        y_tangent_flat = y_tangent.view(batch_size_out * channels_out, length_out, 1)
        
        # Map back to hyperbolic space
        origin_out = torch.zeros_like(y_tangent_flat)
        y_flat = exponential_map(y_tangent_flat, origin_out, c=self.c.item())
        
        # Reshape to final output format
        y = y_flat.view(batch_size_out, channels_out, length_out)
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y
    
    def extra_repr(self) -> str:
        """String representation of layer parameters."""
        s = (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}'
             f', stride={self.stride}')
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if not self.use_bias:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += f', padding_mode={self.padding_mode}'
        s += f', curvature={self.c.item():.3f}'
        return s


class HyperbolicConv2d(HyperbolicModule):
    """
    2D hyperbolic convolutional layer.
    
    This layer performs 2D convolution in hyperbolic space using the same
    approach as HyperbolicConv1d but extended to 2D spatial dimensions.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections (default: 1)
        bias: Whether to use bias (default: True)
        padding_mode: Padding mode (default: 'zeros')
        **kwargs: Additional arguments passed to HyperbolicModule
        
    Example:
        >>> conv = HyperbolicConv2d(3, 64, kernel_size=3, padding=1)
        >>> x = torch.randn(8, 3, 32, 32) * 0.1  # (batch, channels, height, width)
        >>> x = conv.project_to_manifold(x)
        >>> y = conv(x)  # Shape: (8, 64, 32, 32)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.padding_mode = padding_mode
        
        # Standard 2D convolution for tangent space operations
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(f"Initialized HyperbolicConv2d({in_channels} -> {out_channels}, "
                   f"kernel_size={kernel_size})")
    
    def _init_parameters(self):
        """Initialize parameters appropriately for hyperbolic space."""
        with torch.no_grad():
            # Scale down initialization for hyperbolic stability
            fan_in = self.in_channels
            if isinstance(self.kernel_size, (list, tuple)):
                fan_in *= self.kernel_size[0] * self.kernel_size[1]
            else:
                fan_in *= self.kernel_size * self.kernel_size
            
            std = math.sqrt(2.0 / fan_in) * 0.5  # Scale down for hyperbolic space
            nn.init.normal_(self.conv2d.weight, mean=0.0, std=std)
            
            if self.conv2d.bias is not None:
                nn.init.normal_(self.conv2d.bias, mean=0.0, std=std * 0.1)
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space, shape (batch, channels, height, width)
            
        Returns:
            Output tensor in hyperbolic space
        """
        batch_size, channels, height, width = x.shape
        
        # Reshape to apply log/exp maps point-wise
        # Each spatial position is treated as a point in hyperbolic space
        x_flat = x.view(batch_size * channels, height * width, 1)
        
        # Map to tangent space at origin for each spatial position
        origin = torch.zeros_like(x_flat)
        x_tangent_flat = logarithmic_map(origin, x_flat, c=self.c.item())
        
        # Reshape back to convolution format
        x_tangent = x_tangent_flat.view(batch_size, channels, height, width)
        
        # Apply convolution in tangent space
        y_tangent = self.conv2d(x_tangent)
        
        # Get output dimensions
        batch_size_out, channels_out, height_out, width_out = y_tangent.shape
        
        # Reshape for mapping back to hyperbolic space
        y_tangent_flat = y_tangent.view(batch_size_out * channels_out, height_out * width_out, 1)
        
        # Map back to hyperbolic space
        origin_out = torch.zeros_like(y_tangent_flat)
        y_flat = exponential_map(y_tangent_flat, origin_out, c=self.c.item())
        
        # Reshape to final output format
        y = y_flat.view(batch_size_out, channels_out, height_out, width_out)
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y
    
    def extra_repr(self) -> str:
        """String representation of layer parameters."""
        s = (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}'
             f', stride={self.stride}')
        if self.padding != 0:
            s += f', padding={self.padding}'
        if self.dilation != 1:
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if not self.use_bias:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += f', padding_mode={self.padding_mode}'
        s += f', curvature={self.c.item():.3f}'
        return s


class HyperbolicConvTranspose1d(HyperbolicModule):
    """
    1D hyperbolic transposed convolution (deconvolution) layer.
    
    This layer performs transposed convolution in hyperbolic space,
    useful for upsampling operations in hyperbolic neural networks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of blocked connections (default: 1)
        bias: Whether to use bias (default: True)
        dilation: Spacing between kernel elements (default: 1)
        padding_mode: Padding mode (default: 'zeros')
        **kwargs: Additional arguments passed to HyperbolicModule
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        output_padding: Union[int, Tuple[int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int]] = 1,
        padding_mode: str = 'zeros',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        
        # Standard transposed convolution for tangent space operations
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode
        )
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(f"Initialized HyperbolicConvTranspose1d({in_channels} -> {out_channels})")
    
    def _init_parameters(self):
        """Initialize parameters appropriately for hyperbolic space."""
        with torch.no_grad():
            # Scale down initialization for hyperbolic stability
            fan_in = self.in_channels
            if isinstance(self.kernel_size, (list, tuple)):
                fan_in *= self.kernel_size[0]
            else:
                fan_in *= self.kernel_size
            
            std = math.sqrt(2.0 / fan_in) * 0.5
            nn.init.normal_(self.conv_transpose1d.weight, mean=0.0, std=std)
            
            if self.conv_transpose1d.bias is not None:
                nn.init.normal_(self.conv_transpose1d.bias, mean=0.0, std=std * 0.1)
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transposed convolution in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Output tensor in hyperbolic space
        """
        batch_size, channels, length = x.shape
        
        # Map to tangent space
        x_flat = x.view(batch_size * channels, length, 1)
        origin = torch.zeros_like(x_flat)
        x_tangent_flat = logarithmic_map(origin, x_flat, c=self.c.item())
        x_tangent = x_tangent_flat.view(batch_size, channels, length)
        
        # Apply transposed convolution in tangent space
        y_tangent = self.conv_transpose1d(x_tangent)
        
        # Get output dimensions
        batch_size_out, channels_out, length_out = y_tangent.shape
        
        # Map back to hyperbolic space
        y_tangent_flat = y_tangent.view(batch_size_out * channels_out, length_out, 1)
        origin_out = torch.zeros_like(y_tangent_flat)
        y_flat = exponential_map(y_tangent_flat, origin_out, c=self.c.item())
        y = y_flat.view(batch_size_out, channels_out, length_out)
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y


class HyperbolicConvTranspose2d(HyperbolicModule):
    """
    2D hyperbolic transposed convolution (deconvolution) layer.
    
    This layer performs 2D transposed convolution in hyperbolic space,
    useful for upsampling operations in hyperbolic neural networks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution (default: 1)
        padding: Padding added to input (default: 0)
        output_padding: Additional size added to output shape (default: 0)
        groups: Number of blocked connections (default: 1)
        bias: Whether to use bias (default: True)
        dilation: Spacing between kernel elements (default: 1)
        padding_mode: Padding mode (default: 'zeros')
        **kwargs: Additional arguments passed to HyperbolicModule
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding_mode: str = 'zeros',
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.use_bias = bias
        self.dilation = dilation
        self.padding_mode = padding_mode
        
        # Standard transposed convolution for tangent space operations
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode
        )
        
        # Initialize parameters
        self._init_parameters()
        
        logger.info(f"Initialized HyperbolicConvTranspose2d({in_channels} -> {out_channels})")
    
    def _init_parameters(self):
        """Initialize parameters appropriately for hyperbolic space."""
        with torch.no_grad():
            # Scale down initialization for hyperbolic stability
            fan_in = self.in_channels
            if isinstance(self.kernel_size, (list, tuple)):
                fan_in *= self.kernel_size[0] * self.kernel_size[1]
            else:
                fan_in *= self.kernel_size * self.kernel_size
            
            std = math.sqrt(2.0 / fan_in) * 0.5
            nn.init.normal_(self.conv_transpose2d.weight, mean=0.0, std=std)
            
            if self.conv_transpose2d.bias is not None:
                nn.init.normal_(self.conv_transpose2d.bias, mean=0.0, std=std * 0.1)
    
    def hyperbolic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2D transposed convolution in hyperbolic space.
        
        Args:
            x: Input tensor in hyperbolic space
            
        Returns:
            Output tensor in hyperbolic space
        """
        batch_size, channels, height, width = x.shape
        
        # Map to tangent space
        x_flat = x.view(batch_size * channels, height * width, 1)
        origin = torch.zeros_like(x_flat)
        x_tangent_flat = logarithmic_map(origin, x_flat, c=self.c.item())
        x_tangent = x_tangent_flat.view(batch_size, channels, height, width)
        
        # Apply transposed convolution in tangent space
        y_tangent = self.conv_transpose2d(x_tangent)
        
        # Get output dimensions
        batch_size_out, channels_out, height_out, width_out = y_tangent.shape
        
        # Map back to hyperbolic space
        y_tangent_flat = y_tangent.view(batch_size_out * channels_out, height_out * width_out, 1)
        origin_out = torch.zeros_like(y_tangent_flat)
        y_flat = exponential_map(y_tangent_flat, origin_out, c=self.c.item())
        y = y_flat.view(batch_size_out, channels_out, height_out, width_out)
        
        # Ensure output is on manifold
        y = self.project_to_manifold(y)
        
        return y
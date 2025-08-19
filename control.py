"""Motor control logic implementing the Control System Specification.

This module computes motor outputs for a 5-engine AUV system based on:
- Target size error (distance control)
- Horizontal position error (turning control) 
- Vertical position error (depth control)

All outputs are clamped to [-1, 1] range.
"""

from typing import Tuple


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to specified range."""
    return max(min_val, min(max_val, value))


def compute_motor_outputs(
    bbox_area: float,
    target_area: float,
    cx: float,
    cy: float, 
    frame_width: float,
    frame_height: float,
    k_f: float = 0.5,
    k_t: float = 0.5,
    k_y: float = 1.0
) -> Tuple[float, float, float, float, float]:
    """Compute motor outputs for 5-engine AUV system per Control System Specification.
    
    Args:
        bbox_area: Detected bounding box area in pixels
        target_area: Desired bounding box area (proxy for chase distance)
        cx: Centroid X coordinate in pixels
        cy: Centroid Y coordinate in pixels
        frame_width: Video frame width in pixels
        frame_height: Video frame height in pixels
        k_f: Forward/distance gain (default: 0.5)
        k_t: Turning gain (default: 0.5)
        k_y: Vertical gain (default: 1.0)
        
    Returns:
        Tuple of (out1, out2, out3, out4, out5) motor outputs in range [-1, 1]
        
    Control Logic:
        E_size = (bbox_area / target_area) - 1
        X_norm = (cx / W) - 0.5  
        Y_norm = (cy / H) - 0.5
        
        Engines:
        out1 = k_f*E_size + k_t*(-X_norm)  # Left engine
        out2 = k_f*E_size + k_t*(X_norm)   # Right engine
        out3 = out4 = out5 = k_y*(-Y_norm) # Vertical engines
    """
    # Handle edge cases
    if target_area <= 0:
        target_area = 1e-6
    if frame_width <= 0:
        frame_width = 1.0
    if frame_height <= 0:
        frame_height = 1.0
        
    # Compute normalized errors per specification
    E_size = (bbox_area / target_area) - 1.0
    X_norm = (cx / frame_width) - 0.5
    Y_norm = (cy / frame_height) - 0.5
    
    # Compute engine outputs per specification
    # Engines 1 & 2: forward + turning mix
    out1 = k_f * E_size + k_t * (-X_norm)  # Left engine (negative turn for right target)
    out2 = k_f * E_size + k_t * ( X_norm)  # Right engine (positive turn for right target)
    
    # Engines 3, 4, 5: vertical only (identical)
    vertical_output = k_y * (-Y_norm)  # Negative because Y increases downward
    out3 = vertical_output
    out4 = vertical_output 
    out5 = vertical_output
    
    # Clamp all outputs to [-1, 1] range
    out1 = clamp(out1, -1.0, 1.0)
    out2 = clamp(out2, -1.0, 1.0)
    out3 = clamp(out3, -1.0, 1.0)
    out4 = clamp(out4, -1.0, 1.0)
    out5 = clamp(out5, -1.0, 1.0)
    
    return out1, out2, out3, out4, out5


class LowPassFilter:
    """Simple low-pass filter for smoothing motor outputs."""
    
    def __init__(self, alpha: float = 0.0):
        """Initialize filter.
        
        Args:
            alpha: Smoothing factor [0, 1]. 0=no smoothing, higher=more smoothing
        """
        self.alpha = clamp(alpha, 0.0, 1.0)
        self.last_values = [0.0, 0.0, 0.0, 0.0, 0.0]
        
    def filter(self, values: Tuple[float, float, float, float, float]) -> Tuple[float, float, float, float, float]:
        """Apply low-pass filter to motor outputs.
        
        Args:
            values: Current motor output values
            
        Returns:
            Filtered motor output values
        """
        if self.alpha <= 0.0:
            return values
            
        filtered = []
        for i, (prev, curr) in enumerate(zip(self.last_values, values)):
            filtered_val = (self.alpha * prev) + ((1.0 - self.alpha) * curr)
            filtered.append(filtered_val)
            self.last_values[i] = filtered_val
            
        return tuple(filtered)


def compute_motor_outputs_with_filter(
    bbox_area: float,
    target_area: float,
    cx: float,
    cy: float,
    frame_width: float,
    frame_height: float,
    k_f: float = 0.5,
    k_t: float = 0.5,
    k_y: float = 1.0,
    filter_alpha: float = 0.0
) -> Tuple[float, float, float, float, float]:
    """Compute motor outputs with optional low-pass filtering.
    
    This is a convenience function that combines compute_motor_outputs with filtering.
    For repeated calls, it's more efficient to create a LowPassFilter instance and
    use it directly.
    
    Args:
        bbox_area: Detected bounding box area in pixels
        target_area: Desired bounding box area 
        cx: Centroid X coordinate in pixels
        cy: Centroid Y coordinate in pixels
        frame_width: Video frame width in pixels
        frame_height: Video frame height in pixels
        k_f: Forward/distance gain
        k_t: Turning gain
        k_y: Vertical gain
        filter_alpha: Low-pass filter smoothing factor [0, 1]
        
    Returns:
        Tuple of filtered motor outputs in range [-1, 1]
    """
    # Static filter instance to maintain state between calls
    if not hasattr(compute_motor_outputs_with_filter, '_filter'):
        compute_motor_outputs_with_filter._filter = LowPassFilter(filter_alpha)
    
    # Update filter alpha if changed
    if compute_motor_outputs_with_filter._filter.alpha != filter_alpha:
        compute_motor_outputs_with_filter._filter.alpha = clamp(filter_alpha, 0.0, 1.0)
    
    # Compute raw outputs
    outputs = compute_motor_outputs(
        bbox_area, target_area, cx, cy, frame_width, frame_height, k_f, k_t, k_y
    )
    
    # Apply filtering
    return compute_motor_outputs_with_filter._filter.filter(outputs)

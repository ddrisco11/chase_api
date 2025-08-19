"""Unit tests for control logic.

Tests the motor output computation according to the Control System Specification.
"""

import unittest
from ..control import compute_motor_outputs, LowPassFilter


class TestControlLogic(unittest.TestCase):
    """Test cases for motor output computation."""
    
    def test_centered_bbox_half_target_area(self):
        """Test centered bbox with half target area should move forward, no turn."""
        bbox_area = 5000
        target_area = 10000  # bbox is half target size
        cx = 320  # center X
        cy = 240  # center Y
        frame_width = 640
        frame_height = 480
        
        out1, out2, out3, out4, out5 = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height
        )
        
        # E_size = (5000/10000) - 1 = -0.5 (too small, should go forward)
        # X_norm = (320/640) - 0.5 = 0 (centered)
        # Y_norm = (240/480) - 0.5 = 0 (centered)
        
        # out1 = 0.5*(-0.5) + 0.5*(-0) = -0.25 (forward)
        # out2 = 0.5*(-0.5) + 0.5*(0) = -0.25 (forward)
        # out3 = out4 = out5 = 1.0*(-0) = 0 (no vertical)
        
        self.assertAlmostEqual(out1, -0.25, places=3)
        self.assertAlmostEqual(out2, -0.25, places=3)
        self.assertAlmostEqual(out3, 0.0, places=3)
        self.assertAlmostEqual(out4, 0.0, places=3)
        self.assertAlmostEqual(out5, 0.0, places=3)
    
    def test_far_left_target(self):
        """Test target far left should create turning effect."""
        bbox_area = 10000
        target_area = 10000  # same size
        cx = 160  # far left (quarter frame)
        cy = 240  # center Y
        frame_width = 640
        frame_height = 480
        
        out1, out2, out3, out4, out5 = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height
        )
        
        # E_size = (10000/10000) - 1 = 0 (right size)
        # X_norm = (160/640) - 0.5 = -0.25 (left of center)
        # Y_norm = (240/480) - 0.5 = 0 (centered)
        
        # out1 = 0.5*(0) + 0.5*(-(-0.25)) = 0.125 (turn left)
        # out2 = 0.5*(0) + 0.5*(-0.25) = -0.125 (turn left)
        # out3 = out4 = out5 = 1.0*(-0) = 0 (no vertical)
        
        self.assertAlmostEqual(out1, 0.125, places=3)
        self.assertAlmostEqual(out2, -0.125, places=3)
        self.assertAlmostEqual(out3, 0.0, places=3)
        self.assertAlmostEqual(out4, 0.0, places=3)
        self.assertAlmostEqual(out5, 0.0, places=3)
        
        # Left target should create differential: out1 > out2
        self.assertGreater(out1, out2)
    
    def test_far_right_target(self):
        """Test target far right should create symmetric turning effect."""
        bbox_area = 10000
        target_area = 10000  # same size
        cx = 480  # far right (3/4 frame)
        cy = 240  # center Y
        frame_width = 640
        frame_height = 480
        
        out1, out2, out3, out4, out5 = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height
        )
        
        # E_size = (10000/10000) - 1 = 0 (right size)
        # X_norm = (480/640) - 0.5 = 0.25 (right of center)
        # Y_norm = (240/480) - 0.5 = 0 (centered)
        
        # out1 = 0.5*(0) + 0.5*(-0.25) = -0.125 (turn right)
        # out2 = 0.5*(0) + 0.5*(0.25) = 0.125 (turn right)
        # out3 = out4 = out5 = 1.0*(-0) = 0 (no vertical)
        
        self.assertAlmostEqual(out1, -0.125, places=3)
        self.assertAlmostEqual(out2, 0.125, places=3)
        self.assertAlmostEqual(out3, 0.0, places=3)
        self.assertAlmostEqual(out4, 0.0, places=3)
        self.assertAlmostEqual(out5, 0.0, places=3)
        
        # Right target should create differential: out2 > out1
        self.assertGreater(out2, out1)
    
    def test_target_above_center(self):
        """Test target above center should create upward vertical motion."""
        bbox_area = 10000
        target_area = 10000  # same size
        cx = 320  # center X
        cy = 120  # above center (quarter frame)
        frame_width = 640
        frame_height = 480
        
        out1, out2, out3, out4, out5 = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height
        )
        
        # E_size = (10000/10000) - 1 = 0 (right size)
        # X_norm = (320/640) - 0.5 = 0 (centered)
        # Y_norm = (120/480) - 0.5 = -0.25 (above center)
        
        # out1 = 0.5*(0) + 0.5*(-0) = 0 (no forward/turn)
        # out2 = 0.5*(0) + 0.5*(0) = 0 (no forward/turn)
        # out3 = out4 = out5 = 1.0*(-(-0.25)) = 0.25 (upward)
        
        self.assertAlmostEqual(out1, 0.0, places=3)
        self.assertAlmostEqual(out2, 0.0, places=3)
        self.assertAlmostEqual(out3, 0.25, places=3)
        self.assertAlmostEqual(out4, 0.25, places=3)
        self.assertAlmostEqual(out5, 0.25, places=3)
    
    def test_target_below_center(self):
        """Test target below center should create downward vertical motion."""
        bbox_area = 10000
        target_area = 10000  # same size
        cx = 320  # center X
        cy = 360  # below center (3/4 frame)
        frame_width = 640
        frame_height = 480
        
        out1, out2, out3, out4, out5 = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height
        )
        
        # E_size = (10000/10000) - 1 = 0 (right size)
        # X_norm = (320/640) - 0.5 = 0 (centered)
        # Y_norm = (360/480) - 0.5 = 0.25 (below center)
        
        # out1 = 0.5*(0) + 0.5*(-0) = 0 (no forward/turn)
        # out2 = 0.5*(0) + 0.5*(0) = 0 (no forward/turn)
        # out3 = out4 = out5 = 1.0*(-0.25) = -0.25 (downward)
        
        self.assertAlmostEqual(out1, 0.0, places=3)
        self.assertAlmostEqual(out2, 0.0, places=3)
        self.assertAlmostEqual(out3, -0.25, places=3)
        self.assertAlmostEqual(out4, -0.25, places=3)
        self.assertAlmostEqual(out5, -0.25, places=3)
    
    def test_clamping_at_bounds(self):
        """Test that outputs are clamped to [-1, 1] range."""
        # Create scenario that would exceed bounds
        bbox_area = 100  # very small
        target_area = 10000  # large target
        cx = 0  # extreme left
        cy = 0  # extreme top
        frame_width = 640
        frame_height = 480
        k_f = 2.0  # high gains to force clamping
        k_t = 2.0
        k_y = 2.0
        
        out1, out2, out3, out4, out5 = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height, k_f, k_t, k_y
        )
        
        # All outputs should be in valid range
        self.assertGreaterEqual(out1, -1.0)
        self.assertLessEqual(out1, 1.0)
        self.assertGreaterEqual(out2, -1.0)
        self.assertLessEqual(out2, 1.0)
        self.assertGreaterEqual(out3, -1.0)
        self.assertLessEqual(out3, 1.0)
        self.assertGreaterEqual(out4, -1.0)
        self.assertLessEqual(out4, 1.0)
        self.assertGreaterEqual(out5, -1.0)
        self.assertLessEqual(out5, 1.0)
    
    def test_custom_gains(self):
        """Test that custom gains affect outputs correctly."""
        bbox_area = 5000
        target_area = 10000
        cx = 320
        cy = 240
        frame_width = 640
        frame_height = 480
        
        # Test with different gains
        out1_a, out2_a, out3_a, out4_a, out5_a = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height,
            k_f=1.0, k_t=0.5, k_y=1.0
        )
        
        out1_b, out2_b, out3_b, out4_b, out5_b = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height,
            k_f=0.5, k_t=0.5, k_y=1.0
        )
        
        # Higher k_f should produce larger forward/backward response
        self.assertAlmostEqual(abs(out1_a), 2 * abs(out1_b), places=3)
        self.assertAlmostEqual(abs(out2_a), 2 * abs(out2_b), places=3)
    
    def test_zero_division_protection(self):
        """Test protection against zero division errors."""
        bbox_area = 1000
        target_area = 0  # zero target area
        cx = 320
        cy = 240
        frame_width = 0  # zero frame width
        frame_height = 0  # zero frame height
        
        # Should not raise exception
        out1, out2, out3, out4, out5 = compute_motor_outputs(
            bbox_area, target_area, cx, cy, frame_width, frame_height
        )
        
        # Should return valid values in range
        self.assertGreaterEqual(out1, -1.0)
        self.assertLessEqual(out1, 1.0)


class TestLowPassFilter(unittest.TestCase):
    """Test cases for low-pass filter."""
    
    def test_no_filtering(self):
        """Test filter with alpha=0 passes values unchanged."""
        lpf = LowPassFilter(alpha=0.0)
        
        values = (0.5, -0.3, 0.8, -0.1, 0.2)
        filtered = lpf.filter(values)
        
        self.assertEqual(filtered, values)
    
    def test_full_smoothing(self):
        """Test filter with alpha=1 maintains previous values."""
        lpf = LowPassFilter(alpha=1.0)
        
        # First call sets initial values
        values1 = (0.5, -0.3, 0.8, -0.1, 0.2)
        filtered1 = lpf.filter(values1)
        
        # Second call with different values should return first values
        values2 = (1.0, 1.0, 1.0, 1.0, 1.0)
        filtered2 = lpf.filter(values2)
        
        # Should be close to first values (with some numerical difference)
        for f1, f2 in zip(filtered1, filtered2):
            self.assertAlmostEqual(f1, f2, places=3)
    
    def test_partial_smoothing(self):
        """Test filter with intermediate alpha value."""
        lpf = LowPassFilter(alpha=0.5)
        
        # First call
        values1 = (0.0, 0.0, 0.0, 0.0, 0.0)
        filtered1 = lpf.filter(values1)
        
        # Second call with step input
        values2 = (1.0, 1.0, 1.0, 1.0, 1.0)
        filtered2 = lpf.filter(values2)
        
        # With alpha=0.5, output should be (0.5*0 + 0.5*1) = 0.5
        for f in filtered2:
            self.assertAlmostEqual(f, 0.5, places=3)


if __name__ == "__main__":
    unittest.main()

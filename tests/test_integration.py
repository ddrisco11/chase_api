"""Integration tests for chase API.

Tests the complete pipeline with mocked frames to validate output shape and ranges.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from ..inference import chase_stream, InferenceEngine, VideoCapture, select_primary_target
from ..schemas import ChaseOutput


class TestIntegration(unittest.TestCase):
    """Integration tests for complete chase pipeline."""
    
    @patch('cv2.VideoCapture')
    @patch('ultralytics.YOLO')
    def test_chase_stream_with_mock_detection(self, mock_yolo_class, mock_cv2_cap_class):
        """Test one iteration of chase_stream with mocked detection."""
        
        # Mock YOLO model
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Mock detection results
        mock_boxes = MagicMock()
        mock_boxes.conf = [0.8]  # High confidence
        mock_boxes.xyxy = [np.array([100, 100, 200, 200])]  # 100x100 bbox
        
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        
        mock_model.return_value = [mock_result]
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cv2_cap_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640 if prop == mock_cv2_cap_class.CAP_PROP_FRAME_WIDTH else 480
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, test_frame), (False, None)]  # One frame then end
        
        # Test chase stream
        stream = chase_stream(
            video_source="0",
            model_path="test.pt",
            target_area=5000.0,  # Half the detected area (10000)
            confidence_threshold=0.25
        )
        
        # Get first result
        result = next(stream)
        
        # Validate result structure
        self.assertIsInstance(result, ChaseOutput)
        self.assertIsInstance(result.ts, int)
        self.assertGreater(result.ts, 0)
        
        # Validate frame size
        self.assertEqual(result.frameSize.width, 640)
        self.assertEqual(result.frameSize.height, 480)
        
        # Validate detection
        self.assertIsNotNone(result.detection)
        self.assertEqual(len(result.detection.bbox), 4)
        self.assertAlmostEqual(result.detection.area, 10000.0, places=1)  # 100x100
        self.assertAlmostEqual(result.detection.centroid.x, 150.0, places=1)  # Center X
        self.assertAlmostEqual(result.detection.centroid.y, 150.0, places=1)  # Center Y
        self.assertAlmostEqual(result.detection.confidence, 0.8, places=1)
        
        # Validate motor outputs
        self.assertIsNotNone(result.motors)
        motors = result.motors
        
        # All outputs should be in valid range
        self.assertGreaterEqual(motors.out1, -1.0)
        self.assertLessEqual(motors.out1, 1.0)
        self.assertGreaterEqual(motors.out2, -1.0)
        self.assertLessEqual(motors.out2, 1.0)
        self.assertGreaterEqual(motors.out3, -1.0)
        self.assertLessEqual(motors.out3, 1.0)
        self.assertGreaterEqual(motors.out4, -1.0)
        self.assertLessEqual(motors.out4, 1.0)
        self.assertGreaterEqual(motors.out5, -1.0)
        self.assertLessEqual(motors.out5, 1.0)
        
        # Detection area (10000) > target area (5000), so should move forward (negative outputs)
        # Centroid at (150, 150) in 640x480 frame:
        # X_norm = (150/640) - 0.5 = -0.265 (left of center)
        # Y_norm = (150/480) - 0.5 = -0.188 (above center)
        # E_size = (10000/5000) - 1 = 1.0 (too large, move back)
        
        # out1 = 0.5*1.0 + 0.5*(-(-0.265)) = 0.5 + 0.133 = 0.633 (right turn + backward)
        # out2 = 0.5*1.0 + 0.5*(-0.265) = 0.5 - 0.133 = 0.367 (right turn + backward)
        # out3 = out4 = out5 = 1.0*(-(-0.188)) = 0.188 (upward)
        
        self.assertGreater(motors.out1, 0)  # Should be positive (backward + turn)
        self.assertGreater(motors.out2, 0)  # Should be positive (backward + turn)
        self.assertGreater(motors.out1, motors.out2)  # Differential for turning
        self.assertGreater(motors.out3, 0)  # Should be positive (upward)
        self.assertEqual(motors.out3, motors.out4)  # Vertical engines identical
        self.assertEqual(motors.out4, motors.out5)  # Vertical engines identical
    
    @patch('cv2.VideoCapture')
    @patch('ultralytics.YOLO')
    def test_chase_stream_no_detection(self, mock_yolo_class, mock_cv2_cap_class):
        """Test chase_stream with no detections."""
        
        # Mock YOLO model with no detections
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        mock_result = MagicMock()
        mock_result.boxes = None  # No detections
        
        mock_model.return_value = [mock_result]
        
        # Mock video capture
        mock_cap = MagicMock()
        mock_cv2_cap_class.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640 if prop == mock_cv2_cap_class.CAP_PROP_FRAME_WIDTH else 480
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, test_frame), (False, None)]
        
        # Test chase stream
        stream = chase_stream(
            video_source="0",
            model_path="test.pt",
            target_area=5000.0,
            confidence_threshold=0.25
        )
        
        result = next(stream)
        
        # Should have no detection
        self.assertIsNone(result.detection)
        
        # All motor outputs should be zero
        motors = result.motors
        self.assertEqual(motors.out1, 0.0)
        self.assertEqual(motors.out2, 0.0)
        self.assertEqual(motors.out3, 0.0)
        self.assertEqual(motors.out4, 0.0)
        self.assertEqual(motors.out5, 0.0)
    
    def test_select_primary_target(self):
        """Test primary target selection logic."""
        
        # No detections
        result = select_primary_target([])
        self.assertIsNone(result)
        
        # Single detection
        det1 = {'area': 1000, 'confidence': 0.8}
        result = select_primary_target([det1])
        self.assertEqual(result, det1)
        
        # Multiple detections - should pick largest area
        det2 = {'area': 2000, 'confidence': 0.6}  # Larger but lower confidence
        det3 = {'area': 500, 'confidence': 0.9}   # Smaller but higher confidence
        
        result = select_primary_target([det1, det2, det3])
        self.assertEqual(result, det2)  # Should pick det2 (largest area)
    
    def test_chase_distance_to_target_area_conversion(self):
        """Test conversion from chase_distance to target_area."""
        
        with patch('cv2.VideoCapture') as mock_cv2_cap_class, \
             patch('ultralytics.YOLO') as mock_yolo_class:
            
            # Mock empty stream for quick test
            mock_model = MagicMock()
            mock_yolo_class.return_value = mock_model
            mock_result = MagicMock()
            mock_result.boxes = None
            mock_model.return_value = [mock_result]
            
            mock_cap = MagicMock()
            mock_cv2_cap_class.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: 640 if prop == mock_cv2_cap_class.CAP_PROP_FRAME_WIDTH else 480
            mock_cap.read.return_value = (False, None)  # End immediately
            
            # Test with chase_distance instead of target_area
            chase_distance = 2.0
            calibration_k = 1e6
            expected_target_area = calibration_k / chase_distance  # 500000
            
            stream = chase_stream(
                video_source="0",
                model_path="test.pt",
                target_area=None,
                chase_distance=chase_distance,
                calibration_k=calibration_k,
                confidence_threshold=0.25
            )
            
            # Just test that it doesn't crash - the conversion happens internally
            list(stream)  # Consume the empty stream
    
    @patch('cv2.VideoCapture')
    def test_video_capture_error_handling(self, mock_cv2_cap_class):
        """Test video capture error handling."""
        
        # Mock failed video capture
        mock_cap = MagicMock()
        mock_cv2_cap_class.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        # Should raise exception
        with self.assertRaises(RuntimeError):
            VideoCapture("invalid_source")
    
    @patch('ultralytics.YOLO')
    def test_inference_engine_error_handling(self, mock_yolo_class):
        """Test inference engine error handling."""
        
        # Mock failed model loading
        mock_yolo_class.side_effect = Exception("Model load failed")
        
        # Should raise exception
        with self.assertRaises(Exception):
            InferenceEngine("invalid_model.pt")


class TestSchemaValidation(unittest.TestCase):
    """Test Pydantic schema validation."""
    
    def test_chase_output_validation(self):
        """Test ChaseOutput schema validation."""
        from ..schemas import ChaseOutput, FrameSize, MotorOutputs, Detection, Centroid
        
        # Valid output
        valid_output = ChaseOutput(
            ts=1234567890000,
            frameSize=FrameSize(width=640, height=480),
            detection=Detection(
                bbox=[100, 100, 200, 200],
                area=10000.0,
                centroid=Centroid(x=150.0, y=150.0),
                confidence=0.8
            ),
            motors=MotorOutputs(
                out1=0.5, out2=-0.3, out3=0.8, out4=0.8, out5=0.8
            )
        )
        
        # Should serialize to JSON
        json_str = valid_output.model_dump_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("ts", json_str)
        self.assertIn("motors", json_str)
    
    def test_motor_outputs_range_validation(self):
        """Test motor outputs are validated to [-1, 1] range."""
        from ..schemas import MotorOutputs
        from pydantic import ValidationError
        
        # Valid range
        valid_motors = MotorOutputs(
            out1=-1.0, out2=0.0, out3=1.0, out4=0.5, out5=-0.5
        )
        self.assertIsNotNone(valid_motors)
        
        # Invalid range - should raise ValidationError
        with self.assertRaises(ValidationError):
            MotorOutputs(out1=1.5, out2=0.0, out3=0.0, out4=0.0, out5=0.0)
        
        with self.assertRaises(ValidationError):
            MotorOutputs(out1=0.0, out2=-1.5, out3=0.0, out4=0.0, out5=0.0)


if __name__ == "__main__":
    unittest.main()

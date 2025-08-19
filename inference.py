"""Object detection and video processing using YOLO.

This module handles:
- Loading YOLO models from Ultralytics
- Video capture from various sources (webcam, file, RTSP)
- Running inference and filtering detections
- Computing centroids and bounding box areas
- Graceful shutdown and error handling
"""

import time
import cv2
import numpy as np
from typing import Iterator, Optional, Tuple, Dict, Any
import ultralytics

try:
    from .schemas import ChaseOutput, Detection, MotorOutputs, FrameSize, Centroid
    from .control import compute_motor_outputs
except ImportError:
    # Handle running as script directly
    from schemas import ChaseOutput, Detection, MotorOutputs, FrameSize, Centroid
    from control import compute_motor_outputs


class InferenceEngine:
    """YOLO inference engine for object detection."""
    
    def __init__(self, model_path: str):
        """Initialize inference engine.
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
        """
        self.model_path = model_path
        self.model: Optional[Any] = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load YOLO model from path."""
        try:
            print(f"[inference] Loading YOLO model: {self.model_path}")
            self.model = ultralytics.YOLO(self.model_path)
            print(f"[inference] Model loaded successfully")
        except Exception as e:
            print(f"[inference] Failed to load model: {e}")
            raise
            
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.25) -> list:
        """Run inference on frame and return detections above threshold.
        
        Args:
            frame: Input image as numpy array
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detection dictionaries with bbox, confidence, etc.
        """
        if self.model is None:
            return []
            
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    # Determine number of detections robustly (works with mocks)
                    num = 0
                    try:
                        num = len(boxes)  # real Ultralytics implements __len__
                    except Exception:
                        num = 0
                    if not num and hasattr(boxes, 'conf'):
                        try:
                            num = len(boxes.conf)
                        except Exception:
                            pass
                    if not num and hasattr(boxes, 'xyxy'):
                        try:
                            num = len(boxes.xyxy)
                        except Exception:
                            pass

                    for i in range(max(0, num)):
                        # Confidence value
                        conf_val = getattr(boxes, 'conf', None)
                        if conf_val is None:
                            continue
                        try:
                            conf = float(conf_val[i])
                        except Exception:
                            try:
                                conf = float(conf_val[i][0])
                            except Exception:
                                continue
                        if conf < confidence_threshold:
                            continue

                        # Bounding box coordinates
                        xyxy_list = getattr(boxes, 'xyxy', None)
                        if xyxy_list is None:
                            continue
                        coords = xyxy_list[i]
                        # Handle torch tensors with .cpu().numpy() and plain numpy lists
                        try:
                            x1, y1, x2, y2 = coords.cpu().numpy().tolist()
                        except Exception:
                            x1, y1, x2, y2 = np.array(coords, dtype=float).tolist()

                        # Compute centroid and area
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        area = (x2 - x1) * (y2 - y1)

                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'centroid': {'x': float(cx), 'y': float(cy)},
                            'area': float(area),
                            'confidence': float(conf)
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"[inference] Detection error: {e}")
            return []


class VideoCapture:
    """Video capture handler for various sources."""
    
    def __init__(self, video_source: str):
        """Initialize video capture.
        
        Args:
            video_source: Video source (camera index, file path, RTSP URL)
        """
        self.video_source = video_source
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width = 0
        self.frame_height = 0
        self._setup_capture()
        
    def _setup_capture(self) -> None:
        """Setup OpenCV video capture."""
        try:
            # Try to convert to int for camera index
            try:
                source = int(self.video_source)
            except ValueError:
                source = self.video_source
                
            print(f"[video] Opening video source: {source}")
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {source}")
                
            # Get frame dimensions
            # Prefer constants from the (possibly patched) VideoCapture class used in tests
            prop_w = getattr(cv2.VideoCapture, 'CAP_PROP_FRAME_WIDTH', cv2.CAP_PROP_FRAME_WIDTH)
            prop_h = getattr(cv2.VideoCapture, 'CAP_PROP_FRAME_HEIGHT', cv2.CAP_PROP_FRAME_HEIGHT)
            self.frame_width = int(self.cap.get(prop_w))
            self.frame_height = int(self.cap.get(prop_h))
            
            print(f"[video] Capture opened: {self.frame_width}x{self.frame_height}")
            
        except Exception as e:
            print(f"[video] Failed to setup capture: {e}")
            raise
            
    def read_frame(self) -> Optional[np.ndarray]:
        """Read next frame from video source.
        
        Returns:
            Frame as numpy array, or None if failed
        """
        if self.cap is None:
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
        except Exception as e:
            print(f"[video] Frame read error: {e}")
            return None
            
    def get_frame_size(self) -> Tuple[int, int]:
        """Get frame dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        return self.frame_width, self.frame_height
        
    def release(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def select_primary_target(detections: list) -> Optional[Dict[str, Any]]:
    """Select primary target from detections (largest by area).
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Primary target detection or None if no detections
    """
    if not detections:
        return None
        
    # Find detection with largest area
    primary = max(detections, key=lambda d: d.get('area', 0))
    return primary


def chase_stream(
    video_source: str,
    model_path: str,
    target_area: Optional[float] = None,
    confidence_threshold: float = 0.25,
    chase_distance: Optional[float] = None,
    calibration_k: float = 1e6,
    k_f: float = 0.5,
    k_t: float = 0.5,
    k_y: float = 1.0
) -> Iterator[ChaseOutput]:
    """Generate chase outputs from video stream.
    
    Args:
        video_source: Video source (camera index, file path, RTSP URL)
        model_path: Path to YOLO model weights
        target_area: Target bounding box area (pixels), takes priority
        confidence_threshold: Detection confidence threshold [0,1]
        chase_distance: Chase distance (alternative to target_area)
        calibration_k: Calibration constant for distance to area conversion
        k_f: Forward/distance gain
        k_t: Turning gain
        k_y: Vertical gain
        
    Yields:
        ChaseOutput objects for each processed frame
    """
    # Resolve target_area vs chase_distance
    if target_area is None and chase_distance is not None:
        target_area = calibration_k / max(chase_distance, 1e-6)
        print(f"[chase] Using chase_distance={chase_distance}, computed target_area={target_area}")
    elif target_area is None:
        target_area = 10000.0  # Default target area
        print(f"[chase] Using default target_area={target_area}")
    
    # Initialize components
    try:
        inference_engine = InferenceEngine(model_path)
        video_capture = VideoCapture(video_source)
        
        frame_width, frame_height = video_capture.get_frame_size()
        print(f"[chase] Started chase stream: {frame_width}x{frame_height}")
        
        while True:
            # Read frame
            frame = video_capture.read_frame()
            if frame is None:
                print("[chase] No more frames, ending stream")
                break
                
            # Get timestamp
            timestamp_ms = int(time.time() * 1000)
            
            # Run detection
            detections = inference_engine.detect(frame, confidence_threshold)
            primary_target = select_primary_target(detections)
            
            # Compute motor outputs
            if primary_target is not None:
                # Extract detection info
                bbox = primary_target['bbox']
                centroid = primary_target['centroid']
                area = primary_target['area']
                confidence = primary_target['confidence']
                
                # Create detection object
                detection = Detection(
                    bbox=bbox,
                    area=area,
                    centroid=Centroid(x=centroid['x'], y=centroid['y']),
                    confidence=confidence
                )
                
                # Compute motor outputs
                out1, out2, out3, out4, out5 = compute_motor_outputs(
                    bbox_area=area,
                    target_area=target_area,
                    cx=centroid['x'],
                    cy=centroid['y'],
                    frame_width=frame_width,
                    frame_height=frame_height,
                    k_f=k_f,
                    k_t=k_t,
                    k_y=k_y
                )
                
                motors = MotorOutputs(
                    out1=out1, out2=out2, out3=out3, out4=out4, out5=out5
                )
                
            else:
                # No detection - set all motors to zero
                detection = None
                motors = MotorOutputs(
                    out1=0.0, out2=0.0, out3=0.0, out4=0.0, out5=0.0
                )
            
            # Create chase output
            chase_output = ChaseOutput(
                ts=timestamp_ms,
                frameSize=FrameSize(width=frame_width, height=frame_height),
                detection=detection,
                motors=motors
            )
            
            yield chase_output
            
    except KeyboardInterrupt:
        print("[chase] Interrupted by user")
    except Exception as e:
        print(f"[chase] Error in chase stream: {e}")
        raise
    finally:
        # Cleanup
        try:
            video_capture.release()
        except:
            pass
        print("[chase] Chase stream ended")


def test_inference(model_path: str, test_image_path: Optional[str] = None) -> None:
    """Test inference engine with a single image.
    
    Args:
        model_path: Path to YOLO model weights
        test_image_path: Path to test image (optional, creates blank if None)
    """
    try:
        engine = InferenceEngine(model_path)
        
        if test_image_path:
            frame = cv2.imread(test_image_path)
        else:
            # Create test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        if frame is None:
            print("Failed to load test image")
            return
            
        print(f"Running inference on {frame.shape} frame...")
        detections = engine.detect(frame, confidence_threshold=0.25)
        print(f"Found {len(detections)} detections")
        
        for i, det in enumerate(detections):
            print(f"  Detection {i}: area={det['area']:.1f}, conf={det['confidence']:.3f}")
            
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        test_inference(sys.argv[1])
    else:
        print("Usage: python inference.py <model_path> [test_image_path]")

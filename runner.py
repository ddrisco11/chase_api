"""Python API and CLI entrypoint for the chase system.

This module provides:
- High-level Python API for integration
- Command-line interface for standalone usage
- Graceful shutdown handling
"""

import json
import sys
import argparse
import signal
from typing import Callable, Optional

try:
    from .inference import chase_stream
    from .schemas import ChaseOutput
except ImportError:
    # Handle running as script directly
    from inference import chase_stream
    from schemas import ChaseOutput


def run_chase(
    video_source: str,
    model_path: str,
    target_area: Optional[float] = None,
    confidence_threshold: float = 0.25,
    chase_distance: Optional[float] = None,
    calibration_k: float = 1e6,
    k_f: float = 0.5,
    k_t: float = 0.5,
    k_y: float = 1.0,
    on_result: Optional[Callable[[ChaseOutput], None]] = None
) -> None:
    """Run chase system with callback for each result.
    
    This is the main Python API function for integrating the chase system
    into other applications.
    
    Args:
        video_source: Video source (camera index, file path, RTSP URL)
        model_path: Path to YOLO model weights (.pt file)
        target_area: Target bounding box area (pixels), takes priority
        confidence_threshold: Detection confidence threshold [0,1]
        chase_distance: Chase distance (alternative to target_area)
        calibration_k: Calibration constant for distance to area conversion
        k_f: Forward/distance gain
        k_t: Turning gain
        k_y: Vertical gain
        on_result: Callback function called for each ChaseOutput
        
    Example:
        def handle_result(output: ChaseOutput):
            print(f"Motors: {output.motors.out1:.2f}, {output.motors.out2:.2f}")
            
        run_chase(
            video_source="0",
            model_path="yolov10n.pt", 
            target_area=12000,
            on_result=handle_result
        )
    """
    if target_area is None and chase_distance is None:
        raise ValueError("Must provide either target_area or chase_distance")
        
    # Setup signal handler for graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        print(f"\n[runner] Received signal {signum}, shutting down gracefully...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print(f"[runner] Starting chase system...")
        print(f"[runner] Video source: {video_source}")
        print(f"[runner] Model: {model_path}")
        print(f"[runner] Target area: {target_area}")
        print(f"[runner] Chase distance: {chase_distance}")
        print(f"[runner] Confidence threshold: {confidence_threshold}")
        print(f"[runner] Control gains: k_f={k_f}, k_t={k_t}, k_y={k_y}")
        print(f"[runner] Press Ctrl+C to stop")
        
        # Create chase stream
        stream = chase_stream(
            video_source=video_source,
            model_path=model_path,
            target_area=target_area,
            confidence_threshold=confidence_threshold,
            chase_distance=chase_distance,
            calibration_k=calibration_k,
            k_f=k_f,
            k_t=k_t,
            k_y=k_y
        )
        
        # Process chase outputs
        for chase_output in stream:
            if shutdown_requested:
                break
                
            # Call user callback if provided
            if on_result is not None:
                try:
                    on_result(chase_output)
                except Exception as e:
                    print(f"[runner] Error in result callback: {e}")
            else:
                # Default behavior: print JSON
                print(chase_output.model_dump_json())
                
    except KeyboardInterrupt:
        print("\n[runner] Interrupted by user")
    except Exception as e:
        print(f"[runner] Error: {e}")
        raise
    finally:
        print("[runner] Chase system stopped")


def main():
    """Command-line interface for chase system."""
    parser = argparse.ArgumentParser(
        description="Chase API - Target chasing using object detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--video-source",
        required=True,
        help="Video source (camera index like '0', file path, or RTSP URL)"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to YOLO model weights (.pt file)"
    )
    
    # Target specification (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-area",
        type=float,
        help="Target bounding box area in pixels"
    )
    target_group.add_argument(
        "--chase-distance", 
        type=float,
        help="Chase distance (converted to target area using calibration-k)"
    )
    
    # Optional parameters
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Detection confidence threshold [0,1]"
    )
    parser.add_argument(
        "--calibration-k",
        type=float,
        default=1e6,
        help="Calibration constant for chase distance to target area conversion"
    )
    
    # Control gains
    parser.add_argument(
        "--k-f",
        type=float,
        default=0.5,
        help="Forward/distance gain"
    )
    parser.add_argument(
        "--k-t",
        type=float,
        default=0.5,
        help="Turning gain"
    )
    parser.add_argument(
        "--k-y",
        type=float,
        default=1.0,
        help="Vertical gain"
    )
    
    # Output options
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "summary"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational messages"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.confidence_threshold < 0 or args.confidence_threshold > 1:
        parser.error("confidence-threshold must be between 0 and 1")
        
    if args.target_area is not None and args.target_area <= 0:
        parser.error("target-area must be positive")
        
    if args.chase_distance is not None and args.chase_distance <= 0:
        parser.error("chase-distance must be positive")
    
    # Setup output handler based on format
    def create_output_handler(format_type: str):
        if format_type == "json":
            return lambda output: print(output.model_dump_json())
        elif format_type == "csv":
            # Print CSV header once
            header_printed = False
            def csv_handler(output: ChaseOutput):
                nonlocal header_printed
                if not header_printed:
                    print("timestamp,detection_area,detection_confidence,centroid_x,centroid_y,out1,out2,out3,out4,out5")
                    header_printed = True
                
                if output.detection:
                    det_area = output.detection.area
                    det_conf = output.detection.confidence
                    cent_x = output.detection.centroid.x
                    cent_y = output.detection.centroid.y
                else:
                    det_area = det_conf = cent_x = cent_y = 0
                    
                motors = output.motors
                print(f"{output.ts},{det_area},{det_conf},{cent_x},{cent_y},"
                      f"{motors.out1},{motors.out2},{motors.out3},{motors.out4},{motors.out5}")
            return csv_handler
        elif format_type == "summary":
            def summary_handler(output: ChaseOutput):
                if output.detection:
                    det_info = f"area={output.detection.area:.1f}, conf={output.detection.confidence:.3f}"
                    cent_info = f"centroid=({output.detection.centroid.x:.1f},{output.detection.centroid.y:.1f})"
                else:
                    det_info = "no detection"
                    cent_info = ""
                    
                motors = output.motors
                motor_info = f"motors=[{motors.out1:.2f},{motors.out2:.2f},{motors.out3:.2f},{motors.out4:.2f},{motors.out5:.2f}]"
                
                print(f"[{output.ts}] {det_info} {cent_info} {motor_info}")
            return summary_handler
        else:
            return lambda output: print(output.model_dump_json())
    
    output_handler = None if args.quiet else create_output_handler(args.output_format)
    
    # Run chase system
    try:
        run_chase(
            video_source=args.video_source,
            model_path=args.model_path,
            target_area=args.target_area,
            confidence_threshold=args.confidence_threshold,
            chase_distance=args.chase_distance,
            calibration_k=args.calibration_k,
            k_f=args.k_f,
            k_t=args.k_t,
            k_y=args.k_y,
            on_result=output_handler
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

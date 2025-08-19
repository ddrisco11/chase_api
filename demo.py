#!/usr/bin/env python3
"""Demo script to test chase_api control logic without dependencies."""

import sys
import time
from control import compute_motor_outputs

def demo_control_logic():
    """Demonstrate the motor control logic with various scenarios."""
    
    print("Chase API Control Logic Demo")
    print("=" * 40)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Centered target, half size (should go forward)",
            "bbox_area": 5000, "target_area": 10000,
            "cx": 320, "cy": 240, "width": 640, "height": 480
        },
        {
            "name": "Target too large, centered (should go backward)",
            "bbox_area": 15000, "target_area": 10000,
            "cx": 320, "cy": 240, "width": 640, "height": 480
        },
        {
            "name": "Target far left (should turn left)",
            "bbox_area": 10000, "target_area": 10000,
            "cx": 160, "cy": 240, "width": 640, "height": 480
        },
        {
            "name": "Target far right (should turn right)",
            "bbox_area": 10000, "target_area": 10000,
            "cx": 480, "cy": 240, "width": 640, "height": 480
        },
        {
            "name": "Target above center (should go up)",
            "bbox_area": 10000, "target_area": 10000,
            "cx": 320, "cy": 120, "width": 640, "height": 480
        },
        {
            "name": "Target below center (should go down)",
            "bbox_area": 10000, "target_area": 10000,
            "cx": 320, "cy": 360, "width": 640, "height": 480
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Input: area={scenario['bbox_area']}, target={scenario['target_area']}")
        print(f"         centroid=({scenario['cx']}, {scenario['cy']})")
        print(f"         frame={scenario['width']}x{scenario['height']}")
        
        out1, out2, out3, out4, out5 = compute_motor_outputs(
            scenario['bbox_area'], scenario['target_area'],
            scenario['cx'], scenario['cy'],
            scenario['width'], scenario['height']
        )
        
        print(f"  Output: out1={out1:+.3f}, out2={out2:+.3f}, out3={out3:+.3f}, out4={out4:+.3f}, out5={out5:+.3f}")
        
        # Interpret the outputs
        forward = (out1 + out2) / 2
        turn = out1 - out2
        vertical = out3
        
        if abs(forward) > 0.01:
            direction = "forward" if forward < 0 else "backward"
            print(f"         → {direction} motion ({forward:+.3f})")
        
        if abs(turn) > 0.01:
            direction = "left" if turn > 0 else "right"
            print(f"         → turn {direction} ({turn:+.3f})")
            
        if abs(vertical) > 0.01:
            direction = "up" if vertical > 0 else "down"
            print(f"         → vertical {direction} ({vertical:+.3f})")

def demo_json_output():
    """Demonstrate JSON output format without pydantic."""
    print("\n\nJSON Output Format Demo")
    print("=" * 40)
    
    # Simulate a detection result
    timestamp_ms = int(time.time() * 1000)
    bbox_area = 12000
    target_area = 10000
    cx, cy = 350, 200
    frame_width, frame_height = 640, 480
    
    out1, out2, out3, out4, out5 = compute_motor_outputs(
        bbox_area, target_area, cx, cy, frame_width, frame_height
    )
    
    # Manual JSON construction (normally done by pydantic)
    output = {
        "ts": timestamp_ms,
        "frameSize": {"width": frame_width, "height": frame_height},
        "detection": {
            "bbox": [300, 150, 400, 250],  # Example bbox
            "area": bbox_area,
            "centroid": {"x": cx, "y": cy},
            "confidence": 0.85
        },
        "motors": {
            "out1": round(out1, 3),
            "out2": round(out2, 3),
            "out3": round(out3, 3),
            "out4": round(out4, 3),
            "out5": round(out5, 3)
        }
    }
    
    import json
    print("Sample ChaseOutput JSON:")
    print(json.dumps(output, indent=2))

def demo_chase_distance_conversion():
    """Demonstrate chase distance to target area conversion."""
    print("\n\nChase Distance Conversion Demo")
    print("=" * 40)
    
    chase_distances = [0.5, 1.0, 2.0, 5.0]
    calibration_k = 1e6
    
    print(f"Calibration constant: {calibration_k}")
    print("Chase Distance → Target Area:")
    
    for distance in chase_distances:
        target_area = calibration_k / distance
        print(f"  {distance:4.1f}m → {target_area:8.0f} pixels")

if __name__ == "__main__":
    try:
        demo_control_logic()
        demo_json_output()
        demo_chase_distance_conversion()
        print("\n\nDemo completed successfully!")
        print("The chase API is ready for use once dependencies are installed.")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)

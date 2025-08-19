"""Chase API - Minimal target chasing system using object detection.

This module provides a self-contained API for target chasing that:
- Receives video input and runs object detection
- Computes motor outputs per the Control System Specification
- Streams outputs in real-time for integration with existing systems
"""

from .runner import run_chase, chase_stream
from .schemas import ChaseOutput, Detection, MotorOutputs
from .control import compute_motor_outputs

__version__ = "1.0.0"
__all__ = ["run_chase", "chase_stream", "ChaseOutput", "Detection", "MotorOutputs", "compute_motor_outputs"]

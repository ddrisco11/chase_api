"""Pydantic models for chase API inputs and outputs."""

from typing import Optional, List
from pydantic import BaseModel, Field


class Centroid(BaseModel):
    """2D centroid coordinates."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class Detection(BaseModel):
    """Object detection result."""
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    area: float = Field(..., ge=0, description="Bounding box area in pixels")
    centroid: Centroid = Field(..., description="Centroid coordinates")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence [0,1]")


class MotorOutputs(BaseModel):
    """Motor control outputs for 5-engine AUV system.
    
    All values are normalized to [-1, 1] range:
    - out1, out2: Forward/turning engines (left/right)
    - out3, out4, out5: Vertical engines (identical)
    """
    out1: float = Field(..., ge=-1, le=1, description="Engine 1 output [-1,1]")
    out2: float = Field(..., ge=-1, le=1, description="Engine 2 output [-1,1]")
    out3: float = Field(..., ge=-1, le=1, description="Engine 3 output [-1,1]")
    out4: float = Field(..., ge=-1, le=1, description="Engine 4 output [-1,1]")
    out5: float = Field(..., ge=-1, le=1, description="Engine 5 output [-1,1]")


class FrameSize(BaseModel):
    """Video frame dimensions."""
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")


class ChaseOutput(BaseModel):
    """Complete chase system output for a single frame.
    
    Contains timestamp, frame info, detection result (if any), and motor commands.
    If no detection above confidence threshold, detection is null and motors are zero.
    """
    ts: int = Field(..., description="Timestamp in milliseconds since epoch")
    frameSize: FrameSize = Field(..., description="Video frame dimensions")
    detection: Optional[Detection] = Field(None, description="Detection result or null if none")
    motors: MotorOutputs = Field(..., description="Motor control outputs")

    class Config:
        json_encoders = {
            # Ensure proper JSON serialization
        }


class ChaseConfig(BaseModel):
    """Configuration for chase system."""
    video_source: str = Field(..., description="Video source (camera index, file path, RTSP URL)")
    model_path: str = Field(..., description="Path to YOLO model weights (.pt file)")
    target_area: Optional[float] = Field(None, gt=0, description="Target bounding box area (pixels)")
    chase_distance: Optional[float] = Field(None, gt=0, description="Chase distance (alternative to target_area)")
    confidence_threshold: float = Field(0.25, ge=0, le=1, description="Detection confidence threshold")
    calibration_k: float = Field(1e6, gt=0, description="Calibration constant for chase_distance to target_area conversion")
    
    # Control gains
    k_f: float = Field(0.5, description="Forward/distance gain")
    k_t: float = Field(0.5, description="Turning gain") 
    k_y: float = Field(1.0, description="Vertical gain")

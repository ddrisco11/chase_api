# Vision-Based AUV Control System for Object Tracking and Following

David Driscoll — Summer 2025  
Woods Hole Oceanographic Institution — Applied Ocean Physics and Engineering

## Objective

Develop a basic AUV control system to detect and chase objects autonomously.

## Deliverable

A versatile API system that receives video input, detects objects using an AI model, and outputs motor commands according to a five-engine control specification. The system can be used through a command-line tool, a Python API, or a WebSocket service. It can receive inputs in a variety of formats and is highly customizable.

## Future Additions

- Additional computer vision elements: more robust options for tracking, such as zero-shot object detection
- Cross-vehicle compatibility: a system to integrate with vehicles of different engine structures dynamically
- More robust control specs: implement more complex control elements such as PID

## Detailed Overview

The Vision-Based AUV Control System is designed to enable an Autonomous Underwater Vehicle (AUV) to see, track, and follow objects on its own. At its core, the system combines underwater video input with artificial intelligence to detect objects of interest and translate those detections into movement commands for the vehicle’s motors. This allows the AUV to maintain a steady pursuit of a moving target without direct human control.

The system is built around a flexible API (Application Programming Interface) that makes it easy to use in different ways. Operators or developers can interact with it through:

- Command-line tool for quick testing and standalone use
- Python API for programmatic integration into larger projects
- WebSocket service for real-time communication with other systems

This versatility ensures that the system can adapt to research, testing, and real-world applications without being locked into a single method of operation.

### How It Works in Practice

1. Video capture: the AUV receives a video feed from its onboard camera or another video source.
2. Object detection: an AI model processes the video to identify and locate objects.
3. Decision-making: the system calculates where the object is in relation to the AUV—whether it is too far, too close, off to the side, or above/below.
4. Motor control: based on these calculations, the system generates signals for the AUV’s five engines. Two engines control forward/backward movement and turning, while three engines adjust vertical position (up/down).
5. Continuous adjustment: this process repeats in real time, allowing the AUV to smoothly follow its target.

### Benefits and Applications

- Hands-free tracking: once a target is identified, the AUV can autonomously follow it, reducing the need for constant operator input.
- Flexible integration: the system is lightweight and can be embedded into existing AUV frameworks or used as a standalone module.
- Customizable behavior: parameters such as sensitivity, chase distance, and movement responsiveness can be easily adjusted.

## Technical Specifications

The Chase API is a lightweight, real-time control system that enables an Autonomous Underwater Vehicle (AUV) to autonomously track and follow targets using computer vision. It integrates object detection, target selection, and motor command generation into a modular and extensible package.

## API Interfaces

The Chase API is designed for maximum flexibility, offering three primary modes of integration.

### Command-Line Tool (CLI)

Run standalone experiments with webcams, RTSP streams, or video files. Supports adjustable confidence thresholds, chase parameters, and output formats (JSON, CSV).

```bash
python -m chase_api.runner \
    --video-source 0 \
    --model-path yolov10n.pt \
    --target-area 12000 \
    --confidence-threshold 0.25
```

Additional examples:

```bash
# Using chase distance instead of target area
python -m chase_api.runner \
    --video-source 0 \
    --model-path yolov10n.pt \
    --chase-distance 2.0 \
    --confidence-threshold 0.5

# Output in CSV format
python -m chase_api.runner \
    --video-source test_video.mp4 \
    --model-path yolov10n.pt \
    --target-area 8000 \
    --output-format csv

# Custom control gains
python -m chase_api.runner \
    --video-source "rtsp://192.168.1.100/stream" \
    --model-path yolov10n.pt \
    --target-area 15000 \
    --k-f 0.7 \
    --k-t 0.3 \
    --k-y 1.2
```

### Python API

Provides both a high-level callback interface and a low-level generator interface for programmatic use.

```python
from chase_api import run_chase, chase_stream, ChaseOutput

def handle_result(output: ChaseOutput):
    print(output.motors, output.detection)

run_chase(
    video_source="0",
    model_path="yolov10n.pt",
    target_area=12000,
    confidence_threshold=0.25,
    on_result=handle_result
)

# Generator example
for result in chase_stream(
    video_source="test_video.mp4",
    model_path="yolov10n.pt",
    chase_distance=1.5,
    confidence_threshold=0.3
):
    # Custom processing
    pass
```

### WebSocket Service

Enables real-time remote integration with control systems. Results are streamed as JSON messages over a WebSocket connection.

```bash
uvicorn chase_api.service:app --host 0.0.0.0 --port 8088
```

Client connection example:

```javascript
const ws = new WebSocket(
    'ws://localhost:8088/ws/chase?' +
    'video_source=0&model_path=yolov10n.pt&target_area=12000'
);

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log(result.motors, result.detection);
};
```

Health check and API docs:
- Health: `http://localhost:8088/healthz`
- API Documentation: `http://localhost:8088/docs`

## Output Format

Every frame produces a structured JSON message:

```json
{
  "ts": 1755620000000,
  "frameSize": {"width": 1920, "height": 1080},
  "detection": {
    "bbox": [100, 100, 200, 200],
    "area": 10000.0,
    "centroid": {"x": 150.0, "y": 150.0},
    "confidence": 0.82
  },
  "motors": {
    "out1": 0.12,
    "out2": 0.07,
    "out3": -0.34,
    "out4": -0.34,
    "out5": -0.34
  }
}
```

Field descriptions:

- `ts`: timestamp in milliseconds since epoch
- `frameSize`: video frame dimensions
- `detection`: object detection result (null if no detection above threshold)
  - `bbox`: bounding box coordinates [x1, y1, x2, y2]
  - `area`: bounding box area in pixels
  - `centroid`: center point coordinates
  - `confidence`: detection confidence score [0, 1]
- `motors`: motor control outputs in the range [-1, 1]
  - `out1`, `out2`: forward/turning engines (left/right)
  - `out3`, `out4`, `out5`: vertical engines (identical)

## Testing & Validation

- Unit tests validate control logic (distance, turning, vertical alignment, clamping).
- Integration tests cover end-to-end behavior, error handling, and schema validation.

Run tests with:

```bash
python -m pytest tests/ -v
```

## Dependencies

- Ultralytics YOLO (`ultralytics`) for object detection
- OpenCV (`opencv-python`) for video input and frame processing
- FastAPI & Uvicorn for WebSocket service support
- Pydantic for schema validation and serialization
- NumPy for mathematical operations
- websockets for client support

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  YOLO Detection  │───▶│  Target Selection│
│  (Camera/File)  │    │   (inference.py) │    │   (largest area) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  JSON Output    │◀───│ Motor Control    │◀───│   Centroid &    │
│ (ChaseOutput)   │    │  (control.py)    │    │   Area Calc     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  WebSocket API  │    │   Python API     │
│  (service.py)   │    │  (runner.py)     │
└─────────────────┘    └──────────────────┘
```

## Troubleshooting

Common issues:

1. Model loading fails
   ```
   Error: Failed to load model: [Errno 2] No such file or directory: 'yolov10n.pt'
   ```
   Solution: ensure model file exists and path is correct. Download from Ultralytics if needed.

2. Video source not found
   ```
   Error: Failed to open video source: 0
   ```
   Solution: check camera is connected, not in use by other apps, or try a different index.

3. Permission denied on camera
   ```
   Error: Failed to setup capture: Permission denied
   ```
   Solution: grant camera permissions to terminal/application.

4. WebSocket connection refused
   ```
   Error: Connection refused to localhost:8088
   ```
   Solution: ensure service is running with `python -m chase_api.service`.

Performance tips:

- Use lower resolution video sources for better performance
- Adjust confidence threshold to reduce false detections
- Use appropriate target area size for your application
- Consider frame rate vs. processing trade-offs

Debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

```bash
python -c "from chase_api.inference import test_inference; test_inference('yolov10n.pt')"
```

## Control System Specification

Input parameters:

- `bbox_area`: area of detected object’s bounding box
- `target_area`: desired area (proxy for chase distance)
- `(cx, cy)`: centroid of bounding box
- `(W, H)`: frame dimensions

Error calculations:

```
E_size = (bbox_area / target_area) - 1     # Distance error
X_norm = (cx / W) - 0.5                    # Horizontal offset
Y_norm = (cy / H) - 0.5                    # Vertical offset
```

Motor outputs:

```
out1 = k_f * E_size + k_t * (-X_norm)
out2 = k_f * E_size + k_t * (X_norm)
out3 = out4 = out5 = k_y * (-Y_norm)
```

Where:

- `k_f`: forward gain (distance control)
- `k_t`: turning gain (left/right alignment)
- `k_y`: vertical gain

All outputs are clamped to the range [-1, 1]. This ensures the AUV moves forward/backward to maintain distance, turns left/right to keep the target horizontally centered, and adjusts up/down to keep the target vertically centered.

## Installation & Setup

The repository can be cloned and installed directly from GitHub:

```bash
# Clone the repository
git clone https://github.com/ddrisco11/chase_api.git
cd chase_api

# Install dependencies
pip install -r requirements.txt
```

Alternatively, when used inside a larger workspace, navigate to the module directory before installing:

```bash
cd apps/auvctl/chase_api
pip install -r requirements.txt
```

## License

This module is part of the AutonomyV1 AUV control system.

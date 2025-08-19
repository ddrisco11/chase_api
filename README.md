# Chase API

A minimal, self-contained target chasing system using object detection and motor control for AUV (Autonomous Underwater Vehicle) applications.

## Overview

The Chase API provides real-time target chasing capabilities by:
- Running YOLO object detection on video streams
- Computing motor outputs according to the Control System Specification
- Streaming results in real-time for integration with existing control systems

## Features

- **Video Input**: Supports webcam, video files, and RTSP streams
- **Object Detection**: Uses Ultralytics YOLO models for target detection
- **Motor Control**: Implements 5-engine AUV control specification with proper clamping
- **Real-time Streaming**: WebSocket API for live integration
- **Python API**: Clean programmatic interface for custom applications
- **CLI Tool**: Command-line interface for standalone usage

## Quick Start

### Installation

```bash
cd apps/auvctl/chase_api
pip install -r requirements.txt
```

### CLI Usage

```bash
# Basic usage with webcam and target area
python -m chase_api.runner \
    --video-source 0 \
    --model-path yolov10n.pt \
    --target-area 12000 \
    --confidence-threshold 0.25

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

### Python API Usage

```python
from chase_api import run_chase, chase_stream, ChaseOutput

# Method 1: High-level API with callback
def handle_result(output: ChaseOutput):
    """Process each chase result."""
    if output.detection:
        print(f"Target detected: area={output.detection.area:.1f}, conf={output.detection.confidence:.3f}")
    else:
        print("No target detected")
    
    motors = output.motors
    print(f"Motors: [{motors.out1:.2f}, {motors.out2:.2f}, {motors.out3:.2f}, {motors.out4:.2f}, {motors.out5:.2f}]")

run_chase(
    video_source="0",
    model_path="yolov10n.pt",
    target_area=12000,
    confidence_threshold=0.25,
    on_result=handle_result
)

# Method 2: Generator API for custom processing
for result in chase_stream(
    video_source="test_video.mp4",
    model_path="yolov10n.pt", 
    chase_distance=1.5,
    confidence_threshold=0.3
):
    # Custom processing logic
    if result.detection:
        # Forward results to your control system
        send_to_controller(result.motors)
    
    # Log or store results
    with open("chase_log.jsonl", "a") as f:
        f.write(result.model_dump_json() + "\n")
```

### WebSocket Service

Start the FastAPI service:

```bash
# Start service on port 8088
python -m chase_api.service

# Or use uvicorn directly
uvicorn chase_api.service:app --host 0.0.0.0 --port 8088
```

Connect via WebSocket:

```javascript
// JavaScript client example
const ws = new WebSocket(
    'ws://localhost:8088/ws/chase?' + 
    'video_source=0&' +
    'model_path=yolov10n.pt&' +
    'target_area=12000&' +
    'confidence_threshold=0.25'
);

ws.onmessage = function(event) {
    const result = JSON.parse(event.data);
    console.log('Chase result:', result);
    
    // Forward motor commands to your system
    sendMotorCommands(result.motors);
};
```

Health check and API docs:
- Health: `http://localhost:8088/healthz`
- API Documentation: `http://localhost:8088/docs`

## Configuration

### Video Sources

- **Webcam**: Use camera index (e.g., `"0"`, `"1"`)
- **Video File**: Provide file path (e.g., `"video.mp4"`, `"/path/to/video.avi"`)
- **RTSP Stream**: Use RTSP URL (e.g., `"rtsp://192.168.1.100/stream"`)

### Target Specification

Choose one of:

1. **Target Area** (recommended): Specify desired bounding box area in pixels
   ```python
   target_area=12000  # For 640x480 video, roughly 1/25 of frame
   ```

2. **Chase Distance**: Specify desired distance, converted using calibration constant
   ```python
   chase_distance=2.0,
   calibration_k=1e6  # target_area = calibration_k / chase_distance
   ```

### Control Gains

- `k_f` (Forward gain): Controls response to size error (default: 0.5)
- `k_t` (Turning gain): Controls response to horizontal position error (default: 0.5)  
- `k_y` (Vertical gain): Controls response to vertical position error (default: 1.0)

## Output Format

### JSON Message Structure

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

### Field Descriptions

- `ts`: Timestamp in milliseconds since epoch
- `frameSize`: Video frame dimensions
- `detection`: Object detection result (null if no detection above threshold)
  - `bbox`: Bounding box coordinates [x1, y1, x2, y2]
  - `area`: Bounding box area in pixels
  - `centroid`: Center point coordinates  
  - `confidence`: Detection confidence score [0, 1]
- `motors`: Motor control outputs [-1, 1]
  - `out1`, `out2`: Forward/turning engines (left/right)
  - `out3`, `out4`, `out5`: Vertical engines (identical)

### Motor Control Logic

The system implements the Control System Specification:

```
E_size = (bbox_area / target_area) - 1
X_norm = (cx / W) - 0.5
Y_norm = (cy / H) - 0.5

out1 = k_f*E_size + k_t*(-X_norm)  # Left engine
out2 = k_f*E_size + k_t*(X_norm)   # Right engine
out3 = out4 = out5 = k_y*(-Y_norm) # Vertical engines
```

All outputs are clamped to [-1, 1] range.

## Integration Examples

### With existing control-api

```javascript
// In your Node.js control-api
const WebSocket = require('ws');

const chaseWs = new WebSocket('ws://localhost:8088/ws/chase?video_source=0&model_path=yolov10n.pt&target_area=12000');

chaseWs.on('message', (data) => {
    const chaseResult = JSON.parse(data);
    
    // Forward to existing control system
    publishToControlSystem({
        timestamp: chaseResult.ts,
        motors: chaseResult.motors,
        hasTarget: chaseResult.detection !== null
    });
    
    // Send to frontend via existing WebSocket
    broadcastToClients({
        type: 'chase_update',
        data: chaseResult
    });
});
```

### As Python Library

```python
from chase_api import ChaseOutput, chase_stream
import threading
import queue

class ChaseController:
    def __init__(self, video_source, model_path):
        self.output_queue = queue.Queue()
        self.running = False
        
    def start_chase(self):
        def chase_worker():
            for result in chase_stream(
                video_source=self.video_source,
                model_path=self.model_path,
                target_area=12000
            ):
                if not self.running:
                    break
                self.output_queue.put(result)
        
        self.running = True
        self.chase_thread = threading.Thread(target=chase_worker)
        self.chase_thread.start()
    
    def get_latest_result(self) -> ChaseOutput:
        return self.output_queue.get()
```

## Testing

Run the test suite:

```bash
cd apps/auvctl/chase_api
python -m pytest tests/ -v

# Or use unittest
python -m unittest discover tests/
```

### Test Coverage

- **Unit Tests** (`test_control.py`): Motor control logic validation
  - Centered target behavior
  - Left/right turning response
  - Vertical movement response
  - Output clamping
  - Custom gain effects

- **Integration Tests** (`test_integration.py`): End-to-end pipeline validation
  - Mock detection processing
  - Schema validation
  - Error handling
  - No-detection scenarios

## Dependencies

- `ultralytics==8.2.103`: YOLO object detection
- `opencv-python>=4.8,<5`: Video capture and processing
- `fastapi>=0.110,<1`: WebSocket service
- `uvicorn>=0.23,<1`: ASGI server
- `pydantic>=2.6,<3`: Data validation and serialization
- `numpy>=1.24,<2`: Numerical computations
- `websockets>=12.0,<13`: WebSocket client support

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

### Common Issues

1. **Model Loading Fails**
   ```
   Error: Failed to load model: [Errno 2] No such file or directory: 'yolov10n.pt'
   ```
   Solution: Ensure model file exists and path is correct. Download from Ultralytics if needed.

2. **Video Source Not Found**
   ```
   Error: Failed to open video source: 0
   ```
   Solution: Check camera is connected, not in use by other apps, or try different index.

3. **Permission Denied on Camera**
   ```
   Error: Failed to setup capture: Permission denied
   ```
   Solution: Grant camera permissions to terminal/application.

4. **WebSocket Connection Refused**
   ```
   Error: Connection refused to localhost:8088
   ```
   Solution: Ensure service is running with `python -m chase_api.service`.

### Performance Tips

- Use lower resolution video sources for better performance
- Adjust confidence threshold to reduce false detections
- Use appropriate target area size for your application
- Consider frame rate vs. processing trade-offs

### Debugging

Enable verbose output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Test model loading separately:
```bash
python -c "from chase_api.inference import test_inference; test_inference('yolov10n.pt')"
```

## License

This module is part of the AutonomyV1 AUV control system.

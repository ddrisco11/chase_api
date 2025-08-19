"""FastAPI WebSocket service for streaming chase outputs.

This module provides a minimal FastAPI service with:
- WebSocket endpoint for streaming ChaseOutput JSON messages
- Health check endpoint
- Query parameter configuration
"""

import json
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.responses import JSONResponse

try:
    from .inference import chase_stream
    from .schemas import ChaseOutput
except ImportError:
    # Handle running as script directly
    from inference import chase_stream
    from schemas import ChaseOutput


app = FastAPI(
    title="Chase API Service",
    description="Real-time target chasing system using object detection",
    version="1.0.0"
)


@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": "chase-api"}
    )


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Chase API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/chase",
            "health": "/healthz"
        },
        "description": "Real-time target chasing using object detection and motor control"
    }


@app.websocket("/ws/chase")
async def websocket_chase(
    websocket: WebSocket,
    video_source: str = Query(..., description="Video source (camera index, file path, RTSP URL)"),
    model_path: str = Query(..., description="Path to YOLO model weights (.pt file)"),
    target_area: Optional[float] = Query(None, gt=0, description="Target bounding box area (pixels)"),
    chase_distance: Optional[float] = Query(None, gt=0, description="Chase distance (alternative to target_area)"),
    confidence_threshold: float = Query(0.25, ge=0, le=1, description="Detection confidence threshold"),
    calibration_k: float = Query(1e6, gt=0, description="Calibration constant for distance to area conversion"),
    k_f: float = Query(0.5, description="Forward/distance gain"),
    k_t: float = Query(0.5, description="Turning gain"),
    k_y: float = Query(1.0, description="Vertical gain")
):
    """WebSocket endpoint for streaming chase outputs.
    
    Accepts query parameters for configuration and streams ChaseOutput JSON messages
    at video frame rate. Each message contains timestamp, frame info, detection
    result (if any), and motor control outputs.
    
    Example usage:
        ws://localhost:8088/ws/chase?video_source=0&model_path=yolov10n.pt&target_area=12000&confidence_threshold=0.25
    """
    await websocket.accept()
    
    # Validate parameters
    if target_area is None and chase_distance is None:
        await websocket.close(code=1003, reason="Must provide either target_area or chase_distance")
        return
        
    try:
        print(f"[service] Starting chase stream - source: {video_source}, model: {model_path}")
        
        # Create chase stream generator
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
        
        # Stream chase outputs via WebSocket
        async for chase_output in async_chase_stream(stream):
            try:
                # Convert to JSON and send
                message = chase_output.model_dump_json()
                await websocket.send_text(message)
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)  # ~100 FPS max
                
            except WebSocketDisconnect:
                print("[service] Client disconnected")
                break
            except Exception as e:
                print(f"[service] Error sending message: {e}")
                break
                
    except Exception as e:
        error_msg = f"Chase stream error: {str(e)}"
        print(f"[service] {error_msg}")
        try:
            await websocket.close(code=1011, reason=error_msg)
        except:
            pass
    finally:
        print("[service] WebSocket connection closed")


async def async_chase_stream(stream_generator):
    """Convert synchronous chase stream to async generator.
    
    Args:
        stream_generator: Synchronous generator from chase_stream()
        
    Yields:
        ChaseOutput objects asynchronously
    """
    import threading
    import queue
    
    # Use a queue to bridge sync/async
    output_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    exception_holder = [None]
    
    def producer():
        """Producer thread that feeds the queue."""
        try:
            for chase_output in stream_generator:
                if stop_event.is_set():
                    break
                try:
                    output_queue.put(chase_output, timeout=1.0)
                except queue.Full:
                    # Drop frames if consumer can't keep up
                    continue
        except Exception as e:
            exception_holder[0] = e
        finally:
            output_queue.put(None)  # Sentinel to signal end
    
    # Start producer thread
    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    
    try:
        while True:
            # Wait for next item
            while True:
                try:
                    item = output_queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    # Check if producer thread died with exception
                    if exception_holder[0] is not None:
                        raise exception_holder[0]
                    # Check if producer thread finished
                    if not producer_thread.is_alive():
                        item = None
                        break
                    await asyncio.sleep(0.01)
            
            if item is None:
                break  # End of stream
                
            yield item
            
    finally:
        # Signal producer to stop
        stop_event.set()
        
        # Wait for producer thread to finish
        producer_thread.join(timeout=2.0)


@app.get("/test")
async def test_endpoint(
    model_path: str = Query(..., description="Path to YOLO model for testing")
):
    """Test endpoint to validate model loading."""
    try:
        from .inference import InferenceEngine
        
        # Try to load the model
        engine = InferenceEngine(model_path)
        
        return {
            "status": "success",
            "message": f"Model loaded successfully: {model_path}",
            "model_path": model_path
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load model: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Chase API service...")
    print("WebSocket endpoint: ws://localhost:8088/ws/chase")
    print("Health check: http://localhost:8088/healthz")
    print("API docs: http://localhost:8088/docs")
    
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8088,
        reload=False,
        log_level="info"
    )

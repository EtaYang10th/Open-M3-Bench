# server.py
import fnmatch
import os
import base64
import cv2
import time
import threading
from io import BytesIO
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

from mcp.server.fastmcp import FastMCP
from ultralytics import YOLO

# Add this near the top of server.py with other imports
import os.path
import sys
import logging
import contextlib
import logging
import sys
import contextlib
import signal
import atexit

# Set up logging configuration - add this near the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("yolo_service.log"),
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger('yolo_service')

# Global variables for camera control
camera_running = False
camera_thread = None
detection_results = []
camera_last_access_time = 0
CAMERA_INACTIVITY_TIMEOUT = 60  # Auto-shutdown after 60 seconds of inactivity

@contextlib.contextmanager
def redirect_stdout_to_stderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout

def camera_watchdog_thread():
    """Monitor thread that auto-stops the camera after inactivity"""
    global camera_running, camera_last_access_time
    
    logger.info("Camera watchdog thread started")
    
    while True:
        # Sleep for a short time to avoid excessive CPU usage
        time.sleep(5)
        
        # Check if camera is running
        if camera_running:
            current_time = time.time()
            elapsed_time = current_time - camera_last_access_time
            
            # If no access for more than the timeout, auto-stop
            if elapsed_time > CAMERA_INACTIVITY_TIMEOUT:
                logger.info(f"Auto-stopping camera after {elapsed_time:.1f} seconds of inactivity")
                stop_camera_detection()
        else:
            # If camera is not running, no need to check frequently
            time.sleep(10)


def load_image(image_source, is_path=False):
    """
    Load image from file path or base64 data
    
    Args:
        image_source: File path or base64 encoded image data
        is_path: Whether image_source is a file path
        
    Returns:
        PIL Image object
    """
    try:
        if is_path:
            # Load image from file path
            if os.path.exists(image_source):
                return Image.open(image_source)
            else:
                raise FileNotFoundError(f"Image file not found: {image_source}")
        else:
            # Load image from base64 data
            image_bytes = base64.b64decode(image_source)
            return Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")

# Create MCP server
mcp = FastMCP("YOLO_Service")

# Global model cache
models = {}

def get_model(model_name: str = "yolov8n.pt") -> YOLO:
    """Get or load YOLO model from any of the configured model directories"""
    if model_name in models:
        return models[model_name]
    
    # Try to find the model in any of the configured directories
    model_path = None
    for directory in CONFIG["model_dirs"]:
        potential_path = os.path.join(directory, model_name)
        if os.path.exists(potential_path):
            model_path = potential_path
            break
    
    if model_path is None:
        available = list_available_models()
        available_str = ", ".join(available) if available else "none"
        raise FileNotFoundError(f"Model '{model_name}' not found in any configured directories. Available models: {available_str}")
    
    # Load and cache the model - with stdout redirected
    logger.info(f"Loading model: {model_name} from {model_path}")
    with redirect_stdout_to_stderr():
        models[model_name] = YOLO(model_path)
    return models[model_name]

# Global configuration
CONFIG = {
    "model_dirs": [
        ".",  # Current directory
        "./models",  # Models subdirectory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),  # Absolute path to models
        # Add any other potential model directories here
    ]
}



# Add a new tool to get information about model directories
@mcp.tool()
def get_model_directories() -> Dict[str, Any]:
    """Get information about configured model directories and available models"""
    directories = []
    
    for directory in CONFIG["model_dirs"]:
        dir_info = {
            "path": directory,
            "exists": os.path.exists(directory),
            "is_directory": os.path.isdir(directory) if os.path.exists(directory) else False,
            "models": []
        }
        
        if dir_info["exists"] and dir_info["is_directory"]:
            for filename in os.listdir(directory):
                if filename.endswith(".pt"):
                    dir_info["models"].append(filename)
        
        directories.append(dir_info)
    
    return {
        "configured_directories": CONFIG["model_dirs"],
        "directory_details": directories,
        "available_models": list_available_models(),
        "loaded_models": list(models.keys())
    }

@mcp.tool()
def detect_objects(
    image_data: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    save_results: bool = False,
    is_path: bool = False
) -> Dict[str, Any]:
    """
    Detect objects in an image using YOLO
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        model_name: YOLO model name
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        is_path: Whether image_data is a file path
        
    Returns:
        Dictionary containing detection results
    """
    try:
        # Load image (supports path or base64)
        image = load_image(image_data, is_path=is_path)
        
        # Load model and perform detection - with stdout redirected
        model = get_model(model_name)
        with redirect_stdout_to_stderr():  # Ensure all YOLO outputs go to stderr
            results = model.predict(image, conf=confidence, save=save_results)
        
        # Format results
        formatted_results = []
        for result in results:
            boxes = result.boxes
            detections = []
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })
            
            formatted_results.append({
                "detections": detections,
                "image_shape": result.orig_shape
            })
        
        return {
            "results": formatted_results,
            "model_used": model_name,
            "total_detections": sum(len(r["detections"]) for r in formatted_results),
            "source": image_data if is_path else "base64_image"
        }
    except Exception as e:
        logger.error(f"Error in detect_objects: {str(e)}")
        return {
            "error": f"Failed to detect objects: {str(e)}",
            "source": image_data if is_path else "base64_image"
        }

@mcp.tool()
def segment_objects(
    image_data: str,
    model_name: str = "yolov11n-seg.pt",
    confidence: float = 0.25,
    save_results: bool = False,
    is_path: bool = False
) -> Dict[str, Any]:
    """
    Perform instance segmentation on an image using YOLO
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        model_name: YOLO segmentation model name
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        is_path: Whether image_data is a file path
        
    Returns:
        Dictionary containing segmentation results
    """
    try:
        # Load image (supports path or base64)
        image = load_image(image_data, is_path=is_path)
        
        # Load model and perform segmentation
        model = get_model(model_name)
        with redirect_stdout_to_stderr():  # Add this context manager
            results = model.predict(image, conf=confidence, save=save_results)
        
        # Format results
        formatted_results = []
        for result in results:
            if not hasattr(result, 'masks') or result.masks is None:
                continue
                
            boxes = result.boxes
            masks = result.masks
            segments = []
            
            for i in range(len(boxes)):
                box = boxes[i]
                mask = masks[i].data[0].cpu().numpy() if masks else None
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                segment = {
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                }
                
                if mask is not None:
                    # Convert binary mask to simplified format for API response
                    segment["mask"] = mask.tolist()
                
                segments.append(segment)
            
            formatted_results.append({
                "segments": segments,
                "image_shape": result.orig_shape
            })
        
        return {
            "results": formatted_results,
            "model_used": model_name,
            "total_segments": sum(len(r["segments"]) for r in formatted_results),
            "source": image_data if is_path else "base64_image"
        }
    except Exception as e:
        return {
            "error": f"Failed to segment objects: {str(e)}",
            "source": image_data if is_path else "base64_image"
        }


@mcp.tool()
def classify_image(
    image_data: str,
    model_name: str = "yolov11n-cls.pt",
    top_k: int = 5,
    save_results: bool = False,
    is_path: bool = False
) -> Dict[str, Any]:
    """
    Classify an image using YOLO classification model
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        model_name: YOLO classification model name
        top_k: Number of top categories to return
        save_results: Whether to save results to disk
        is_path: Whether image_data is a file path
        
    Returns:
        Dictionary containing classification results
    """
    try:
        # Load image (supports path or base64)
        image = load_image(image_data, is_path=is_path)
        
        # Load model and perform classification
        model = get_model(model_name)
        with redirect_stdout_to_stderr():  # Add this context manager
            results = model.predict(image, save=save_results)
        
        # Format results
        formatted_results = []
        for result in results:
            if not hasattr(result, 'probs') or result.probs is None:
                continue
                
            probs = result.probs
            top_indices = probs.top5
            top_probs = probs.top5conf.tolist()
            top_classes = [result.names[idx] for idx in top_indices]
            
            classifications = [
                {"class_id": int(idx), "class_name": name, "probability": float(prob)}
                for idx, name, prob in zip(top_indices[:top_k], top_classes[:top_k], top_probs[:top_k])
            ]
            
            formatted_results.append({
                "classifications": classifications,
                "image_shape": result.orig_shape
            })
        
        return {
            "results": formatted_results,
            "model_used": model_name,
            "top_k": top_k,
            "source": image_data if is_path else "base64_image"
        }
    except Exception as e:
        return {
            "error": f"Failed to classify image: {str(e)}",
            "source": image_data if is_path else "base64_image"
        }


@mcp.tool()
def track_objects(
    image_data: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    tracker: str = "bytetrack.yaml",
    save_results: bool = False
) -> Dict[str, Any]:
    """
    Track objects in an image sequence using YOLO
    
    Args:
        image_data: Base64 encoded image
        model_name: YOLO model name
        confidence: Detection confidence threshold
        tracker: Tracker name to use (e.g., 'bytetrack.yaml', 'botsort.yaml')
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary containing tracking results
    """
    # Decode Base64 image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Load model and perform tracking
    model = get_model(model_name)
    # Add redirect_stdout_to_stderr context manager
    with redirect_stdout_to_stderr():
        results = model.track(image, conf=confidence, tracker=tracker, save=save_results)
    
    # Format results
    formatted_results = []
    for result in results:
        if not hasattr(result, 'boxes') or result.boxes is None:
            continue
            
        boxes = result.boxes
        tracks = []
        
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Extract track ID (if any)
            track_id = int(box.id[0]) if box.id is not None else None
            
            track = {
                "box": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "track_id": track_id
            }
            
            tracks.append(track)
        
        formatted_results.append({
            "tracks": tracks,
            "image_shape": result.orig_shape
        })
    
    return {
        "results": formatted_results,
        "model_used": model_name,
        "tracker": tracker,
        "total_tracks": sum(len(r["tracks"]) for r in formatted_results)
    }

# 3. FIX train_model FUNCTION TO USE REDIRECTION:
@mcp.tool()
def train_model(
    dataset_path: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    name: str = "yolo_custom_model",
    project: str = "runs/train"
) -> Dict[str, Any]:
    """
    Train a YOLO model on a custom dataset
    
    Args:
        dataset_path: Path to YOLO format dataset
        model_name: Base model to start with
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        name: Name for the training run
        project: Project directory
        
    Returns:
        Dictionary containing training results
    """
    # Validate dataset path
    if not os.path.exists(dataset_path):
        return {"error": f"Dataset not found: {dataset_path}"}
    
    # Initialize model
    model = get_model(model_name)
    
    # Train model
    try:
        # Add redirect_stdout_to_stderr context manager
        with redirect_stdout_to_stderr():
            results = model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                name=name,
                project=project
            )
        
        # Get best model path
        best_model_path = os.path.join(project, name, "weights", "best.pt")
        
        return {
            "status": "success",
            "model_path": best_model_path,
            "epochs_completed": epochs,
            "final_metrics": {
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0))
            }
        }
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}

# 4. FIX validate_model FUNCTION TO USE REDIRECTION:
@mcp.tool()
def validate_model(
    model_path: str,
    data_path: str,
    imgsz: int = 640,
    batch: int = 16
) -> Dict[str, Any]:
    """
    Validate a YOLO model on a dataset
    
    Args:
        model_path: Path to YOLO model (.pt file)
        data_path: Path to YOLO format validation dataset
        imgsz: Image size for validation
        batch: Batch size
        
    Returns:
        Dictionary containing validation results
    """
    # Validate model path
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}
    
    # Validate dataset path
    if not os.path.exists(data_path):
        return {"error": f"Dataset not found: {data_path}"}
    
    # Load model
    try:
        model = get_model(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}
    
    # Validate model
    try:
        # Add redirect_stdout_to_stderr context manager
        with redirect_stdout_to_stderr():
            results = model.val(data=data_path, imgsz=imgsz, batch=batch)
        
        return {
            "status": "success",
            "metrics": {
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0))
            }
        }
    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}

# 5. FIX export_model FUNCTION TO USE REDIRECTION:
@mcp.tool()
def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640
) -> Dict[str, Any]:
    """
    Export a YOLO model to different formats
    
    Args:
        model_path: Path to YOLO model (.pt file)
        format: Export format (onnx, torchscript, openvino, etc.)
        imgsz: Image size for export
        
    Returns:
        Dictionary containing export results
    """
    # Validate model path
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}
    
    # Valid export formats
    valid_formats = [
        "torchscript", "onnx", "openvino", "engine", "coreml", "saved_model", 
        "pb", "tflite", "edgetpu", "tfjs", "paddle"
    ]
    
    if format not in valid_formats:
        return {"error": f"Invalid export format: {format}. Valid formats include: {', '.join(valid_formats)}"}
    
    # Load model
    try:
        model = get_model(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}
    
    # Export model
    try:
        # Add redirect_stdout_to_stderr context manager
        with redirect_stdout_to_stderr():
            export_path = model.export(format=format, imgsz=imgsz)
        
        return {
            "status": "success",
            "export_path": str(export_path),
            "format": format
        }
    except Exception as e:
        return {"error": f"Export failed: {str(e)}"}

# 6. ADD REDIRECTION TO get_model_info FUNCTION:
@mcp.resource("model_info/{model_name}")
def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a YOLO model
    
    Args:
        model_name: YOLO model name
        
    Returns:
        Dictionary containing model information
    """
    try:
        model = get_model(model_name)
        
        # Get model task
        task = 'detect'  # Default task
        if 'seg' in model_name:
            task = 'segment'
        elif 'pose' in model_name:
            task = 'pose'
        elif 'cls' in model_name:
            task = 'classify'
        elif 'obb' in model_name:
            task = 'obb'
        
        # Make sure any model property access that might trigger output is wrapped
        with redirect_stdout_to_stderr():
            yaml_str = str(model.yaml)
            pt_path = str(model.pt_path) if hasattr(model, 'pt_path') else None
            class_names = model.names
        
        # Get model info
        return {
            "model_name": model_name,
            "task": task,
            "yaml": yaml_str,
            "pt_path": pt_path,
            "class_names": class_names
        }
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

# 7. MODIFY list_available_models to use logging instead of print
@mcp.tool()
def list_available_models() -> List[str]:
    """List available YOLO models that actually exist on disk in any configured directory"""
    # Common YOLO model patterns
    model_patterns = [
        "yolov11*.pt", 
        "yolov8*.pt"
    ]
    
    # Find all existing models in all configured directories
    available_models = set()
    for directory in CONFIG["model_dirs"]:
        if not os.path.exists(directory):
            continue
            
        # Check for model files directly
        for filename in os.listdir(directory):
            if filename.endswith(".pt") and any(
                fnmatch.fnmatch(filename, pattern) for pattern in model_patterns
            ):
                available_models.add(filename)
    
    # Convert to sorted list
    result = sorted(list(available_models))
    
    if not result:
        # Replace print with logger
        logger.warning("No model files found in configured directories.")
        return ["No models available - download models to any of these directories: " + ", ".join(CONFIG["model_dirs"])]
    
    return result
@mcp.resource("model_info/{model_name}")
def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a YOLO model
    
    Args:
        model_name: YOLO model name
        
    Returns:
        Dictionary containing model information
    """
    try:
        model = get_model(model_name)
        
        # Get model task
        task = 'detect'  # Default task
        if 'seg' in model_name:
            task = 'segment'
        elif 'pose' in model_name:
            task = 'pose'
        elif 'cls' in model_name:
            task = 'classify'
        elif 'obb' in model_name:
            task = 'obb'
        
        # Get model info
        return {
            "model_name": model_name,
            "task": task,
            "yaml": str(model.yaml),
            "pt_path": str(model.pt_path) if hasattr(model, 'pt_path') else None,
            "class_names": model.names
        }
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

@mcp.tool()
def list_available_models() -> List[str]:
    """List available YOLO models that actually exist on disk in any configured directory"""
    # Common YOLO model patterns
    model_patterns = [
        "yolov11*.pt", 
        "yolov8*.pt"
    ]
    
    # Find all existing models in all configured directories
    available_models = set()
    for directory in CONFIG["model_dirs"]:
        if not os.path.exists(directory):
            continue
            
        # Check for model files directly
        for filename in os.listdir(directory):
            if filename.endswith(".pt") and any(
                fnmatch.fnmatch(filename, pattern) for pattern in model_patterns
            ):
                available_models.add(filename)
    
    # Convert to sorted list
    result = sorted(list(available_models))
    
    if not result:
        print("Warning: No model files found in configured directories.")
        return ["No models available - download models to any of these directories: " + ", ".join(CONFIG["model_dirs"])]
    
    return result



# Camera detection background thread
camera_thread = None
camera_running = False
detection_results = []

def camera_detection_thread(model_name, confidence, fps_limit=30, camera_id=0):
    """Background thread for camera detection"""
    global camera_running, detection_results
    
    # Load model
    try:
        with redirect_stdout_to_stderr():
            model = get_model(model_name)
        logger.info(f"Model {model_name} loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        camera_running = False
        detection_results.append({
            "timestamp": time.time(),
            "error": f"Failed to load model: {str(e)}",
            "detections": []
        })
        return
    
    # Rest of the function...
    # Try to open camera with multiple attempts and multiple camera IDs if necessary
    cap = None
    error_message = ""
    
    # Try camera IDs from 0 to 2
    for cam_id in range(3):
        try:
            logger.info(f"Attempting to open camera with ID {cam_id}...")
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                logger.info(f"Successfully opened camera {cam_id}")
                break
        except Exception as e:
            error_message = f"Error opening camera {cam_id}: {str(e)}"
            logger.error(error_message)
    
    # Check if any camera was successfully opened
    if cap is None or not cap.isOpened():
        logger.error("Error: Could not open any camera.")
        camera_running = False
        detection_results.append({
            "timestamp": time.time(),
            "error": "Failed to open camera. Make sure camera is connected and not in use by another application.",
            "camera_status": "unavailable",
            "detections": []
        })
        return
    
    # Get camera properties for diagnostics
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Camera properties: {width}x{height} at {fps} FPS")
    
    # Calculate frame interval based on fps_limit
    frame_interval = 1.0 / fps_limit
    frame_count = 0
    error_count = 0
    
    while camera_running:
        start_time = time.time()
        
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Error: Failed to capture frame (attempt {error_count+1}).")
                error_count += 1
                
                # Add error to detection results
                detection_results.append({
                    "timestamp": time.time(),
                    "error": f"Failed to capture frame (attempt {error_count})",
                    "camera_status": "error",
                    "detections": []
                })
                
                # If we have consistent failures, try to restart the camera
                if error_count >= 5:
                    logger.warning("Too many frame capture errors, attempting to restart camera...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(camera_id)
                    error_count = 0
                    if not cap.isOpened():
                        logger.error("Failed to reopen camera after errors.")
                        break
                
                time.sleep(1)  # Wait before trying again
                continue
            
            # Reset error count on successful frame capture
            error_count = 0
            frame_count += 1
            
            # Perform detection on frame
            with redirect_stdout_to_stderr():  # Add this context manager
                results = model.predict(frame, conf=confidence)
            
            # Update detection results (only keep the last 10)
            if len(detection_results) >= 10:
                detection_results.pop(0)
                
            # Format results
            for result in results:
                boxes = result.boxes
                detections = []
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": class_name
                    })
                
                detection_results.append({
                    "timestamp": time.time(),
                    "frame_count": frame_count,
                    "detections": detections,
                    "camera_status": "running",
                    "image_shape": result.orig_shape
                })
            
            # Log occasional status
            if frame_count % 30 == 0:
                logger.info(f"Camera running: processed {frame_count} frames")
                detection_count = sum(len(r.get("detections", [])) for r in detection_results if "detections" in r)
                logger.info(f"Total detections in current buffer: {detection_count}")
            
            # Limit FPS by waiting if necessary
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                
        except Exception as e:
            logger.error(f"Error in camera thread: {str(e)}")
            detection_results.append({
                "timestamp": time.time(),
                "error": f"Exception in camera processing: {str(e)}",
                "camera_status": "error",
                "detections": []
            })
            time.sleep(1)  # Wait before continuing
    
    # Clean up
    logger.info("Shutting down camera...")
    if cap is not None:
        cap.release()

@mcp.tool()
def start_camera_detection(
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    camera_id: int = 0
) -> Dict[str, Any]:
    """
    Start realtime object detection using the computer's camera
    
    Args:
        model_name: YOLO model name to use
        confidence: Detection confidence threshold
        camera_id: Camera device ID (0 is usually the default camera)
        
    Returns:
        Status of camera detection
    """
    global camera_thread, camera_running, detection_results, camera_last_access_time
    
    # Check if already running
    if camera_running:
        # Update last access time
        camera_last_access_time = time.time()
        return {"status": "success", "message": "Camera detection is already running"}
    
    # Clear previous results
    detection_results = []
    
    # First, try to check if OpenCV is properly installed
    try:
        cv2_version = cv2.__version__
        logger.info(f"OpenCV version: {cv2_version}")
    except Exception as e:
        logger.error(f"OpenCV not properly installed: {str(e)}")
        return {
            "status": "error",
            "message": f"OpenCV not properly installed: {str(e)}",
            "solution": "Please check OpenCV installation"
        }
    
    # Start detection thread
    camera_running = True
    camera_last_access_time = time.time()  # Update access time
    camera_thread = threading.Thread(
        target=camera_detection_thread,
        args=(model_name, confidence, 30, camera_id),
        daemon=True
    )
    camera_thread.start()
    
    # Add initial status to detection results
    detection_results.append({
        "timestamp": time.time(),
        "system_info": {
            "os": platform.system() if 'platform' in globals() else "Unknown",
            "opencv_version": cv2.__version__,
            "camera_id": camera_id
        },
        "camera_status": "starting",
        "detections": []
    })
    
    return {
        "status": "success",
        "message": f"Started camera detection using model {model_name}",
        "model": model_name,
        "confidence": confidence,
        "camera_id": camera_id,
        "auto_shutdown": f"Camera will auto-shutdown after {CAMERA_INACTIVITY_TIMEOUT} seconds of inactivity",
        "note": "If camera doesn't work, try different camera_id values (0, 1, or 2)"
    }


@mcp.tool()
def stop_camera_detection() -> Dict[str, Any]:
    """
    Stop realtime camera detection
    
    Returns:
        Status message
    """
    global camera_running
    
    if not camera_running:
        return {"status": "error", "message": "Camera detection is not running"}
    
    logger.info("Stopping camera detection by user request")
    camera_running = False
    
    # Wait for thread to terminate
    if camera_thread and camera_thread.is_alive():
        camera_thread.join(timeout=2.0)
    
    return {
        "status": "success",
        "message": "Stopped camera detection"
    }


@mcp.tool()
def get_camera_detections() -> Dict[str, Any]:
    """
    Get the latest detections from the camera
    
    Returns:
        Dictionary with recent detections
    """
    global detection_results, camera_thread, camera_last_access_time
    
    # Update the last access time whenever this function is called
    if camera_running:
        camera_last_access_time = time.time()
    
    # Check if thread is alive
    thread_alive = camera_thread is not None and camera_thread.is_alive()
    
    # If camera_running is True but thread is dead, there's an issue
    if camera_running and not thread_alive:
        return {
            "status": "error", 
            "message": "Camera thread has stopped unexpectedly",
            "is_running": False,
            "camera_status": "error",
            "thread_alive": thread_alive,
            "detections": detection_results,
            "count": len(detection_results),
            "solution": "Please try restart the camera with a different camera_id"
        }
    
    if not camera_running:
        return {
            "status": "error", 
            "message": "Camera detection is not running",
            "is_running": False,
            "camera_status": "stopped"
        }
    
    # Check for errors in detection results
    errors = [result.get("error") for result in detection_results if "error" in result]
    recent_errors = errors[-5:] if errors else []
    
    # Count actual detections
    detection_count = sum(len(result.get("detections", [])) for result in detection_results if "detections" in result)
    
    return {
        "status": "success",
        "is_running": camera_running,
        "thread_alive": thread_alive,
        "detections": detection_results,
        "count": len(detection_results),
        "total_detections": detection_count,
        "recent_errors": recent_errors if recent_errors else None,
        "camera_status": "error" if recent_errors else "running",
        "inactivity_timeout": {
            "seconds_remaining": int(CAMERA_INACTIVITY_TIMEOUT - (time.time() - camera_last_access_time)),
            "last_access": camera_last_access_time
        }
    }

def cleanup_resources():
    """Clean up resources when the server is shutting down"""
    global camera_running
    
    logger.info("Cleaning up resources...")
    
    # Stop camera if it's running
    if camera_running:
        logger.info("Shutting down camera during server exit")
        camera_running = False
        
        # Give the camera thread a moment to clean up
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=2.0)
    
    logger.info("Cleanup complete")
atexit.register(cleanup_resources)

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def start_watchdog():
    """Start the camera watchdog thread"""
    watchdog = threading.Thread(
        target=camera_watchdog_thread,
        daemon=True
    )
    watchdog.start()
    return watchdog

@mcp.tool()
def comprehensive_image_analysis(
    image_path: str,
    confidence: float = 0.25,
    save_results: bool = False
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on an image by combining multiple model results
    
    Args:
        image_path: Path to the image file
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary containing comprehensive analysis results
    """
    try:
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
        
        # Load image
        image = load_image(image_path, is_path=True)
        
        analysis_results = {}
        
        # 1. Object detection
        object_model = get_model("yolov11n.pt")
        with redirect_stdout_to_stderr():  # Add this context manager
            object_results = object_model.predict(image, conf=confidence, save=save_results)
        
        # Process object detection results
        detected_objects = []
        for result in object_results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                detected_objects.append({
                    "class_name": class_name,
                    "confidence": conf
                })
        analysis_results["objects"] = detected_objects
        
        # 2. Scene classification
        try:
            cls_model = get_model("yolov8n-cls.pt")
            with redirect_stdout_to_stderr():  # Add this context manager
                cls_results = cls_model.predict(image, save=False)
            
            scene_classifications = []
            for result in cls_results:
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    top_indices = probs.top5
                    top_probs = probs.top5conf.tolist()
                    top_classes = [result.names[idx] for idx in top_indices]
                    
                    for idx, name, prob in zip(top_indices[:3], top_classes[:3], top_probs[:3]):
                        scene_classifications.append({
                            "class_name": name,
                            "probability": float(prob)
                        })
            analysis_results["scene"] = scene_classifications
        except Exception as e:
            analysis_results["scene_error"] = str(e)
        
        # 3. Human pose detection
        try:
            pose_model = get_model("yolov8n-pose.pt")
            with redirect_stdout_to_stderr():  # Add this context manager
                pose_results = pose_model.predict(image, conf=confidence, save=False)
            
            detected_poses = []
            for result in pose_results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    boxes = result.boxes
                    keypoints = result.keypoints
                    
                    for i in range(len(boxes)):
                        box = boxes[i]
                        conf = float(box.conf[0])
                        detected_poses.append({
                            "person_confidence": conf,
                            "has_keypoints": keypoints[i].data.shape[1] if keypoints else 0
                        })
            analysis_results["poses"] = detected_poses
        except Exception as e:
            analysis_results["pose_error"] = str(e)
        
        # Rest of the function remains the same...
        # 4. Comprehensive task description
        tasks = []
        
        # Detect main objects
        main_objects = [obj["class_name"] for obj in detected_objects if obj["confidence"] > 0.5]
        if "person" in main_objects:
            tasks.append("Person Detection")
        
        # Check for weapon objects
        weapon_objects = ["sword", "knife", "katana", "gun", "pistol", "rifle"]
        weapons = [obj for obj in main_objects if any(weapon in obj.lower() for weapon in weapon_objects)]
        if weapons:
            tasks.append(f"Weapon Detection ({', '.join(weapons)})")
        
        # Count people
        person_count = main_objects.count("person")
        if person_count > 0:
            tasks.append(f"Person Count ({person_count} people)")
        
        # Pose analysis
        if "poses" in analysis_results and analysis_results["poses"]:
            tasks.append("Human Pose Analysis")
        
        # Scene classification
        if "scene" in analysis_results and analysis_results["scene"]:
            scene_types = [scene["class_name"] for scene in analysis_results["scene"][:2]]
            tasks.append(f"Scene Classification ({', '.join(scene_types)})")
        
        analysis_results["identified_tasks"] = tasks
        
        # Return comprehensive results
        return {
            "status": "success",
            "image_path": image_path,
            "analysis": analysis_results,
            "summary": "Tasks identified in the image: " + ", ".join(tasks) if tasks else "No clear tasks identified"
        }
    except Exception as e:
        return {
            "status": "error",
            "image_path": image_path,
            "error": f"Comprehensive analysis failed: {str(e)}"
        }


@mcp.tool()
def analyze_image_from_path(
    image_path: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    save_results: bool = False
) -> Dict[str, Any]:
    """
    Analyze image from file path using YOLO
    
    Args:
        image_path: Path to the image file
        model_name: YOLO model name
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary containing detection results
    """
    try:
        # Call detect_objects function with is_path=True
        return detect_objects(
            image_data=image_path,
            model_name=model_name,
            confidence=confidence,
            save_results=save_results,
            is_path=True
        )
    except Exception as e:
        return {
            "error": f"Failed to analyze image: {str(e)}",
            "image_path": image_path
        }

@mcp.tool()
def test_connection() -> Dict[str, Any]:
    """
    Test if YOLO MCP service is running properly
    
    Returns:
        Status information and available tools
    """
    return {
        "status": "YOLO MCP service is running normally",
        "available_models": list_available_models(),
        "available_tools": [
            "list_available_models", "detect_objects", "segment_objects", 
            "classify_image", "detect_poses", "detect_oriented_objects", 
            "track_objects", "train_model", "validate_model", 
            "export_model", "start_camera_detection", "stop_camera_detection", 
            "get_camera_detections", "test_connection",
            # Additional tools
            "analyze_image_from_path",
            "comprehensive_image_analysis"
        ],
        "new_features": [
            "Support for loading images directly from file paths",
            "Support for comprehensive image analysis with task identification",
            "All detection functions support both file paths and base64 data"
        ]
    }

# Modify the main execution section
if __name__ == "__main__":
    logger.info("Starting YOLO MCP service")
    
    # Start the camera watchdog thread
    watchdog_thread = start_watchdog()
    
    # Initialize and run server
    mcp.run(transport='stdio')
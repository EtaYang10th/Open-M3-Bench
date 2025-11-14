# server.py - CLI version (command return only)
import fnmatch
import os
import base64
import time
import threading
import json
import tempfile
import platform
from io import BytesIO
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

from mcp.server.fastmcp import FastMCP

# Set up logging configuration
import os.path
import sys
import logging
import contextlib
import signal
import atexit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("yolo_service.log"),
        logging.StreamHandler(sys.stderr)
    ]
)
camera_startup_status = None  # Will store error details if startup fails
camera_last_error = None
logger = logging.getLogger('yolo_service')

# Global variables for camera control
camera_running = False
camera_thread = None
detection_results = []
camera_last_access_time = 0
CAMERA_INACTIVITY_TIMEOUT = 60  # Auto-shutdown after 60 seconds of inactivity

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

# Modified function to just return the command string
def run_yolo_cli(command_args, capture_output=True, timeout=60):
    """
    Return the YOLO CLI command string without executing it
    
    Args:
        command_args: List of command arguments to pass to yolo CLI
        capture_output: Not used, kept for compatibility with original function
        timeout: Not used, kept for compatibility with original function
        
    Returns:
        Dictionary containing the command string
    """
    # Build the complete command
    cmd = ["yolo"] + command_args
    cmd_str = " ".join(cmd)
    
    # Log the command
    logger.info(f"Would run YOLO CLI command: {cmd_str}")
    
    # Return the command string in a similar structure as the original function
    return {
        "success": True,
        "command": cmd_str,
        "would_execute": True,
        "note": "CLI execution disabled, showing command only"
    }

# Create MCP server
mcp = FastMCP("YOLO_Service")

# Global configuration
CONFIG = {
    "model_dirs": [
        ".",  # Current directory
        "./models",  # Models subdirectory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    ]
}

# Function to save base64 data to temp file
def save_base64_to_temp(base64_data, prefix="image", suffix=".jpg"):
    """Save base64 encoded data to a temporary file and return the path"""
    try:
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        
        # Write data to file
        with os.fdopen(fd, 'wb') as temp_file:
            temp_file.write(image_data)
            
        return temp_path
    except Exception as e:
        logger.error(f"Error saving base64 to temp file: {str(e)}")
        raise ValueError(f"Failed to save base64 data: {str(e)}")

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
        "loaded_models": []  # No longer track loaded models with CLI approach
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
    Return the YOLO CLI command for object detection without executing it
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        model_name: YOLO model name
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        is_path: Whether image_data is a file path
        
    Returns:
        Dictionary containing command that would be executed
    """
    try:
        # Determine source path
        if is_path:
            source_path = image_data
            if not os.path.exists(source_path):
                return {
                    "error": f"Image file not found: {source_path}",
                    "source": source_path
                }
        else:
            # For base64, we would save to temp file, but we'll just indicate this
            source_path = "[temp_file_from_base64]"
        
        # Determine full model path
        model_path = None
        for directory in CONFIG["model_dirs"]:
            potential_path = os.path.join(directory, model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                break
        
        if model_path is None:
            available = list_available_models()
            available_str = ", ".join(available) if available else "none"
            return {
                "error": f"Model '{model_name}' not found in any configured directories. Available models: {available_str}",
                "source": image_data if is_path else "base64_image"
            }
        
        # Setup output directory for save_results
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_results")
        
        # Build YOLO CLI command
        cmd_args = [
            "detect",  # Task
            "predict",  # Mode
            f"model={model_path}",
            f"source={source_path}",
            f"conf={confidence}",
            "format=json",  # Request JSON output for parsing
        ]
        
        if save_results:
            cmd_args.append(f"project={output_dir}")
            cmd_args.append("save=True")
        else:
            cmd_args.append("save=False")
        
        # Get command string without executing
        result = run_yolo_cli(cmd_args)
        
        # Return command information
        return {
            "status": "command_generated",
            "model_used": model_name,
            "model_path": model_path,
            "source": source_path,
            "command": result["command"],
            "note": "Command generated but not executed - detection results would be returned from actual execution",
            "parameters": {
                "confidence": confidence,
                "save_results": save_results,
                "is_path": is_path,
                "output_dir": output_dir if save_results else None
            }
        }
            
    except Exception as e:
        logger.error(f"Error in detect_objects command generation: {str(e)}")
        return {
            "error": f"Failed to generate detection command: {str(e)}",
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
    Return the YOLO CLI command for segmentation without executing it
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        model_name: YOLO segmentation model name
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        is_path: Whether image_data is a file path
        
    Returns:
        Dictionary containing command that would be executed
    """
    try:
        # Determine source path
        if is_path:
            source_path = image_data
            if not os.path.exists(source_path):
                return {
                    "error": f"Image file not found: {source_path}",
                    "source": source_path
                }
        else:
            # For base64, we would save to temp file, but we'll just indicate this
            source_path = "[temp_file_from_base64]"
        
        # Determine full model path
        model_path = None
        for directory in CONFIG["model_dirs"]:
            potential_path = os.path.join(directory, model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                break
        
        if model_path is None:
            available = list_available_models()
            available_str = ", ".join(available) if available else "none"
            return {
                "error": f"Model '{model_name}' not found in any configured directories. Available models: {available_str}",
                "source": image_data if is_path else "base64_image"
            }
        
        # Setup output directory for save_results
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_results")
        
        # Build YOLO CLI command
        cmd_args = [
            "segment",  # Task
            "predict",  # Mode
            f"model={model_path}",
            f"source={source_path}",
            f"conf={confidence}",
            "format=json",  # Request JSON output for parsing
        ]
        
        if save_results:
            cmd_args.append(f"project={output_dir}")
            cmd_args.append("save=True")
        else:
            cmd_args.append("save=False")
        
        # Get command string without executing
        result = run_yolo_cli(cmd_args)
        
        # Return command information
        return {
            "status": "command_generated",
            "model_used": model_name,
            "model_path": model_path,
            "source": source_path,
            "command": result["command"],
            "note": "Command generated but not executed - segmentation results would be returned from actual execution",
            "parameters": {
                "confidence": confidence,
                "save_results": save_results,
                "is_path": is_path,
                "output_dir": output_dir if save_results else None
            }
        }
            
    except Exception as e:
        logger.error(f"Error in segment_objects command generation: {str(e)}")
        return {
            "error": f"Failed to generate segmentation command: {str(e)}",
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
    Return the YOLO CLI command for image classification without executing it
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        model_name: YOLO classification model name
        top_k: Number of top categories to return
        save_results: Whether to save results to disk
        is_path: Whether image_data is a file path
        
    Returns:
        Dictionary containing command that would be executed
    """
    try:
        # Determine source path
        if is_path:
            source_path = image_data
            if not os.path.exists(source_path):
                return {
                    "error": f"Image file not found: {source_path}",
                    "source": source_path
                }
        else:
            # For base64, we would save to temp file, but we'll just indicate this
            source_path = "[temp_file_from_base64]"
        
        # Determine full model path
        model_path = None
        for directory in CONFIG["model_dirs"]:
            potential_path = os.path.join(directory, model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                break
        
        if model_path is None:
            available = list_available_models()
            available_str = ", ".join(available) if available else "none"
            return {
                "error": f"Model '{model_name}' not found in any configured directories. Available models: {available_str}",
                "source": image_data if is_path else "base64_image"
            }
        
        # Setup output directory for save_results
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_results")
        
        # Build YOLO CLI command
        cmd_args = [
            "classify",  # Task
            "predict",  # Mode
            f"model={model_path}",
            f"source={source_path}",
            "format=json",  # Request JSON output for parsing
        ]
        
        if save_results:
            cmd_args.append(f"project={output_dir}")
            cmd_args.append("save=True")
        else:
            cmd_args.append("save=False")
        
        # Get command string without executing
        result = run_yolo_cli(cmd_args)
        
        # Return command information
        return {
            "status": "command_generated",
            "model_used": model_name,
            "model_path": model_path,
            "source": source_path,
            "command": result["command"],
            "note": "Command generated but not executed - classification results would be returned from actual execution",
            "parameters": {
                "top_k": top_k,
                "save_results": save_results,
                "is_path": is_path,
                "output_dir": output_dir if save_results else None
            }
        }
            
    except Exception as e:
        logger.error(f"Error in classify_image command generation: {str(e)}")
        return {
            "error": f"Failed to generate classification command: {str(e)}",
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
    Return the YOLO CLI command for object tracking without executing it
    
    Args:
        image_data: Base64 encoded image
        model_name: YOLO model name
        confidence: Detection confidence threshold
        tracker: Tracker name to use (e.g., 'bytetrack.yaml', 'botsort.yaml')
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary containing command that would be executed
    """
    try:
        # For base64, we would save to temp file, but we'll just indicate this
        source_path = "[temp_file_from_base64]"
        
        # Determine full model path
        model_path = None
        for directory in CONFIG["model_dirs"]:
            potential_path = os.path.join(directory, model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                break
        
        if model_path is None:
            available = list_available_models()
            available_str = ", ".join(available) if available else "none"
            return {
                "error": f"Model '{model_name}' not found in any configured directories. Available models: {available_str}"
            }
        
        # Setup output directory for save_results
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_track_results")
        
        # Build YOLO CLI command
        cmd_args = [
            "track",  # Combined task and mode for tracking
            f"model={model_path}",
            f"source={source_path}",
            f"conf={confidence}",
            f"tracker={tracker}",
            "format=json",  # Request JSON output for parsing
        ]
        
        if save_results:
            cmd_args.append(f"project={output_dir}")
            cmd_args.append("save=True")
        else:
            cmd_args.append("save=False")
        
        # Get command string without executing
        result = run_yolo_cli(cmd_args)
        
        # Return command information
        return {
            "status": "command_generated",
            "model_used": model_name,
            "model_path": model_path,
            "source": source_path,
            "command": result["command"],
            "note": "Command generated but not executed - tracking results would be returned from actual execution",
            "parameters": {
                "confidence": confidence,
                "tracker": tracker,
                "save_results": save_results,
                "output_dir": output_dir if save_results else None
            }
        }
            
    except Exception as e:
        logger.error(f"Error in track_objects command generation: {str(e)}")
        return {
            "error": f"Failed to generate tracking command: {str(e)}"
        }

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
    Return the YOLO CLI command for model training without executing it
    
    Args:
        dataset_path: Path to YOLO format dataset
        model_name: Base model to start with
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        name: Name for the training run
        project: Project directory
        
    Returns:
        Dictionary containing command that would be executed
    """
    # Validate dataset path
    if not os.path.exists(dataset_path):
        return {"error": f"Dataset not found: {dataset_path}"}
    
    # Determine full model path
    model_path = None
    for directory in CONFIG["model_dirs"]:
        potential_path = os.path.join(directory, model_name)
        if os.path.exists(potential_path):
            model_path = potential_path
            break
    
    if model_path is None:
        available = list_available_models()
        available_str = ", ".join(available) if available else "none"
        return {
            "error": f"Model '{model_name}' not found in any configured directories. Available models: {available_str}"
        }
    
    # Determine task type based on model name
    task = "detect"  # Default task
    if "seg" in model_name:
        task = "segment"
    elif "pose" in model_name:
        task = "pose"
    elif "cls" in model_name:
        task = "classify"
    elif "obb" in model_name:
        task = "obb"
    
    # Build YOLO CLI command
    cmd_args = [
        task,  # Task
        "train",  # Mode
        f"model={model_path}",
        f"data={dataset_path}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"name={name}",
        f"project={project}"
    ]
    
    # Get command string without executing
    result = run_yolo_cli(cmd_args)
    
    # Return command information
    return {
        "status": "command_generated",
        "model_used": model_name,
        "model_path": model_path,
        "command": result["command"],
        "note": "Command generated but not executed - training would start with actual execution",
        "parameters": {
            "dataset_path": dataset_path,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "name": name,
            "project": project,
            "task": task
        }
    }

@mcp.tool()
def validate_model(
    model_path: str,
    data_path: str,
    imgsz: int = 640,
    batch: int = 16
) -> Dict[str, Any]:
    """
    Return the YOLO CLI command for model validation without executing it
    
    Args:
        model_path: Path to YOLO model (.pt file)
        data_path: Path to YOLO format validation dataset
        imgsz: Image size for validation
        batch: Batch size
        
    Returns:
        Dictionary containing command that would be executed
    """
    # Validate model path
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}
    
    # Validate dataset path
    if not os.path.exists(data_path):
        return {"error": f"Dataset not found: {data_path}"}
    
    # Determine task type based on model name
    model_name = os.path.basename(model_path)
    task = "detect"  # Default task
    if "seg" in model_name:
        task = "segment"
    elif "pose" in model_name:
        task = "pose"
    elif "cls" in model_name:
        task = "classify"
    elif "obb" in model_name:
        task = "obb"
    
    # Build YOLO CLI command
    cmd_args = [
        task,  # Task
        "val",  # Mode
        f"model={model_path}",
        f"data={data_path}",
        f"imgsz={imgsz}",
        f"batch={batch}"
    ]
    
    # Get command string without executing
    result = run_yolo_cli(cmd_args)
    
    # Return command information
    return {
        "status": "command_generated",
        "model_path": model_path,
        "command": result["command"],
        "note": "Command generated but not executed - validation would begin with actual execution",
        "parameters": {
            "data_path": data_path,
            "imgsz": imgsz,
            "batch": batch,
            "task": task
        }
    }

@mcp.tool()
def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640
) -> Dict[str, Any]:
    """
    Return the YOLO CLI command for model export without executing it
    
    Args:
        model_path: Path to YOLO model (.pt file)
        format: Export format (onnx, torchscript, openvino, etc.)
        imgsz: Image size for export
        
    Returns:
        Dictionary containing command that would be executed
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
    
    # Build YOLO CLI command
    cmd_args = [
        "export",  # Combined task and mode for export
        f"model={model_path}",
        f"format={format}",
        f"imgsz={imgsz}"
    ]
    
    # Get command string without executing
    result = run_yolo_cli(cmd_args)
    
    # Return command information
    return {
        "status": "command_generated",
        "model_path": model_path,
        "command": result["command"],
        "note": "Command generated but not executed - export would begin with actual execution",
        "parameters": {
            "format": format,
            "imgsz": imgsz,
            "expected_output": f"{os.path.splitext(model_path)[0]}.{format}"
        }
    }

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
        logger.warning("No model files found in configured directories.")
        return ["No models available - download models to any of these directories: " + ", ".join(CONFIG["model_dirs"])]
    
    return result

@mcp.tool()
def start_camera_detection(
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    camera_id: int = 0
) -> Dict[str, Any]:
    """
    Return the YOLO CLI command for starting camera detection without executing it
    
    Args:
        model_name: YOLO model name to use
        confidence: Detection confidence threshold
        camera_id: Camera device ID (0 is usually the default camera)
        
    Returns:
        Dictionary containing command that would be executed
    """
    # Determine full model path
    model_path = None
    for directory in CONFIG["model_dirs"]:
        potential_path = os.path.join(directory, model_name)
        if os.path.exists(potential_path):
            model_path = potential_path
            break
    
    if model_path is None:
        available = list_available_models()
        available_str = ", ".join(available) if available else "none"
        return {
            "error": f"Model '{model_name}' not found in any configured directories. Available models: {available_str}"
        }
    
    # Determine task type based on model name
    task = "detect"  # Default task
    if "seg" in model_name:
        task = "segment"
    elif "pose" in model_name:
        task = "pose"
    elif "cls" in model_name:
        task = "classify"
    
    # Build YOLO CLI command
    cmd_args = [
        task,  # Task
        "predict",  # Mode
        f"model={model_path}",
        f"source={camera_id}",  # Camera source ID
        f"conf={confidence}",
        "format=json",
        "save=False",  # Don't save frames by default
        "show=True"   # Show GUI window for camera view
    ]
    
    # Get command string without executing
    result = run_yolo_cli(cmd_args)
    
    # Return command information
    return {
        "status": "command_generated",
        "model_used": model_name,
        "model_path": model_path,
        "command": result["command"],
        "note": "Command generated but not executed - camera would start with actual execution",
        "parameters": {
            "confidence": confidence,
            "camera_id": camera_id,
            "task": task
        }
    }

@mcp.tool()
def stop_camera_detection() -> Dict[str, Any]:
    """
    Simulate stopping camera detection (no actual command to execute)
    
    Returns:
        Information message
    """
    return {
        "status": "command_generated",
        "message": "To stop camera detection, close the YOLO window or press 'q' in the terminal",
        "note": "Since commands are not executed, no actual camera is running"
    }

@mcp.tool()
def get_camera_detections() -> Dict[str, Any]:
    """
    Simulate getting latest camera detections (no actual command to execute)
    
    Returns:
        Information message
    """
    return {
        "status": "command_generated",
        "message": "Camera detections would be returned here if a camera was running",
        "note": "Since commands are not executed, no camera is running and no detections are available"
    }

@mcp.tool()
def comprehensive_image_analysis(
    image_path: str,
    confidence: float = 0.25,
    save_results: bool = False
) -> Dict[str, Any]:
    """
    Return the YOLO CLI commands for comprehensive image analysis without executing them
    
    Args:
        image_path: Path to the image file
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary containing commands that would be executed
    """
    if not os.path.exists(image_path):
        return {"error": f"Image file not found: {image_path}"}
    
    commands = []
    
    # 1. Object detection
    detect_result = detect_objects(
        image_data=image_path,
        model_name="yolov11n.pt",
        confidence=confidence,
        save_results=save_results,
        is_path=True
    )
    if "command" in detect_result:
        commands.append({
            "task": "object_detection",
            "command": detect_result["command"]
        })
    
    # 2. Scene classification
    try:
        cls_result = classify_image(
            image_data=image_path,
            model_name="yolov8n-cls.pt",
            top_k=3,
            save_results=save_results,
            is_path=True
        )
        if "command" in cls_result:
            commands.append({
                "task": "classification",
                "command": cls_result["command"]
            })
    except Exception as e:
        logger.error(f"Error generating classification command: {str(e)}")
    
    # 3. Pose detection if available
    for directory in CONFIG["model_dirs"]:
        pose_model_path = os.path.join(directory, "yolov8n-pose.pt")
        if os.path.exists(pose_model_path):
            # Build YOLO CLI command for pose detection
            cmd_args = [
                "pose",  # Task
                "predict",  # Mode
                f"model={pose_model_path}",
                f"source={image_path}",
                f"conf={confidence}",
                "format=json",
            ]
            
            if save_results:
                output_dir = os.path.join(tempfile.gettempdir(), "yolo_pose_results")
                cmd_args.append(f"project={output_dir}")
                cmd_args.append("save=True")
            else:
                cmd_args.append("save=False")
            
            result = run_yolo_cli(cmd_args)
            
            commands.append({
                "task": "pose_detection",
                "command": result["command"]
            })
            break
    
    return {
        "status": "commands_generated",
        "image_path": image_path,
        "commands": commands,
        "note": "Commands generated but not executed - comprehensive analysis would occur with actual execution",
        "parameters": {
            "confidence": confidence,
            "save_results": save_results
        }
    }

@mcp.tool()
def analyze_image_from_path(
    image_path: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    save_results: bool = False
) -> Dict[str, Any]:
    """
    Return the YOLO CLI command for image analysis without executing it
    
    Args:
        image_path: Path to the image file
        model_name: YOLO model name
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary containing command that would be executed
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
            "error": f"Failed to generate analysis command: {str(e)}",
            "image_path": image_path
        }

@mcp.tool()
def test_connection() -> Dict[str, Any]:
    """
    Test if YOLO CLI service is available
    
    Returns:
        Status information and available tools
    """
    # Build a simple YOLO CLI version command
    cmd_args = ["--version"]
    result = run_yolo_cli(cmd_args)
    
    return {
        "status": "YOLO CLI command generator is running",
        "command_mode": "Command generation only, no execution",
        "version_command": result["command"],
        "available_models": list_available_models(),
        "available_tools": [
            "list_available_models", "detect_objects", "segment_objects", 
            "classify_image", "track_objects", "train_model", "validate_model", 
            "export_model", "start_camera_detection", "stop_camera_detection", 
            "get_camera_detections", "test_connection",
            # Additional tools
            "analyze_image_from_path",
            "comprehensive_image_analysis"
        ],
        "note": "This service only generates YOLO commands without executing them"
    }

# Modify the main execution section
if __name__ == "__main__":
    import platform
    
    logger.info("Starting YOLO CLI command generator service")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info("⚠️ Commands will be generated but NOT executed")
    
    # Initialize and run server
    logger.info("Starting MCP server...")
    mcp.run(transport='stdio')
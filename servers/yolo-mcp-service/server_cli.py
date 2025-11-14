# server.py - CLI version
import fnmatch
import os
import base64
import cv2
import time
import threading
import subprocess
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

# New function to run YOLO CLI commands
def run_yolo_cli(command_args, capture_output=True, timeout=60):
    """
    Run YOLO CLI command and return the results
    
    Args:
        command_args: List of command arguments to pass to yolo CLI
        capture_output: Whether to capture and return command output
        timeout: Command timeout in seconds
        
    Returns:
        Command output or success status
    """
    # Build the complete command
    cmd = ["yolo"] + command_args
    
    # Log the command
    logger.info(f"Running YOLO CLI command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            timeout=timeout
        )
        
        # Check for errors
        if result.returncode != 0:
            logger.error(f"YOLO CLI command failed with code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return {
                "success": False,
                "error": result.stderr,
                "command": " ".join(cmd),
                "returncode": result.returncode
            }
        
        # Return the result
        if capture_output:
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
        else:
            return {"success": True, "command": " ".join(cmd)}
        
    except subprocess.TimeoutExpired:
        logger.error(f"YOLO CLI command timed out after {timeout} seconds")
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds",
            "command": " ".join(cmd)
        }
    except Exception as e:
        logger.error(f"Error running YOLO CLI command: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "command": " ".join(cmd)
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
    Detect objects in an image using YOLO CLI
    
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
        # Determine source path
        if is_path:
            source_path = image_data
            if not os.path.exists(source_path):
                return {
                    "error": f"Image file not found: {source_path}",
                    "source": source_path
                }
        else:
            # Save base64 data to temp file
            source_path = save_base64_to_temp(image_data)
        
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
        
        # Setup output directory if saving results
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_results")
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
        
        # Run YOLO CLI command
        result = run_yolo_cli(cmd_args)
        
        # Clean up temp file if we created one
        if not is_path:
            try:
                os.remove(source_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {source_path}: {str(e)}")
        
        # Check for command success
        if not result["success"]:
            return {
                "error": f"YOLO CLI command failed: {result.get('error', 'Unknown error')}",
                "command": result.get("command", ""),
                "source": image_data if is_path else "base64_image"
            }
        
        # Parse JSON output from stdout
        try:
            # Try to find JSON in the output
            json_start = result["stdout"].find("{")
            json_end = result["stdout"].rfind("}")
            
            if json_start >= 0 and json_end > json_start:
                json_str = result["stdout"][json_start:json_end+1]
                detection_data = json.loads(json_str)
            else:
                # If no JSON found, create a basic response with info from stderr
                return {
                    "results": [],
                    "model_used": model_name,
                    "total_detections": 0,
                    "source": image_data if is_path else "base64_image",
                    "command_output": result["stderr"]
                }
            
            # Format results
            formatted_results = []
            
            # Parse detection data from YOLO JSON output
            if "predictions" in detection_data:
                detections = []
                
                for pred in detection_data["predictions"]:
                    # Extract box coordinates
                    box = pred.get("box", {})
                    x1, y1, x2, y2 = box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)
                    
                    # Extract class information
                    confidence = pred.get("confidence", 0)
                    class_name = pred.get("name", "unknown")
                    class_id = pred.get("class", -1)
                    
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })
                
                # Get image dimensions if available
                image_shape = [
                    detection_data.get("width", 0),
                    detection_data.get("height", 0)
                ]
                
                formatted_results.append({
                    "detections": detections,
                    "image_shape": image_shape
                })
            
            return {
                "results": formatted_results,
                "model_used": model_name,
                "total_detections": sum(len(r["detections"]) for r in formatted_results),
                "source": image_data if is_path else "base64_image",
                "save_dir": output_dir if save_results else None
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from YOLO output: {e}")
            logger.error(f"Output: {result['stdout']}")
            
            return {
                "error": f"Failed to parse YOLO results: {str(e)}",
                "command": result.get("command", ""),
                "source": image_data if is_path else "base64_image",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
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
    Perform instance segmentation on an image using YOLO CLI
    
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
        # Determine source path
        if is_path:
            source_path = image_data
            if not os.path.exists(source_path):
                return {
                    "error": f"Image file not found: {source_path}",
                    "source": source_path
                }
        else:
            # Save base64 data to temp file
            source_path = save_base64_to_temp(image_data)
        
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
        
        # Setup output directory if saving results
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_results")
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
        
        # Run YOLO CLI command
        result = run_yolo_cli(cmd_args)
        
        # Clean up temp file if we created one
        if not is_path:
            try:
                os.remove(source_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {source_path}: {str(e)}")
        
        # Check for command success
        if not result["success"]:
            return {
                "error": f"YOLO CLI command failed: {result.get('error', 'Unknown error')}",
                "command": result.get("command", ""),
                "source": image_data if is_path else "base64_image"
            }
        
        # Parse JSON output from stdout
        try:
            # Try to find JSON in the output
            json_start = result["stdout"].find("{")
            json_end = result["stdout"].rfind("}")
            
            if json_start >= 0 and json_end > json_start:
                json_str = result["stdout"][json_start:json_end+1]
                segmentation_data = json.loads(json_str)
            else:
                # If no JSON found, create a basic response with info from stderr
                return {
                    "results": [],
                    "model_used": model_name,
                    "total_segments": 0,
                    "source": image_data if is_path else "base64_image",
                    "command_output": result["stderr"]
                }
            
            # Format results
            formatted_results = []
            
            # Parse segmentation data from YOLO JSON output
            if "predictions" in segmentation_data:
                segments = []
                
                for pred in segmentation_data["predictions"]:
                    # Extract box coordinates
                    box = pred.get("box", {})
                    x1, y1, x2, y2 = box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)
                    
                    # Extract class information
                    confidence = pred.get("confidence", 0)
                    class_name = pred.get("name", "unknown")
                    class_id = pred.get("class", -1)
                    
                    segment = {
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    }
                    
                    # Extract mask if available
                    if "mask" in pred:
                        segment["mask"] = pred["mask"]
                    
                    segments.append(segment)
                
                # Get image dimensions if available
                image_shape = [
                    segmentation_data.get("width", 0),
                    segmentation_data.get("height", 0)
                ]
                
                formatted_results.append({
                    "segments": segments,
                    "image_shape": image_shape
                })
            
            return {
                "results": formatted_results,
                "model_used": model_name,
                "total_segments": sum(len(r["segments"]) for r in formatted_results),
                "source": image_data if is_path else "base64_image",
                "save_dir": output_dir if save_results else None
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from YOLO output: {e}")
            logger.error(f"Output: {result['stdout']}")
            
            return {
                "error": f"Failed to parse YOLO results: {str(e)}",
                "command": result.get("command", ""),
                "source": image_data if is_path else "base64_image",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }
            
    except Exception as e:
        logger.error(f"Error in segment_objects: {str(e)}")
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
    Classify an image using YOLO classification model via CLI
    
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
        # Determine source path
        if is_path:
            source_path = image_data
            if not os.path.exists(source_path):
                return {
                    "error": f"Image file not found: {source_path}",
                    "source": source_path
                }
        else:
            # Save base64 data to temp file
            source_path = save_base64_to_temp(image_data)
        
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
        
        # Setup output directory if saving results
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_results")
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
        
        # Run YOLO CLI command
        result = run_yolo_cli(cmd_args)
        
        # Clean up temp file if we created one
        if not is_path:
            try:
                os.remove(source_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {source_path}: {str(e)}")
        
        # Check for command success
        if not result["success"]:
            return {
                "error": f"YOLO CLI command failed: {result.get('error', 'Unknown error')}",
                "command": result.get("command", ""),
                "source": image_data if is_path else "base64_image"
            }
        
        # Parse JSON output from stdout
        try:
            # Try to find JSON in the output
            json_start = result["stdout"].find("{")
            json_end = result["stdout"].rfind("}")
            
            if json_start >= 0 and json_end > json_start:
                json_str = result["stdout"][json_start:json_end+1]
                classification_data = json.loads(json_str)
            else:
                # If no JSON found, create a basic response with info from stderr
                return {
                    "results": [],
                    "model_used": model_name,
                    "top_k": top_k,
                    "source": image_data if is_path else "base64_image",
                    "command_output": result["stderr"]
                }
            
            # Format results
            formatted_results = []
            
            # Parse classification data from YOLO JSON output
            if "predictions" in classification_data:
                classifications = []
                predictions = classification_data["predictions"]
                
                # Predictions could be an array of classifications
                for i, pred in enumerate(predictions[:top_k]):
                    class_name = pred.get("name", f"class_{i}")
                    confidence = pred.get("confidence", 0)
                    
                    classifications.append({
                        "class_id": i,
                        "class_name": class_name,
                        "probability": confidence
                    })
                
                # Get image dimensions if available
                image_shape = [
                    classification_data.get("width", 0),
                    classification_data.get("height", 0)
                ]
                
                formatted_results.append({
                    "classifications": classifications,
                    "image_shape": image_shape
                })
            
            return {
                "results": formatted_results,
                "model_used": model_name,
                "top_k": top_k,
                "source": image_data if is_path else "base64_image",
                "save_dir": output_dir if save_results else None
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from YOLO output: {e}")
            logger.error(f"Output: {result['stdout']}")
            
            return {
                "error": f"Failed to parse YOLO results: {str(e)}",
                "command": result.get("command", ""),
                "source": image_data if is_path else "base64_image",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }
            
    except Exception as e:
        logger.error(f"Error in classify_image: {str(e)}")
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
    Track objects in an image sequence using YOLO CLI
    
    Args:
        image_data: Base64 encoded image
        model_name: YOLO model name
        confidence: Detection confidence threshold
        tracker: Tracker name to use (e.g., 'bytetrack.yaml', 'botsort.yaml')
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary containing tracking results
    """
    try:
        # Save base64 data to temp file
        source_path = save_base64_to_temp(image_data)
        
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
        
        # Setup output directory if saving results
        output_dir = os.path.join(tempfile.gettempdir(), "yolo_track_results")
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
        
        # Run YOLO CLI command
        result = run_yolo_cli(cmd_args)
        
        # Clean up temp file
        try:
            os.remove(source_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {source_path}: {str(e)}")
        
        # Check for command success
        if not result["success"]:
            return {
                "error": f"YOLO CLI command failed: {result.get('error', 'Unknown error')}",
                "command": result.get("command", ""),
            }
        
        # Parse JSON output from stdout
        try:
            # Try to find JSON in the output
            json_start = result["stdout"].find("{")
            json_end = result["stdout"].rfind("}")
            
            if json_start >= 0 and json_end > json_start:
                json_str = result["stdout"][json_start:json_end+1]
                tracking_data = json.loads(json_str)
            else:
                # If no JSON found, create a basic response
                return {
                    "results": [],
                    "model_used": model_name,
                    "tracker": tracker,
                    "total_tracks": 0,
                    "command_output": result["stderr"]
                }
            
            # Format results
            formatted_results = []
            
            # Parse tracking data from YOLO JSON output
            if "predictions" in tracking_data:
                tracks = []
                
                for pred in tracking_data["predictions"]:
                    # Extract box coordinates
                    box = pred.get("box", {})
                    x1, y1, x2, y2 = box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)
                    
                    # Extract class and tracking information
                    confidence = pred.get("confidence", 0)
                    class_name = pred.get("name", "unknown")
                    class_id = pred.get("class", -1)
                    track_id = pred.get("id", -1)
                    
                    track = {
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                        "track_id": track_id
                    }
                    
                    tracks.append(track)
                
                # Get image dimensions if available
                image_shape = [
                    tracking_data.get("width", 0),
                    tracking_data.get("height", 0)
                ]
                
                formatted_results.append({
                    "tracks": tracks,
                    "image_shape": image_shape
                })
            
            return {
                "results": formatted_results,
                "model_used": model_name,
                "tracker": tracker,
                "total_tracks": sum(len(r["tracks"]) for r in formatted_results),
                "save_dir": output_dir if save_results else None
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from YOLO output: {e}")
            logger.error(f"Output: {result['stdout']}")
            
            return {
                "error": f"Failed to parse YOLO results: {str(e)}",
                "command": result.get("command", ""),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "")
            }
            
    except Exception as e:
        logger.error(f"Error in track_objects: {str(e)}")
        return {
            "error": f"Failed to track objects: {str(e)}"
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
    Train a YOLO model on a custom dataset using CLI
    
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
    
    # Create project directory if it doesn't exist
    if not os.path.exists(project):
        os.makedirs(project)
    
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
    
    # Run YOLO CLI command - with longer timeout
    logger.info(f"Starting model training with {epochs} epochs - this may take a while...")
    result = run_yolo_cli(cmd_args, timeout=epochs * 300)  # 5 minutes per epoch
    
    # Check for command success
    if not result["success"]:
        return {
            "error": f"Training failed: {result.get('error', 'Unknown error')}",
            "command": result.get("command", ""),
            "stderr": result.get("stderr", "")
        }
    
    # Determine path to best model weights
    best_model_path = os.path.join(project, name, "weights", "best.pt")
    
    # Determine metrics from stdout if possible
    metrics = {}
    try:
        # Look for metrics in output
        stdout = result.get("stdout", "")
        
        # Extract metrics from training output
        import re
        precision_match = re.search(r"Precision: ([\d\.]+)", stdout)
        recall_match = re.search(r"Recall: ([\d\.]+)", stdout)
        map50_match = re.search(r"mAP50: ([\d\.]+)", stdout)
        map_match = re.search(r"mAP50-95: ([\d\.]+)", stdout)
        
        if precision_match:
            metrics["precision"] = float(precision_match.group(1))
        if recall_match:
            metrics["recall"] = float(recall_match.group(1))
        if map50_match:
            metrics["mAP50"] = float(map50_match.group(1))
        if map_match:
            metrics["mAP50-95"] = float(map_match.group(1))
    except Exception as e:
        logger.warning(f"Failed to parse metrics from training output: {str(e)}")
    
    return {
        "status": "success",
        "model_path": best_model_path,
        "epochs_completed": epochs,
        "final_metrics": metrics,
        "training_log_sample": result.get("stdout", "")[:1000] + "..." if len(result.get("stdout", "")) > 1000 else result.get("stdout", "")
    }

@mcp.tool()
def validate_model(
    model_path: str,
    data_path: str,
    imgsz: int = 640,
    batch: int = 16
) -> Dict[str, Any]:
    """
    Validate a YOLO model on a dataset using CLI
    
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
    
    # Run YOLO CLI command
    result = run_yolo_cli(cmd_args, timeout=300)  # 5 minute timeout
    
    # Check for command success
    if not result["success"]:
        return {
            "error": f"Validation failed: {result.get('error', 'Unknown error')}",
            "command": result.get("command", ""),
            "stderr": result.get("stderr", "")
        }
    
    # Extract metrics from validation output
    metrics = {}
    try:
        stdout = result.get("stdout", "")
        
        import re
        precision_match = re.search(r"Precision: ([\d\.]+)", stdout)
        recall_match = re.search(r"Recall: ([\d\.]+)", stdout)
        map50_match = re.search(r"mAP50: ([\d\.]+)", stdout)
        map_match = re.search(r"mAP50-95: ([\d\.]+)", stdout)
        
        if precision_match:
            metrics["precision"] = float(precision_match.group(1))
        if recall_match:
            metrics["recall"] = float(recall_match.group(1))
        if map50_match:
            metrics["mAP50"] = float(map50_match.group(1))
        if map_match:
            metrics["mAP50-95"] = float(map_match.group(1))
    except Exception as e:
        logger.warning(f"Failed to parse metrics from validation output: {str(e)}")
    
    return {
        "status": "success",
        "metrics": metrics,
        "validation_output": result.get("stdout", "")[:1000] + "..." if len(result.get("stdout", "")) > 1000 else result.get("stdout", "")
    }

@mcp.tool()
def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640
) -> Dict[str, Any]:
    """
    Export a YOLO model to different formats using CLI
    
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
    
    # Build YOLO CLI command
    cmd_args = [
        "export",  # Combined task and mode for export
        f"model={model_path}",
        f"format={format}",
        f"imgsz={imgsz}"
    ]
    
    # Run YOLO CLI command
    result = run_yolo_cli(cmd_args, timeout=300)  # 5 minute timeout
    
    # Check for command success
    if not result["success"]:
        return {
            "error": f"Export failed: {result.get('error', 'Unknown error')}",
            "command": result.get("command", ""),
            "stderr": result.get("stderr", "")
        }
    
    # Try to determine export path
    export_path = None
    try:
        # Model path without extension
        base_path = os.path.splitext(model_path)[0]
        
        # Expected export paths based on format
        format_extensions = {
            "torchscript": ".torchscript",
            "onnx": ".onnx",
            "openvino": "_openvino_model",
            "engine": ".engine",
            "coreml": ".mlmodel",
            "saved_model": "_saved_model",
            "pb": ".pb",
            "tflite": ".tflite",
            "edgetpu": "_edgetpu.tflite",
            "tfjs": "_web_model",
            "paddle": "_paddle_model"
        }
        
        expected_ext = format_extensions.get(format, "")
        expected_path = base_path + expected_ext
        
        # Check if the exported file exists
        if os.path.exists(expected_path) or os.path.isdir(expected_path):
            export_path = expected_path
    except Exception as e:
        logger.warning(f"Failed to determine export path: {str(e)}")
    
    return {
        "status": "success",
        "export_path": export_path,
        "format": format,
        "export_output": result.get("stdout", "")[:1000] + "..." if len(result.get("stdout", "")) > 1000 else result.get("stdout", "")
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

# Camera detection functions using CLI instead of Python API
def camera_detection_thread(model_name, confidence, fps_limit=30, camera_id=0):
    """Background thread for camera detection using YOLO CLI"""
    global camera_running, detection_results, camera_last_access_time, camera_startup_status, camera_last_error
    
    try:
        # Create a unique directory for camera results
        output_dir = os.path.join(tempfile.gettempdir(), f"yolo_camera_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine full model path
        model_path = None
        for directory in CONFIG["model_dirs"]:
            potential_path = os.path.join(directory, model_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                break
        
        if model_path is None:
            error_msg = f"Model {model_name} not found in any configured directories"
            logger.error(error_msg)
            camera_running = False
            camera_startup_status = {
                "success": False,
                "error": error_msg,
                "timestamp": time.time()
            }
            detection_results.append({
                "timestamp": time.time(),
                "error": f"Failed to load model: Model not found",
                "camera_status": "error",
                "detections": []
            })
            return
        
        # Log camera start
        logger.info(f"Starting camera detection with model {model_name}, camera ID {camera_id}")
        detection_results.append({
            "timestamp": time.time(),
            "system_info": {
                "os": platform.system() if 'platform' in globals() else "Unknown",
                "camera_id": camera_id
            },
            "camera_status": "starting",
            "detections": []
        })
        
        # Determine task type based on model name
        task = "detect"  # Default task
        if "seg" in model_name:
            task = "segment"
        elif "pose" in model_name:
            task = "pose"
        elif "cls" in model_name:
            task = "classify"
        
        # Build YOLO CLI command
        base_cmd_args = [
            task,  # Task
            "predict",  # Mode
            f"model={model_path}",
            f"source={camera_id}",  # Camera source ID
            f"conf={confidence}",
            "format=json",
            "save=False",  # Don't save frames by default
            "show=False"   # Don't show GUI window
        ]
        
        # First verify YOLO command is available
        logger.info("Verifying YOLO CLI availability before starting camera...")
        check_cmd = ["yolo", "--version"]
        try:
            check_result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=10
            )
            
            if check_result.returncode != 0:
                error_msg = f"YOLO CLI check failed with code {check_result.returncode}: {check_result.stderr}"
                logger.error(error_msg)
                camera_running = False
                camera_startup_status = {
                    "success": False,
                    "error": error_msg,
                    "timestamp": time.time()
                }
                detection_results.append({
                    "timestamp": time.time(),
                    "error": error_msg,
                    "camera_status": "error",
                    "detections": []
                })
                return
                
            logger.info(f"YOLO CLI is available: {check_result.stdout.strip()}")
        except Exception as e:
            error_msg = f"Error checking YOLO CLI: {str(e)}"
            logger.error(error_msg)
            camera_running = False
            camera_startup_status = {
                "success": False,
                "error": error_msg,
                "timestamp": time.time()
            }
            detection_results.append({
                "timestamp": time.time(),
                "error": error_msg,
                "camera_status": "error",
                "detections": []
            })
            return
            
        # Set up subprocess for ongoing camera capture
        process = None
        frame_count = 0
        error_count = 0
        start_time = time.time()
        
        # Start YOLO CLI process
        cmd_str = "yolo " + " ".join(base_cmd_args)
        logger.info(f"Starting YOLO CLI process: {cmd_str}")
        
        try:
            process = subprocess.Popen(
                ["yolo"] + base_cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )
            
            # Wait a moment to check if the process immediately fails
            time.sleep(1)
            if process.poll() is not None:
                error_msg = f"YOLO process failed to start (exit code {process.returncode})"
                stderr_output = process.stderr.read()
                logger.error(f"{error_msg} - STDERR: {stderr_output}")
                
                camera_running = False
                camera_startup_status = {
                    "success": False,
                    "error": error_msg,
                    "stderr": stderr_output,
                    "timestamp": time.time()
                }
                detection_results.append({
                    "timestamp": time.time(),
                    "error": error_msg,
                    "stderr": stderr_output,
                    "camera_status": "error",
                    "detections": []
                })
                return
                
            # Process started successfully
            camera_startup_status = {
                "success": True,
                "timestamp": time.time()
            }
            
            # Handle camera stream
            while camera_running:
                # Read output line from process
                stdout_line = process.stdout.readline().strip()
                
                if not stdout_line:
                    # Check if process is still running
                    if process.poll() is not None:
                        error_msg = f"YOLO process ended unexpectedly with code {process.returncode}"
                        stderr_output = process.stderr.read()
                        logger.error(f"{error_msg} - STDERR: {stderr_output}")
                        
                        camera_running = False
                        camera_last_error = {
                            "error": error_msg,
                            "stderr": stderr_output,
                            "timestamp": time.time()
                        }
                        detection_results.append({
                            "timestamp": time.time(),
                            "error": error_msg,
                            "camera_status": "error",
                            "stderr": stderr_output,
                            "detections": []
                        })
                        break
                    
                    time.sleep(0.1)  # Short sleep to avoid CPU spin
                    continue
                
                # Try to parse JSON output from YOLO
                try:
                    # Find JSON in the output line
                    json_start = stdout_line.find("{")
                    if json_start >= 0:
                        json_str = stdout_line[json_start:]
                        detection_data = json.loads(json_str)
                        
                        frame_count += 1
                        
                        # Process detection data
                        if "predictions" in detection_data:
                            detections = []
                            
                            for pred in detection_data["predictions"]:
                                # Extract box coordinates
                                box = pred.get("box", {})
                                x1, y1, x2, y2 = box.get("x1", 0), box.get("y1", 0), box.get("x2", 0), box.get("y2", 0)
                                
                                # Extract class information
                                confidence = pred.get("confidence", 0)
                                class_name = pred.get("name", "unknown")
                                class_id = pred.get("class", -1)
                                
                                detections.append({
                                    "box": [x1, y1, x2, y2],
                                    "confidence": confidence,
                                    "class_id": class_id,
                                    "class_name": class_name
                                })
                            
                            # Update detection results (keep only the last 10)
                            if len(detection_results) >= 10:
                                detection_results.pop(0)
                                
                            # Get image dimensions if available
                            image_shape = [
                                detection_data.get("width", 0),
                                detection_data.get("height", 0)
                            ]
                            
                            detection_results.append({
                                "timestamp": time.time(),
                                "frame_count": frame_count,
                                "detections": detections,
                                "camera_status": "running",
                                "image_shape": image_shape
                            })
                            
                            # Update last access time when processing frames
                            camera_last_access_time = time.time()
                            
                            # Log occasional status
                            if frame_count % 30 == 0:
                                fps = frame_count / (time.time() - start_time)
                                logger.info(f"Camera running: processed {frame_count} frames ({fps:.1f} FPS)")
                                detection_count = sum(len(r.get("detections", [])) for r in detection_results if "detections" in r)
                                logger.info(f"Total detections in current buffer: {detection_count}")
                        
                except json.JSONDecodeError:
                    # Not all lines will be valid JSON, that's normal
                    pass
                except Exception as e:
                    error_msg = f"Error processing camera output: {str(e)}"
                    logger.warning(error_msg)
                    error_count += 1
                    
                    if error_count > 10:
                        logger.error("Too many processing errors, stopping camera")
                        camera_running = False
                        camera_last_error = {
                            "error": "Too many processing errors",
                            "timestamp": time.time()
                        }
                        break
                
        except Exception as e:
            error_msg = f"Error in camera process management: {str(e)}"
            logger.error(error_msg)
            camera_running = False
            camera_startup_status = {
                "success": False,
                "error": error_msg,
                "timestamp": time.time()
            }
            detection_results.append({
                "timestamp": time.time(),
                "error": error_msg,
                "camera_status": "error",
                "detections": []
            })
            return
            
    except Exception as e:
        error_msg = f"Error in camera thread: {str(e)}"
        logger.error(error_msg)
        camera_running = False
        camera_startup_status = {
            "success": False,
            "error": error_msg,
            "timestamp": time.time()
        }
        detection_results.append({
            "timestamp": time.time(),
            "error": error_msg,
            "camera_status": "error",
            "detections": []
        })
    
    finally:
        # Clean up
        logger.info("Shutting down camera...")
        camera_running = False
        
        if process is not None and process.poll() is None:
            try:
                # Terminate process
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if terminate doesn't work
            except Exception as e:
                logger.error(f"Error terminating YOLO process: {str(e)}")
        
        logger.info("Camera detection stopped")


@mcp.tool()
def start_camera_detection(
    model_name: str = "yolov8n.pt",
    confidence: float = 0.25,
    camera_id: int = 0
) -> Dict[str, Any]:
    """
    Start realtime object detection using the computer's camera via YOLO CLI
    
    Args:
        model_name: YOLO model name to use
        confidence: Detection confidence threshold
        camera_id: Camera device ID (0 is usually the default camera)
        
    Returns:
        Status of camera detection
    """
    global camera_thread, camera_running, detection_results, camera_last_access_time, camera_startup_status, camera_last_error
    
    # Reset status variables
    camera_startup_status = None
    camera_last_error = None
    
    # Check if already running
    if camera_running:
        # Update last access time
        camera_last_access_time = time.time()
        return {"status": "success", "message": "Camera detection is already running"}
    
    # Clear previous results
    detection_results = []
    
    # First check if YOLO CLI is available
    try:
        version_check = run_yolo_cli(["--version"], timeout=10)
        if not version_check["success"]:
            return {
                "status": "error",
                "message": "YOLO CLI not available or not properly installed",
                "details": version_check.get("error", "Unknown error"),
                "solution": "Please make sure the 'yolo' command is in your PATH"
            }
    except Exception as e:
        error_msg = f"Error checking YOLO CLI: {str(e)}"
        logger.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "solution": "Please make sure the 'yolo' command is in your PATH"
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
    
    # Give the thread a moment to initialize and potentially fail
    time.sleep(2)
    
    # Check if the thread has reported any startup issues
    if camera_startup_status and not camera_startup_status.get("success", False):
        # Camera thread encountered an error during startup
        return {
            "status": "error",
            "message": "Camera detection failed to start",
            "details": camera_startup_status,
            "solution": "Check logs for detailed error information"
        }
    
    # Thread is running, camera should be starting
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
    global detection_results, camera_thread, camera_last_access_time, camera_startup_status, camera_last_error
    
    # Update the last access time whenever this function is called
    if camera_running:
        camera_last_access_time = time.time()
    
    # Check if thread is alive
    thread_alive = camera_thread is not None and camera_thread.is_alive()
    
    # If camera_running is False, check if we have startup status information
    if not camera_running and camera_startup_status and not camera_startup_status.get("success", False):
        return {
            "status": "error", 
            "message": "Camera detection failed to start",
            "is_running": False,
            "camera_status": "error",
            "startup_error": camera_startup_status,
            "solution": "Check logs for detailed error information"
        }
    
    # If camera_running is True but thread is dead, there's an issue
    if camera_running and not thread_alive:
        return {
            "status": "error", 
            "message": "Camera thread has stopped unexpectedly",
            "is_running": False,
            "camera_status": "error",
            "thread_alive": thread_alive,
            "last_error": camera_last_error,
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


@mcp.tool()
def comprehensive_image_analysis(
    image_path: str,
    confidence: float = 0.25,
    save_results: bool = False
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on an image by combining multiple CLI model results
    
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
        
        analysis_results = {}
        
        # 1. Object detection
        logger.info("Running object detection for comprehensive analysis")
        object_result = detect_objects(
            image_data=image_path,
            model_name="yolov11n.pt",
            confidence=confidence,
            save_results=save_results,
            is_path=True
        )
        
        # Process object detection results
        detected_objects = []
        if "results" in object_result and object_result["results"]:
            for result in object_result["results"]:
                for obj in result.get("detections", []):
                    detected_objects.append({
                        "class_name": obj.get("class_name", "unknown"),
                        "confidence": obj.get("confidence", 0)
                    })
        analysis_results["objects"] = detected_objects
        
        # 2. Scene classification
        try:
            logger.info("Running classification for comprehensive analysis")
            cls_result = classify_image(
                image_data=image_path,
                model_name="yolov8n-cls.pt",
                top_k=3,
                save_results=False,
                is_path=True
            )
            
            scene_classifications = []
            if "results" in cls_result and cls_result["results"]:
                for result in cls_result["results"]:
                    for cls in result.get("classifications", []):
                        scene_classifications.append({
                            "class_name": cls.get("class_name", "unknown"),
                            "probability": cls.get("probability", 0)
                        })
            analysis_results["scene"] = scene_classifications
        except Exception as e:
            logger.error(f"Error during scene classification: {str(e)}")
            analysis_results["scene_error"] = str(e)
        
        # 3. Human pose detection (if pose model is available)
        try:
            # Check if pose model exists
            pose_model_exists = False
            for directory in CONFIG["model_dirs"]:
                if os.path.exists(os.path.join(directory, "yolov8n-pose.pt")):
                    pose_model_exists = True
                    break
            
            if pose_model_exists:
                logger.info("Running pose detection for comprehensive analysis")
                # Build YOLO CLI command for pose detection
                cmd_args = [
                    "pose",  # Task
                    "predict",  # Mode
                    f"model=yolov8n-pose.pt",
                    f"source={image_path}",
                    f"conf={confidence}",
                    "format=json",
                ]
                
                result = run_yolo_cli(cmd_args)
                
                if result["success"]:
                    # Parse JSON output
                    json_start = result["stdout"].find("{")
                    json_end = result["stdout"].rfind("}")
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = result["stdout"][json_start:json_end+1]
                        pose_data = json.loads(json_str)
                        
                        detected_poses = []
                        if "predictions" in pose_data:
                            for pred in pose_data["predictions"]:
                                confidence = pred.get("confidence", 0)
                                keypoints = pred.get("keypoints", [])
                                
                                detected_poses.append({
                                    "person_confidence": confidence,
                                    "has_keypoints": len(keypoints) if keypoints else 0
                                })
                        
                        analysis_results["poses"] = detected_poses
            else:
                analysis_results["pose_error"] = "Pose model not available"
                
        except Exception as e:
            logger.error(f"Error during pose detection: {str(e)}")
            analysis_results["pose_error"] = str(e)
        
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
    Analyze image from file path using YOLO CLI
    
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
    Test if YOLO CLI service is running properly
    
    Returns:
        Status information and available tools
    """
    # Test YOLO CLI availability
    try:
        version_result = run_yolo_cli(["--version"], timeout=10)
        yolo_version = version_result.get("stdout", "Unknown") if version_result.get("success") else "Not available"
        
        # Clean up version string
        if "ultralytics" in yolo_version.lower():
            yolo_version = yolo_version.strip()
        else:
            yolo_version = "YOLO CLI not found or not responding correctly"
    except Exception as e:
        yolo_version = f"Error checking YOLO CLI: {str(e)}"
    
    return {
        "status": "YOLO CLI service is running normally",
        "yolo_version": yolo_version,
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
        "features": [
            "All detection functions use YOLO CLI rather than Python API",
            "Support for loading images directly from file paths",
            "Support for comprehensive image analysis with task identification",
            "Support for camera detection using YOLO CLI"
        ]
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

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, shutting down...")
    cleanup_resources()
    sys.exit(0)

def start_watchdog():
    """Start the camera watchdog thread"""
    watchdog = threading.Thread(
        target=camera_watchdog_thread,
        daemon=True
    )
    watchdog.start()
    return watchdog

# Register cleanup functions
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Modify the main execution section
if __name__ == "__main__":
    import platform
    
    logger.info("Starting YOLO CLI service")
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    
    # Test if YOLO CLI is available
    try:
        test_result = run_yolo_cli(["--version"], timeout=10)
        if test_result["success"]:
            logger.info(f"YOLO CLI available: {test_result.get('stdout', '').strip()}")
        else:
            logger.warning(f"YOLO CLI test failed: {test_result.get('stderr', '')}")
            logger.warning("Service may not function correctly without YOLO CLI available")
    except Exception as e:
        logger.error(f"Error testing YOLO CLI: {str(e)}")
        logger.warning("Service may not function correctly without YOLO CLI available")
    
    # Start the camera watchdog thread
    watchdog_thread = start_watchdog()
    
    # Initialize and run server
    logger.info("Starting MCP server...")
    mcp.run(transport='stdio')
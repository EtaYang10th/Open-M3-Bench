# servers/crop_mcp.py
import os
import cv2
import logging
import numpy as np
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional, Literal, Annotated
from pydantic import BaseModel, Field

server = FastMCP("crop")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("crop_mcp")

@server.tool()
def crop(
    input_path: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    output_path: Optional[str] = None
) -> dict:
    """
    Crop an image using OpenCV's slicing.

    Args:
        input_path (str): Full path to the input image.
        x1 (int): X-coordinate of the top-left corner.
        y1 (int): Y-coordinate of the top-left corner.
        x2 (int): X-coordinate of the bottom-right corner.
        y2 (int): Y-coordinate of the bottom-right corner.
        output_path (Optional[str]): Full path to save the cropped image.
            If not provided, a new file with '_cropped' suffix will be saved.
            you should not give output_path in this task.

    Returns:
        images: Path to the cropped image.
    """
    logger.info(f"[crop] Requested for image: {input_path}, region=({x1}, {y1}, {x2}, {y2})")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_cropped{ext}"
        logger.info(f"[crop] Output path not provided, generated: {output_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")
    logger.info(f"[crop] Image loaded. Shape={img.shape}")

    h, w = img.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"Invalid crop coordinates: ({x1},{y1})→({x2},{y2}) for image size {w}x{h}")

    cropped = img[y1:y2, x1:x2]
    logger.info(f"[crop] Cropped image shape: {cropped.shape}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cv2.imwrite(output_path, cropped)
    logger.info(f"[crop] Saved cropped image -> {output_path}")

    return {"images": [output_path]}

DEFAULT_BLUR_STRENGTH = 15

@server.tool()
def blur(
    input_path: str,
    areas: List[Dict[str, Any]],
    invert_areas: bool = False,
    output_path: Optional[str] = None
) -> dict:
    """
    Blur specified rectangular or polygonal areas of an image using OpenCV.

    Args:
        input_path (str): Full path to input image.
        areas (List[Dict]): List of blur regions.
            Each can be:
              - {'x1': int, 'y1': int, 'x2': int, 'y2': int, 'blur_strength': int (optional)}
              - {'polygon': [[x, y], ...], 'blur_strength': int (optional)}
        invert_areas (bool): If True, blur everything EXCEPT the specified areas.
        output_path (Optional[str]): Full path to save output image. If not given, adds "_blurred" suffix.

    Returns:
        images: Path to the saved blurred image.
    """
    logger.info(f"[blur] Requested for image: {input_path}, {len(areas)} area(s), invert={invert_areas}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_blurred{ext}"

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")
    logger.info(f"[blur] Image loaded. Shape={img.shape}")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for area in areas:
        if "polygon" in area:
            pts = np.array(area["polygon"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        elif all(k in area for k in ("x1", "y1", "x2", "y2")):
            cv2.rectangle(mask, (int(area["x1"]), int(area["y1"])), (int(area["x2"]), int(area["y2"])), 255, -1)
        else:
            logger.warning("[blur] Skipped an invalid area (missing keys).")

    if invert_areas:
        mask = cv2.bitwise_not(mask)
        logger.info("[blur] Inverted mask -> blurring background only.")
    else:
        logger.info("[blur] Applying blur to specified regions.")

    blur_strength = areas[0].get("blur_strength", DEFAULT_BLUR_STRENGTH) if areas else DEFAULT_BLUR_STRENGTH
    if blur_strength % 2 == 0:
        blur_strength += 1
        logger.warning(f"[blur] Adjusted blur strength to odd number: {blur_strength}")

    blurred = cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)
    result = np.where(mask[:, :, None] == 255, blurred, img)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, result)
    logger.info(f"[blur] Saved blurred image -> {output_path}")

    return {"images": [output_path]}

@server.tool()
def draw_arrows(
    input_path: str,
    arrows: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> dict:
    """
    Draw arrows on an image using OpenCV.

    Args:
        input_path (str): Full path to input image.
        arrows (List[Dict]): List of arrows, each with keys: x1, y1, x2, y2, and optional color, thickness, tip_length.
        output_path (Optional[str]): Path to save output image. Defaults to '_with_arrows' suffix.

    Returns:
        images: Path to image with arrows drawn.
    """
    logger.info(f"Draw arrows tool requested for image: {input_path} with {len(arrows)} arrows")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        file_name, file_ext = os.path.splitext(input_path)
        output_path = f"{file_name}_with_arrows{file_ext}"

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")
    logger.info(f"Image read successfully. Shape: {img.shape}")

    for i, arrow_item in enumerate(arrows):
        x1, y1 = arrow_item["x1"], arrow_item["y1"]
        x2, y2 = arrow_item["x2"], arrow_item["y2"]
        color = arrow_item.get("color", [0, 0, 0])
        thickness = arrow_item.get("thickness", 1)
        tip_length = arrow_item.get("tip_length", 0.1)
        cv2.arrowedLine(img, (x1, y1), (x2, y2), color, thickness, tipLength=tip_length)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    logger.info(f"Image with arrows saved to: {output_path}")

    return {"images": [output_path]}

@server.tool()
def change_color(
    input_path: str,
    palette: Literal["grayscale", "sepia"],
    output_path: Optional[str] = None
) -> dict:
    """
    Change the color palette of an image.

    Args:
        input_path (str): Full path to input image.
        palette (Literal): 'grayscale' or 'sepia'.
        output_path (Optional[str]): Path to save output. Defaults to '_grayscale' or '_sepia'.

    Returns:
        images: Path to the image with new color palette.
    """
    logger.info(f"Change color tool requested for image: {input_path} with palette: {palette}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        file_name, file_ext = os.path.splitext(input_path)
        output_path = f"{file_name}_{palette}{file_ext}"

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")

    if palette == "grayscale":
        output_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif palette == "sepia":
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(img, sepia_kernel)
        output_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, output_img)
    logger.info(f"Transformed image saved to: {output_path}")

    return {"images": [output_path]}

class CircleItem(BaseModel):
    """Represents a circle to be drawn on an image."""
    center_x: int
    center_y: int
    radius: int
    color: List[int] = [0, 0, 0]  # BGR
    thickness: int = 1
    filled: bool = False


@server.tool()
def draw_circles(input_path: str, circles: List[CircleItem], output_path: Optional[str] = None) -> dict:
    """
    Draw one or more circles on an image using OpenCV.

    Each circle is defined by:
    - `center_x` (int): X-coordinate of the center of the circle.
    - `center_y` (int): Y-coordinate of the center of the circle.
    - `radius` (int): Radius of the circle in pixels.
    - `color` (list of int): BGR color of the circle, e.g., [255, 0, 0] for blue.
    - `thickness` (int): Line thickness. Ignored if `filled=True`.
    - `filled` (bool): If True, the circle will be filled regardless of `thickness`.

    Args:
        input_path (str): Full path to the input image.
        circles (List[CircleItem]): List of circle definitions.
        output_path (Optional[str]): Optional path to save the resulting image. If not provided,
                                     will save to a file with '_with_circles' suffix.

    Returns:
        images: Full path to the saved image with drawn circles.
    """
    logger.info(f"[draw_circles] Requested on image: {input_path}, circles={len(circles)}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_with_circles{ext}"

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")

    for i, circle in enumerate(circles):
        center = (circle.center_x, circle.center_y)
        radius = circle.radius
        color = tuple(circle.color)
        thickness = -1 if circle.filled else circle.thickness
        cv2.circle(img, center, radius, color, thickness)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    return {"images": [output_path]}


@server.tool()
def draw_lines(input_path: str, lines: List[Dict[str, Any]], output_path: Optional[str] = None) -> dict:
    """
    Draw one or more lines on an image using OpenCV.

    Each line is defined by:
    - `x1`, `y1`: Coordinates of the line start point.
    - `x2`, `y2`: Coordinates of the line end point.
    - `color`: Optional. BGR color as list of 3 integers (default: [0, 0, 0]).
    - `thickness`: Optional. Line thickness (default: 1).

    Args:
        input_path (str): Path to the input image file.
        lines (List[Dict[str, Any]]): List of lines to draw, each represented as a dictionary.
        output_path (Optional[str]): Path to save the output image. If not provided,
                                     '_with_lines' will be appended to the input filename.

    Returns:
        images: Path to the resulting image with drawn lines.
    """
    logger.info(f"[draw_lines] Requested for image: {input_path} with {len(lines)} lines")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_with_lines{ext}"

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")

    for i, line_item in enumerate(lines):
        x1, y1 = line_item["x1"], line_item["y1"]
        x2, y2 = line_item["x2"], line_item["y2"]
        color = tuple(line_item.get("color", [0, 0, 0]))
        thickness = line_item.get("thickness", 1)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    return {"images": [output_path]}


@server.tool()
def draw_texts(input_path: str, texts: List[Dict[str, Any]], output_path: Optional[str] = None) -> dict:
    """
    Draw one or more text elements on an image using OpenCV.

    Each text item supports:
    - `text` (str): Text string to be drawn.
    - `x`, `y` (int): Bottom-left corner position for the text.
    - `font_scale` (float, optional): Scale of the font (default: 1.0).
    - `color` (list of int): BGR color [B, G, R] (default: [0, 0, 0]).
    - `thickness` (int): Thickness of the text (default: 1).
    - `font_face` (str): Optional font style. Must be one of:
        'FONT_HERSHEY_SIMPLEX', 'FONT_HERSHEY_PLAIN', 'FONT_HERSHEY_DUPLEX',
        'FONT_HERSHEY_COMPLEX', 'FONT_HERSHEY_TRIPLEX', 'FONT_HERSHEY_COMPLEX_SMALL',
        'FONT_HERSHEY_SCRIPT_SIMPLEX', 'FONT_HERSHEY_SCRIPT_COMPLEX'.

    Args:
        input_path (str): Path to the input image.
        texts (List[Dict[str, Any]]): List of text specifications.
        output_path (Optional[str]): Optional output path. If not given, saves to
                                     '<input>_with_text<ext>'.

    Returns:
        images: Path to the image with rendered text.
    """
    logger.info(f"[draw_texts] Drawing {len(texts)} texts on {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_with_text{ext}"

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")

    font_faces = {
        "FONT_HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
        "FONT_HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
        "FONT_HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
        "FONT_HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
        "FONT_HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
        "FONT_HERSHEY_COMPLEX_SMALL": cv2.FONT_HERSHEY_COMPLEX_SMALL,
        "FONT_HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        "FONT_HERSHEY_SCRIPT_COMPLEX": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    }

    for i, item in enumerate(texts):
        text = item["text"]
        x, y = item["x"], item["y"]
        font_scale = item.get("font_scale", 1.0)
        color = tuple(item.get("color", [0, 0, 0]))
        thickness = item.get("thickness", 1)
        face_name = item.get("font_face", "FONT_HERSHEY_SIMPLEX")
        font_face = font_faces.get(face_name, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    return {"images": [output_path]}

if __name__ == "__main__":
    server.run(transport="stdio")

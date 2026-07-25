# servers/mcp-yolo/server.py
"""
mcp-yolo: drop-in local replacement for DINO-X-MCP.

The external tool surface is intentionally identical to
servers/DINO-X-MCP/src/servers/stdio-server.ts so that GT trajectories
that reference `dinox-mcp/detect-all-objects` / `dinox-mcp/detect-objects-by-text`
can be re-routed to `mcp-yolo/<same tool>` without changing arguments.

Design:
- detect-all-objects -> Ultralytics YOLO11 (COCO 80 classes)
- detect-objects-by-text -> Ultralytics YOLO-World (open-vocabulary)
- Input / output shape matches DINO-X:
    * args: imageFileUri (str, local path or file:// URI), textPrompt (str, "."-joined),
            includeDescription (bool)
    * returns 3 TextContent blocks, same wording and key names as DINO-X.
- Local inference: no cloud key, no quota.
- Caveats vs DINO-X (documented here so downstream eval knows):
    * `includeDescription=True` uses "<category> (conf=XX)" as `description`
      because no local VLM captioner runs on this machine; DINO-X uses a
      multimodal captioner. Field name and block layout are preserved.
    * YOLO weights live next to this file at `servers/mcp-yolo/weights/`
      by default. Override with `MCP_YOLO_WEIGHTS_DIR` if you want a
      shared cache elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional
from urllib.parse import urlparse, unquote

# --- silence ultralytics banner on stdout; FastMCP stdio uses stdout ---
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "False")

# Ultralytics (and its download progress bars) can leak to stdout. FastMCP's
# stdio transport transmits JSONRPC on stdout, so any stray stdout chatter
# would corrupt the framing. Redirect OS-level FD 1 to stderr, but keep the
# original stdout FD reachable through `sys.stdout` so that MCP still writes
# JSONRPC there.
_ORIG_STDOUT_FD = os.dup(sys.stdout.fileno())
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = os.fdopen(_ORIG_STDOUT_FD, "w", buffering=1, encoding="utf-8", newline="\n")

logging.basicConfig(
    level=os.environ.get("MCP_YOLO_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s mcp-yolo %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("mcp-yolo")

from mcp.server.fastmcp import FastMCP

# Heavy deps imported lazily so that list_tools does not pay for model init.
_YOLO_DETECT = None           # type: ignore[var-annotated]
_YOLO_WORLD = None            # type: ignore[var-annotated]

# Default weights live next to this file so the benchmark is self-contained
# and we never pollute the repo root or the user's home cache.
_HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = Path(os.environ.get("MCP_YOLO_WEIGHTS_DIR", _HERE / "weights"))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DETECT_WEIGHTS = os.environ.get("MCP_YOLO_DETECT_WEIGHTS", "yolo11n.pt")
WORLD_WEIGHTS = os.environ.get("MCP_YOLO_WORLD_WEIGHTS", "yolov8s-world.pt")
CONF_THRESHOLD = float(os.environ.get("MCP_YOLO_CONF", "0.25"))
IMG_SIZE = int(os.environ.get("MCP_YOLO_IMGSZ", "640"))

server = FastMCP("mcp-yolo")


# ------------------------- helpers -------------------------
def _resolve_image_path(image_file_uri: str) -> str:
    s = (image_file_uri or "").strip()
    if not s:
        raise ValueError("imageFileUri is empty")
    if s.startswith("file://"):
        parsed = urlparse(s)
        return unquote(parsed.path)
    return s


def _parse_text_prompt(text_prompt: str) -> List[str]:
    if not text_prompt:
        return []
    # DINO-X uses "person.hat.vest" with dot-separated categories.
    # We also accept commas and pipes for robustness.
    parts = re.split(r"[.,|]", text_prompt)
    return [p.strip() for p in parts if p and p.strip()]


def _resolve_weights(name_or_path: str) -> str:
    """Return an existing absolute weights path. If missing, return just the
    model *name*: ultralytics will download it into the current working
    directory. Callers chdir to WEIGHTS_DIR before triggering the download so
    weights land there instead of in the repo root."""
    if os.path.isabs(name_or_path):
        return name_or_path
    candidate = WEIGHTS_DIR / name_or_path
    if candidate.exists():
        return str(candidate)
    return name_or_path


def _load_model(loader, name_or_path: str):
    """Load a weights file, downloading into WEIGHTS_DIR on first use."""
    w = _resolve_weights(name_or_path)
    if os.path.isabs(w):
        return loader(w)
    # Need to download; confine the side effect to WEIGHTS_DIR.
    prev_cwd = os.getcwd()
    try:
        os.chdir(WEIGHTS_DIR)
        return loader(w)
    finally:
        os.chdir(prev_cwd)


def _load_detect():
    global _YOLO_DETECT
    if _YOLO_DETECT is None:
        from ultralytics import YOLO
        log.info("loading detect weights: %s (dir=%s)", DETECT_WEIGHTS, WEIGHTS_DIR)
        _YOLO_DETECT = _load_model(YOLO, DETECT_WEIGHTS)
    return _YOLO_DETECT


def _load_world():
    global _YOLO_WORLD
    if _YOLO_WORLD is None:
        from ultralytics import YOLOWorld
        log.info("loading yolo-world weights: %s (dir=%s)", WORLD_WEIGHTS, WEIGHTS_DIR)
        _YOLO_WORLD = _load_model(YOLOWorld, WORLD_WEIGHTS)
    return _YOLO_WORLD


def _round_bbox(xyxy: Iterable[float]) -> dict:
    # DINO-X returns bbox as {xmin, ymin, xmax, ymax}, rounded to 1 decimal.
    v = [round(float(x), 1) for x in xyxy]
    return {"xmin": v[0], "ymin": v[1], "xmax": v[2], "ymax": v[3]}


def _describe(cat: str, conf: float) -> str:
    return f"{cat} (conf={conf:.2f})"


def _format_response(objects_info: List[dict], categories: dict, bbox_note: str) -> list[dict]:
    """Produce the exact 3-text-block layout DINO-X stdio-server emits.

    `bbox_note` is the `{...}` / `[...]` variant wording used by each tool."""
    cat_summary = ", ".join(
        f"{cat} ({len(items)})" for cat, items in categories.items()
    ) or "none"
    return [
        {
            "type": "text",
            "text": f"Objects detected in image: {cat_summary}.",
        },
        {
            "type": "text",
            "text": f"Detailed object detection results: {json.dumps(objects_info, ensure_ascii=False, indent=2)}",
        },
        {"type": "text", "text": bbox_note},
    ]


# Exact wording from DINO-X stdio-server.
NOTE_ALL = (
    "Note: The bbox coordinates are in {xmin, ymin, xmax, ymax} format, "
    "where the origin (0,0) is at the top-left corner of the image. "
    "These coordinates help determine the exact position and spatial "
    "relationships of objects in the image."
)
NOTE_BY_TEXT = (
    "Note: The bbox coordinates are in [xmin, ymin, xmax, ymax] format, "
    "where the origin (0,0) is at the top-left corner of the image. "
    "These coordinates help determine the exact position and spatial "
    "relationships of objects in the image."
)


def _error_block(msg: str) -> list[dict]:
    return [{"type": "text", "text": msg}]


# ------------------------- tools -------------------------
@server.tool(name="detect-all-objects")
def detect_all_objects(imageFileUri: str, includeDescription: bool = False):  # noqa: N803
    """Analyze an image and detect all objects.

      Args:
        imageFileUri (string): Local image file path or file:// URI.
        includeDescription (boolean): Whether to include a text description per object.

      Returns:
        text (string): Summary of object categories, counts, and JSON details.
    """
    try:
        path = _resolve_image_path(imageFileUri)
        if not os.path.exists(path):
            return _error_block(f"Failed to detect objects from image: file not found: {path}")
        model = _load_detect()
        results = model.predict(source=path, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        objects_info: List[dict] = []
        categories: dict = {}
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
            for box, conf, c in zip(xyxy, confs, cls):
                cat = names[c]
                item: dict = {"name": cat, "bbox": _round_bbox(box)}
                if includeDescription:
                    item["description"] = _describe(cat, conf)
                objects_info.append(item)
                categories.setdefault(cat, []).append(item)
        return _format_response(objects_info, categories, NOTE_ALL)
    except Exception as e:  # noqa: BLE001
        log.exception("detect-all-objects failed")
        return _error_block(f"Failed to detect objects from image: {e}")


@server.tool(name="detect-objects-by-text")
def detect_objects_by_text(imageFileUri: str, textPrompt: str, includeDescription: bool = False):  # noqa: N803
    """Detect objects in an image by a text prompt (open-vocabulary).

      Args:
        imageFileUri (string): Local image file path or file:// URI.
        textPrompt (string): Dot-separated category list, e.g. "person.ladder.hardhat".
        includeDescription (boolean): Whether to include a text description per object.

      Returns:
        text (string): Summary of object categories, counts, and JSON details.
    """
    try:
        if not imageFileUri or not textPrompt:
            return _error_block("Image file URI and text prompt are required")
        path = _resolve_image_path(imageFileUri)
        if not os.path.exists(path):
            return _error_block(f"Failed to detect objects from image: file not found: {path}")
        classes = _parse_text_prompt(textPrompt)
        if not classes:
            return _error_block("textPrompt contains no valid categories")
        model = _load_world()
        model.set_classes(classes)
        results = model.predict(source=path, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        objects_info: List[dict] = []
        categories: dict = {}
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().tolist()
            confs = r.boxes.conf.cpu().numpy().tolist()
            cls = r.boxes.cls.cpu().numpy().astype(int).tolist()
            for box, conf, c in zip(xyxy, confs, cls):
                cat = names[c] if c in names else classes[c] if c < len(classes) else str(c)
                item: dict = {"name": cat, "bbox": _round_bbox(box)}
                if includeDescription:
                    item["description"] = _describe(cat, conf)
                objects_info.append(item)
                categories.setdefault(cat, []).append(item)
        return _format_response(objects_info, categories, NOTE_BY_TEXT)
    except Exception as e:  # noqa: BLE001
        log.exception("detect-objects-by-text failed")
        return _error_block(f"Failed to detect objects from image: {e}")


if __name__ == "__main__":
    server.run(transport="stdio")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Context Protocol (MCP) Server: pyzbar_mcp
=================================================

This MCP server provides barcode and QR code **scanning capabilities**
based on `pyzbar` and OpenCV. It exposes a single tool: `scan_barcode`.

Features:
- Scans and decodes barcodes or QR codes from image files.
- Automatically falls back to OpenCV’s `QRCodeDetector` if `pyzbar` is unavailable.
- Compatible with both **stdio MCP protocol** and **file-transport mode**
  (used by call_mcp_tool in your tool selector).
- Returns rich structured JSON with bounding boxes.

Author: Bob’s MCP system integration
"""

import json
import sys
import cv2
import argparse
from mcp.server.fastmcp import FastMCP

server = FastMCP("pyzbar")

try:
    from pyzbar.pyzbar import decode as zbar_decode
    _HAS_PYZBAR = True
except Exception:
    _HAS_PYZBAR = False


# ------------------------------------------------------------
# Prevent recursive triggering: safe output wrapper
# ------------------------------------------------------------
def _sanitize_output(obj: dict) -> dict:
    """Prevent recursive triggering: move top-level fields like isbn/data into metadata"""
    if not isinstance(obj, dict):
        return obj

    safe = {}
    metadata = {}
    for k, v in obj.items():
        if k in {"results", "isbn", "data", "type", "rect"}:
            metadata[k] = v
        else:
            safe[k] = v
    if metadata:
        safe["metadata"] = metadata
    return safe


@server.tool()
def scan_barcode(image_paths: list[str] | str) -> str:
    """
      Scan barcodes and QR codes in an image and return decoded values with bounding boxes.
      Args:
        image_paths (list[str] | str): Single image path or list where the first path is used.
      Returns:
        result (str): JSON string summarizing detections, metadata results, or error information.
    """
    try:
        # Accept image_paths as either str or list
        if isinstance(image_paths, list):
            image_path = image_paths[0]
        else:
            image_path = image_paths

        img = cv2.imread(image_path)
        if img is None:
            return json.dumps(_sanitize_output({"error": f"Unable to read image file: {image_path}"}), ensure_ascii=False)

        results = []
        if _HAS_PYZBAR:
            decoded = zbar_decode(img)
            for d in decoded:
                rect = d.rect
                code_data = d.data.decode("utf-8", errors="ignore")
                entry = {
                    "type": d.type,
                    "data": code_data,
                    "rect": {
                        "x": rect.left,
                        "y": rect.top,
                        "width": rect.width,
                        "height": rect.height
                    }
                }

                # If ISBN or EAN13, explicitly add isbn field
                if d.type.upper() in ["EAN13", "ISBN"]:
                    entry["isbn"] = code_data
                results.append(entry)
        else:
            detector = cv2.QRCodeDetector()
            retval, infos, points, _ = detector.detectAndDecodeMulti(img)
            if retval and infos:
                for i, text in enumerate(infos):
                    if text:
                        rect = {}
                        if points is not None and len(points) > i:
                            pts = points[i].astype(int).tolist()
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            rect = {
                                "x": int(min(xs)),
                                "y": int(min(ys)),
                                "width": int(max(xs) - min(xs)),
                                "height": int(max(ys) - min(ys))
                            }
                        results.append({
                            "type": "QRCODE",
                            "data": text,
                            "rect": rect
                        })

        # If no barcodes or QR codes were detected
        if not results:
            return json.dumps(_sanitize_output({
                "results": [],
                "message": "Barcode scan completed successfully. No barcodes or QR codes were detected. You can proceed to other tools if needed, but do not re-scan the same image."
            }), ensure_ascii=False)

        # Extract ISBN information for subsequent call hints
        isbn_codes = [r["isbn"] for r in results if "isbn" in r]

        if isbn_codes:
            msg = (
                "Barcode scan completed successfully. "
                f"Found ISBN(s): {', '.join(isbn_codes)}. "
                "No further barcode scanning is needed — proceed directly to book lookup."
            )
        else:
            msg = (
                "Barcode scan completed successfully. "
                "Detected non-ISBN barcodes. You may now use other tools to look up related items."
            )

        result_obj = {"results": results, "message": msg}
        return json.dumps(_sanitize_output(result_obj), ensure_ascii=False)

    except Exception as e:
        return json.dumps(_sanitize_output({"error": f"Execution error: {str(e)}"}), ensure_ascii=False)


# ==================================================================
# File-transport mode (compatible with call_mcp_tool)
# ==================================================================
def _file_transport_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True, help="Input request JSON path")
    parser.add_argument("--out", dest="output_path", required=True, help="Output response JSON path")
    args = parser.parse_args()

    try:
        req = json.loads(open(args.input_path, "r", encoding="utf-8").read())
        name = req.get("name")
        args_ = req.get("arguments", {})
        if name != "scan_barcode":
            raise ValueError(f"Unknown tool '{name}' for pyzbar MCP")

        image_paths = args_.get("image_paths")
        result_json = scan_barcode(image_paths)
        result_obj = json.loads(result_json)
        resp = {"success": True, "result": _sanitize_output(result_obj)}
    except Exception as e:
        resp = {"success": False, "error": str(e)}

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=2)


# ==================================================================
# Entry point dispatcher
# ==================================================================
if __name__ == "__main__":
    if "--in" in sys.argv and "--out" in sys.argv:
        _file_transport_main()
    else:
        server.run(transport="stdio")

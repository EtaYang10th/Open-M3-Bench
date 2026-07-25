#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Context Protocol (MCP) Server: ocr_mcp
============================================

Local, repo-side replacement for the third-party ``mcp_ocr`` package.

Why this exists
---------------
The pip-installed ``mcp_ocr`` package constructs ``mcp.types.ErrorData`` with
positional arguments (e.g. ``ErrorData(INTERNAL_ERROR, msg)``). The installed
``mcp`` release exposes ``ErrorData`` as a pydantic ``BaseModel`` that only
accepts keyword arguments, so *every* error branch (invalid path, no text
detected, etc.) raises::

    BaseModel.__init__() takes 1 positional argument but 3 were given

masking the real error and failing the tool call. This server keeps the exact
same tool surface (``perform_ocr`` / ``get_supported_languages`` with identical
parameter names/types) so GT matching is unaffected, while using plain
exceptions for error surfacing (matching the other local servers in this repo).

Tool signatures (must stay identical for GT binding):
- perform_ocr(input_data: str, language: str = "eng", config: str = "--oem 3 --psm 6") -> str
- get_supported_languages() -> list[str]
"""

import os
import urllib.parse

import cv2
import numpy as np
import httpx
import pytesseract
from mcp.server.fastmcp import FastMCP

server = FastMCP("ocr")


def _load_image(input_data):
    """Load an image from a file path, URL, or raw bytes into a BGR ndarray."""
    if isinstance(input_data, str):
        scheme = urllib.parse.urlparse(input_data).scheme
        if scheme in ("http", "https"):
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(input_data)
                resp.raise_for_status()
                nparr = np.frombuffer(resp.content, np.uint8)
        elif os.path.exists(input_data):
            nparr = np.fromfile(input_data, np.uint8)
        else:
            raise ValueError(
                f"Invalid input: {input_data} is neither a valid URL nor an existing file"
            )
    else:
        nparr = np.frombuffer(input_data, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image data")
    return img


@server.tool()
async def perform_ocr(
    input_data: str,
    language: str = "eng",
    config: str = "--oem 3 --psm 6",
) -> str:
    """Perform OCR on the provided input.

    Args:
        input_data: File path to an image, URL to an image, or raw image bytes.
        language: Tesseract language code (default: "eng").
        config: Tesseract configuration options (default: "--oem 3 --psm 6").

    Returns:
        Extracted text from the image.
    """
    available_langs = pytesseract.get_languages()
    if language not in available_langs:
        raise ValueError(
            f"Unsupported language: {language}. Available languages: {', '.join(available_langs)}"
        )

    img = _load_image(input_data)
    text = pytesseract.image_to_string(img, lang=language, config=config)
    if not text.strip():
        raise ValueError("No text detected in image")
    return text.strip()


@server.tool()
async def get_supported_languages() -> list[str]:
    """Get list of supported OCR languages."""
    langs = pytesseract.get_languages()
    if not langs:
        raise ValueError(
            "No supported languages found. Please check Tesseract installation."
        )
    return langs


if __name__ == "__main__":
    server.run(transport="stdio")

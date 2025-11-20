#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Server: Unsplash Main Image Downloader
------------------------------------------
Description:
This MCP server uses the official Unsplash API to download a photo
given a public Unsplash photo URL.

Instead of scraping HTML (which can be blocked by anti-bot mechanisms),
it directly queries the Unsplash REST API endpoint:
    https://api.unsplash.com/photos/{photo_id}?client_id={ACCESS_KEY}

Workflow:
1. Extract the photo ID from the provided Unsplash page URL.
2. Query the official Unsplash API to obtain metadata and image URLs.
3. Download the full-resolution image and save it as a .jpg file
   in the local "images/" directory.

Returned JSON:
- image_url: Direct URL to the full-resolution image.
- author: Photographer’s name.
- saved_path: Absolute path to the downloaded image.
- message: Status of the operation.
- error: Error message if any issue occurred.
"""

import json
import sys
import argparse
import os
import requests
from mcp.server.fastmcp import FastMCP

ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

# Initialize MCP server
server = FastMCP("unsplash_image_fetcher")

def extract_photo_id(photo_url: str) -> str:
    """Extract the Unsplash photo ID from a photo page URL."""
    if not photo_url:
        raise ValueError("No photo URL provided.")
    return photo_url.rstrip("/").split("-")[-1]


def download_unsplash_image(photo_url: str) -> dict:
    """Download an image from Unsplash using the official API."""
    try:
        photo_id = extract_photo_id(photo_url)
        api_url = f"https://api.unsplash.com/photos/{photo_id}?client_id={ACCESS_KEY}"

        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return {"error": f"Failed to fetch photo metadata. HTTP {resp.status_code}"}

        data = resp.json()

        image_url = data["urls"]["full"]
        author = data["user"]["name"]
        save_name = f"{photo_id}.jpg"

        os.makedirs("images", exist_ok=True)
        img_data = requests.get(image_url, timeout=20).content
        save_path = os.path.abspath(os.path.join("images", save_name))
        with open(save_path, "wb") as f:
            f.write(img_data)

        return {
            "message": f"Downloaded '{author}' photo successfully.",
            "image_url": image_url,
            "author": author,
            "saved_path": save_path,
        }

    except Exception as e:
        return {"error": f"Exception occurred: {e}"}


@server.tool()
def fetch_unsplash_image(url: str) -> str:
    """
      Download the main image from an Unsplash photo page using the official API.
      Args:
        url (str): Full Unsplash photo URL for the target image.
      Returns:
        result (str): JSON string describing download status, image URL, author, and saved path.
    """
    result = download_unsplash_image(url)
    return json.dumps(result, ensure_ascii=False)


def _file_transport_main():
    """Support file-based MCP call (used by call_mcp_tool)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    try:
        req = json.loads(open(args.input_path, "r", encoding="utf-8").read())
        name = req.get("name")
        if name != "fetch_unsplash_image":
            raise ValueError(f"Unknown tool name: {name}")

        url = req.get("arguments", {}).get("url")
        result_json = fetch_unsplash_image(url)
        resp = {"success": True, "result": json.loads(result_json)}
    except Exception as e:
        resp = {"success": False, "error": str(e)}

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if "--in" in sys.argv and "--out" in sys.argv:
        _file_transport_main()
    else:
        server.run(transport="stdio")

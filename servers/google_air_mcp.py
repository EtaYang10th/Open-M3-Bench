# servers/air_quality_mcp.py
import os, json, base64
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import httpx
from mcp.server.fastmcp import FastMCP


# ==== Configuration ====
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_MAPS_API_KEY is required")

CLIENT = httpx.AsyncClient(timeout=30.0)
server = FastMCP("air-quality")

BASE = "https://airquality.googleapis.com/v1"


# ==== Tool 1: Current Conditions ====
@server.tool()
async def current_conditions(lat: float, lng: float) -> Dict[str, Any]:
    """
      Retrieve current air-quality conditions at a specific latitude and longitude.
      Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
      Returns:
        data (Dict[str, Any]): JSON payload from Google Air Quality API with AQI and pollutants.
    """
    url = f"{BASE}/currentConditions:lookup"
    params = {"key": API_KEY}
    body = {"location": {"latitude": lat, "longitude": lng}}

    resp = await CLIENT.post(url, params=params, json=body)
    resp.raise_for_status()
    return resp.json()


# ==== Tool 2: Forecast ====
@server.tool()
async def forecast(lat: float, lng: float, hours: Optional[int] = None) -> Dict[str, Any]:
    """
      Retrieve air-quality forecasts for a location for a configurable number of hours.
      Args:
        lat (float): Latitude of the forecast location.
        lng (float): Longitude of the forecast location.
        hours (Optional[int]): Number of forecast hours to request (1–96).
      Returns:
        data (Dict[str, Any]): JSON payload with forecasted pollutant levels and AQI indexes.
    """
    url = f"{BASE}/forecast:lookup"
    params = {"key": API_KEY}
    body: Dict[str, Any] = {"location": {"latitude": lat, "longitude": lng}}
    # New Air Quality API forecast:lookup no longer accepts a top-level `hours`.
    # Convert the requested `hours` into a `period` (startTime/endTime, ISO-8601 UTC).
    span = hours if hours else 24
    span = max(1, min(int(span), 96))
    # Forecast requires the time range to be at least one rounded hour into the
    # future; the API rounds timestamps down to the hour, so start at the next
    # whole hour to keep the whole period in the future.
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now + timedelta(hours=1)
    end = start + timedelta(hours=span)
    body["period"] = {
        "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    resp = await CLIENT.post(url, params=params, json=body)
    resp.raise_for_status()
    return resp.json()


# ==== Tool 3: Historical Data ====
@server.tool()
async def history(lat: float, lng: float, startTime: str, endTime: str) -> Dict[str, Any]:
    """
      Retrieve historical air-quality records for a location over a time range.
      Args:
        lat (float): Latitude of the query location.
        lng (float): Longitude of the query location.
        startTime (str): Start time in ISO-8601 format.
        endTime (str): End time in ISO-8601 format.
      Returns:
        data (Dict[str, Any]): JSON with past AQI and pollutant data for the interval.
    """
    url = f"{BASE}/history:lookup"
    params = {"key": API_KEY}
    # New Air Quality API history:lookup requires the time range wrapped in `period`.
    body = {
        "location": {"latitude": lat, "longitude": lng},
        "period": {"startTime": startTime, "endTime": endTime},
    }

    resp = await CLIENT.post(url, params=params, json=body)
    resp.raise_for_status()
    return resp.json()


# ==== Tool 4: Heatmap Tile ====
@server.tool()
async def heatmap_tile(z: int, x: int, y: int, indexType: str = "UNIVERSAL_AQI") -> Dict[str, Any]:
    """
      Retrieve a heat-map tile representing air-quality levels for a map tile coordinate.
      Args:
        z (int): Zoom level of the map tile.
        x (int): Tile X coordinate.
        y (int): Tile Y coordinate.
        indexType (str): AQI index type to visualize.
      Returns:
        data (Dict[str, Any]): JSON with tile metadata or encoded image data.
    """
    url = f"{BASE}/mapTypes/{indexType}/heatmapTiles/{z}/{x}/{y}"
    params = {"key": API_KEY}

    resp = await CLIENT.get(url, params=params)
    resp.raise_for_status()
    # heatmapTiles returns a binary PNG tile, not JSON. Encode as base64.
    return {
        "contentType": resp.headers.get("content-type"),
        "imageBase64": base64.b64encode(resp.content).decode("ascii"),
        "z": z,
        "x": x,
        "y": y,
        "indexType": indexType,
    }


if __name__ == "__main__":
    server.run(transport="stdio")

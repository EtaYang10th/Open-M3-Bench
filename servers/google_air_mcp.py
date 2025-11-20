# servers/air_quality_mcp.py
import os, json
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
    body = {"location": {"latitude": lat, "longitude": lng}}
    if hours:
        body["hours"] = hours

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
    body = {
        "location": {"latitude": lat, "longitude": lng},
        "startTime": startTime,
        "endTime": endTime,
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
    return resp.json()


if __name__ == "__main__":
    server.run(transport="stdio")

from __future__ import annotations
"""
Google Maps Platform MCP Server (FastMCP, stdio)

Exposes a pragmatic superset of Google Maps Platform REST endpoints via MCP tools:
- Geocoding & Reverse Geocoding (maps.googleapis.com)
- Places API (New): Text Search, Nearby Search, Place Details, Place Photos (v1)
- Routes API v2: computeRoutes
- Distance Matrix (legacy)
- Directions (legacy) — convenient fallback when you don't need Routes v2
- Time Zone
- Elevation
- Roads API: snapToRoads, nearestRoads, speedLimits
- Geolocation (cell towers, Wi‑Fi)
- Static Maps / Street View (returns ready-to-use image URLs)

Requirements
-----------
- Environment: GOOGLE_MAPS_API_KEY must be set
- Python deps: fastmcp, httpx (\n    pip install 'mcp[fastmcp]' httpx\n  )

Register in mcp_servers.json
----------------------------
{
  "servers": {
    "google-maps": {
      "command": "python",
      "args": ["servers/google_maps_mcp.py"],
      "env": {"GOOGLE_MAPS_API_KEY": "${YOUR_KEY}"}
    }
  }
}

Notes
-----
- Places (New) and Routes v2 **require** a response field mask via header `X-Goog-FieldMask` or `$fields` param.
- This server defaults to sensible field masks; you can override via the `fields` argument on each tool.
- All tools return JSON objects (not plain text) so your host can keep rich structure.
"""

import os, json, time, base64, urllib.parse as urlparse
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP

# ---------- Config ----------
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_MAPS_API_KEY is required in environment")

HTTP_TIMEOUT = 30.0
CLIENT = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

server = FastMCP("google-maps")

# ---------- Helpers ----------

def _params_with_key(params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params or {})
    q["key"] = API_KEY
    return q

async def _get(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        r = await CLIENT.get(url, params=params, headers=headers)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Surface Google API error payloads (often JSON) to the host/LLM
        try:
            payload = e.response.json()
        except Exception:
            payload = {"text": e.response.text}
        raise RuntimeError(
            f"HTTP {e.response.status_code} {e.response.reason_phrase} at {url}: {payload}"
        ) from None

    ct = r.headers.get("content-type", "")
    if ct.startswith("application/json"):
        return r.json()
    # Non-JSON (e.g., /media photo endpoint). Return as a data-url for convenience.
    data_url = f"data:{ct};base64,{base64.b64encode(r.content).decode()}"
    return {"content_type": ct, "data_url": data_url}

async def _post_json(url: str, json_body: Dict[str, Any], headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        r = await CLIENT.post(url, json=json_body, headers=headers, params=params)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        try:
            payload = e.response.json()
        except Exception:
            payload = {"text": e.response.text}
        raise RuntimeError(
            f"HTTP {e.response.status_code} {e.response.reason_phrase} at {url}: {payload}"
        ) from None
    return r.json()


# ---------- Geocoding ----------
@server.tool()
async def geocode(address: str, region: Optional[str] = None, language: Optional[str] = None) -> Dict[str, Any]:
    """
      Forward geocode a human-readable address using Google Geocoding API.
      Args:
        address (str): Free-text address to geocode.
        region (Optional[str]): Region bias such as "us".
        language (Optional[str]): Response language code such as "en".
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and API response.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = _params_with_key({"address": address})
    if region: params["region"] = region
    if language: params["language"] = language
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}

@server.tool()
async def reverse_geocode(lat: float, lng: float, language: Optional[str] = None) -> Dict[str, Any]:
    """
      Reverse geocode coordinates into addresses using Google Geocoding API.
      Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        language (Optional[str]): Response language code such as "en".
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and API response.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = _params_with_key({"latlng": f"{lat},{lng}"})
    if language: params["language"] = language
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}

# ---------- Places API (New) v1 ----------
PLACES_V1 = "https://places.googleapis.com/v1"

_DEF_PLACE_FIELDS = ",".join([
    # Array: places[*].field
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.primaryType",
    "places.primaryTypeDisplayName",
    "places.rating",
    "places.userRatingCount",
    "places.internationalPhoneNumber",
    "places.websiteUri",
    "places.currentOpeningHours",
    "places.photos.name",
])

@server.tool()
async def places_text_search(query: str,
                             languageCode: Optional[str] = None,
                             regionCode: Optional[str] = None,
                             locationBias: Optional[Dict[str, Any]] = None,
                             maxResultCount: int = 10,
                             fields: Optional[str] = None) -> Dict[str, Any]:
    """
      Search for places using a free-text query via Google Places Text Search.
      Args:
        query (str): Text query describing the place.
        languageCode (Optional[str]): Response language code.
        regionCode (Optional[str]): Region bias code.
        locationBias (Optional[Dict[str, Any]]): Optional circle bias for location preference.
        maxResultCount (int): Maximum number of results to keep client-side.
        fields (Optional[str]): Unused field mask parameter for signature parity.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and trimmed Places response.
    """
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params: Dict[str, Any] = {"query": query, "key": API_KEY}
    if languageCode: params["language"] = languageCode
    if regionCode: params["region"] = regionCode
    # If caller gave a circle bias, translate to legacy location+radius
    if locationBias and isinstance(locationBias, dict) and "circle" in locationBias:
        circle = locationBias["circle"]
        center = circle.get("center", {})
        lat, lng = center.get("latitude"), center.get("longitude")
        if lat is not None and lng is not None:
            params["location"] = f"{lat},{lng}"
        if "radius" in circle:
            params["radius"] = int(circle["radius"])  # meters
    res = await _get(url, params)
    # Trim to requested count for parity with v1
    if isinstance(res, dict) and isinstance(res.get("results"), list):
        res["results"] = res["results"][: int(maxResultCount)]
    return {"endpoint": url, "request": params, "response": res}

@server.tool()
async def places_nearby_search(location: Dict[str, float],
                               radiusMeters: int,
                               includedTypes: Optional[List[str]] = None,
                               languageCode: Optional[str] = None,
                               regionCode: Optional[str] = None,
                               fields: Optional[str] = None) -> Dict[str, Any]:
    """
      Search nearby places using Places API (New) v1.
      Args:
        location (Dict[str, float]): Latitude and longitude dictionary.
        radiusMeters (int): Search radius in meters.
        includedTypes (Optional[List[str]]): Optional place type filters.
        languageCode (Optional[str]): Response language code.
        regionCode (Optional[str]): Region bias.
        fields (Optional[str]): Optional field mask override for response.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, field mask, request body, and API response.
    """
    url = f"{PLACES_V1}/places:searchNearby"
    body = {"location": location, "radiusMeters": int(radiusMeters)}
    if includedTypes: body["includedTypes"] = includedTypes
    if languageCode: body["languageCode"] = languageCode
    if regionCode: body["regionCode"] = regionCode
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": fields or _DEF_PLACE_FIELDS,
    }
    res = await _post_json(url, body, headers=headers)
    return {"endpoint": url, "headers": {"X-Goog-FieldMask": headers["X-Goog-FieldMask"]}, "request": body, "response": res}

@server.tool()
async def place_details(place_id: str, fields: Optional[str] = None) -> Dict[str, Any]:
    """
      Fetch place details by place ID using Places API (New) v1.
      Args:
        place_id (str): Google Place ID.
        fields (Optional[str]): Optional field mask override.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, field mask, and place details response.
    """
    url = f"{PLACES_V1}/places/{place_id}"
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": fields or ",".join([
            "id","displayName","formattedAddress","location",
            "primaryType","primaryTypeDisplayName","rating","userRatingCount",
            "internationalPhoneNumber","websiteUri","currentOpeningHours","photos.name"
        ]),
    }
    res = await _get(url, params={}, headers=headers)
    return {"endpoint": url, "headers": {"X-Goog-FieldMask": headers["X-Goog-FieldMask"]}, "response": res}

@server.tool()
async def place_photo_media(photo_resource: str,
                            maxWidthPx: Optional[int] = None,
                            maxHeightPx: Optional[int] = None) -> Dict[str, Any]:
    """
      Download a place photo asset and return it wrapped as a data URL or JSON.
      Args:
        photo_resource (str): Places photo resource path.
        maxWidthPx (Optional[int]): Optional maximum image width in pixels.
        maxHeightPx (Optional[int]): Optional maximum image height in pixels.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, query parameters, and photo response.
    """
    # Build path: .../media
    base = f"{PLACES_V1}/{photo_resource}/media"
    params = {}
    if maxWidthPx: params["maxWidthPx"] = int(maxWidthPx)
    if maxHeightPx: params["maxHeightPx"] = int(maxHeightPx)
    headers = {"X-Goog-Api-Key": API_KEY}
    res = await _get(base, params=params, headers=headers)
    return {"endpoint": base, "request": params, "response": res}

# ---------- Routes API v2 ----------
ROUTES_V2 = "https://routes.googleapis.com/directions/v2:computeRoutes"

_DEF_ROUTE_FIELDS = ",".join([
    "routes.distanceMeters",
    "routes.duration",
    "routes.polyline.encodedPolyline",
    "routes.legs",
    "routes.travelAdvisory",
])

@server.tool()
async def compute_route(origin: Dict[str, Any],
                        destination: Dict[str, Any],
                        travelMode: str = "DRIVE",
                        routingPreference: Optional[str] = None,
                        avoidTolls: Optional[bool] = None,
                        optimizeWaypointOrder: Optional[bool] = None,
                        intermediates: Optional[List[Dict[str, Any]]] = None,
                        fields: Optional[str] = None) -> Dict[str, Any]:
    """
      Compute a route between origin and destination using Routes API v2.
      Args:
        origin (Dict[str, Any]): Origin location using placeId or latLng structure.
        destination (Dict[str, Any]): Destination location using placeId or latLng.
        travelMode (str): Travel mode such as DRIVE or WALK.
        routingPreference (Optional[str]): Optional routing preference.
        avoidTolls (Optional[bool]): Whether to avoid toll roads.
        optimizeWaypointOrder (Optional[bool]): Whether to optimize waypoint order.
        intermediates (Optional[List[Dict[str, Any]]]): Optional waypoints.
        fields (Optional[str]): Optional field mask override for response.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, field mask, request body, and route response.
    """
    body: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "travelMode": travelMode,
    }
    if intermediates: body["intermediates"] = intermediates
    if routingPreference: body["routingPreference"] = routingPreference
    if avoidTolls is not None:
        body["routeModifiers"] = {"avoidTolls": bool(avoidTolls)}
    if optimizeWaypointOrder is not None:
        body["optimizeWaypointOrder"] = bool(optimizeWaypointOrder)

    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": fields or _DEF_ROUTE_FIELDS,
    }
    res = await _post_json(ROUTES_V2, body, headers=headers)
    return {"endpoint": ROUTES_V2, "headers": {"X-Goog-FieldMask": headers["X-Goog-FieldMask"]}, "request": body, "response": res}

# ---------- Legacy Directions & Distance Matrix ----------
@server.tool()
async def directions_legacy(origin: str, destination: str, mode: str = "driving", waypoints: Optional[List[str]] = None, departure_time: Optional[str] = None) -> Dict[str, Any]:
    """
      Get directions between two locations using the legacy Directions API.
      Args:
        origin (str): Origin address or "lat,lng" string.
        destination (str): Destination address or "lat,lng" string.
        mode (str): Travel mode string such as "driving".
        waypoints (Optional[List[str]]): Optional waypoint addresses.
        departure_time (Optional[str]): Optional departure time parameter.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and directions response.
    """
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params: Dict[str, Any] = {"origin": origin, "destination": destination, "mode": mode}
    if waypoints: params["waypoints"] = "|".join(waypoints)
    if departure_time: params["departure_time"] = departure_time
    res = await _get(url, _params_with_key(params))
    return {"endpoint": url, "request": params, "response": res}

@server.tool()
async def distance_matrix(origins: List[str], destinations: List[str], mode: str = "driving", departure_time: Optional[str] = None) -> Dict[str, Any]:
    """
      Compute travel times and distances using the legacy Distance Matrix API.
      Args:
        origins (List[str]): Origin addresses or "lat,lng" strings.
        destinations (List[str]): Destination addresses or "lat,lng" strings.
        mode (str): Travel mode such as "driving".
        departure_time (Optional[str]): Optional departure time parameter.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and matrix response.
    """
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params: Dict[str, Any] = {"origins": "|".join(origins), "destinations": "|".join(destinations), "mode": mode}
    if departure_time: params["departure_time"] = departure_time
    res = await _get(url, _params_with_key(params))
    return {"endpoint": url, "request": params, "response": res}

# ---------- Time Zone & Elevation ----------
@server.tool()
async def timezone(lat: float, lng: float, timestamp: Optional[int] = None) -> Dict[str, Any]:
    """
      Retrieve time zone information for a coordinate using the Time Zone API.
      Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        timestamp (Optional[int]): Unix epoch seconds for DST calculation.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and time zone response.
    """
    if timestamp is None:
        timestamp = int(time.time())
    url = "https://maps.googleapis.com/maps/api/timezone/json"
    params = _params_with_key({"location": f"{lat},{lng}", "timestamp": timestamp})
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}

@server.tool()
async def elevation_by_locations(locations: List[str]) -> Dict[str, Any]:
    """
      Retrieve elevation for one or more discrete locations.
      Args:
        locations (List[str]): Coordinate strings such as "lat,lng".
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and elevation results.
    """
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = _params_with_key({"locations": "|".join(locations)})
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}

@server.tool()
async def elevation_along_path(path: List[str], samples: int) -> Dict[str, Any]:
    """
      Retrieve elevation sampled along a path.
      Args:
        path (List[str]): Path coordinates as "lat,lng" strings.
        samples (int): Number of evenly spaced samples to return.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and elevation samples.
    """
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = _params_with_key({"path": "|".join(path), "samples": int(samples)})
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}

# ---------- Roads API ----------
@server.tool()
async def roads_snap_to_roads(path: List[str], interpolate: bool = False) -> Dict[str, Any]:
    """
      Snap GPS points to the road network using the Roads API.
      Args:
        path (List[str]): Coordinate strings such as "lat,lng".
        interpolate (bool): Whether to add interpolated points along the road.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and snapped points.
    """
    url = "https://roads.googleapis.com/v1/snapToRoads"
    params = _params_with_key({"path": "|".join(path)})
    if interpolate:
        params["interpolate"] = "true"
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}


@server.tool()
async def roads_nearest_roads(points: List[str]) -> Dict[str, Any]:
    """
      Find nearest road segments for input points using the Roads API.
      Args:
        points (List[str]): Coordinate strings such as "lat,lng".
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and snapped road segments.
    """
    url = "https://roads.googleapis.com/v1/nearestRoads"
    params = _params_with_key({"points": "|".join(points)})
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}

@server.tool()
async def roads_speed_limits(placeIds: Optional[List[str]] = None,
                             path: Optional[List[str]] = None,
                             units: str = "KPH") -> Dict[str, Any]:
    """
      Retrieve posted speed limits along road segments by place IDs or path.
      Args:
        placeIds (Optional[List[str]]): One or more road segment place IDs.
        path (Optional[List[str]]): Coordinate path strings as an alternative to placeIds.
        units (str): Speed units "KPH" or "MPH".
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request parameters, and speed limit results.
    """
    url = "https://roads.googleapis.com/v1/speedLimits"
    params: Dict[str, Any] = {}
    if placeIds: params["placeId"] = placeIds
    if path: params["path"] = "|".join(path)
    params["units"] = units
    res = await _get(url, _params_with_key(params))
    return {"endpoint": url, "request": params, "response": res}

# ---------- Geolocation (Cell/Wi‑Fi) ----------
@server.tool()
async def geolocate_home(cellTowers: Optional[List[Dict[str, Any]]] = None,
                         wifiAccessPoints: Optional[List[Dict[str, Any]]] = None,
                         considerIp: bool = True) -> Dict[str, Any]:
    """
      Estimate device location from cell tower, Wi‑Fi, and IP signals using Geolocation API.
      Args:
        cellTowers (Optional[List[Dict[str, Any]]]): Optional list of cell tower observations.
        wifiAccessPoints (Optional[List[Dict[str, Any]]]): Optional list of Wi‑Fi access point observations.
        considerIp (bool): Whether to allow IP-based fallback geolocation.
      Returns:
        data (Dict[str, Any]): JSON with endpoint, request body, and geolocation result.
    """
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={API_KEY}"
    body: Dict[str, Any] = {"considerIp": bool(considerIp)}
    if cellTowers: body["cellTowers"] = cellTowers
    if wifiAccessPoints: body["wifiAccessPoints"] = wifiAccessPoints
    res = await _post_json(url, json_body=body)
    return {"endpoint": url, "request": body, "response": res}

# ---------- Health check ----------
@server.tool()
async def ping() -> Dict[str, Any]:
    """
      Perform a basic health check of the Google Maps MCP server.
      Args:
      Returns:
        data (Dict[str, Any]): JSON with ok flag and list of enabled service names.
    """
    return {
        "ok": True,
        "services": [
            "geocoding", "places_v1", "routes_v2", "distance_matrix", "directions_legacy",
            "timezone", "elevation", "roads", "geolocation", "static_maps", "street_view"
        ]
    }

# ---------- Static Images ----------
@server.tool()
async def static_map(center: Optional[str] = None,
                     zoom: Optional[int] = None,
                     size: str = "640x640",
                     markers: Optional[List[str]] = None,
                     path: Optional[str] = None,
                     scale: int = 2,
                     maptype: str = "roadmap") -> Dict[str, Any]:
    """
      Build a ready-to-use Google Static Maps image URL without making the HTTP request.
      Args:
        center (Optional[str]): Map center as address or "lat,lng".
        zoom (Optional[int]): Map zoom level.
        size (str): Image size string such as "640x640".
        markers (Optional[List[str]]): Optional marker parameter strings.
        path (Optional[str]): Optional path parameter string.
        scale (int): Scale factor for higher resolution.
        maptype (str): Map type such as "roadmap".
      Returns:
        data (Dict[str, Any]): Dictionary containing the generated image_url.
    """
    base = "https://maps.googleapis.com/maps/api/staticmap"
    params: Dict[str, Any] = {"size": size, "scale": scale, "maptype": maptype, "key": API_KEY}
    if center is not None: params["center"] = center
    if zoom is not None: params["zoom"] = zoom
    if markers:
        # multiple markers entries allowed
        for i, m in enumerate(markers):
            params[f"markers{i}"] = m
    if path:
        params["path"] = path
    # Build URL manually to preserve repeated markers
    q = []
    for k, v in params.items():
        if isinstance(v, list):
            for item in v:
                q.append((k, item))
        else:
            q.append((k, v))
    url = base + "?" + urlparse.urlencode(q, doseq=True)
    return {"image_url": url}

@server.tool()
async def street_view_image(location: str,
                            size: str = "640x640",
                            heading: Optional[int] = None,
                            pitch: Optional[int] = None,
                            fov: Optional[int] = None) -> Dict[str, Any]:
    """
      Build a ready-to-use Street View Static image URL.
      Args:
        location (str): Address or "lat,lng" of the panorama location.
        size (str): Image size string such as "640x640".
        heading (Optional[int]): Optional camera heading.
        pitch (Optional[int]): Optional camera pitch.
        fov (Optional[int]): Optional field of view.
      Returns:
        data (Dict[str, Any]): Dictionary containing the generated image_url.
    """
    base = "https://maps.googleapis.com/maps/api/streetview"
    params: Dict[str, Any] = {"location": location, "size": size, "key": API_KEY}
    if heading is not None: params["heading"] = heading
    if pitch is not None: params["pitch"] = pitch
    if fov is not None: params["fov"] = fov
    url = base + "?" + urlparse.urlencode(params)
    return {"image_url": url}


if __name__ == "__main__":
    # Run stdio server
    server.run()

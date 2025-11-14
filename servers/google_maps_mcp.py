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
    Forward geocoding for a human-readable address using Google Geocoding API.

    Args:
        address (str): Free-text address to geocode.
        region (Optional[str]): Region bias (e.g., "us").
        language (Optional[str]): Response language code (e.g., "en").

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters sent (including the API key).
            - response (dict): Full Google Geocoding API JSON response.
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
    Reverse geocoding for coordinates using Google Geocoding API.

    Args:
        lat (float): Latitude.
        lng (float): Longitude.
        language (Optional[str]): Response language code (e.g., "en").

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters sent (including the API key).
            - response (dict): Full Google Geocoding API JSON response.
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
    Text search for places. Uses legacy Places Text Search as a pragmatic fallback.

    Args:
        query (str): Free-text place search query.
        languageCode (Optional[str]): Response language code.
        regionCode (Optional[str]): Region bias.
        locationBias (Optional[Dict[str, Any]]): Optional circle bias for v1 parity.
        maxResultCount (int): Maximum results to return (trimmed client-side).
        fields (Optional[str]): Ignored here; included for signature parity.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters used.
            - response (dict): Places Text Search JSON (results may be trimmed to maxResultCount).
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
    Nearby search for places using Places API (New) v1.

    Args:
        location (Dict[str, float]): {"latitude": float, "longitude": float}.
        radiusMeters (int): Search radius in meters.
        includedTypes (Optional[List[str]]): Filter by place types.
        languageCode (Optional[str]): Response language code.
        regionCode (Optional[str]): Region bias.
        fields (Optional[str]): Field mask override for the response.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - headers (dict): Effective X-Goog-FieldMask sent.
            - request (dict): JSON body used.
            - response (dict): Places API v1 response JSON.
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
    Fetch place details by `place_id` using Places API (New) v1.

    Args:
        place_id (str): Google Place ID.
        fields (Optional[str]): Field mask override; defaults to common fields.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - headers (dict): Effective X-Goog-FieldMask sent.
            - response (dict): Place details JSON.
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
    Download a place photo asset and return a data URL wrapper.

    Args:
        photo_resource (str): Resource path `places/{placeId}/photos/{photoRef}`.
        maxWidthPx (Optional[int]): Maximum width constraint.
        maxHeightPx (Optional[int]): Maximum height constraint.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters used.
            - response (dict): Either JSON or {content_type, data_url} when non-JSON.
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
    Compute a route using Routes API v2.

    Args:
        origin (Dict[str, Any]): {placeId: str} or {latLng: {latitude, longitude}}.
        destination (Dict[str, Any]): Same structure as origin.
        travelMode (str): One of DRIVE|BICYCLE|WALK|TWO_WHEELER|TRANSIT.
        routingPreference (Optional[str]): Routing preference.
        avoidTolls (Optional[bool]): Avoid toll roads.
        optimizeWaypointOrder (Optional[bool]): Optimize waypoint order.
        intermediates (Optional[List[Dict[str, Any]]]): Waypoints.
        fields (Optional[str]): Field mask override for response trimming.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - headers (dict): Effective X-Goog-FieldMask sent.
            - request (dict): JSON body used.
            - response (dict): Routes API response JSON.
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
    Get directions using the legacy Directions API (classic JSON format).

    Args:
        origin (str): Origin address or "lat,lng".
        destination (str): Destination address or "lat,lng".
        mode (str): Travel mode, e.g., "driving".
        waypoints (Optional[List[str]]): Optional waypoint addresses.
        departure_time (Optional[str]): Departure time (as accepted by API).

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str)
            - request (dict)
            - response (dict): Directions API JSON response.
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
    Compute travel times/distances using the legacy Distance Matrix API.

    Args:
        origins (List[str]): List of origins (addresses or "lat,lng").
        destinations (List[str]): List of destinations (addresses or "lat,lng").
        mode (str): Travel mode, e.g., "driving".
        departure_time (Optional[str]): Departure time parameter.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str)
            - request (dict)
            - response (dict): Distance Matrix API JSON response.
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
    Time Zone API for a coordinate.

    Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        timestamp (Optional[int]): Unix epoch seconds used to compute DST offsets.
            Defaults to current time if omitted.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters sent (including the API key and timestamp).
            - response (dict): Time Zone API JSON (e.g., timeZoneId, timeZoneName, dstOffset, rawOffset, status).
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
    Elevation for one or more discrete locations.

    Args:
        locations (List[str]): List like ["lat,lng", ...]. Max ~512 per request.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters sent.
            - response (dict): Elevation API JSON (e.g., results[*].elevation, location, resolution).
    """
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = _params_with_key({"locations": "|".join(locations)})
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}

@server.tool()
async def elevation_along_path(path: List[str], samples: int) -> Dict[str, Any]:
    """
    Elevation sampled along a path polyline.

    Args:
        path (List[str]): Path coordinates as ["lat,lng", ...].
        samples (int): Total number of evenly spaced samples to return.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters sent.
            - response (dict): Elevation API JSON (e.g., results[*].elevation along the path).
    """
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = _params_with_key({"path": "|".join(path), "samples": int(samples)})
    res = await _get(url, params)
    return {"endpoint": url, "request": params, "response": res}

# ---------- Roads API ----------
@server.tool()
async def roads_snap_to_roads(path: List[str], interpolate: bool = False) -> Dict[str, Any]:
    """
    Snap GPS points to the road network.

    Args:
        path (List[str]): Points as ["lat,lng", ...]. Up to ~100 points typical.
        interpolate (bool): If True, inserts additional points to better follow geometry.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters sent.
            - response (dict): Roads API JSON (e.g., snappedPoints[*].location, placeId, originalIndex).
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
    Find the closest road segments for up to 100 points.

    Args:
        points (List[str]): Points as ["lat,lng", ...], max ~100.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters sent.
            - response (dict): Roads API JSON (e.g., snappedPoints grouped by input point).
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
    Get posted speed limits by place IDs or path.

    Args:
        placeIds (Optional[List[str]]): One or more Road Segment place IDs.
        path (Optional[List[str]]): Alternative to placeIds; ["lat,lng", ...] path.
        units (str): Units for speed. One of "KPH" or "MPH". Defaults to "KPH".

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): Query parameters sent.
            - response (dict): Roads API JSON (e.g., speedLimits[*].speedLimit, units, placeId).
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
    Geolocation API: estimate device location from radio/Wi‑Fi/IP signals.

    Args:
        cellTowers (Optional[List[Dict[str, Any]]]): Cell tower observations, each like
            {"cellId": int, "locationAreaCode": int, "mobileCountryCode": int,
             "mobileNetworkCode": int, "signalStrength": int, ...}.
        wifiAccessPoints (Optional[List[Dict[str, Any]]]): Wi‑Fi AP observations, each like
            {"macAddress": str, "signalStrength": int, "channel": int, ...}.
        considerIp (bool): If True, allows IP-based fallback geolocation.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - endpoint (str): The called URL.
            - request (dict): JSON body sent.
            - response (dict): Geolocation API JSON (e.g., location.lat/lng, accuracy in meters).
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
    Basic health check.

    Returns:
        Dict[str, Any]: JSON object with keys:
            - ok (bool): Always True if the server is up.
            - services (List[str]): List of enabled capability names.
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
    Generate a ready-to-use Static Maps image URL (no request performed here).

    Args:
        center (Optional[str]): Map center as address or "lat,lng".
        zoom (Optional[int]): Zoom level.
        size (str): Image size, e.g., "640x640".
        markers (Optional[List[str]]): Repeated marker parameters.
        path (Optional[str]): Path parameter string.
        scale (int): Scale factor.
        maptype (str): Map type, e.g., "roadmap".

    Returns:
        Dict[str, Any]: {"image_url": str} with the static maps URL.
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
    Generate a ready-to-use Street View Static image URL.

    Args:
        location (str): Address or "lat,lng" of the panorama location.
        size (str): Image size, e.g., "640x640".
        heading (Optional[int]): Camera heading.
        pitch (Optional[int]): Camera pitch.
        fov (Optional[int]): Field of view.

    Returns:
        Dict[str, Any]: {"image_url": str} with the Street View image URL.
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

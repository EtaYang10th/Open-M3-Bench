# servers/weather_mcp.py
import requests, asyncio, re
from typing import Literal, Optional, Tuple
from mcp.server.fastmcp import FastMCP
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

server = FastMCP("weather")


_session = requests.Session()
_session.headers.update({
    "User-Agent": "mcp-weather/0.1 (contact@example.com)",  # replace with your contact info
    "Accept": "application/json",
})
_retry = Retry(
    total=3, backoff_factor=0.8,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"])
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))

_ZIP_RE = re.compile(r"^(\d{5})(?:-\d{4})?$")

# State abbreviation mapping (mainly used to filter geocoding results by state)
_STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

def _parse_city_state(text: str) -> Tuple[str, Optional[str]]:
    """Parse “City, ST” / “City ST” / “City, StateName” / “City” → (city, state_abbr|None)."""
    s = text.strip()
    # Comma takes precedence
    if "," in s:
        city_part, state_part = [p.strip() for p in s.split(",", 1)]
    else:
        # Trailing token may be a state abbreviation
        m = re.match(r"^(.*?)[\s,]+([A-Za-z]{2,})$", s)
        if m:
            city_part, state_part = m.group(1).strip(), m.group(2).strip()
        else:
            return s, None

    # Normalize to two-letter abbreviation
    sp = state_part.upper()
    if sp in _STATE_ABBR_TO_NAME:
        return city_part, sp
    # Map full state name back to abbreviation
    for abbr, full in _STATE_ABBR_TO_NAME.items():
        if full.lower() == state_part.lower():
            return city_part, abbr
    # Unrecognized
    return city_part, None

def _geocode_zip_us(zip_text: str) -> Optional[Tuple[float, float, str]]:
    m = _ZIP_RE.match(zip_text.strip())
    if not m:
        return None
    zip5 = m.group(1)
    r = _session.get(f"https://api.zippopotam.us/us/{zip5}", timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    js = r.json()
    places = js.get("places") or []
    if not places:
        return None
    p0 = places[0]
    lat = float(p0.get("latitude"))
    lon = float(p0.get("longitude"))
    place = p0.get("place name") or js.get("place name") or zip5
    state_abbr = p0.get("state abbreviation") or ""
    resolved = f"{place}, {state_abbr} {zip5}".strip()
    return lat, lon, resolved

def _pick_result(results, want_city: Optional[str], want_state_abbr: Optional[str]):
    if not results:
        return None
    # Filter by state first
    if want_state_abbr:
        want_state_full = _STATE_ABBR_TO_NAME.get(want_state_abbr)
        filtered = [r for r in results if (r.get("admin1") or "").lower() == (want_state_full or "").lower()]
        if filtered:
            results = filtered
    # Prefer approximate city-name match (ignore suffixes like Township/City)
    def norm(n: str) -> str:
        return re.sub(r"\b(township|city|village|borough)\b", "", (n or "").lower()).strip()
    if want_city:
        wc = norm(want_city)
        exacts = [r for r in results if norm(r.get("name", "")) == wc]
        if exacts:
            return exacts[0]
    return results[0]

def _geocode_us(name: str):
    """Geocoding: support ZIP or City/State, prefer US.
    Returns (lat, lon, resolved_name)
    """
    # 1) ZIP fallback (preferred, precise)
    hit = _geocode_zip_us(name)
    if hit:
        return hit

    # 2) City[, State]
    city, st = _parse_city_state(name)
    r = _session.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 10,
            "language": "en",
            "format": "json",
            "country": "US",
        },
        timeout=15,
    )
    r.raise_for_status()
    js = r.json()
    results = js.get("results") or []
    if not results:
        raise ValueError(f"Geocoding failed for '{name}'")
    pick = _pick_result(results, city, st)
    lat = float(pick["latitude"])  # type: ignore[index]
    lon = float(pick["longitude"]) # type: ignore[index]
    # Prefer displaying "City, ST"
    admin1 = pick.get("admin1") or ""
    abbr = next((k for k,v in _STATE_ABBR_TO_NAME.items() if v.lower()==admin1.lower()), None)
    resolved = pick.get("name") or city
    if abbr:
        resolved = f"{resolved}, {abbr}"
    return lat, lon, resolved

@server.tool()
def get_weather(location: str, units: Literal["us","metric"]="us") -> str:
    """
    Get current weather for a U.S. location using Open‑Meteo, with robust ZIP/city geocoding.

    Args:
        location (str): City/state like "Piscataway, NJ" or a 5‑digit ZIP.
        units (Literal["us","metric"]): "us" for °F/mph, "metric" for °C/km/h.

    Returns:
        str: Plain-text sentence, e.g.,
            "Current weather in City, ST: 72°F, wind 5 mph, code 3."
            Includes weathercode and resolves the canonical place name.
    """
    lat, lon, resolved = _geocode_us(location)
    r = _session.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon, "current_weather": True,
            "temperature_unit": "fahrenheit" if units == "us" else "celsius",
            "windspeed_unit": "mph" if units == "us" else "kmh",
            "precipitation_unit": "inch" if units == "us" else "mm",
        },
        timeout=15,
    )
    r.raise_for_status()
    cur = (r.json().get("current_weather") or {})
    t = cur.get("temperature")
    w = cur.get("windspeed")
    code = cur.get("weathercode")
    return f"Current weather in {resolved}: {t}{'°F' if units=='us' else '°C'}, wind {w} {'mph' if units=='us' else 'km/h'}, code {code}."

if __name__ == "__main__":
    server.run(transport="stdio")  # you can also call server.run() directly

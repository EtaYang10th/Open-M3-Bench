import os
import requests

# Read the FIPE bearer token from the environment (see .env: CAR_API_KEY).
# Never hardcode credentials in source that gets pushed to a remote.
API_KEY = os.getenv("CAR_API_KEY", "")

# Default to the free, no-auth parallelum FIPE mirror. Only attach a Bearer
# token when CAR_API_KEY is actually set (fipe.online requires one; parallelum
# does not and rejects stale tokens with 401).
BASE_URL = os.getenv("FIPE_API_URL", "https://parallelum.com.br/fipe/api/v1")
HEADERS = {"Accept": "application/json"}
# Only fipe.online requires a Bearer token; parallelum is free/no-auth and
# rejects stray tokens with 401. Attach auth only for the fipe.online host.
if API_KEY and "fipe.online" in BASE_URL:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

import json as _json
import time as _time
from pathlib import Path as _Path

# Local cache for准静态 FIPE data (brands/models/prices). Isolated + git-ignored.
_CACHE_DIR = _Path(__file__).resolve().parents[2] / "tools" / "mock_runtime" / "cache" / "_fipe_http"
_CACHE_TTL = int(os.getenv("CAR_CACHE_TTL", "604800"))  # 7 days


def _cache_get(url):
    fp = _CACHE_DIR / (str(abs(hash(url))) + ".json")
    if fp.exists() and (_time.time() - fp.stat().st_mtime) < _CACHE_TTL:
        try:
            return _json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_put(url, data):
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / (str(abs(hash(url))) + ".json")).write_text(
            _json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def cached_get_json(url, *, max_retries=4, base_delay=2.0, timeout=10):
    """Cache-first GET with exponential backoff on 429/5xx. Returns parsed JSON
    or None. Cache makes准静态 price data resilient to free-tier throttling."""
    cached = _cache_get(url)
    if cached is not None:
        return cached
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                _cache_put(url, data)
                return data
            if resp.status_code == 429 or resp.status_code >= 500:
                _time.sleep(delay)
                delay *= 2
                continue
            return None
        except requests.RequestException:
            _time.sleep(delay)
            delay *= 2
    return None


# ------------------------------
# 1) Get all car brands
# ------------------------------
def getCarBrands() -> str:
    """
    Get all car brands from FIPE API (via fipe.online).
    """
    try:
        url = f"{BASE_URL}/carros/marcas"
        brands = cached_get_json(url)

        if not brands:
            return "Error: Could not fetch car brands (rate-limited or empty after retries)"

        formatted = "Car Brands Available\n\n"
        for b in brands[:20]:
            formatted += f"- {b['nome']} (Code: {b['codigo']})\n"

        formatted += f"\nTotal brands: {len(brands)}"
        return formatted

    except Exception as e:
        return f"Error: {e}"


# ------------------------------
# 2) Brand + Model family search
# ------------------------------
def searchBrandModelPrice(brand_name: str, model_keyword: str) -> str:
    """
    Search all models under a brand whose names contain model_keyword.
    Return ALL matched models with latest prices.
    """
    try:
        # Step 1: get all brands
        url = f"{BASE_URL}/carros/marcas"
        brands = cached_get_json(url)
        if not brands:
            return "Error: could not fetch brands (rate-limited or empty after retries)"

        # Step 2: find brand
        brand_lower = brand_name.lower()
        target_brand = None
        for b in brands:
            if brand_lower in b["nome"].lower():
                target_brand = b
                break

        if not target_brand:
            return f"Brand '{brand_name}' not found."

        brand_code = target_brand["codigo"]

        # Step 3: get all models under this brand
        url = f"{BASE_URL}/carros/marcas/{brand_code}/modelos"
        _mres = cached_get_json(url) or {}
        modelos = _mres.get("modelos", [])
        if not modelos:
            return f"No models found for brand '{target_brand['nome']}'."

        # Step 4: fuzzy match ALL models
        model_kw = model_keyword.lower()
        matched = [m for m in modelos if model_kw in m["nome"].lower()]

        if not matched:
            suggestions = [m["nome"] for m in modelos[:10]]
            return (
                f"No models matching '{model_keyword}' under brand '{target_brand['nome']}'.\n"
                f"Examples: {suggestions}"
            )

        # Step 5: fetch latest price for EACH matched model
        result = f"Matched Models for {target_brand['nome']} — '{model_keyword}'\n\n"

        for idx, chosen in enumerate(matched, start=1):
            model_code = chosen["codigo"]

            # fetch years
            url = f"{BASE_URL}/carros/marcas/{brand_code}/modelos/{model_code}/anos"
            anos = cached_get_json(url)

            if not anos:
                result += f"{idx}. {chosen['nome']} — No year data.\n\n"
                continue

            latest_year = anos[0]["codigo"]

            # fetch price
            price_url = f"{BASE_URL}/carros/marcas/{brand_code}/modelos/{model_code}/anos/{latest_year}"
            price = cached_get_json(price_url)
            if not price:
                result += f"{idx}. {chosen['nome']} — Price unavailable.\n\n"
                continue

            # append formatted result
            result += (
                f"{idx}. {chosen['nome']}\n"
                f"   Year: {price.get('AnoModelo')}\n"
                f"   Fuel: {price.get('Combustivel')}\n"
                f"   Price: {price.get('Valor')}\n"
                f"   FIPE Code: {price.get('CodigoFipe')}\n"
                f"   Reference: {price.get('MesReferencia')}\n\n"
            )

        return result

    except Exception as e:
        return f"Error: {e}"


# =======================================================
# Local test (optional)
# =======================================================
if __name__ == "__main__":
    print("=== Brand List ===")
    print(getCarBrands())

    print("\n=== Example: Brand + Model ===")
    print(searchBrandModelPrice("Toyota", "Corolla Cross"))

# servers/amazon_mcp.py
import os, requests
from mcp.server.fastmcp import FastMCP

server = FastMCP("amazon")

# ==== Rainforest API key ====
RAINFOREST_API_KEY = os.getenv("RAINFOREST_API_KEY")
if not RAINFOREST_API_KEY:
    print("[amazon_mcp] WARNING: Missing Rainforest API Key. Set RAINFOREST_API_KEY.")

BASE_URL = "https://api.rainforestapi.com/request"


def _price_from_search_item(it: dict) -> str:
    price = (it.get("price") or {}).get("raw")
    if price:
        return price
    bw = (it.get("buybox_winner") or {}).get("price") or {}
    if "raw" in bw:
        return bw["raw"]
    return "N/A"


def _price_from_product(p: dict) -> str:
    price = ((p.get("buybox_winner") or {}).get("price") or {}).get("raw")
    if price:
        return price
    price = (p.get("price") or {}).get("raw")
    return price or "N/A"


# ==== Tool 1: search products by keywords ====
@server.tool()
def search_products(
    keywords: str,
    n: int = 1,
    page: int = 1,
    amazon_domain: str = "amazon.com",
) -> str:
    """
      Search Amazon products via Rainforest API and return a concise plain-text list.
      Args:
        keywords (str): Product search keywords.
        n (int): Number of results to include (1–10).
        page (int): Result page index starting from 1.
        amazon_domain (str): Amazon domain such as "amazon.com".
      Returns:
        result (str): Plain-text list of up to n products or an error message.
    """
    if not RAINFOREST_API_KEY:
        return "Rainforest API key not configured."

    kw = (keywords or "").strip()
    if not kw:
        return "Missing 'keywords'."

    try:
        k = int(n)
    except Exception:
        k = 1
    k = max(1, min(k, 10))

    dom = (amazon_domain or "amazon.com").strip() or "amazon.com"

    params = {
        "api_key": RAINFOREST_API_KEY,
        "type": "search",
        "amazon_domain": dom,
        "search_term": kw,
        "page": max(1, int(page or 1)),
    }

    r = requests.get(BASE_URL, params=params, timeout=20)
    if r.status_code != 200:
        return f"Error: {r.status_code} {r.text}"

    js = r.json()
    items = (js.get("search_results") or [])[:k]
    if not items:
        return "No results."

    lines = []
    for it in items:
        title = it.get("title", "")
        url = it.get("link", "")
        img = it.get("image", "")
        price = _price_from_search_item(it)
        lines.append(f"- {title}\n  {url}\n  {img}\n  Price: {price}")
    return "\n".join(lines)

# ==== Tool 2: get product details by ASIN ====
@server.tool()
def get_product(
    asin: str,
    amazon_domain: str = "amazon.com",
) -> str:
    """
      Retrieve product details for a specific Amazon ASIN via Rainforest API.
      Args:
        asin (str): Amazon Standard Identification Number.
        amazon_domain (str): Amazon domain such as "amazon.com".
      Returns:
        result (str): Plain-text block with title, URLs, price, features, or an error message.
    """
    if not RAINFOREST_API_KEY:
        return "Rainforest API key not configured."

    the_asin = (asin or "").strip()
    if not the_asin:
        return "Missing 'asin'."

    dom = (amazon_domain or "amazon.com").strip() or "amazon.com"

    params = {
        "api_key": RAINFOREST_API_KEY,
        "type": "product",
        "amazon_domain": dom,
        "asin": the_asin,
    }

    r = requests.get(BASE_URL, params=params, timeout=20)
    if r.status_code != 200:
        return f"Error: {r.status_code} {r.text}"

    js = r.json()
    product = js.get("product") or {}
    if not product:
        return "No product found."

    title = product.get("title", "")
    url = product.get("link", "")
    img = (product.get("main_image") or {}).get("link", "")
    price = _price_from_product(product)
    features = product.get("features") or []

    body = [f"**{title}**", url, img, f"Price: {price}", ""]
    if features:
        body.append("Features:")
        body.extend(f"- {f}" for f in features)
    return "\n".join(body)

if __name__ == "__main__":
    server.run(transport="stdio")

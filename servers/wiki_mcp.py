# servers/wiki_mcp.py
import requests, asyncio
from urllib.parse import quote
from mcp.server.fastmcp import FastMCP
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

server = FastMCP("wiki")

# --- new: shared session with proper headers & retries ---
_session = requests.Session()
_session.headers.update({
    "User-Agent": "mcp-wiki/0.1 (eta@rutgers.edu)",   # <-- replace with your contact info
    "Accept": "application/json",
})
_retry = Retry(
    total=3, backoff_factor=1.0,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"])
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))

@server.tool()
def search(query: str, n: int = 3) -> str:
    """
      Search Wikipedia for pages matching a query and return top titles.
      Args:
        query (str): Free-text search query.
        n (int): Number of results to include (1–10).
      Returns:
        result (str): Plain-text list of up to n titles or a no-results message.
    """
    params = {
        "action": "query", "format": "json", "list": "search",
        "utf8": 1, "srlimit": max(1, min(int(n), 10)), "srsearch": query,
        "maxlag": 5,  # be friendly to avoid backend overload when busy
    }
    
    r = _session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=15)
    if r.status_code == 403:
        # fallback: REST search (no token required; backup)
        r2 = _session.get(
            "https://api.wikimedia.org/core/v1/wikipedia/en/search/page",
            params={"q": query, "limit": max(1, min(int(n), 10))},
            timeout=15,
            headers={"Api-User-Agent": _session.headers.get("User-Agent", "mcp-wiki/0.1")}
        )
        r2.raise_for_status()
        js = r2.json()
        hits = [p.get("title", "") for p in js.get("pages", [])]
        return "Top results:\n" + "\n".join(f"- {t}" for t in hits) if hits else "No results."
    r.raise_for_status()
    hits = [h["title"] for h in r.json().get("query", {}).get("search", [])]
    return "Top results:\n" + "\n".join(f"- {t}" for t in hits) if hits else "No results."

@server.tool()
def summary(title: str) -> str:
    """
      Fetch a short Wikipedia summary for the given page title.
      Args:
        title (str): Exact or redirectable page title.
      Returns:
        result (str): Plain-text block with title, short description, and extract.
    """
    
    r = _session.get(
        "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title),
        params={"redirect": "true"},  # follow redirects for robustness
        timeout=15
    )
    if r.status_code == 403:
        r = _session.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title),
            params={"redirect": "true"},
            timeout=15,
            headers={"Api-User-Agent": _session.headers.get("User-Agent", "mcp-wiki/0.1")}
        )
    r.raise_for_status()
    js = r.json()
    t = js.get("title","") or ""
    desc  = js.get("description","") or ""
    extract = js.get("extract","") or ""
    return f"**{t}**\n{desc}\n\n{extract}"

if __name__ == "__main__":
    server.run(transport="stdio")

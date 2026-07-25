"""Functional health-check: invoke one representative tool per MCP server and
verify it returns a non-empty, non-error response.

Output: tools/mcp_functional_report.json
Usage: python tools/functional_test_mcp_servers.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "mcp_servers.json"

# Per-server probe: (tool_name, arguments). None => "skip, list_tools only".
PROBES: dict[str, tuple[str, dict[str, Any]] | None] = {
    "weather":           ("get_weather",    {"location": "New York", "units": "us"}),
    "wiki":              ("search",         {"query": "Albert Einstein", "n": 3}),
    "ocr":               ("get_supported_languages", {}),
    "amazon":            ("search_products", {"keywords": "laptop", "n": 1}),
    "google-maps":       ("geocode",        {"address": "1600 Amphitheatre Parkway, Mountain View, CA"}),
    "tmdb":              ("search_movies",  {"query": "Inception"}),
    "pyzbar-mcp":        None,  # needs a barcode image, skip heavy
    "openlibrary_mcp":   ("get_book_info",  {"isbn": "9780140328721"}),
    "imagesorcery-mcp":  None,  # needs a local image & writes output
    "healthcare-mcp":    ("fda_drug_lookup", {"drug_name": "aspirin"}),
    "food_nutrition_mcp": ("get_nutrition", {"query": "apple"}),
    "mcp-yolo":          None,  # heavy: model weights & image
    "linkimage-mcp":     ("fetch_unsplash_image",
                          {"url": "https://unsplash.com/photos/a-laptop-computer-sitting-on-top-of-a-wooden-table-eMP4sYPJ9x0"}),
    "google-air":        ("current_conditions", {"lat": 40.7128, "lng": -74.0060}),
    "ppt":               ("get_server_info", {}),
    "Reddit-MCP-Server": ("search_hot_posts", {"subreddit": "news", "limit": 1}),
    "excel":             ("create_workbook", {"filepath": "media/_mcp_healthcheck.xlsx"}),
    "nationalparks":     ("findParks",      {"q": "Yellowstone", "limit": 1}),
    "paper_search":      ("search_arxiv",   {"query": "quantum computing", "max_results": 1}),
    "metmuseum-mcp":     ("list-departments", {"__intent": "healthcheck"}),
    "nasa-mcp":          ("get_astronomy_picture_of_day", {}),
    "okx":               ("get_price",      {"instrument": "BTC-USDT"}),
    "hugeicons-mcp":     ("search_icons",   {"query": "home"}),
    "yahoo-finance":     ("get_stock_info", {"ticker": "AAPL"}),
    "math":              ("add",            {"firstNumber": 2, "secondNumber": 3}),
    "nixos":             ("nix",            {"action": "search", "query": "git", "type": "packages", "limit": 1}),
    "car-price":         ("get_car_brands", {}),
}

PER_CALL_TIMEOUT = 45.0   # seconds per tool call
START_TIMEOUT = 30.0      # seconds to initialize server


def load_env_file() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        os.environ[k] = v


def _extract_text(result: Any) -> tuple[bool, str]:
    """Return (is_error, short_text_preview)."""
    is_error = bool(getattr(result, "isError", False))
    parts: list[str] = []
    for c in (getattr(result, "content", None) or []):
        t = getattr(c, "type", None)
        if t == "text":
            parts.append(getattr(c, "text", "") or "")
        elif t == "json":
            try:
                parts.append(json.dumps(getattr(c, "data", None), ensure_ascii=False)[:400])
            except Exception:  # noqa: BLE001
                pass
        elif t == "image":
            parts.append("[image]")
    preview = " | ".join(p.strip().replace("\n", " ") for p in parts if p)[:600]
    return is_error, preview or "[empty content]"


async def test_server(name: str, scfg: dict) -> dict:
    merged_env = dict(os.environ)
    merged_env.update(scfg.get("env") or {})
    params = StdioServerParameters(
        command=scfg["command"],
        args=scfg.get("args", []),
        env=merged_env,
    )
    probe = PROBES.get(name)

    res: dict[str, Any] = {"server": name, "init": False}
    try:
        async with asyncio.timeout(START_TIMEOUT + PER_CALL_TIMEOUT + 10):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), START_TIMEOUT)
                    res["init"] = True
                    tools_resp = await asyncio.wait_for(session.list_tools(), 15)
                    res["num_tools"] = len(tools_resp.tools)

                    if probe is None:
                        res["status"] = "LIST_ONLY"
                        res["note"] = "no probe configured (skipped)"
                        return res

                    tool_name, arguments = probe
                    res["tool"] = tool_name
                    res["arguments"] = arguments
                    try:
                        call = await asyncio.wait_for(
                            session.call_tool(tool_name, arguments),
                            PER_CALL_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        res["status"] = "TIMEOUT"
                        res["error"] = f"call_tool > {PER_CALL_TIMEOUT}s"
                        return res
                    is_err, preview = _extract_text(call)
                    res["is_error"] = is_err
                    res["preview"] = preview
                    res["status"] = "ERROR" if is_err else "OK"
                    return res
    except asyncio.TimeoutError:
        res["status"] = "START_TIMEOUT"
        res["error"] = "init or whole call exceeded overall timeout"
        return res
    except Exception as e:  # noqa: BLE001
        res["status"] = "CRASH"
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc(limit=3)
        return res


async def main():
    load_env_file()
    os.chdir(REPO)
    (REPO / "media").mkdir(exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    servers = cfg.get("servers", {})

    results: list[dict] = []
    for name, scfg in servers.items():
        if scfg.get("disabled"):
            print(f"--- {name}: DISABLED (skipped) ---")
            results.append({"server": name, "status": "DISABLED"})
            continue
        print(f"--- {name} ---", flush=True)
        r = await test_server(name, scfg)
        results.append(r)
        msg = r.get("preview") or r.get("error") or r.get("note") or ""
        print(f"    [{r.get('status')}] tool={r.get('tool')} -> {msg[:200]}", flush=True)

    out = REPO / "tools" / "mcp_functional_report.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved report: {out}")

    # Summary table
    width_s, width_t = 20, 12
    print("\n" + "=" * 100)
    print(f"{'Server':<22} | {'Status':<14} | {'Tool':<28} | preview/error")
    print("-" * 100)
    for r in results:
        s = r.get("server", "?")
        st = r.get("status", "?")
        tn = r.get("tool") or "-"
        msg = r.get("preview") or r.get("error") or r.get("note") or ""
        print(f"{s:<22} | {st:<14} | {tn:<28} | {msg[:60]}")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())

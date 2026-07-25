# tools/mock_fallback.py
"""
Local 3-tier fallback for unstable / dead external MCP APIs
(StableToolBench-style: real -> cache -> LLM simulator).

DEFAULT OFF. Enable with env M3_MOCK_FALLBACK=1.

Flow (invoked by mcp_host.call ONLY after a real call failed):
  1. real call succeeds        -> host uses real result (this module untouched)
  2. record-replay CACHE hit   -> deterministic replay from mock_runtime/cache/
  3. LLM simulator             -> generate a lively, schema-valid payload,
                                  then persist to cache so future calls replay it
  4. generic synthetic template (graceful degradation; never raises)

Allow-list = "has a fixture OR is listed in _MOCKABLE". Visual/OCR tools have
neither, so they are never mocked.

Runtime artifacts are isolated under tools/mock_runtime/ (git-ignored):
  fixtures/<server>/<tool>.json   committed sample fixtures (curated)
  cache/<server>/<tool>.jsonl     record-replay cache (runtime, ignored)
  logs/_mock_calls.log            audit log of every served mock (ignored)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_RT = Path(__file__).resolve().parent / "mock_runtime"
_FIXTURE_DIR = _RT / "fixtures"
_CACHE_DIR = _RT / "cache"
_LOG_PATH = _RT / "logs" / "_mock_calls.log"

# Explicit allow-list of tools we permit to be simulated (deterministic/dict-like).
# Visual/OCR/file tools are intentionally excluded.
_MOCKABLE = {
    "amazon/search_products",
    "amazon/get_product",
    "car-price/get_car_brands",
    "car-price/search_brand_model_price",
    "nasa-mcp/get_solar_flare",
    "nasa-mcp/get_coronal_mass_ejection",
    "nasa-mcp/get_notifications",
    "paper_search/search_arxiv",
}

# Output-contract hints per tool, used to steer the LLM simulator toward the
# exact shape the real server returns (see servers/amazon_mcp.py etc.).
_CONTRACT_HINTS = {
    "amazon/search_products": (
        "Plain text. One block per product, up to n products:\n"
        "- <title>\n  <product url>\n  <image url>\n  Price: <$X.XX>\n"
        "Return only the list, no header."
    ),
    "amazon/get_product": (
        "Plain text block:\n**<title>**\n<url>\n<image url>\nPrice: <$X.XX>\n\n"
        "Features:\n- <feature 1>\n- <feature 2>"
    ),
}

_ERROR_MARKERS = (
    "Rainforest API key", "key is deactivated", "[Tool error]",
    "Error: 4", "Status: 4", "status 4", "OVER_RATE_LIMIT",
    "Too Many Requests", "HTTP 401", "HTTP 403", "HTTP 429", "not configured",
    "rate-limited or empty", "could not fetch", "Could not fetch",
)


def mock_enabled() -> bool:
    return os.getenv("M3_MOCK_FALLBACK", "0").strip() in ("1", "true", "True", "yes")


def sim_enabled() -> bool:
    """LLM simulator tier; on by default when fallback is on. Disable with M3_MOCK_LLM=0."""
    return os.getenv("M3_MOCK_LLM", "1").strip() in ("1", "true", "True", "yes")


def mark_inline() -> bool:
    return os.getenv("M3_MOCK_MARK_INLINE", "0").strip() in ("1", "true", "True", "yes")


def looks_like_error(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        s = result.strip()
        if not s or s == "[no textual content]":
            return True
        return any(m in s for m in _ERROR_MARKERS)
    return False


def _norm_args(arguments: Any) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(arguments)


def _args_key(arguments: Any) -> str:
    return hashlib.sha1(_norm_args(arguments).encode("utf-8")).hexdigest()[:16]


def _to_str(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


# ----------------------------- tiers -----------------------------

def _fixture_lookup(server: str, tool: str, arguments: Any) -> Optional[Any]:
    fp = _FIXTURE_DIR / server / f"{tool}.json"
    if not fp.exists():
        return None
    try:
        records = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(records, list):
        return None
    target = _norm_args(arguments)
    default = None
    for rec in records:
        ra = rec.get("arguments")
        if ra == "*":
            default = rec.get("result")
        elif _norm_args(ra) == target:
            return rec.get("result")
    return default


def _cache_path(server: str, tool: str) -> Path:
    return _CACHE_DIR / server / f"{tool}.jsonl"


def _cache_lookup(server: str, tool: str, arguments: Any) -> Optional[Any]:
    fp = _cache_path(server, tool)
    if not fp.exists():
        return None
    key = _args_key(arguments)
    try:
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("key") == key:
                return rec.get("result")
    except Exception:
        return None
    return None


def cache_store(server: str, tool: str, arguments: Any, result: Any, source: str = "recorded") -> None:
    fp = _cache_path(server, tool)
    fp.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "key": _args_key(arguments),
        "arguments": arguments,
        "result": result,
        "source": source,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _generic_template(server: str, tool: str, arguments: Any) -> str:
    return json.dumps(
        {"mocked": True, "note": "generic synthetic fallback (LLM unavailable)",
         "tool": f"{server}/{tool}", "arguments": arguments}, ensure_ascii=False)


def _log(server: str, tool: str, arguments: Any, tier: str) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "server": server, "tool": tool, "arguments": arguments,
                "tier": tier, "mocked": True,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def try_fallback(server: str, tool: str, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return (served, payload). served=False -> keep the real error."""
    if not mock_enabled():
        return False, None
    qn = f"{server}/{tool}"
    has_fixture = (_FIXTURE_DIR / server / f"{tool}.json").exists()
    if qn not in _MOCKABLE and not has_fixture:
        return False, None  # allow-list gate

    tier = None
    result = None

    # Tier 2a: curated fixture (highest trust)
    result = _fixture_lookup(server, tool, arguments)
    if result is not None:
        tier = "fixture"
    # Tier 2b: record-replay cache
    if result is None:
        result = _cache_lookup(server, tool, arguments)
        if result is not None:
            tier = "cache"
    # Tier 3: LLM simulator (then persist to cache)
    if result is None and sim_enabled():
        try:
            from tools.llm_simulator import simulate
        except Exception:
            simulate = None
        if simulate is not None:
            hint = _CONTRACT_HINTS.get(qn, "Return a realistic, structurally valid payload.")
            sim = simulate(server, tool, arguments, hint)
            if sim:
                result = sim
                tier = "llm"
                cache_store(server, tool, arguments, sim, source="llm")
    # Tier 4: generic synthetic template (graceful degradation)
    if result is None:
        result = _generic_template(server, tool, arguments)
        tier = "generic"

    _log(server, tool, arguments, tier)
    payload = _to_str(result)
    if mark_inline():
        payload = json.dumps({"mocked": True, "tier": tier, "result": result}, ensure_ascii=False)
    return True, payload

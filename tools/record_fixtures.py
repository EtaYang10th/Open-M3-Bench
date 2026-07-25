"""Record real MCP tool returns into the record-replay cache.

While an external API still works, call it with the *real GT arguments* and
store successful returns under tools/mock_runtime/cache/<server>/<tool>.jsonl.
Later, when the API is down, tools/mock_fallback.py replays these deterministically.

Usage:
  python tools/record_fixtures.py --servers car-price paper_search nasa-mcp
  python tools/record_fixtures.py --servers car-price --limit 3
  python tools/record_fixtures.py --list           # show planned calls only

Notes:
- Reads GT arguments from json/test_mcp_GT.json (never modifies it).
- Respects rate limits with --delay (default 4s) between calls.
- NASA needs a real NASA_API_KEY to avoid DEMO_KEY 429; otherwise it is skipped
  with a note (record later once the key is set).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "mcp_servers.json"
GT = REPO / "json" / "test_mcp_GT.json"

sys.path.insert(0, str(REPO))
from tools.mock_fallback import cache_store, looks_like_error  # noqa: E402

PER_CALL_TIMEOUT = 90.0


def load_env_file() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if (v[:1], v[-1:]) in (('"', '"'), ("'", "'")):
            v = v[1:-1]
        os.environ[k] = v


def collect_gt_calls(servers: List[str]) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """server -> list of (tool, arguments) de-duplicated, taken from GT."""
    data = json.loads(GT.read_text(encoding="utf-8"))
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {s: [] for s in servers}
    seen: Dict[str, set] = {s: set() for s in servers}
    for task in data:
        for step in task.get("steps", []):
            for call in step.get("calls", []):
                name = call.get("name", "")
                if "/" not in name:
                    continue
                srv, tool = name.split("/", 1)
                if srv not in out:
                    continue
                args = call.get("arguments", {}) or {}
                key = json.dumps(args, sort_keys=True, ensure_ascii=False)
                if key in seen[srv]:
                    continue
                seen[srv].add(key)
                out[srv].append((tool, args))
    return out


def _extract(result: Any) -> Tuple[bool, str]:
    is_err = bool(getattr(result, "isError", False))
    parts: List[str] = []
    for c in (getattr(result, "content", None) or []):
        t = getattr(c, "type", None)
        if t == "text":
            parts.append(getattr(c, "text", "") or "")
        elif t == "json":
            parts.append(json.dumps(getattr(c, "data", None), ensure_ascii=False))
    return is_err, "\n".join(p for p in parts if p)


async def record_server(name: str, scfg: dict, calls: List[Tuple[str, Dict[str, Any]]],
                        delay: float, limit: int) -> dict:
    merged = dict(os.environ)
    merged.update(scfg.get("env") or {})
    params = StdioServerParameters(command=scfg["command"], args=scfg.get("args", []), env=merged)
    recorded, failed = 0, 0
    if limit > 0:
        calls = calls[:limit]
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool, args in calls:
                try:
                    call = await asyncio.wait_for(session.call_tool(tool, args), PER_CALL_TIMEOUT)
                    is_err, payload = _extract(call)
                except Exception as e:  # noqa: BLE001
                    is_err, payload = True, f"[Tool error] {type(e).__name__}: {e}"
                if is_err or looks_like_error(payload):
                    failed += 1
                    print(f"    [SKIP] {name}/{tool} {args} -> {payload[:100]}", flush=True)
                else:
                    cache_store(name, tool, args, payload, source="recorded")
                    recorded += 1
                    print(f"    [OK]   {name}/{tool} {args} -> {payload[:80]}", flush=True)
                await asyncio.sleep(delay)
    return {"server": name, "recorded": recorded, "failed": failed}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", nargs="+",
                    default=["car-price", "paper_search", "nasa-mcp"])
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="max calls per server (0=all)")
    ap.add_argument("--list", action="store_true", help="only print planned calls")
    args = ap.parse_args()

    load_env_file()
    os.chdir(REPO)
    cfg = json.loads(CONFIG.read_text())
    servers = cfg.get("servers", {})
    plan = collect_gt_calls(args.servers)

    if args.list:
        for srv, calls in plan.items():
            print(f"# {srv}: {len(calls)} unique GT calls")
            for tool, a in calls:
                print(f"    {tool} {json.dumps(a, ensure_ascii=False)}")
        return

    summary = []
    for srv in args.servers:
        scfg = servers.get(srv)
        if not scfg or scfg.get("disabled"):
            print(f"--- {srv}: missing/disabled, skip ---", flush=True)
            continue
        if srv == "nasa-mcp" and not (os.environ.get("NASA_API_KEY") or "").strip():
            print(f"--- nasa-mcp: no NASA_API_KEY (DEMO_KEY will 429); "
                  f"skip, record later once key is set ---", flush=True)
            continue
        print(f"--- recording {srv} ({len(plan.get(srv, []))} calls) ---", flush=True)
        summary.append(await record_server(srv, scfg, plan.get(srv, []), args.delay, args.limit))
    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['server']}: recorded={s['recorded']} failed={s['failed']}")


if __name__ == "__main__":
    asyncio.run(main())

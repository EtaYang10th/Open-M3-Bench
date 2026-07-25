"""Functional verification of EVERY tool of EVERY enabled MCP server.

For each non-disabled server in mcp_servers.json, this script opens a real
stdio session and issues a real `call_tool` for each tool using realistic
fixture arguments (see FIXTURES below). It then classifies the outcome:

  OK                -> real, non-empty, non-error content returned
  EMPTY             -> call succeeded but content is empty / no useful data
  ERROR             -> tool returned isError or raised a tool-level error
  DEPRECATED_OR_AUTH-> error text looks like API removed / 401 / 403 / quota /
                       invalid key / billing (suspected dead external API/key)
  CRASH             -> server would not start or the call timed out
  SKIPPED           -> no fixture provided for this tool

Usage
-----
  # test everything (uses the mcp_app conda env python on PATH)
  python tools/verify_all_tools.py

  # only one (or several) servers
  python tools/verify_all_tools.py --server google-maps --server google-air

  # only list what WOULD be tested (no calls)
  python tools/verify_all_tools.py --dry-run

  # tune timeout / output
  python tools/verify_all_tools.py --timeout 90 --out tools/verify_report.json

IMPORTANT: launch with an interpreter/PATH that can spawn the `python`, `node`,
`uv`, `uvx` based servers. The single-file python servers are started with the
bare command `python`, so put the right env's bin on PATH, e.g.:

  PATH=/path/to/envs/mcp_app/bin:$PATH \\
      python tools/verify_all_tools.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "mcp_servers.json"
MEDIA = REPO / "media"

# ---- realistic fixtures -------------------------------------------------
SAMPLE_PNG = str((MEDIA / "00000000.png").resolve())
QR_PNG = str((MEDIA / "_qr_healthcheck.png").resolve())
XLSX = str((MEDIA / "_mcp_healthcheck.xlsx").resolve())
CROP_OUT = str((MEDIA / "_crop_healthcheck.png").resolve())

# Status labels
OK, EMPTY, ERROR, DEP, CRASH, SKIP = (
    "OK", "EMPTY", "ERROR", "DEPRECATED_OR_AUTH", "CRASH", "SKIPPED",
)

# Substrings (lowercase) that strongly indicate a dead API / auth / quota issue
# NOTE: keep these as specific phrases. Bare "401"/"403" are intentionally NOT
# used because those digit sequences occur inside legitimate JSON payloads
# (coordinates, ids, distances) and cause false positives.
AUTH_MARKERS = [
    "api key not valid", "invalid api key", "api_key_invalid",
    "permission_denied", "permission denied", "request_denied", "requestdenied",
    "unauthorized", "unauthenticated", "http 401", "http 403", "403 forbidden",
    "401 unauthorized", "status: forbidden",
    "quota exceeded", "rate limit", "over_rate_limit", "over_query_limit",
    "too many requests", "http error: 429", "status: 429", "(status: 429)",
    "billing", "has not been used in project",
    "is not enabled", "api not enabled", "disabled for this project",
    "token expired", "invalid_grant", "invalid access token", "access denied",
    "no longer available", "has been shut down", "api has been shut down",
    "not authorized", "keyinvalid", "over_query_limit", "not available for this project",
]
EMPTY_MARKERS = [
    "[empty content]", "[no textual content]", "no results", "not found",
    "empty", "[]", "{}", "no data", "0 results",
]


def load_env_file() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if (v.startswith('"') and v.endswith('"')) or (
            v.startswith("'") and v.endswith("'")
        ):
            v = v[1:-1]
        os.environ[k] = v


def _extract(result: Any) -> tuple[bool, str]:
    """Return (is_error, text_preview) from a CallToolResult."""
    is_err = bool(getattr(result, "isError", False))
    parts: list[str] = []
    for c in (getattr(result, "content", None) or []):
        t = getattr(c, "type", None)
        if t == "text":
            parts.append(getattr(c, "text", "") or "")
        elif t == "json":
            parts.append(json.dumps(getattr(c, "data", None), ensure_ascii=False))
        elif t == "image":
            parts.append("[image]")
        else:
            parts.append(str(getattr(c, "text", "") or ""))
    # structuredContent fallback (fastmcp returns dicts here)
    sc = getattr(result, "structuredContent", None)
    if sc:
        parts.append(json.dumps(sc, ensure_ascii=False))
    joined = " | ".join(p.strip().replace("\n", " ") for p in parts if p and p.strip())
    return is_err, joined


def classify(is_err: bool, text: str, exc: str | None) -> str:
    # Auth/deprecation markers are ONLY meaningful on an error path. A
    # successful tool result may legitimately contain words like "billing" or
    # "unauthorized" inside its content (e.g. icon tags, movie overviews), so
    # we never scan a successful body for those markers.
    if exc and not text:
        blob = exc.lower()
        if "timeout" in blob or "timed out" in blob or "cancel" in blob:
            return CRASH
        if any(m in blob for m in AUTH_MARKERS):
            return DEP
        return ERROR
    if is_err:
        blob = f"{text} {exc or ''}".lower()
        if any(m in blob for m in AUTH_MARKERS):
            return DEP
        return ERROR
    # success path
    if not text.strip():
        return EMPTY
    stripped = text.strip()
    if stripped in ("[]", "{}", "null", "[empty content]", "[no textual content]"):
        return EMPTY
    low = stripped.lower()
    # Some servers wrap failures inside a *successful* response body, e.g.
    # {"error": "Failed to fetch ... HTTP 401"}. Detect an error-shaped payload
    # (short, dominated by an "error" field) and classify by its content.
    looks_like_error_payload = (
        len(stripped) < 400
        and ('"error"' in low or low.startswith("error") or "traceback" in low
             or "api error" in low or '"status": "error"' in low
             or "returned unexpected content type" in low)
    )
    # Rate-limit / auth markers anywhere in a *short* body are also telling
    # (long successful payloads that merely mention these words are excluded by
    # the length guard and the earlier success checks).
    rate_or_auth = len(stripped) < 500 and any(m in low for m in AUTH_MARKERS)
    if looks_like_error_payload or rate_or_auth:
        if any(m in low for m in AUTH_MARKERS):
            return DEP
        return ERROR
    # short "no results" style responses
    if len(stripped) < 80 and any(m in low for m in ("no results", "not found", "no data", "0 results", "empty")):
        return EMPTY
    return OK


try:
    from verify_fixtures import FIXTURES  # when run from tools/
except ImportError:  # when run from repo root
    sys.path.insert(0, str(REPO / "tools"))
    from verify_fixtures import FIXTURES


async def prep_excel_fixture(servers: dict) -> None:
    """Create a clean, sufficiently large workbook the excel fixtures rely on.

    Prefer openpyxl (fast, local). If it is not importable in the launcher
    interpreter, fall back to driving the excel MCP server's own
    create_workbook / write_data_to_excel tools so the script stays
    self-contained regardless of the launcher env.
    """
    p = MEDIA / "_verify_excel.xlsx"
    newp = MEDIA / "_verify_excel_new.xlsx"
    if p.exists():
        p.unlink()
    if newp.exists():
        newp.unlink()

    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["a", "b", "c"])
        for i in range(1, 30):  # 29 data rows -> bounds ops stay valid
            ws.append([i, i * 2, i * 3])
        wb.save(str(p))
        return
    except Exception:
        pass

    # Fallback: build it via the excel MCP server.
    scfg = servers.get("excel")
    if not scfg:
        return
    merged = dict(os.environ)
    merged.update(scfg.get("env") or {})
    params = StdioServerParameters(command=scfg["command"], args=scfg.get("args", []), env=merged)
    data = [["a", "b", "c"]] + [[i, i * 2, i * 3] for i in range(1, 30)]
    try:
        async with asyncio.timeout(120):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await session.call_tool("create_workbook", {"filepath": str(p)})
                    await session.call_tool("write_data_to_excel",
                                            {"filepath": str(p), "sheet_name": "Sheet1",
                                             "data": data, "start_cell": "A1"})
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not prep excel fixture: {e}", flush=True)


async def test_one_server(name: str, scfg: dict, timeout: float,
                          only_tool: str | None = None) -> list[dict]:
    """Open a session and call every tool of one server with its fixture."""
    merged = dict(os.environ)
    merged.update(scfg.get("env") or {})
    params = StdioServerParameters(
        command=scfg["command"], args=scfg.get("args", []), env=merged,
    )
    results: list[dict] = []
    fixtures = FIXTURES.get(name, {})
    try:
        async with asyncio.timeout(timeout + 30):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    tools = [t.name for t in resp.tools]
                    for tool in tools:
                        if only_tool and tool != only_tool:
                            continue
                        if tool not in fixtures:
                            results.append({"server": name, "tool": tool,
                                            "status": SKIP, "detail": "no fixture defined"})
                            continue
                        args = fixtures[tool]
                        if args is None:
                            results.append({"server": name, "tool": tool,
                                            "status": SKIP, "detail": "intentionally skipped (needs prior state / not testable in isolation)"})
                            continue
                        t0 = time.time()
                        try:
                            call = await asyncio.wait_for(
                                session.call_tool(tool, args), timeout)
                            is_err, text = _extract(call)
                            status = classify(is_err, text, None)
                            results.append({"server": name, "tool": tool, "status": status,
                                            "detail": text[:500], "secs": round(time.time() - t0, 1)})
                        except Exception as e:  # noqa: BLE001
                            exc = f"{type(e).__name__}: {e}"
                            status = classify(False, "", exc)
                            results.append({"server": name, "tool": tool, "status": status,
                                            "detail": exc[:500], "secs": round(time.time() - t0, 1)})
    except Exception as e:  # noqa: BLE001
        # server would not start / session-level timeout -> all tools CRASH
        exc = f"{type(e).__name__}: {e}"
        # If we already tested some tools, keep them; mark server-level failure too.
        tested = {r["tool"] for r in results}
        for tool in fixtures:
            if tool not in tested:
                results.append({"server": name, "tool": tool, "status": CRASH,
                                "detail": f"server/session failure: {exc}"[:500]})
        if not results:
            results.append({"server": name, "tool": "<session>", "status": CRASH,
                            "detail": exc[:500]})
    return results


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="append", default=None,
                    help="only test this server (repeatable)")
    ap.add_argument("--tool", default=None, help="only this tool (needs single --server)")
    ap.add_argument("--timeout", type=float, default=90.0, help="per-call timeout seconds")
    ap.add_argument("--out", default=str(REPO / "tools" / "verify_report.json"))
    ap.add_argument("--dry-run", action="store_true", help="list planned calls only")
    args = ap.parse_args()

    load_env_file()
    os.chdir(REPO)

    cfg = json.loads(CONFIG.read_text())
    servers = cfg.get("servers", {})
    await prep_excel_fixture(servers)
    wanted = set(args.server) if args.server else None

    plan = []
    for name, scfg in servers.items():
        if scfg.get("disabled"):
            continue
        if wanted and name not in wanted:
            continue
        plan.append((name, scfg))

    if args.dry_run:
        for name, _ in plan:
            fx = FIXTURES.get(name, {})
            tested = [t for t, v in fx.items() if v is not None]
            skipped = [t for t, v in fx.items() if v is None]
            print(f"{name}: {len(tested)} to-call, {len(skipped)} skip, "
                  f"undefined-fixture tools will be SKIPPED")
        return

    all_results: list[dict] = []
    for name, scfg in plan:
        print(f"\n=== {name} ===", flush=True)
        res = await test_one_server(name, scfg, args.timeout, args.tool)
        for r in res:
            print(f"  [{r['status']:18s}] {r['tool']:32s} {r.get('detail','')[:90]}", flush=True)
        all_results.extend(res)

    # summary
    from collections import Counter
    by_status = Counter(r["status"] for r in all_results)
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totals": dict(by_status),
        "results": all_results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n========== SUMMARY ==========")
    for s in (OK, EMPTY, ERROR, DEP, CRASH, SKIP):
        print(f"  {s:18s}: {by_status.get(s, 0)}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

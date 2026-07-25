"""List tools exposed by every enabled MCP server in mcp_servers.json.

Usage:
    python tools/list_mcp_tools.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "mcp_servers.json"


def load_env_file() -> None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if (v.startswith('"') and v.endswith('"')) or (
            v.startswith("'") and v.endswith("'")
        ):
            v = v[1:-1]
        os.environ[k] = v


async def _list_one(name: str, scfg: dict, timeout: float = 20.0) -> dict:
    merged_env = dict(os.environ)
    merged_env.update(scfg.get("env") or {})
    params = StdioServerParameters(
        command=scfg["command"],
        args=scfg.get("args", []),
        env=merged_env,
    )
    try:
        async with asyncio.timeout(timeout):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    return {
                        "ok": True,
                        "tools": [
                            {
                                "name": t.name,
                                "description": (t.description or "").strip(),
                                "schema": t.inputSchema,
                            }
                            for t in resp.tools
                        ],
                    }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


async def main():
    load_env_file()
    cfg = json.loads(CONFIG.read_text())
    os.chdir(REPO)
    results: dict[str, dict] = {}
    for name, scfg in cfg.get("servers", {}).items():
        if scfg.get("disabled"):
            results[name] = {"ok": False, "error": "disabled"}
            continue
        print(f"-> listing {name} ...", flush=True)
        results[name] = await _list_one(name, scfg)
        if results[name]["ok"]:
            print(
                f"   [{len(results[name]['tools'])} tools] "
                + ", ".join(t["name"] for t in results[name]["tools"])
            )
        else:
            print(f"   FAIL: {results[name]['error']}")
    out = REPO / "tools" / "mcp_tools_dump.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    asyncio.run(main())

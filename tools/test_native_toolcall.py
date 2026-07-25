"""Minimal end-to-end test of NATIVE tool calling via the apicursor endpoint.

Flow:
  1. Load .env (CURSOR_API_BASE_URL / CURSOR_API_KEY).
  2. Start the lightweight `math` MCP server over stdio and list its tools.
  3. Build OpenAI-style function schemas (with escaped names) from those tools.
  4. Ask CursorAPIClient.generate_with_tools() to solve a small arithmetic task.
  5. Restore qualified names, dispatch the structured tool_calls to the MCP
     server, and print the real tool results.

Usage:
    python tools/test_native_toolcall.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from models.api_clients import CursorAPIClient
from models.tool_schema import build_openai_tools


def load_env_file() -> None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        os.environ.setdefault(k, v)


# Server to exercise (name -> stdio config). `math` is fast, no network/keys.
SERVER_NAME = "math"
SERVER_CMD = "node"
SERVER_ARGS = ["servers/math-mcp/build/index.js"]

MODEL = os.environ.get("CURSOR_TEST_MODEL", "cursor:claude-opus-4.5")
TASK = "Please add the two numbers 17 and 25 using the available tool, then tell me the sum."


def _extract_text(result) -> str:
    parts = []
    for c in (getattr(result, "content", None) or []):
        t = getattr(c, "type", None)
        if t == "text":
            parts.append(getattr(c, "text", "") or "")
        elif t == "json":
            parts.append(json.dumps(getattr(c, "data", None), ensure_ascii=False))
    return " | ".join(p for p in parts if p) or "[empty]"


async def main() -> int:
    load_env_file()
    os.chdir(REPO)

    params = StdioServerParameters(command=SERVER_CMD, args=SERVER_ARGS, env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()

            # Build host.tools-like map: qualified -> (server, tool, desc, schema)
            host_tools = {}
            for t in resp.tools:
                qn = f"{SERVER_NAME}/{t.name}"
                host_tools[qn] = (SERVER_NAME, t.name, t.description or "", t.inputSchema)
            tool_names = list(host_tools.keys())
            print(f"[MCP] {SERVER_NAME} exposes {len(tool_names)} tools: "
                  + ", ".join(tool_names))

            tools, name_map = build_openai_tools(host_tools, tool_names)
            print(f"[SCHEMA] built {len(tools)} function schemas; "
                  f"escaped names: {[t['function']['name'] for t in tools]}")

            # Native tool-calling round-trip via apicursor endpoint.
            model_name = MODEL.split(":", 1)[1] if MODEL.startswith("cursor:") else MODEL
            client = CursorAPIClient(model_name=model_name, max_new_tokens=1024)
            print(f"[MODEL] calling {model_name} via {os.environ.get('CURSOR_API_BASE_URL')} ...")

            messages = [{"role": "user", "content": TASK}]
            # The shared endpoint occasionally returns transient 503s; retry a few times.
            visible, raw_calls = "", []
            for attempt in range(1, 6):
                try:
                    visible, raw_calls = client.generate_with_tools(messages, tools, tool_choice="auto")
                    break
                except Exception as e:
                    print(f"[RETRY {attempt}] endpoint error: {str(e)[:120]}")
                    await asyncio.sleep(3)
            else:
                print("[FAIL] endpoint unavailable after retries.")
                return 1
            print(f"[MODEL] visible text: {visible!r}")
            print(f"[MODEL] raw structured tool_calls: {json.dumps(raw_calls, ensure_ascii=False)}")

            if not raw_calls:
                print("[FAIL] Model returned no native tool_calls.")
                return 1

            # Restore qualified names and dispatch to the MCP server.
            ok = False
            for c in raw_calls:
                esc = c.get("name")
                qn = name_map.get(esc, esc)
                args = c.get("arguments", {}) or {}
                print(f"\n[DISPATCH] escaped={esc} -> qualified={qn} args={args}")
                _server, tool_name, _desc, _schema = host_tools[qn]
                result = await session.call_tool(tool_name, args)
                text = _extract_text(result)
                print(f"[RESULT] {tool_name} -> {text}")
                ok = True

            print("\n[PASS] Native tool calling round-trip succeeded." if ok
                  else "[FAIL] No tool dispatched.")
            return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

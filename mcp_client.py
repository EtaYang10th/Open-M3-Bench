import json, subprocess, sys, uuid, os
from pathlib import Path
from typing import Dict, Any

def _load_config(config_path: Path) -> Dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return cfg

def call_mcp_tool(config_path: Path, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    File-transport MCP call:
      1) Write request JSON to a temp file
      2) Spawn server process with --in/--out
      3) Read response JSON
    An MCP-flavored schema (call_tool) is used to keep semantics familiar.
    """
    cfg = _load_config(config_path)
    servers = cfg.get("servers", {})
    if server_name not in servers:
        raise RuntimeError(f"Unknown server '{server_name}'. Check {config_path}")

    server_cfg = servers[server_name]
    server_cmd = server_cfg.get("cmd")
    if not server_cmd:
        raise RuntimeError(f"No 'cmd' configured for server '{server_name}'")

    tmp_dir = Path(server_cfg.get("io_dir", ".")).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    req_id = str(uuid.uuid4())
    req_path = tmp_dir / f"{req_id}.req.json"
    resp_path = tmp_dir / f"{req_id}.resp.json"

    request = {
        "version": "mcp.file.v1",
        "server": server_name,
        "request_id": req_id,
        "action": "call_tool",
        "name": tool_name,
        "arguments": arguments,
    }
    req_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build subprocess command: append --in/--out paths
    cmd = list(server_cmd) + ["--in", str(req_path), "--out", str(resp_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"Server '{server_name}' failed: {proc.stderr.strip()}")

    if not resp_path.exists():
        raise RuntimeError(f"Server '{server_name}' wrote no response file: {resp_path}")

    try:
        resp = json.loads(resp_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Invalid JSON response from '{server_name}': {e}")

    if not resp.get("success", False):
        # Standardize error surfacing
        raise RuntimeError(resp.get("error", "Unknown MCP server error"))

    return resp

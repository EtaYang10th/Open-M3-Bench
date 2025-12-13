import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from models.model_loader import create_model_driver


ALLOWED_CATEGORIES: List[str] = [
  "Weather & Air Quality",
  "Geography & Travel",
  "Health",
  "Science & Space",
  "Academic & Knowledge",
  "Office Automation",
  "Computer Vision",
  "E-commerce & Finance"
]

BBOX_OPEN = "<<BBOX>>"
BBOX_CLOSE = "<</BBOX>>"
_BBOX_RE = re.compile(r"<<BBOX>>(.*?)<</BBOX>>", re.DOTALL)


def _load_env_from_file(env_path: Path) -> None:
  """
  Read simple .env (KEY=VALUE), ignore comments/blank lines, inject into os.environ.
  No third-party dependency required.
  """
  if not env_path.exists():
    return
  try:
    with env_path.open("r", encoding="utf-8") as f:
      for line in f:
        s = line.strip()
        if (not s) or s.startswith("#"):
          continue
        if "=" not in s:
          continue
        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and (key not in os.environ):
          os.environ[key] = val
  except Exception:
    # Fail silently; do not block main flow
    pass


def _build_messages(tool_full_name: str, description: str) -> List[Dict[str, str]]:
  system_prompt = (
    "You are a strict single-label classifier. You must select exactly one category "
    "from the following list, and the output must exactly match one of these strings:\n"
    f"{', '.join(ALLOWED_CATEGORIES)}\n\n"
    "Output constraint: respond with ONLY one bbox-wrapped final answer, no other text or punctuation.\n"
    f"Format: {BBOX_OPEN}CATEGORY{BBOX_CLOSE}. For example: {BBOX_OPEN}{ALLOWED_CATEGORIES[0]}{BBOX_CLOSE}.\n"
    "Do not explain, do not add newlines, and do not add any prefixes or suffixes."
  )
  user_prompt = (
    "Given the tool name and description, choose the single best category.\n"
    f"Tool: {tool_full_name}\n"
    f"Description: {description}"
  )
  return [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ]


def _extract_bbox_label(text: str) -> Optional[str]:
  m = _BBOX_RE.search(text or "")
  if not m:
    return None
  return (m.group(1) or "").strip()


def _maybe_abs_path(token: str, base: Path) -> str:
  """
  Convert a repo-relative path token to absolute if the target exists.
  Leave flags (e.g., '-m') and non-existent paths untouched to avoid breaking commands.
  """
  if not isinstance(token, str):
    return token
  if token.startswith("-") or os.path.isabs(token):
    return token
  candidate = base / token
  return str(candidate) if candidate.exists() else token


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
  if not path.exists():
    return {}
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


async def _list_tools_for_server(server_name: str, scfg: Dict[str, Any], repo_root: Path) -> Dict[str, Dict[str, str]]:
  """
  Start one MCP server via stdio, run list_tools, and return {tool: {description}}.
  """
  if scfg.get("disabled", False):
    print(f"[SKIP] {server_name}: disabled in mcp_servers.json")
    return {}

  command = _maybe_abs_path(scfg.get("command", ""), repo_root)
  args = [_maybe_abs_path(a, repo_root) for a in scfg.get("args", [])]
  merged_env = dict(os.environ)
  merged_env.update(scfg.get("env") or {})

  params = StdioServerParameters(
    command=command,
    args=args,
    env=merged_env,
    cwd=str(repo_root),
  )

  async with AsyncExitStack() as stack:
    stdio = await stack.enter_async_context(stdio_client(params))
    read, write = stdio
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    resp = await session.list_tools()

  tools: Dict[str, Dict[str, str]] = {}
  for tool in resp.tools:
    tools[tool.name] = {"description": tool.description or ""}
  print(f"[OK] {server_name}: discovered {len(tools)} tools")
  return tools


async def discover_tools(config_path: Path, repo_root: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
  """
  Load mcp_servers.json and enumerate tools for each active server.
  Returns structure: { server: { tool: {description: str} } }
  """
  if not config_path.exists():
    raise FileNotFoundError(f"Missing MCP config: {config_path}")

  cfg = json.loads(config_path.read_text(encoding="utf-8"))
  servers = cfg.get("servers", {}) or {}

  discovered: Dict[str, Dict[str, Dict[str, str]]] = {}
  for name, scfg in servers.items():
    if not isinstance(scfg, dict):
      continue
    try:
      discovered[name] = await _list_tools_for_server(name, scfg, repo_root)
    except Exception as e:
      print(f"[WARN] {name}: failed to list tools ({e})")
  return discovered


def merge_catalog(existing: Dict[str, Any], discovered: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Any]:
  """
  Merge newly discovered tools into the catalog without overriding prior metadata.
  - Keep existing entries (including 'catalogry') intact.
  - Add new servers/tools.
  - If an existing tool lacks a description, backfill it.
  """
  merged = json.loads(json.dumps(existing))  # deep copy to avoid mutating the original
  for server, tools in (discovered or {}).items():
    if not isinstance(tools, dict):
      continue
    server_bucket = merged.setdefault(server, {})
    for tool_name, payload in tools.items():
      if not isinstance(payload, dict):
        continue
      if tool_name not in server_bucket:
        server_bucket[tool_name] = {"description": payload.get("description", "")}
      else:
        dest = server_bucket[tool_name]
        if not dest.get("description"):
          dest["description"] = payload.get("description", "")
  return merged


def classify_one(driver, tool_full_name: str, description: str, max_retries: int = 3) -> str:
  messages = _build_messages(tool_full_name, description)
  attempt = 0
  while attempt < max_retries:
    attempt += 1
    visible, _ = driver.generate_once(messages)
    label = _extract_bbox_label(visible)
    if label in ALLOWED_CATEGORIES:
      return label  # valid
    # Retry if not matched
  raise RuntimeError(f"Invalid model output: {visible}, tool: {tool_full_name}")


def main() -> None:
  # Compute project root
  script_path = Path(__file__).resolve()
  repo_root = script_path.parent.parent

  # Preload .env (same path as scripts/evaluate_final_answer.sh)
  _load_env_from_file(repo_root / ".env")

  # Target JSON file (create/merge before classification)
  json_path = repo_root / "save" / "mcp_tools_with_desc.json"
  json_path.parent.mkdir(parents=True, exist_ok=True)

  # Step 1) Discover tools from running each MCP server
  config_path = repo_root / "mcp_servers.json"
  discovered = asyncio.run(discover_tools(config_path, repo_root))

  # Step 2) Merge with any existing catalog to preserve prior 'catalogry' labels
  existing_data = _load_json_if_exists(json_path)
  data: Dict[str, Any] = merge_catalog(existing_data, discovered)
  if data != existing_data:
    with json_path.open("w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] Saved merged catalog to {json_path}")
  else:
    print("[INFO] Catalog up to date; no merge changes.")

  # Step 3) Classify only tools missing 'catalogry'
  # Create gpt-5-mini driver (same choice as evaluate_final_answer.sh)
  driver = create_model_driver("gemini-2.5-flash-lite", max_new_tokens=10240)

  updated = False
  # Iterate structure: top-level groups (services), second-level tools with description
  for group_name, group_obj in (data or {}).items():
    if not isinstance(group_obj, dict):
      continue
    for tool_name, tool_obj in list(group_obj.items()):
      if not isinstance(tool_obj, dict):
        continue
      # Only process items missing 'catalogry' to avoid rework
      if "catalogry" in tool_obj:
        continue
      desc = tool_obj.get("description", "")
      tool_full_name = f"{group_name}.{tool_name}"

      try:
        label = classify_one(driver, tool_full_name, desc)
      except Exception as e:
        # If classification fails, warn and skip; try next run
        print(f"[WARN] Classification failed: {tool_full_name}: {e}")
        continue

      tool_obj["catalogry"] = label
      updated = True
      print(f"[OK] {tool_full_name} -> {label}")
      # Write back each item to persist incrementally
      with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

  if updated:
    with json_path.open("w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated and wrote: {json_path}")
  else:
    print("No updates needed: no tools missing 'catalogry'.")


if __name__ == "__main__":
  main()



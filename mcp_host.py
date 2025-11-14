# mcp_host.py
import json, asyncio, re, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPHost:
    """Manage multiple MCP server sessions via stdio; cache tool schemas; route calls."""
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        # qualified tool name -> (server, tool_name, description, inputSchema)
        self.tools: Dict[str, Tuple[str, str, str, Dict[str, Any]]] = {}

    async def start(self):
        servers = self.cfg.get("servers", {})
        for name, scfg in servers.items():
            # Skip servers explicitly disabled in configuration
            if isinstance(scfg, dict) and scfg.get("disabled", False):
                continue
            merged_env = None
            if "env" in scfg and isinstance(scfg.get("env"), dict):
                merged_env = dict(os.environ)
                merged_env.update(scfg.get("env") or {})
            params = StdioServerParameters(
                command=scfg["command"],
                args=scfg.get("args", []),
                env=merged_env,
            )
            stdio = await self.exit_stack.enter_async_context(stdio_client(params))
            read, write = stdio
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session

            # discover tools
            resp = await session.list_tools()
            for tool in resp.tools:
                qn = f"{name}/{tool.name}"
                self.tools[qn] = (name, tool.name, tool.description or "", tool.inputSchema)

    async def stop(self):
        await self.exit_stack.aclose()

    async def call(self, qualified_name: str, arguments: Dict[str, Any]) -> str:
        if qualified_name not in self.tools:
            raise ValueError(f"Unknown tool: {qualified_name}")
        server, tool = self.tools[qualified_name][0], self.tools[qualified_name][1]
        session = self.sessions[server]
        try:
            result = await session.call_tool(tool, arguments)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Tool call timed out: {qualified_name} (> {timeout}s)")

        bundle = {"text_parts": [], "json_parts": [], "images": []}
        for c in (result.content or []):
            t = getattr(c, "type", None)
            if t == "text":
                bundle["text_parts"].append(c.text)
            elif t == "json":
                bundle["json_parts"].append(c.data)  # preserve raw structure
            elif t == "image":
                if hasattr(c, "data") and isinstance(c.data, str):
                    bundle["images"].append(c.data)

        # If there is only text, return plain text for compatibility; otherwise return JSON string
        if bundle["json_parts"] or bundle["images"]:
            return json.dumps(bundle, ensure_ascii=False)
        return "\n".join(bundle["text_parts"]) if bundle["text_parts"] else "[no textual content]"

    def select_tools_for(self, user_query: str, k: int = 6) -> List[str]:
        """Very simple lexical selection: choose top-K tools by name/desc token overlap."""
        uq = user_query.lower()
        scored = []
        for qn, (_, tname, desc, schema) in self.tools.items():
            hay = f"{qn} {tname} {desc} {json.dumps(schema, ensure_ascii=False)}".lower()
            # naive overlap score
            score = sum(1 for w in set(re.findall(r"[a-z0-9_]+", uq)) if w in hay)
            scored.append((score, qn))
        scored.sort(reverse=True)
        return [qn for s, qn in scored[:k] if s > 0] or list(self.tools.keys())[:min(k, len(self.tools))]

    def render_tool_specs(self, tool_names: List[str]) -> str:
        """Render compact specs for prompt injection when using local HF models."""
        lines = []
        for qn in tool_names:
            server, tname, desc, schema = self.tools[qn]
            lines.append(
                f"- name: {server}/{tname}\n  desc: {desc}\n  schema: {json.dumps(schema, ensure_ascii=False)}"
            )
        return "Available tools (select one if useful):\n" + "\n".join(lines)

"""Helpers to build native function-calling tool schemas from MCP tools.

MCP qualified tool names look like "server/tool" (e.g. "math/add"). Most
providers (OpenAI, Anthropic, Gemini, ...) forbid "/" in a function name and
only allow ``[a-zA-Z0-9_-]``. We therefore *escape* qualified names when
sending them to the provider and keep a ``name_map`` (escaped -> qualified) so
the round runner can restore the original qualified name before dispatching the
call to :class:`MCPHost`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_INVALID_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
# OpenAI limits function names to 64 chars.
_MAX_NAME_LEN = 64


def escape_tool_name(qualified_name: str) -> str:
    """Turn a qualified MCP name into a provider-safe function name."""
    esc = _INVALID_NAME_RE.sub("_", qualified_name or "")
    if len(esc) > _MAX_NAME_LEN:
        esc = esc[:_MAX_NAME_LEN]
    return esc or "tool"


def _ensure_object_schema(schema: Any) -> Dict[str, Any]:
    """Return a JSON-schema object suitable for a function ``parameters`` field."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out = dict(schema)
    if out.get("type") != "object":
        out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


def build_name_map(tool_names: List[str]) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Build ``[(qualified, escaped)]`` pairs plus an ``escaped -> qualified`` map.

    Collisions after escaping are resolved by appending a numeric suffix so the
    map stays bijective and every original name is recoverable.
    """
    pairs: List[Tuple[str, str]] = []
    name_map: Dict[str, str] = {}
    used: set = set()
    for qn in tool_names:
        esc = escape_tool_name(qn)
        base = esc
        i = 1
        while esc in used:
            suffix = f"_{i}"
            esc = (base[: _MAX_NAME_LEN - len(suffix)]) + suffix
            i += 1
        used.add(esc)
        name_map[esc] = qn
        pairs.append((qn, esc))
    return pairs, name_map


def build_openai_tools(
    host_tools: Dict[str, Any],
    tool_names: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Build OpenAI-style ``tools`` list and an ``escaped -> qualified`` map.

    ``host_tools`` is ``MCPHost.tools`` (qualified -> (server, tool, desc, schema)).
    Returns ``(tools, name_map)``. Only names present in ``host_tools`` are kept.
    """
    valid = [qn for qn in tool_names if qn in host_tools]
    pairs, name_map = build_name_map(valid)
    tools: List[Dict[str, Any]] = []
    for qn, esc in pairs:
        _server, _tname, desc, schema = host_tools[qn]
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": esc,
                    "description": desc or "",
                    "parameters": _ensure_object_schema(schema),
                },
            }
        )
    return tools, name_map


def _clean_gemini_schema(schema: Any) -> Dict[str, Any]:
    """Strip JSON-schema keys the Gemini function-declaration parser rejects."""
    drop = {"$schema", "additionalProperties", "title", "$id", "$ref", "definitions", "examples"}
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if isinstance(v, dict):
            out[k] = _clean_gemini_schema(v)
        elif isinstance(v, list):
            out[k] = [_clean_gemini_schema(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    if "properties" in out and out.get("type") is None:
        out["type"] = "object"
    return out


def openai_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to Anthropic ``tools`` (name/description/input_schema)."""
    out: List[Dict[str, Any]] = []
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", "") or "",
                "input_schema": _ensure_object_schema(fn.get("parameters")),
            }
        )
    return out


def openai_tools_to_gemini_decls(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to Gemini function-declaration dicts."""
    out: List[Dict[str, Any]] = []
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", "") or "",
                "parameters": _clean_gemini_schema(fn.get("parameters")),
            }
        )
    return out

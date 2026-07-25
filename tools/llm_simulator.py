# tools/llm_simulator.py
"""
LLM-driven API simulator (StableToolBench-style).

Given a tool's (name, description, inputSchema, output-contract hint) plus the
current call arguments, ask an LLM to synthesize a *realistic* return payload:
- related keywords -> plausible near-match results
- unrelated keywords -> unrelated-but-valid results

Uses the repo's already-wired apicursor endpoint (OpenAI-compatible) via
CURSOR_API_BASE_URL / CURSOR_API_KEY. Falls back to a generic synthetic
template if the endpoint is unavailable, so callers never crash.

No hard dependency on models/api_clients.py: we talk to the OpenAI-compatible
endpoint directly to keep this module import-light.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

# Model known to work on the apicursor endpoint (per prior verification).
_SIM_MODEL = os.getenv("M3_MOCK_LLM_MODEL", "claude-opus-4.5")


def _sim_timeout(default: float = 60.0) -> float:
    """Per-request timeout for the simulator LLM call (env M3_MOCK_LLM_TIMEOUT)."""
    try:
        return float(os.environ.get("M3_MOCK_LLM_TIMEOUT", default))
    except (TypeError, ValueError):
        return default


def _sim_max_retries(default: int = 0) -> int:
    try:
        return max(0, int(os.environ.get("M3_MOCK_LLM_MAX_RETRIES", default)))
    except (TypeError, ValueError):
        return default


def _tool_meta(server: str, tool: str) -> Dict[str, Any]:
    """Load {description, schema} for server/tool from the tools dump if present."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        dump = json.load(open(os.path.join(here, "mcp_tools_dump.json"), encoding="utf-8"))
        for t in dump.get(server, {}).get("tools", []):
            if t.get("name") == tool:
                return {"description": t.get("description", ""), "schema": t.get("schema", {})}
    except Exception:
        pass
    return {"description": "", "schema": {}}


def _build_prompt(server: str, tool: str, arguments: Dict[str, Any], contract_hint: str) -> str:
    meta = _tool_meta(server, tool)
    return (
        "You are a high-fidelity SIMULATOR of a real external API used by an MCP tool.\n"
        "Produce ONLY the tool's return payload — no explanations, no markdown fences.\n\n"
        f"Tool: {server}/{tool}\n"
        f"Description:\n{meta['description']}\n\n"
        f"Input JSON schema (properties):\n{json.dumps(meta['schema'].get('properties', {}), ensure_ascii=False)}\n\n"
        f"Output contract (match this exact shape/style):\n{contract_hint}\n\n"
        f"This call's arguments:\n{json.dumps(arguments, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "- Behave like the REAL API: if the query keywords are meaningful, return\n"
        "  plausible near-matching items with realistic titles/URLs/prices/fields.\n"
        "- If the query is unrelated/nonsense, return unrelated but structurally valid results.\n"
        "- Respect any count argument (e.g. n) and keep values self-consistent.\n"
        "- Output must be directly usable as the tool's return (same type/shape as the contract)."
    )


def _openai_compatible_call(prompt: str) -> Optional[str]:
    base_url = os.environ.get("CURSOR_API_BASE_URL")
    api_key = os.environ.get("CURSOR_API_KEY")
    if not base_url or not api_key:
        return None
    try:
        from openai import OpenAI  # local import; optional dependency
    except Exception:
        return None
    try:
        # This runs on the asyncio event-loop thread (mcp_host.call -> try_fallback),
        # so an unbounded request here freezes every worker, not just this task.
        client = OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=_sim_timeout(), max_retries=_sim_max_retries(),
        )
        import time as _time

        _deadline = _time.monotonic() + _sim_timeout()
        stream = client.chat.completions.create(
            model=_SIM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            stream=True,
            timeout=_sim_timeout(),
        )
        parts = []
        for chunk in stream:
            if _time.monotonic() > _deadline:
                raise TimeoutError("simulator stream exceeded deadline")
            try:
                ct = chunk.choices[0].delta.content
            except Exception:
                ct = None
            if ct:
                parts.append(str(ct))
        text = "".join(parts).strip()
        return text or None
    except Exception:
        return None


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


def simulate(server: str, tool: str, arguments: Dict[str, Any], contract_hint: str) -> Optional[str]:
    """Return an LLM-simulated payload string, or None if the endpoint is unusable."""
    prompt = _build_prompt(server, tool, arguments, contract_hint)
    out = _openai_compatible_call(prompt)
    if out is None:
        return None
    return _strip_fences(out)

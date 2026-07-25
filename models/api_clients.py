import os
import re
import json
import base64
import mimetypes
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI  # type: ignore
import requests

from .tool_schema import (
    openai_tools_to_anthropic,
    openai_tools_to_gemini_decls,
)

# Structured tool-call return type: list of {"name": <escaped>, "arguments": dict}
ToolCall = Dict[str, Any]


def _llm_timeout(default: float = 300.0) -> float:
    """Per-request LLM timeout (seconds). Env M3_LLM_TIMEOUT overrides default.

    Guards against endpoints (e.g. apicursor.com) that never return, which
    would otherwise hang the whole benchmark batch.
    """
    try:
        return float(os.environ.get("M3_LLM_TIMEOUT", default))
    except (TypeError, ValueError):
        return default


def _llm_max_retries(default: int = 1) -> int:
    """SDK-level retry budget for LLM HTTP calls (env M3_LLM_MAX_RETRIES).

    The OpenAI SDK defaults to 2 retries and applies ``timeout`` *per attempt*,
    so a stalled endpoint costs ``timeout * (1 + retries)`` plus backoff before
    the caller ever sees an exception. Capping it keeps the wall clock bounded
    and predictable. Set M3_LLM_MAX_RETRIES=2 to restore the SDK default.
    """
    try:
        return max(0, int(os.environ.get("M3_LLM_MAX_RETRIES", default)))
    except (TypeError, ValueError):
        return default


_LLM_STATS: Dict[str, Any] = {
    "requests": 0,
    "text_chars": 0,
    "tools_chars": 0,
    "image_inlines": 0,
    "image_b64_chars": 0,
    # path -> {"uploads": n, "b64_chars": total}
    "image_by_path": {},
}


def _stats_enabled() -> bool:
    return str(os.environ.get("M3_LLM_STATS", "0")).lower() not in ("", "0", "false", "no")


def _stats_flush_every(default: int = 10) -> int:
    """Flush the stats file every N requests. Env M3_LLM_STATS_FLUSH_EVERY."""
    try:
        return max(1, int(os.environ.get("M3_LLM_STATS_FLUSH_EVERY", default)))
    except (TypeError, ValueError):
        return default


def _record_request(
    messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Count one outgoing LLM request (opt-in via M3_LLM_STATS=1).

    ``text_chars`` counts prompt text only; base64 image payloads are tracked
    separately by :func:`_record_image_inline` so the two can be compared.
    """
    if not _stats_enabled():
        return
    n = 0
    for m in messages or []:
        c = m.get("content", "")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    n += len(part.get("text") or "")
    _LLM_STATS["requests"] += 1
    _LLM_STATS["text_chars"] += n
    _record_tools(tools)
    if _LLM_STATS["requests"] % _stats_flush_every() == 0:
        dump_llm_stats()


def _record_tools(tools: Optional[List[Dict[str, Any]]]) -> None:
    if not _stats_enabled() or not tools:
        return
    try:
        _LLM_STATS["tools_chars"] += len(json.dumps(tools, ensure_ascii=False))
    except Exception:
        pass


def _record_image_inline(data_url: str, path: str = "") -> None:
    if not _stats_enabled() or not data_url:
        return
    _LLM_STATS["image_inlines"] += 1
    _LLM_STATS["image_b64_chars"] += len(data_url)
    slot = _LLM_STATS["image_by_path"].setdefault(
        os.path.basename(path) or "?", {"uploads": 0, "b64_chars": 0}
    )
    slot["uploads"] += 1
    slot["b64_chars"] += len(data_url)


def dump_llm_stats() -> None:
    """Write collected stats to M3_LLM_STATS_FILE (JSON), if configured.

    Written atomically and called both periodically (from ``_record_request``)
    and at exit, because a batch killed by SIGTERM/SIGKILL never runs atexit
    handlers and would otherwise lose every counter it collected.
    """
    if not _stats_enabled():
        return
    path = os.environ.get("M3_LLM_STATS_FILE")
    if not path:
        return
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_LLM_STATS, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


import atexit as _atexit

_atexit.register(dump_llm_stats)


class BaseAPIClient:
    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        raise NotImplementedError

    def supports_native_tools(self) -> bool:
        """Whether this client can do provider-native function/tool calling."""
        return False

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        """Native tool-calling round-trip.

        ``tools`` is an OpenAI-style function-schema list whose function names
        are already escaped (see ``models.tool_schema``). Returns
        ``(visible_text, tool_calls)`` where each tool call is
        ``{"name": <escaped fn name>, "arguments": <dict>}``.
        Callers restore the qualified MCP name via the escaped->qualified map.
        """
        raise NotImplementedError


def _extract_image_paths(msgs: List[Dict[str, str]]) -> List[str]:
    """Collect ``file=<abs path>`` marked images, deduplicated in first-seen order.

    The same path legitimately appears in several messages (the initial user
    message plus every round's image hint). Deduplication keeps one inline copy
    per request; first-seen order keeps the prefix stable across rounds, which
    matters for provider-side prompt caching.
    """
    paths: List[str] = []
    seen: set = set()
    file_re = re.compile(r"file=([^\s]+)")
    for m in msgs:
        content = m.get("content", "") or ""
        if not isinstance(content, str):
            continue
        for match in file_re.findall(content):
            if match in seen:
                continue
            if os.path.isabs(match) and os.path.exists(match):
                seen.add(match)
                paths.append(match)
    return paths


_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def _image_max_edge() -> int:
    """Longest edge (px) for images sent to the LLM. 0 disables downscaling.

    Only the copy sent to the model is resized; files on disk stay untouched so
    MCP tools keep operating on the originals. Every request re-uploads the full
    base64 payload, so a 2.4 MB source image costs ~3.2 M characters per round.
    """
    try:
        return max(0, int(os.environ.get("M3_IMAGE_MAX_EDGE", 1568)))
    except (TypeError, ValueError):
        return 1568


def _image_recode_over_kb() -> int:
    """Re-encode images larger than this (KB) as JPEG. 0 disables.

    Most benchmark PNGs are stored at 2-3 bytes/pixel (essentially
    uncompressed), so they stay huge even when their dimensions are modest and
    resizing alone does not help. Re-encoding preserves the pixel grid.
    """
    try:
        return max(0, int(os.environ.get("M3_IMAGE_RECODE_OVER_KB", 512)))
    except (TypeError, ValueError):
        return 512


def _maybe_downscale(raw: bytes, mime: str) -> Tuple[bytes, str]:
    """Shrink an oversized image for transport, or return it unchanged."""
    limit = _image_max_edge()
    recode_over = _image_recode_over_kb() * 1024
    if mime == "image/gif" or (not limit and not recode_over):
        return raw, mime
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(raw))
        oversized = bool(limit) and max(img.size) > limit
        heavy = bool(recode_over) and len(raw) > recode_over
        if not oversized and not heavy:
            return raw, mime
        if oversized:
            img.thumbnail((limit, limit))
        if heavy:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88, optimize=True)
            out = buf.getvalue()
            return (out, "image/jpeg") if len(out) < len(raw) else (raw, mime)
        if img.mode in ("RGBA", "LA", "P") and mime != "image/png":
            img = img.convert("RGB")
        buf = io.BytesIO()
        fmt = "PNG" if mime == "image/png" else "JPEG"
        img.save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
        return buf.getvalue(), ("image/png" if fmt == "PNG" else "image/jpeg")
    except Exception:
        return raw, mime


def guess_media_type(path: str, head: Optional[bytes] = None) -> str:
    """Media type from the file's magic bytes, falling back to its extension.

    Several benchmark images carry a ``.png`` name while holding JPEG or WebP
    data. Trusting the extension makes providers that validate the declared
    media type (Anthropic) reject the request with a 400 that no retry can fix,
    so the bytes win whenever they identify a known format.
    """
    try:
        if head is None:
            with open(path, "rb") as f:
                head = f.read(16)
        for sig, mime in _IMAGE_MAGIC:
            if head.startswith(sig):
                return mime
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "image/webp"
    except Exception:
        pass
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _file_to_data_url(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            b = f.read()
        mime = guess_media_type(path, b[:16])
        b, mime = _maybe_downscale(b, mime)
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _messages_to_chat(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Filter to system/user/assistant text messages, injecting images (OpenAI parts)."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "user", "assistant"):
            out.append({"role": role, "content": m.get("content", "")})
    image_paths = _extract_image_paths(messages)
    if image_paths:
        last_user_idx = None
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            out.append({"role": "user", "content": ""})
            last_user_idx = len(out) - 1
        user_text = out[last_user_idx].get("content", "") or ""
        parts: List[Dict[str, Any]] = []
        if user_text:
            parts.append({"type": "text", "text": user_text})
        for p in image_paths:
            data_url = _file_to_data_url(p)
            if data_url:
                _record_image_inline(data_url, p)
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
        out[last_user_idx]["content"] = parts
    return out


def _openai_chat_tool_call(
    client: "OpenAI",
    model_name: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    tool_choice: str = "auto",
    max_tokens: Optional[int] = None,
    stream: bool = False,
) -> Tuple[str, List[ToolCall]]:
    """Shared OpenAI-compatible chat.completions native tool-calling round-trip."""
    chat_messages = _messages_to_chat(messages)
    _record_request(chat_messages, tools)
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": chat_messages,
        "tools": tools,
        "tool_choice": tool_choice if tools else "none",
        "stream": stream,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    timeout = _llm_timeout()
    kwargs["timeout"] = timeout
    if kwargs.get("stream"):
        return _accumulate_stream_tool_calls(
            client.chat.completions.create(**kwargs), timeout=timeout
        )
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    visible = (getattr(msg, "content", None) or "").strip()
    calls: List[ToolCall] = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", None) or ""
        raw_args = getattr(fn, "arguments", None)
        args = _parse_args(raw_args)
        if name:
            calls.append({"name": name, "arguments": args})
    return visible, calls


def _parse_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    try:
        return json.loads(raw_args) if raw_args else {}
    except Exception:
        return {}


def _accumulate_stream_tool_calls(
    stream, timeout: Optional[float] = None
) -> Tuple[str, List[ToolCall]]:
    """Accumulate visible text and tool_calls from an OpenAI-style SSE stream.

    Needed for endpoints (e.g. apicursor.com) that always stream, emitting
    tool-call name/arguments across multiple delta chunks keyed by index.

    A wall-clock ``timeout`` guards against streams that stall or never end
    (the OpenAI SDK client timeout may not cover an idle SSE iteration).
    """
    import time as _time
    _start = _time.monotonic()
    text_parts: List[str] = []
    # index -> {"name": str, "arguments": str}
    acc: Dict[int, Dict[str, str]] = {}
    for chunk in stream:
        if timeout is not None and (_time.monotonic() - _start) > timeout:
            raise TimeoutError(f"LLM stream timed out > {timeout}s")
        try:
            delta = chunk.choices[0].delta
        except Exception:
            continue
        if delta is None:
            continue
        ct = getattr(delta, "content", None)
        if ct:
            text_parts.append(str(ct))
        for tc in (getattr(delta, "tool_calls", None) or []):
            idx = getattr(tc, "index", 0) or 0
            slot = acc.setdefault(idx, {"name": "", "arguments": ""})
            fn = getattr(tc, "function", None)
            if fn is not None:
                nm = getattr(fn, "name", None)
                if nm:
                    slot["name"] = nm
                ar = getattr(fn, "arguments", None)
                if ar:
                    slot["arguments"] += ar
    calls: List[ToolCall] = []
    for idx in sorted(acc.keys()):
        slot = acc[idx]
        if slot["name"]:
            calls.append({"name": slot["name"], "arguments": _parse_args(slot["arguments"])})
    return "".join(text_parts).strip(), calls


class OpenAIAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._client = OpenAI(timeout=_llm_timeout(), max_retries=_llm_max_retries())

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:

        """
        Construct OpenAI Responses API call:
        - Convert to multi-turn input (system/user/assistant -> input_text)
        - If uploaded image paths are parsed, append images as data URLs to the last user content as input_image
        """

        def _extract_image_paths(msgs: List[Dict[str, str]]) -> List[str]:
            paths: List[str] = []
            file_re = re.compile(r"file=([^\s]+)")
            for m in msgs:
                content = m.get("content", "") or ""
                for match in file_re.findall(content):
                    if os.path.isabs(match) and os.path.exists(match):
                        paths.append(match)
            return paths

        def _file_to_data_url(path: str) -> Optional[str]:
            try:
                with open(path, "rb") as f:
                    b = f.read()
                mime = guess_media_type(path, b[:16])
                b, mime = _maybe_downscale(b, mime)
                b64 = base64.b64encode(b).decode("ascii")
                return f"data:{mime};base64,{b64}"
            except Exception:
                return None

        # 1) Convert all messages into Responses API input
        input_content: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role in ("system", "user", "assistant"):
                content_type = "output_text" if role == "assistant" else "input_text"
                input_content.append({
                    "role": role,
                    "content": [{"type": content_type, "text": m.get("content", "")}],
                })

        # 2) Append images to the last user entry
        image_paths = _extract_image_paths(messages)
        if image_paths:
            # Find the last user item
            last_user_idx = None
            for i in range(len(input_content) - 1, -1, -1):
                if input_content[i].get("role") == "user":
                    last_user_idx = i
                    break
            if last_user_idx is None:
                # If absent, create a user entry
                input_content.append({"role": "user", "content": []})
                last_user_idx = len(input_content) - 1

            # Append images as input_image (data URL to avoid public URL constraints)
            for p in image_paths:
                data_url = _file_to_data_url(p)
                if data_url:
                    input_content[last_user_idx]["content"].append({
                        "type": "input_image",
                        "image_url": data_url,
                    })

        resp = self._client.responses.create(
            model=self.model_name,
            input=input_content,
            max_output_tokens=self.max_new_tokens,
        )
        visible = getattr(resp, "output_text", None) or ""
        full = visible
        return visible, full

    def supports_native_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        # OpenAI chat.completions supports native tool calling uniformly.
        # Omit max_tokens: reasoning models (gpt-5*) reject `max_tokens` here.
        return _openai_chat_tool_call(
            self._client, self.model_name, messages, tools, tool_choice,
            max_tokens=None,
        )


class DeepseekAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        self._client = OpenAI(
            api_key=api_key, base_url="https://api.deepseek.com",
            timeout=_llm_timeout(), max_retries=_llm_max_retries(),
        )

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        chat_messages = []
        for m in messages:
            if m.get("role") in ("system", "user", "assistant"):
                chat_messages.append({"role": m["role"], "content": m.get("content", "")})
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            stream=False,
            timeout=60,
        )
        visible = resp.choices[0].message.content.strip()
        full = visible
        return visible, full

    def supports_native_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        return _openai_chat_tool_call(
            self._client, self.model_name, messages, tools, tool_choice,
            max_tokens=self.max_new_tokens,
        )


class InternAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        api_key = os.environ.get("INTERN_API_KEY")
        if not api_key:
            raise RuntimeError("INTERN_API_KEY not set")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        self._api_base = 'https://chat.intern-ai.org.cn/api/v1/chat/completions'

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Construct Intern Chat Completions request:
        - Keep multi-turn messages structure
        - If image paths detected, change last user content to array: text + multiple image_url (data URL)
        - Deepseek remains plain text (not handled here)
        """

        def _extract_image_paths(msgs: List[Dict[str, str]]) -> List[str]:
            paths: List[str] = []
            file_re = re.compile(r"file=([^\s]+)")
            for m in msgs:
                content = m.get("content", "") or ""
                for match in file_re.findall(content):
                    if os.path.isabs(match) and os.path.exists(match):
                        paths.append(match)
            return paths

        def _file_to_data_url(path: str) -> Optional[str]:
            try:
                with open(path, "rb") as f:
                    b = f.read()
                mime = guess_media_type(path, b[:16])
                b, mime = _maybe_downscale(b, mime)
                b64 = base64.b64encode(b).decode("ascii")
                return f"data:{mime};base64,{b64}"
            except Exception:
                return None

        # Split messages and find last user index
        payload_messages: List[Dict[str, Any]] = []
        last_user_idx: Optional[int] = None
        for m in messages:
            role = m.get("role")
            if role in ("system", "user", "assistant"):
                payload_messages.append({"role": role, "content": m.get("content", "")})
        for i in range(len(payload_messages) - 1, -1, -1):
            if payload_messages[i].get("role") == "user":
                last_user_idx = i
                break

        # Parse images and inject into last user's content array
        image_paths = _extract_image_paths(messages)
        if image_paths:
            if last_user_idx is None:
                payload_messages.append({"role": "user", "content": ""})
                last_user_idx = len(payload_messages) - 1

            user_text = payload_messages[last_user_idx].get("content", "") or ""
            # Intern API multimodal requires array: text + image_url
            content_items: List[Dict[str, Any]] = []
            if user_text:
                content_items.append({"type": "text", "text": user_text})
            for p in image_paths:
                data_url = _file_to_data_url(p)
                if data_url:
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
            payload_messages[last_user_idx]["content"] = content_items

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": payload_messages,
            "stream": False,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.7,
        }

        url = f"{self._api_base}"
        # requests defaults to waiting forever; an unresponsive endpoint would
        # hang the whole batch the way apicursor did on task 00240000.
        r = requests.post(url, headers=self._headers, json=payload, timeout=_llm_timeout())
        r.raise_for_status()
        data = r.json()
        visible = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        full = visible
        return visible, full



class GeminiAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 2048) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        # Lazy import; only required when using Gemini
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as e:
            raise RuntimeError("google-genai SDK not installed. Please install `google-genai`." ) from e

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
        # Create client once
        self._client = genai.Client(api_key=api_key)
        self._types = types

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Construct Gemini generate_content call.
        - Extract any absolute file paths like `file=/abs/path.jpg` from messages
        - Read as bytes and pass via types.Part.from_bytes
        - Concatenate textual context into a single prompt
        """

        def _extract_image_paths(msgs: List[Dict[str, str]]) -> List[str]:
            paths: List[str] = []
            file_re = re.compile(r"file=([^\s]+)")
            for m in msgs:
                content = m.get("content", "") or ""
                for match in file_re.findall(content):
                    if os.path.isabs(match) and os.path.exists(match):
                        paths.append(match)
            return paths

        def _read_file_bytes(path: str) -> Optional[Tuple[bytes, str]]:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                mime = guess_media_type(path, data[:16])
                data, mime = _maybe_downscale(data, mime)
                return data, mime
            except Exception:
                return None

        # Build a compact textual prompt from the conversation
        text_segments: List[str] = []
        for m in messages:
            role = m.get("role")
            if role in ("system", "user", "assistant"):
                text = m.get("content", "") or ""
                if text:
                    text_segments.append(f"[{role}] {text}")
        prompt_text = "\n".join(text_segments) if text_segments else ""

        # Prepare parts: images (if any) + prompt text (as str)
        parts: List[Any] = []
        for p in _extract_image_paths(messages):
            rb = _read_file_bytes(p)
            if rb is None:
                continue
            data, mime = rb
            try:
                parts.append(self._types.Part.from_bytes(data=data, mime_type=mime))
            except Exception:
                # Skip invalid image
                continue
        if prompt_text:
            parts.append(prompt_text)

        # Fallback to at least a space if both are empty (Gemini requires some content)
        if not parts:
            parts = [" "]

        # Optional generation config
        gen_config = None
        try:
            gen_config = self._types.GenerateContentConfig(
                max_output_tokens=int(self.max_new_tokens),
                temperature=0.7,
            )
        except Exception:
            gen_config = None

        # Call Gemini API
        try:
            if gen_config is not None:
                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    config=gen_config,
                )
            else:
                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                )
        except Exception as e:
            raise RuntimeError(f"Gemini generate_content failed: {e}")

        visible = getattr(resp, "text", None) or ""
        full = visible
        return visible, full

    def supports_native_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        types = self._types
        # Build a single text prompt from the conversation (same style as generate_once).
        segments = [f"[{m.get('role')}] {m.get('content','')}" for m in messages
                    if m.get("role") in ("system", "user", "assistant") and m.get("content")]
        prompt_text = "\n".join(segments) or " "

        decls = openai_tools_to_gemini_decls(tools)
        tool_obj = types.Tool(function_declarations=decls)
        mode = "ANY" if tool_choice == "required" else "AUTO"
        try:
            fcc = types.FunctionCallingConfig(mode=mode)
            tool_cfg = types.ToolConfig(function_calling_config=fcc)
            config = types.GenerateContentConfig(
                tools=[tool_obj],
                tool_config=tool_cfg,
                max_output_tokens=int(self.max_new_tokens),
            )
        except Exception:
            config = types.GenerateContentConfig(tools=[tool_obj])

        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt_text,
            config=config,
        )
        calls: List[ToolCall] = []
        for fc in (getattr(resp, "function_calls", None) or []):
            name = getattr(fc, "name", None) or ""
            args = getattr(fc, "args", None) or {}
            if not isinstance(args, dict):
                try:
                    args = dict(args)
                except Exception:
                    args = {}
            if name:
                calls.append({"name": name, "arguments": args})
        visible = ""
        try:
            visible = getattr(resp, "text", None) or ""
        except Exception:
            visible = ""
        return visible, calls


class AnthropicAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as e:
            raise RuntimeError("anthropic SDK not installed. Please install `anthropic`." ) from e

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            # Anthropic() reads from env vars; we proactively check for clear error
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        # Create client relying on environment variables
        from anthropic import Anthropic  # type: ignore
        self._client = Anthropic()

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Use Anthropic Messages API:
        - Aggregate system content into system field
        - Convert multi-turn messages (user/assistant)
        - If image paths detected (file=/abs/path), change last user content to blocks: text + multiple image(base64)
        """

        def _extract_image_paths(msgs: List[Dict[str, str]]) -> List[str]:
            paths: List[str] = []
            file_re = re.compile(r"file=([^\s]+)")
            for m in msgs:
                content = m.get("content", "") or ""
                for match in file_re.findall(content):
                    if os.path.isabs(match) and os.path.exists(match):
                        paths.append(match)
            return paths

        def _image_block_from_path(path: str) -> Optional[Dict[str, Any]]:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                media_type = guess_media_type(path, raw[:16])
                raw, media_type = _maybe_downscale(raw, media_type)
                data = base64.b64encode(raw).decode("utf-8")
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            except Exception:
                return None

        # Aggregate system text
        system_segments: List[str] = []
        for m in messages:
            if m.get("role") == "system":
                text = m.get("content", "") or ""
                if text:
                    system_segments.append(text)
        system_text = "\n".join(system_segments) if system_segments else ""

        # Copy user/assistant conversation
        payload_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role in ("user", "assistant"):
                payload_messages.append({"role": role, "content": m.get("content", "")})

        # Find last user index
        last_user_idx: Optional[int] = None
        for i in range(len(payload_messages) - 1, -1, -1):
            if payload_messages[i].get("role") == "user":
                last_user_idx = i
                break

        # Inject image blocks into the last user message
        image_paths = _extract_image_paths(messages)
        if image_paths:
            if last_user_idx is None:
                payload_messages.append({"role": "user", "content": ""})
                last_user_idx = len(payload_messages) - 1

            user_text = payload_messages[last_user_idx].get("content", "") or ""
            content_blocks: List[Dict[str, Any]] = []
            # Per guidance: images first, text after
            for p in image_paths:
                blk = _image_block_from_path(p)
                if blk:
                    content_blocks.append(blk)
            if user_text:
                content_blocks.append({"type": "text", "text": user_text})
            payload_messages[last_user_idx]["content"] = content_blocks

        # Call Anthropic Messages API
        try:
            # system field optional; include only when non-empty
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": int(self.max_new_tokens),
                "messages": payload_messages,
                "temperature": 0.7,
            }
            if system_text:
                kwargs["system"] = system_text
            # Use streaming to support long-running requests (>10 minutes)
            with self._client.messages.stream(**kwargs) as stream:  # type: ignore
                stream.until_done()
                resp = stream.get_final_message()
            
        except Exception as e:
            raise RuntimeError(f"Anthropic messages.create failed: {e}")

        # Extract visible text (concatenate all text blocks)
        visible_parts: List[str] = []
        try:
            content_list = getattr(resp, "content", [])
            for block in content_list:
                # Compatible with object or dict
                btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
                if btype == "text":
                    text = getattr(block, "text", None)
                    if text is None and isinstance(block, dict):
                        text = block.get("text")
                    if text:
                        visible_parts.append(str(text))
        except Exception:
            pass
        visible = "".join(visible_parts).strip()
        full = visible
        return visible, full

    def supports_native_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        # Aggregate system text; keep only user/assistant turns as messages.
        system_segments = [m.get("content", "") for m in messages
                           if m.get("role") == "system" and m.get("content")]
        system_text = "\n".join(system_segments)
        payload_messages = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in messages if m.get("role") in ("user", "assistant")
        ]
        if not payload_messages:
            payload_messages = [{"role": "user", "content": system_text or " "}]
        anth_tools = openai_tools_to_anthropic(tools)
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(self.max_new_tokens),
            "messages": payload_messages,
            "tools": anth_tools,
        }
        if system_text:
            kwargs["system"] = system_text
        if tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        with self._client.messages.stream(**kwargs) as stream:  # type: ignore
            stream.until_done()
            resp = stream.get_final_message()
        visible_parts: List[str] = []
        calls: List[ToolCall] = []
        for block in (getattr(resp, "content", []) or []):
            btype = getattr(block, "type", None)
            if btype == "text":
                txt = getattr(block, "text", None)
                if txt:
                    visible_parts.append(str(txt))
            elif btype == "tool_use":
                name = getattr(block, "name", None) or ""
                args = getattr(block, "input", None) or {}
                if not isinstance(args, dict):
                    args = {}
                if name:
                    calls.append({"name": name, "arguments": args})
        return "".join(visible_parts).strip(), calls


class TogetherAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        # Lazy import to avoid dependency unless used
        try:
            from together import Together  # type: ignore
        except Exception as e:
            raise RuntimeError("together SDK not installed. Please install `together`.") from e
        api_key = os.environ.get("TOGETHER_API_KEY")
        if api_key:
            self._client = Together(api_key=api_key)  # type: ignore
        else:
            # Together SDK will try env var internally as well
            self._client = Together()  # type: ignore

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Use Together Chat Completions:
        - Keep system/user/assistant roles
        - If image paths detected (file=/abs/path), change last user content to parts: text + multiple image_url (data URL)
        """

        def _extract_image_paths(msgs: List[Dict[str, str]]) -> List[str]:
            paths: List[str] = []
            file_re = re.compile(r"file=([^\s]+)")
            for m in msgs:
                content = m.get("content", "") or ""
                for match in file_re.findall(content):
                    if os.path.isabs(match) and os.path.exists(match):
                        paths.append(match)
            return paths

        def _file_to_data_url(path: str) -> Optional[str]:
            try:
                with open(path, "rb") as f:
                    b = f.read()
                mime = guess_media_type(path, b[:16])
                b, mime = _maybe_downscale(b, mime)
                b64 = base64.b64encode(b).decode("ascii")
                return f"data:{mime};base64,{b64}"
            except Exception:
                return None

        # Build base messages
        payload_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role in ("system", "user", "assistant"):
                payload_messages.append({"role": role, "content": m.get("content", "")})

        # Find the last user
        last_user_idx: Optional[int] = None
        for i in range(len(payload_messages) - 1, -1, -1):
            if payload_messages[i].get("role") == "user":
                last_user_idx = i
                break

        # Inject image parts
        image_paths = _extract_image_paths(messages)
        if image_paths:
            if last_user_idx is None:
                payload_messages.append({"role": "user", "content": ""})
                last_user_idx = len(payload_messages) - 1

            user_text = payload_messages[last_user_idx].get("content", "") or ""
            parts: List[Dict[str, Any]] = []
            if user_text:
                parts.append({"type": "text", "text": user_text})
            for p in image_paths:
                data_url = _file_to_data_url(p)
                if data_url:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
            payload_messages[last_user_idx]["content"] = parts

        # Estimate tokens for input messages (rough heuristic: ~4 chars ≈ 1 token)
        def _estimate_tokens_from_messages(msgs: List[Dict[str, Any]]) -> int:
            total_chars = 0
            for msg in msgs:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            total_chars += len(part.get("text", ""))
            # Rough estimate: 4 chars ≈ 1 token
            return max(0, int(total_chars / 4))

        prompt_tokens = _estimate_tokens_from_messages(payload_messages)
        gen_max_tokens = max(1, int(self.max_new_tokens) - prompt_tokens-100)
        #print(f"prompt_tokens: {prompt_tokens}, gen_max_tokens: {gen_max_tokens}")
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=payload_messages,
                max_tokens=gen_max_tokens,
                temperature=0.7,
                stream=False, 
            )
        except Exception as e:
            raise RuntimeError(f"Together chat.completions failed: {e}")

        visible = (resp.choices[0].message.content or "").strip()
        return visible, visible

    def supports_native_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        # Together SDK exposes an OpenAI-compatible chat.completions interface.
        # Cap output tokens modestly; tool-call outputs are small and passing the
        # full budget risks exceeding the model context window.
        return _openai_chat_tool_call(
            self._client, self.model_name, messages, tools, tool_choice,
            max_tokens=min(4096, int(self.max_new_tokens)),
        )


class ZhipuAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        try:
            from zai import ZhipuAiClient  # type: ignore
        except Exception as e:
            raise RuntimeError("zai SDK not installed. Please install `zai`.") from e

        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise RuntimeError("ZHIPU_API_KEY not set")

        # Initialize Zhipu client
        from zai import ZhipuAiClient  # type: ignore
        self._client = ZhipuAiClient(api_key=api_key)

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Use ZAI Chat Completions (streaming):
        - Pass multi-turn system/user/assistant messages
        - Detect local image paths (file=/abs/path) and change last user content to parts: multiple image_url(data URL) + text
        - Enable thinking; accumulate reasoning_content and content as visible output
        """

        def _extract_image_paths(msgs: List[Dict[str, str]]) -> List[str]:
            paths: List[str] = []
            file_re = re.compile(r"file=([^\s]+)")
            for m in msgs:
                content = m.get("content", "") or ""
                for match in file_re.findall(content):
                    if os.path.isabs(match) and os.path.exists(match):
                        paths.append(match)
            return paths

        def _file_to_data_url(path: str) -> Optional[str]:
            with open(path, "rb") as f:
                b = f.read()
            mime = guess_media_type(path, b[:16])
            b, mime = _maybe_downscale(b, mime)
            b64 = base64.b64encode(b).decode("ascii")
            return f"data:{mime};base64,{b64}"

        # Copy multi-turn messages
        payload_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role in ("system", "user", "assistant"):
                payload_messages.append({"role": role, "content": m.get("content", "")})

        # Find last user
        last_user_idx: Optional[int] = None
        for i in range(len(payload_messages) - 1, -1, -1):
            if payload_messages[i].get("role") == "user":
                last_user_idx = i
                break

        # Inject images as data URL (image_url.url)
        image_paths = _extract_image_paths(messages)
        if image_paths:
            if last_user_idx is None:
                payload_messages.append({"role": "user", "content": ""})
                last_user_idx = len(payload_messages) - 1

            user_text = payload_messages[last_user_idx].get("content", "") or ""
            parts: List[Dict[str, Any]] = []
            # Per examples and common guidance: images first, text after
            for p in image_paths:
                data_url = _file_to_data_url(p)
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
            if user_text:
                parts.append({"type": "text", "text": user_text})
            payload_messages[last_user_idx]["content"] = parts


        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=payload_messages,
                thinking={"type": "enabled"},
                stream=True,
            )
        except Exception as e:
            raise RuntimeError(f"ZAI chat.completions failed: {e}")

        visible_chunks: List[str] = []
        try:
            for chunk in response:
                try:
                    delta = chunk.choices[0].delta  # type: ignore[attr-defined]
                except Exception:
                    delta = None
                if delta is None:
                    continue
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    visible_chunks.append(str(rc))
                ct = getattr(delta, "content", None)
                if ct:
                    visible_chunks.append(str(ct))
        except Exception as e:
            raise RuntimeError(f"ZAI streaming response handling failed: {e}")

        visible = "".join(visible_chunks).strip()
        full = visible
        return visible, full

    def supports_native_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        # ZhipuAiClient exposes an OpenAI-compatible chat.completions with tools.
        return _openai_chat_tool_call(
            self._client, self.model_name, messages, tools, tool_choice,
            max_tokens=None,
        )

class GrokAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        """
        xAI / Grok is OpenAI-compatible:
        - base_url: https://api.x.ai
        - endpoint: /v1/chat/completions
        - params: model, messages, max_tokens, temperature, stream, ...
        Docs: https://docs.x.ai/docs/api-reference  (OpenAI-style) :contentReference[oaicite:0]{index=0}
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        api_key = (
            os.environ.get("XAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError("XAI_API_KEY / GROK_API_KEY not set")

        # xAI follows OpenAI SDK semantics; just change base_url
        self._client = OpenAI(
            api_key=api_key, base_url="https://api.x.ai/v1",
            timeout=_llm_timeout(), max_retries=_llm_max_retries(),
        )

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Same as other APIClients:
        - Keep system / user / assistant
        - Support local images: "file=/abs/path/to/img.png" -> convert to OpenAI/xAI image_url part
        - Map max_new_tokens -> max_tokens
        """
        def _extract_image_paths(msgs: List[Dict[str, str]]) -> List[str]:
            paths: List[str] = []
            file_re = re.compile(r"file=([^\s]+)")
            for m in msgs:
                content = m.get("content", "") or ""
                for match in file_re.findall(content):
                    if os.path.isabs(match) and os.path.exists(match):
                        paths.append(match)
            return paths

        def _file_to_data_url(path: str) -> Optional[str]:
            try:
                with open(path, "rb") as f:
                    b = f.read()
                mime = guess_media_type(path, b[:16])
                b, mime = _maybe_downscale(b, mime)
                b64 = base64.b64encode(b).decode("ascii")
                return f"data:{mime};base64,{b64}"
            except Exception:
                return None

        # 1) Base messages
        payload_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role in ("system", "user", "assistant"):
                payload_messages.append({"role": role, "content": m.get("content", "")})

        # 2) Find last user
        last_user_idx: Optional[int] = None
        for i in range(len(payload_messages) - 1, -1, -1):
            if payload_messages[i].get("role") == "user":
                last_user_idx = i
                break

        # 3) Inject images
        image_paths = _extract_image_paths(messages)
        if image_paths:
            if last_user_idx is None:
                payload_messages.append({"role": "user", "content": ""})
                last_user_idx = len(payload_messages) - 1

            user_text = payload_messages[last_user_idx].get("content", "") or ""
            parts: List[Dict[str, Any]] = []
            if user_text:
                parts.append({"type": "text", "text": user_text})
            for p in image_paths:
                data_url = _file_to_data_url(p)
                if data_url:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
            payload_messages[last_user_idx]["content"] = parts

        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=payload_messages,
                max_tokens=int(self.max_new_tokens),
                temperature=0.7,
                stream=False,
            )
        except Exception as e:
            raise RuntimeError(f"Grok(chat.completions) failed: {e}")

        visible = (resp.choices[0].message.content or "").strip()
        return visible, visible

    def supports_native_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        return _openai_chat_tool_call(
            self._client, self.model_name, messages, tools, tool_choice,
            max_tokens=self.max_new_tokens,
        )


class CursorAPIClient(BaseAPIClient):
    """OpenAI-compatible client for a custom endpoint (e.g. apicursor.com).

    Configured via env vars ``CURSOR_API_BASE_URL`` and ``CURSOR_API_KEY``.
    Supports native tool calling through chat.completions. Used for models
    routed with the ``cursor:`` prefix (see model_loader).
    """

    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        base_url = os.environ.get("CURSOR_API_BASE_URL")
        api_key = os.environ.get("CURSOR_API_KEY")
        if not base_url:
            raise RuntimeError("CURSOR_API_BASE_URL not set")
        if not api_key:
            raise RuntimeError("CURSOR_API_KEY not set")
        # Client-level timeout so a non-returning endpoint can't hang forever.
        self._client = OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=_llm_timeout(), max_retries=_llm_max_retries(),
        )

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        # This endpoint streams responses; accumulate the text deltas.
        chat_messages = _messages_to_chat(messages)
        _record_request(chat_messages)
        _timeout = _llm_timeout()
        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            max_tokens=int(self.max_new_tokens),
            stream=True,
            timeout=_timeout,
        )
        import time as _time
        _start = _time.monotonic()
        parts: List[str] = []
        for chunk in stream:
            if (_time.monotonic() - _start) > _timeout:
                raise TimeoutError(f"LLM stream timed out > {_timeout}s")
            try:
                ct = chunk.choices[0].delta.content
            except Exception:
                ct = None
            if ct:
                parts.append(str(ct))
        visible = "".join(parts).strip()
        return visible, visible

    def supports_native_tools(self) -> bool:
        return True

    def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[ToolCall]]:
        return _openai_chat_tool_call(
            self._client, self.model_name, messages, tools, tool_choice,
            max_tokens=self.max_new_tokens, stream=True,
        )

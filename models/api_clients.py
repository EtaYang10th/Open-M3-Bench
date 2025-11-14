import os
import re
import base64
import mimetypes
from typing import Any, Dict, List, Tuple, Optional
from openai import OpenAI  # type: ignore
import requests
import ipdb

class BaseAPIClient:
    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        raise NotImplementedError


class OpenAIAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._client = OpenAI()

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
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "application/octet-stream"
                with open(path, "rb") as f:
                    b = f.read()
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


class DeepseekAPIClient(BaseAPIClient):
    def __init__(self, model_name: str, max_new_tokens: int = 32768) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

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
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "application/octet-stream"
                with open(path, "rb") as f:
                    b = f.read()
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
        r = requests.post(url, headers=self._headers, json=payload)
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
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "application/octet-stream"
                with open(path, "rb") as f:
                    data = f.read()
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
                mime, _ = mimetypes.guess_type(path)
                media_type = mime or "image/png"
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
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
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "application/octet-stream"
                with open(path, "rb") as f:
                    b = f.read()
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
            mime, _ = mimetypes.guess_type(path)
            mime = mime or "application/octet-stream"
            with open(path, "rb") as f:
                b = f.read()
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
        self._client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

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
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "application/octet-stream"
                with open(path, "rb") as f:
                    b = f.read()
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

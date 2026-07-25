#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, asyncio, shutil, random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from models.tool_schema import build_openai_tools


THINK = ("<think>", "</think>")
tool_box = ("<|begin_of_box|>", "<|end_of_box|>")


def _env_on(name: str, default: bool = True) -> bool:
    """Read a boolean optimization switch. Optimizations default to ON; set 0 to revert."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def _tool_result_max_chars(default: int = 2000) -> int:
    try:
        v = int(os.environ.get("M3_TOOL_RESULT_MAX_CHARS", default))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else 0


def strip_think(txt: str) -> str:
    s, e = map(re.escape, THINK)
    return re.sub(f"{s}.*?{e}", "", txt, flags=re.DOTALL).strip()



def strip_tool_box(txt: str) -> str:
    """
    just serves for glm-4.5v
    """
    s, e = map(re.escape, tool_box)
    s_txt = (txt or "")
    m = re.search(f"{s}(.*?){e}", s_txt, flags=re.DOTALL)
    if m:
        return (m.group(1) or "").strip()
    return s_txt.strip()


def parse_tool_calls(txt: str) -> List[Dict[str, Any]]:
    """
    Parse JSON-based tool calls from model output.
    Accepts either a top-level object with key 'tool_calls' (list),
    or a bare list of call objects. Each call requires 'name' and 'arguments'.
    """
    s = (txt or "").strip()
    if not s:
        return []
    # Accept fenced JSON blocks like ```json\n{...}\n``` or ```\n{...}\n```
    # Also tolerate extra prose around the fenced block
    m = re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```", s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except Exception:
        return []
    seq: List[Any]
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        seq = obj.get("tool_calls")
    elif isinstance(obj, list):
        seq = obj
    else:
        return []
    calls: List[Dict[str, Any]] = []
    for it in seq:
        if isinstance(it, dict) and "name" in it and "arguments" in it:
            calls.append(it)
    return calls


def extract_selected_tools(prepare_output: str, all_tools: List[str]) -> List[str]:
    found: List[str] = []
    if not prepare_output:
        return found
    # Match patterns like: "server/tool: ..." or server/tool: ...
    for qn in all_tools:
        # strict prefix match followed by ':'
        pattern = re.compile(rf"(^|[\"\n\r\t ,]){re.escape(qn)}\s*:\s", re.IGNORECASE)
        if pattern.search(prepare_output):
            found.append(qn)
            continue
        # also support quoted key format: "server/tool": ...
        pattern_quoted = re.compile(rf"(^|[\"\n\r\t ,])\"{re.escape(qn)}\"\s*:\s", re.IGNORECASE)
        if pattern_quoted.search(prepare_output):
            found.append(qn)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for qn in found:
        if qn not in seen:
            uniq.append(qn)
            seen.add(qn)
    return uniq


def to_image_paths(uploaded_file_paths: Optional[List[str]]) -> List[str]:
    if not uploaded_file_paths:
        return []
    return [str(Path(p)) for p in uploaded_file_paths if isinstance(p, str) and p]


def build_workspace_hint(ws_dir: Optional[Path]) -> str:
    """Tell the model where writes are allowed, matching what the host enforces."""
    if ws_dir is None:
        return ""
    return (
        f"\nYour working directory for this task is {ws_dir.resolve()}. Write every "
        f"output file (cropped images, spreadsheets, slides) inside it by passing an "
        f"absolute path under that directory; do not invent paths elsewhere."
    )


def _instruction_role() -> str:
    """Role used for the per-round instruction appended after history.

    Every round previously appended its instruction as ``system`` after a run of
    ``assistant`` tool-result messages, so the request never ended on a ``user``
    turn. Chat endpoints read that as "the user said nothing" and reply with
    "your message came through empty", wasting the round. Sending the
    instruction as ``user`` keeps the turn structure valid. Set
    ``M3_INSTRUCTION_ROLE=system`` to restore the old behaviour.
    """
    role = (os.environ.get("M3_INSTRUCTION_ROLE") or "user").strip().lower()
    return role if role in ("user", "system") else "user"


def build_image_hint(image_paths: List[str]) -> str:
    """Render the per-round image section for a work/prepare system prompt.

    Each entry carries a ``file=<absolute path>`` marker because that marker is
    the contract ``models.api_clients._extract_image_paths`` uses to decide which
    images to inline as base64. Without it, tool-produced images (crops,
    annotated frames) would be named but never actually shown to the model, so
    the model could only guess from filenames. The bare path is kept alongside so
    the model can still copy it verbatim into a tool argument.
    """
    if not image_paths:
        return ""
    lines: List[str] = []
    seen: set = set()
    for p in image_paths:
        try:
            ap = os.path.abspath(p)
        except Exception:
            continue
        if ap in seen:
            continue
        seen.add(ap)
        lines.append(f"  - path: {ap} file={ap}")
    if not lines:
        return ""
    return (
        "\nImages currently available. Their visual content is attached to this "
        "request; pass the path itself when a tool needs an image/file argument:\n"
        + "\n".join(lines)
    )


def truncate_tool_result(text: str, max_chars: Optional[int] = None) -> str:
    """Shorten a tool result for the model-facing history copy only.

    The on-disk record keeps the full original text; this copy exists purely to
    stop history from growing without bound across rounds. For JSON payloads we
    keep the structural head (keys/opening braces read first) which is where the
    useful summary usually lives; plain text is head-truncated the same way.
    """
    if not isinstance(text, str):
        return text
    limit = _tool_result_max_chars() if max_chars is None else max_chars
    if limit <= 0 or len(text) <= limit:
        return text
    dropped = len(text) - limit
    head = text[:limit]
    return f"{head}\n...[truncated {dropped} chars]"


# Argument keys across the image-consuming MCP tools (pyzbar/scan_barcode,
# mcp-yolo/detect-*, imagesorcery/*, ocr/perform_ocr). Only INPUT image args are
# listed; output-path args are intentionally excluded so we never redirect writes.
_IMAGE_ARG_KEYS = {
    "input_path", "input_data", "imagefileuri", "image_path",
    "image_paths", "img_path", "image_uri", "image",
}
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")

# Argument keys through which a tool is told WHERE TO WRITE. When a per-task
# workspace exists these are anchored to it: the model chooses the filename, the
# host chooses the directory. That turns "the model invented a path" from the
# largest error class into a no-op, because /workspace/out.png, ./out.png and
# C:\out.png all resolve to <workspace>/out.png and the call simply succeeds.
_OUTPUT_ARG_KEYS = {
    "output_path", "output_file", "outputpath", "save_path", "dest_path",
    "destination", "filepath", "file_path", "output",
}
_WRITABLE_EXTS = _IMAGE_EXTS + (".xlsx", ".xls", ".csv", ".pptx", ".ppt", ".pdf", ".json", ".txt")


def anchor_output_path(value: str, ws_dir: Path) -> Optional[str]:
    """Rewrite a model-supplied output path to sit directly inside ``ws_dir``.

    Returns None when the value does not look like a writable file path (so we
    never rewrite an unrelated string parameter that happens to share a key
    name), or when it already points inside the workspace.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    if v.startswith("file://"):
        v = v[len("file://"):]
    if v.startswith(("http://", "https://")):
        return None
    name = os.path.basename(v.rstrip("/\\").replace("\\", "/"))
    if not name or not name.lower().endswith(_WRITABLE_EXTS):
        return None
    target = ws_dir / name
    try:
        if os.path.abspath(v) == str(target.resolve()):
            return None
    except Exception:
        pass
    return str(target)


def normalize_output_args(args: Dict[str, Any], ws_dir: Optional[Path]) -> Dict[str, Any]:
    """Confine every explicit output path of a tool call to the task workspace."""
    if not isinstance(args, dict) or ws_dir is None:
        return args
    try:
        ws_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return args
    for k, val in list(args.items()):
        if k.lower() not in _OUTPUT_ARG_KEYS:
            continue
        if isinstance(val, str):
            fixed = anchor_output_path(val, ws_dir)
            if fixed:
                args[k] = fixed
        elif isinstance(val, list):
            new_list = list(val)
            changed = False
            for i, item in enumerate(val):
                if isinstance(item, str):
                    fixed = anchor_output_path(item, ws_dir)
                    if fixed:
                        new_list[i] = fixed
                        changed = True
            if changed:
                args[k] = new_list
    return args


def _workspace_candidates(ws_dir: Optional[Path]) -> List[str]:
    """Existing files inside the task workspace, as absolute paths."""
    if ws_dir is None:
        return []
    try:
        # abspath, not resolve: a symlinked input must stay addressed through the
        # workspace so tool-derived output paths also stay in the workspace.
        return [
            os.path.abspath(str(p))
            for p in sorted(ws_dir.iterdir())
            if p.is_file() or p.is_symlink()
        ]
    except Exception:
        return []


def _resolve_one_image_ref(value: str, image_paths: List[str]) -> Optional[str]:
    """Map a single (possibly fabricated) image reference to a real local path.

    Returns a corrected absolute path, or None to leave the value unchanged.
    Only acts when we actually have task image(s) and the given value does not
    already point at an existing file.
    """
    if not isinstance(value, str) or not value.strip() or not image_paths:
        return None
    v = value.strip()
    if v.startswith("file://"):
        v = v[len("file://"):]
    # URLs are valid inputs for some tools (e.g. ocr) -> never rewrite.
    if v.startswith(("http://", "https://")):
        return None
    # Already a readable file -> keep as-is.
    try:
        if os.path.exists(v):
            return None
    except Exception:
        pass
    base = os.path.basename(v.rstrip("/"))
    # 1) Basename match against known real paths (uploaded + tool-generated).
    for p in image_paths:
        if os.path.basename(p) == base and os.path.exists(p):
            return p
    # 2) Looks like an image filename but doesn't exist -> substitute the
    #    task's primary uploaded image (first entry) if it exists.
    low = v.lower()
    if low.endswith(_IMAGE_EXTS) or "image" in low or "photo" in low or "input" in low:
        for p in image_paths:
            if os.path.exists(p):
                return p
    return None


def normalize_excel_filepath(args: Dict[str, Any], ws_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Make excel-mcp 'filepath' absolute (server rejects relative paths in stdio).

    Maps any relative / non-writable path to an absolute path using just the
    basename, so the tool executes instead of erroring on the path shape. The
    target directory is the per-task workspace when one exists, else ./media.
    """
    if not isinstance(args, dict):
        return args
    fp = args.get("filepath")
    if not isinstance(fp, str) or not fp.strip():
        return args
    media_dir = (ws_dir if ws_dir is not None else Path("media")).resolve()
    try:
        media_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    p = Path(fp)
    # Absolute and inside a writable location we can keep -> only reroute if the
    # parent dir is missing/unwritable (e.g. /workspace). Otherwise normalize.
    if p.is_absolute():
        parent = p.parent
        if parent.exists() and os.access(str(parent), os.W_OK):
            return args
        args["filepath"] = str(media_dir / p.name)
        return args
    args["filepath"] = str(media_dir / p.name)
    return args


def normalize_image_args(args: Dict[str, Any], image_paths: List[str]) -> Dict[str, Any]:
    """Normalize fabricated image-path arguments to real task image paths.

    Low-invasive: only touches keys in _IMAGE_ARG_KEYS, only when the model gave
    a non-existent path and a real image exists. Leaves everything else intact so
    native tool-calling argument structure and non-image params are preserved.
    """
    if not isinstance(args, dict) or not image_paths:
        return args
    for k, val in list(args.items()):
        if k.lower() not in _IMAGE_ARG_KEYS:
            continue
        if isinstance(val, str):
            fixed = _resolve_one_image_ref(val, image_paths)
            if fixed:
                args[k] = fixed
        elif isinstance(val, list):
            new_list = list(val)
            changed = False
            for i, item in enumerate(val):
                if isinstance(item, str):
                    fixed = _resolve_one_image_ref(item, image_paths)
                    if fixed:
                        new_list[i] = fixed
                        changed = True
            if changed:
                args[k] = new_list
    return args


class RoundRunner:
    def __init__(
        self,
        host,
        model_driver,
        max_step: int,
        max_concurrent: int,
        top_tools: int = 4,
        num_context_tools: int = 0,
        gt_tools: Optional[List[str]] = None,
        ws_dir: Optional[Path] = None,
    ) -> None:
        self.host = host
        self.model_driver = model_driver
        self.max_step = max(1, int(max_step))
        self.max_concurrent = max(1, int(max_concurrent))
        self.top_tools = max(1, int(top_tools))
        self.num_context_tools = num_context_tools
        self.gt_tools = gt_tools or []
        # Per-task workspace. When set, every tool output path is anchored here
        # and fabricated input paths are resolved against this directory first.
        self.ws_dir = Path(ws_dir) if ws_dir is not None else None

    # ---------------- optimization switches ----------------
    def _skip_end_stage(self) -> bool:
        return _env_on("M3_SKIP_END_STAGE")

    def _merge_prepare_work(self) -> bool:
        return _env_on("M3_MERGE_PREPARE_WORK")

    def _list_all_tool_descriptions(self, tool_names: Optional[List[str]] = None) -> List[str]:
        lines: List[str] = []
        # If tool_names is provided, use it; otherwise use all
        target_tools = tool_names if tool_names is not None else self.host.tools.keys()
        
        for qn in target_tools:
            if qn in self.host.tools:
                _server, _tname, desc, _schema = self.host.tools[qn]
                lines.append(f"{qn}: {desc}")
        return lines

    def _describe_selected(self, tool_names: List[str]) -> List[str]:
        lines: List[str] = []
        for qn in tool_names:
            _server, _tname, desc, _schema = self.host.tools.get(qn, ("", "", "", {}))
            lines.append(f"{qn}: {desc}")
        return lines

    def _supports_native_tools(self) -> bool:
        try:
            fn = getattr(self.model_driver, "supports_native_tools", None)
            return bool(fn()) if callable(fn) else False
        except Exception:
            return False

    async def _prepare_work_merged(
        self,
        history: List[Dict[str, str]],
        context_tool_names: List[str],
        image_paths: List[str],
        round_idx: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Single native call replacing the prepare + work stages.

        With native function calling the model already receives every candidate
        tool as a JSON schema, so the separate natural-language "pick tools"
        turn (which re-sent all 50 descriptions as plain text) is redundant.
        Returns ``(visible_text, tool_calls)``.
        """
        tools, name_map = build_openai_tools(self.host.tools, context_tool_names)
        if not tools:
            return "", []
        img_hint = build_image_hint(image_paths) + build_workspace_hint(self.ws_dir)
        system = (
            f"You are an assistant with MCP tool invocation capability. Complete the task by "
            f"calling MCP tools, possibly across multiple rounds. This is round {round_idx} of "
            f"{self.max_step}. Consider completed steps from the conversation history and skip "
            f"unnecessary steps; extra calls may reduce your score.\n"
            f"Call 1 to {self.max_concurrent} tools now via function calling. Briefly state which "
            f"tools you chose and why in your text reply. If the task is already complete, reply "
            f"with the conclusion and call no tools."
            + img_hint
        )
        messages = history + [{"role": _instruction_role(), "content": system}]

        def _run():
            return self.model_driver.generate_with_tools(messages, tools, tool_choice="auto")

        visible, raw_calls = await asyncio.get_event_loop().run_in_executor(None, _run)
        tcs: List[Dict[str, Any]] = []
        for c in raw_calls:
            esc = c.get("name")
            tcs.append({"name": name_map.get(esc, esc), "arguments": c.get("arguments", {}) or {}})
        return strip_think(visible or ""), tcs

    async def _work_native(
        self,
        history: List[Dict[str, str]],
        selected_tools: List[str],
        image_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Native tool-calling work stage. Returns parsed tool_calls (qualified names)."""
        tools, name_map = build_openai_tools(self.host.tools, selected_tools)
        if not tools:
            return []
        img_hint = build_image_hint(image_paths) + build_workspace_hint(self.ws_dir)
        work_system = (
            f"Now call the necessary MCP tools for this round using function calling. "
            f"Call 1 to {self.max_concurrent} tools. Consider completed steps from the "
            f"conversation history and skip unnecessary steps; extra calls may reduce your score."
            + img_hint
        )
        work_messages = history + [{"role": _instruction_role(), "content": work_system}]

        def _run():
            return self.model_driver.generate_with_tools(work_messages, tools, tool_choice="auto")

        _visible, raw_calls = await asyncio.get_event_loop().run_in_executor(None, _run)
        tcs: List[Dict[str, Any]] = []
        for c in raw_calls:
            esc = c.get("name")
            qn = name_map.get(esc, esc)
            args = c.get("arguments", {}) or {}
            tcs.append({"name": qn, "arguments": args})
        return tcs

    

    async def _call_tools_concurrently(self, tool_calls: List[Dict[str, Any]], image_paths: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], str]]:
        # Known paths come first so the "substitute the task's primary image"
        # fallback keeps picking the uploaded original. Workspace files are
        # appended to widen only the basename-match domain, which is now this
        # task's handful of files rather than a shared media/ with thousands.
        img_pool = list(image_paths or []) + _workspace_candidates(self.ws_dir)
        seen_pool: set = set()
        img_pool = [p for p in img_pool if not (p in seen_pool or seen_pool.add(p))]

        async def prepare_and_call(tc: Dict[str, Any]):
            raw_args = tc.get("arguments", {}) or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}
            args = raw_args
            # Low-invasive image-path normalization: if the model fabricated an
            # image path (e.g. "image.png"), remap it to the real task image so
            # the tool actually receives the uploaded file. Mutates tc so the
            # corrected args are what gets logged/recorded.
            if isinstance(args, dict) and img_pool:
                args = normalize_image_args(args, img_pool)
                tc["arguments"] = args
            # Anchor every write to the per-task workspace (no-op when there is
            # no workspace). Runs before the excel-specific rule so 'filepath'
            # is already absolute-and-inside-workspace by the time it is checked.
            if isinstance(args, dict) and self.ws_dir is not None:
                args = normalize_output_args(args, self.ws_dir)
                tc["arguments"] = args
            # excel-mcp requires an absolute filepath in stdio mode. Normalize any
            # relative filepath to an absolute writable path so
            # create_workbook/create_worksheet actually execute.
            if isinstance(args, dict) and str(tc.get("name", "")).startswith("excel/"):
                args = normalize_excel_filepath(args, self.ws_dir)
                tc["arguments"] = args
            try:
                res = await self.host.call(tc["name"], args)
            except Exception as e:
                res = f"[Tool error] {e}"
            return tc, res

        sem = asyncio.Semaphore(self.max_concurrent)

        async def with_sem(tc: Dict[str, Any]):
            async with sem:
                return await prepare_and_call(tc)

        return await asyncio.gather(*[with_sem(tc) for tc in tool_calls])


    async def run(
        self,
        history: List[Dict[str, str]],
        last_user: str,
        uploaded_file_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        image_paths = to_image_paths(uploaded_file_paths)
        
        # --- Tool Selection Logic ---
        all_tool_names = list(self.host.tools.keys())
        context_tool_names = all_tool_names
        
        if self.num_context_tools > 0:
            # If we have GT tools, start with them
            selected_set = set()
            # Filter GT tools that actually exist in host.tools
            valid_gt = [t for t in self.gt_tools if t in self.host.tools]
            selected_set.update(valid_gt)
            
            # Fill the rest with random tools
            needed = self.num_context_tools - len(selected_set)
            if needed > 0:
                candidates = [t for t in all_tool_names if t not in selected_set]
                if len(candidates) <= needed:
                    selected_set.update(candidates)
                else:
                    selected_set.update(random.sample(candidates, needed))
            
            context_tool_names = list(selected_set)
        # -----------------------------

        round_groups: List[List[Dict[str, Any]]] = []
        dialogues: List[Dict[str, str]] = []
        results_flat: List[Tuple[Dict[str, Any], str]] = []

        merge_native = self._merge_prepare_work() and self._supports_native_tools()

        for i in range(1, self.max_step + 1):
            # ===== Merged prepare+work stage (native tool calling only) =====
            if merge_native:
                try:
                    prep_visible, tcs = await self._prepare_work_merged(
                        history, context_tool_names, image_paths, i,
                    )
                except Exception as e:
                    print(f"[WARN] Merged prepare+work failed ({e}); falling back to two-stage path.")
                    prep_visible, tcs = "", None
                if tcs is not None:
                    tcs = tcs[: self.max_concurrent]
                    if not tcs:
                        # No tool calls -> the model considers the task done.
                        # ``stop_reason`` distinguishes this from a malformed
                        # tool_calls payload, which also yields an empty list but
                        # is a genuine protocol failure (see evaluate_calls.py).
                        dialogues.append({
                            "prepare": prep_visible,
                            "work": {"tool_calls": []},
                            "end": "yes",
                            "stop_reason": "completed",
                        })
                        break
                    results = await self._call_tools_concurrently(tcs, image_paths)
                    image_paths = await self._finalize_round(
                        tcs, results, history, round_groups, results_flat,
                        dialogues, prep_visible, last_user, image_paths,
                    )
                    if dialogues and dialogues[-1].get("end") == "yes":
                        break
                    continue

            # ---------------- Prepare Stage ----------------
            tool_lines = self._list_all_tool_descriptions(context_tool_names)
            with open("tool_lines.txt", "w") as f:
                f.write("\n".join(tool_lines))
            prepare_system = "".join((
                f"You are an assistant with MCP tool invocation capability. "
                f"You must call one or more MCP tools, possibly in multiple rounds, to complete the task. Consider necessary steps as much as possible, consider completed steps in conjunction with conversation history, and skip unnecessary steps. Extra steps may reduce your final score\n"
                f"This is round {i} of {self.max_step}.\n"
                f"Now decide which tools to use for this round.\n"
                f"Output EXACTLY in the following format (no extra text):\n\n"
                f"\"server_name/tool_name: <description> \\n Args: <arguments (type): description> \\n Returns: <returns(type): description> \"",
                f"\"server/tool: <description> \\n Args: <arguments (type): description> \\n Returns: <returns(type): description> \"\n\n"
                f"<plan for how to use>\n\n"
                f"Here are the MCP tools for you. "
                f"All tools list (name: description):\n" + "\n".join(tool_lines)
            )).strip()

            prep_messages = history + [{"role": _instruction_role(), "content": prepare_system}]
            prep_visible, _prep_full = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model_driver.generate_once(prep_messages)
            )

            prep_visible = strip_think(prep_visible)
            selected_tools = extract_selected_tools(prep_visible, context_tool_names)
            if not selected_tools:
                # Robust fallback: never assign the free-form prepare text as a
                # "tool list". Use lexical search to get a valid List[str].
                selected_tools = self.host.select_tools_for(last_user, k=self.top_tools)
                if not selected_tools:
                    # Last resort: first top_tools from the context so the work
                    # stage always has at least some tools to work with.
                    selected_tools = list(context_tool_names)[: self.top_tools]

            # ---------------- Work Stage ----------------
            # Normalize selected tools to a list of valid qualified names.
            if isinstance(selected_tools, str):
                selected_list = extract_selected_tools(selected_tools, context_tool_names)
                if not selected_list:
                    selected_list = self.host.select_tools_for(last_user, k=self.top_tools)
            else:
                selected_list = list(selected_tools)
            valid_selected = [t for t in selected_list if t in self.host.tools]

            # ===== Native tool-calling path (preferred) =====
            if self._supports_native_tools() and valid_selected:
                try:
                    tcs = await self._work_native(history, valid_selected, image_paths)
                except Exception as e:
                    print(f"[WARN] Native tool calling failed ({e}); falling back to prompt path.")
                    tcs = []
                if tcs:
                    tcs = tcs[: self.max_concurrent]
                    results = await self._call_tools_concurrently(tcs, image_paths)
                    image_paths = await self._finalize_round(
                        tcs, results, history, round_groups, results_flat,
                        dialogues, prep_visible, last_user, image_paths,
                    )
                    if dialogues and dialogues[-1].get("end") == "yes":
                        break
                    continue
                # else: fall through to prompt-based path as fallback

            # ===== Prompt-based fallback path =====
            # Use the validated list (names guaranteed to exist in host.tools).
            # Fall back to selected_list if validation filtered everything out.
            describe_tools = valid_selected or selected_list
            selected_desc = self._describe_selected(describe_tools)
            img_section = (build_image_hint(image_paths) + build_workspace_hint(self.ws_dir)).lstrip("\n")

            work_system = (
                f"Now execute the selected tools for this round. Return STRICT JSON ONLY with a single top-level key 'tool_calls' containing 1 to {self.max_concurrent} items. "
                f"Consider necessary steps as much as possible, consider completed steps in conjunction with conversation history, and skip unnecessary steps. Extra tool calls may reduce your final score\n"
                f"Each item must be an object: {{\"name\": \"server_name/tool_name\", \"arguments\": {{ ... }} }}.\n"
                f"Do NOT include markdown fences or any extra text or special delimiters and tokens.\n"
                f"If a tool requires images/files, include the appropriate arguments per the tool's schema. You may use the image pool paths listed below if applicable.\n"
                f"Selected tools (name: description):\n"
                + "\n".join(selected_desc)
                + (f"\n{img_section}" if img_section else "")
            ).strip()
            work_messages = history + [{"role": _instruction_role(), "content": work_system}]

            work_visible, work_full = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model_driver.generate_once(work_messages)
            )
            
            work_visible = strip_think(work_visible)
            tcs = parse_tool_calls(work_visible)
            if not tcs:
                work_visible = strip_tool_box(work_visible)
                tcs = parse_tool_calls(work_visible)

            if not tcs:
                # Retry once to generate valid tool_calls
                print("[WARN] Invalid tool_calls JSON detected; prompting model to retry with strict JSON.")
                retry_messages = work_messages + [{
                    "role": "system",
                    "content": (
                        "Reminder: Your last output was not valid JSON and could not be parsed. "
                        "Retry now and return STRICT JSON ONLY with a single top-level key 'tool_calls' "
                        f"containing 1 to {self.max_concurrent} items. Do NOT include markdown fences or any extra text."
                    ),
                }]
                work_visible_retry, _work_full_retry = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.model_driver.generate_once(retry_messages)
                )
                work_visible_retry = strip_think(work_visible_retry)
                tcs = parse_tool_calls(work_visible_retry)
                if not tcs:
                    work_visible_retry = strip_tool_box(work_visible_retry)
                    tcs = parse_tool_calls(work_visible_retry)

            if not tcs:
                # No valid tool calls after retry: a real protocol failure.
                dialogues.append({
                    "prepare": prep_visible,
                    "work": {"tool_calls": []},
                    "end": "no",
                    "stop_reason": "invalid_tool_calls",
                })
                history.append({
                    "role": "assistant",
                    "content": "[TOOL_ERROR] The model did not produce a valid tool_calls JSON; the function call format is invalid and could not be parsed. Continuing to the next round.",
                })
                continue

            tcs = tcs[: self.max_concurrent]
            results = await self._call_tools_concurrently(tcs, image_paths)
            image_paths = await self._finalize_round(
                tcs, results, history, round_groups, results_flat,
                dialogues, prep_visible, last_user, image_paths,
            )
            if dialogues and dialogues[-1].get("end") == "yes":
                break

        return {
            "round_groups": round_groups,
            "results_flat": results_flat,
            "dialogues": dialogues,
        }

    def _collect_new_image_paths(self, results: List[Tuple[Dict[str, Any], str]]) -> List[str]:
        new_image_paths: List[str] = []
        for _tc, tool_result in results:
            try:
                obj = json.loads(tool_result)
            except Exception:
                continue
            if isinstance(obj, dict):
                imgs = obj.get("images")
                if isinstance(imgs, list):
                    for p in imgs:
                        if isinstance(p, str) and p:
                            new_image_paths.append(p)
                json_parts = obj.get("json_parts")
                if isinstance(json_parts, list):
                    for jp in json_parts:
                        if isinstance(jp, dict) and isinstance(jp.get("images"), list):
                            for p in jp.get("images"):
                                if isinstance(p, str) and p:
                                    new_image_paths.append(p)
        return new_image_paths

    async def _finalize_round(
        self,
        tcs: List[Dict[str, Any]],
        results: List[Tuple[Dict[str, Any], str]],
        history: List[Dict[str, str]],
        round_groups: List[List[Dict[str, Any]]],
        results_flat: List[Tuple[Dict[str, Any], str]],
        dialogues: List[Dict[str, Any]],
        prep_visible: Any,
        last_user: str,
        image_paths: List[str],
    ) -> List[str]:
        """Record round results, propagate images, and run the end stage.

        Shared by both the native and prompt-based work paths. Returns the
        (possibly extended) list of image paths for the next round.
        """
        # record for history and round logging
        # Two copies: the on-disk record keeps the full original tool output
        # (evaluation data), while the model-facing history copy is truncated so
        # prompt size stops growing linearly with the number of rounds.
        aggregated_parts: List[str] = []
        round_entries: List[Dict[str, Any]] = []
        for tc, tool_result in results:
            aggregated_parts.append(truncate_tool_result(tool_result))
            entry = dict(tc)
            entry["result"] = tool_result
            round_entries.append(entry)
            results_flat.append((tc, tool_result))
        if aggregated_parts:
            history.append({"role": "assistant", "content": "\n\n".join(aggregated_parts)})
        if round_entries:
            round_groups.append(round_entries)

        # ---------------- End Stage ----------------
        new_image_paths = self._collect_new_image_paths(results)
        if new_image_paths:
            seen_paths = set()
            merged: List[str] = []
            for p in list(image_paths) + new_image_paths:
                if p not in seen_paths:
                    seen_paths.add(p)
                    merged.append(p)
            image_paths = merged

        if self._skip_end_stage():
            # Termination is inferred from the work stage: the round produced
            # tool calls, so there is still work in flight -> keep going. An
            # empty tool_calls round is handled by the caller, which records
            # end="yes" and breaks without spending an extra LLM request.
            yn = "no" if tcs else "yes"
        else:
            end_system = (
                f"If the task can be considered complete based on current tool results and question, Here is the original question: {last_user}, \n"
                "answer 'yes'. Otherwise 'no'.\n"
                "Answer strictly with 'yes' or 'no', No extra words."
            ).strip()
            end_messages = history + [{"role": _instruction_role(), "content": end_system}]
            end_visible, _end_full = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model_driver.generate_once(end_messages)
            )
            end_visible = strip_think(end_visible).strip().lower()
            yn = "yes" if end_visible.startswith("y") else ("no" if end_visible.startswith("n") else "no")
        dialogues.append({"prepare": prep_visible, "work": {"tool_calls": tcs}, "end": yn})
        return image_paths

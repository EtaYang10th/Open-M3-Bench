#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, asyncio, shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import ipdb


THINK = ("<think>", "</think>")
tool_box = ("<|begin_of_box|>", "<|end_of_box|>")


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


class RoundRunner:
    def __init__(
        self,
        host,
        model_driver,
        max_step: int,
        max_concurrent: int,
        top_tools: int = 4,
    ) -> None:
        self.host = host
        self.model_driver = model_driver
        self.max_step = max(1, int(max_step))
        self.max_concurrent = max(1, int(max_concurrent))
        self.top_tools = max(1, int(top_tools))

    def _list_all_tool_descriptions(self) -> List[str]:
        lines: List[str] = []
        for qn, (_server, _tname, desc, _schema) in self.host.tools.items():
            lines.append(f"{qn}: {desc}")
        return lines

    def _describe_selected(self, tool_names: List[str]) -> List[str]:
        lines: List[str] = []
        for qn in tool_names:
            _server, _tname, desc, _schema = self.host.tools.get(qn, ("", "", "", {}))
            lines.append(f"{qn}: {desc}")
        return lines

    

    async def _call_tools_concurrently(self, tool_calls: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
        async def prepare_and_call(tc: Dict[str, Any]):
            raw_args = tc.get("arguments", {}) or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}
            args = raw_args
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
        all_tool_names = list(self.host.tools.keys())

        round_groups: List[List[Dict[str, Any]]] = []
        dialogues: List[Dict[str, str]] = []
        results_flat: List[Tuple[Dict[str, Any], str]] = []

        for i in range(1, self.max_step + 1):
            # ---------------- Prepare Stage ----------------
            tool_lines = self._list_all_tool_descriptions()
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

            prep_messages = history + [{"role": "system", "content": prepare_system}]
            prep_visible, _prep_full = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model_driver.generate_once(prep_messages)
            )

            prep_visible = strip_think(prep_visible)
            selected_tools = extract_selected_tools(prep_visible, all_tool_names)
            if not selected_tools:
                if prep_visible:
                    selected_tools = prep_visible
                else:
                    selected_tools = self.host.select_tools_for(last_user, k=self.top_tools)

            # ---------------- Work Stage ----------------
            selected_desc = self._describe_selected(selected_tools)
            img_section = ""
            if image_paths:
                img_lines = [f"  - path: {p}" for p in image_paths]
                img_section = "image list:\n" + "\n".join(img_lines)

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
            work_messages = history + [{"role": "system", "content": work_system}]

            work_visible, work_full = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model_driver.generate_once(work_messages)
            )
            
            work_visible = strip_think(work_visible)
            tcs = parse_tool_calls(work_visible)
            #import ipdb; ipdb.set_trace()
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
                # No valid tool calls after retry: record and continue to next round
                dialogues.append({"prepare": prep_visible, "work": {"tool_calls": []}, "end": "no"})
                history.append({
                    "role": "assistant",
                    "content": "[TOOL_ERROR] The model did not produce a valid tool_calls JSON; the function call format is invalid and could not be parsed. Continuing to the next round.",
                })
                continue

            tcs = tcs[: self.max_concurrent]
            results = await self._call_tools_concurrently(tcs)

            # record for history and round logging
            aggregated_parts: List[str] = []
            round_entries: List[Dict[str, Any]] = []
            for tc, tool_result in results:
                aggregated_parts.append(tool_result)
                entry = dict(tc)
                entry["result"] = tool_result
                round_entries.append(entry)
                results_flat.append((tc, tool_result))
            if aggregated_parts:
                history.append({"role": "assistant", "content": "\n\n".join(aggregated_parts)})
            if round_entries:
                round_groups.append(round_entries)


            # ---------------- End Stage ----------------
            # Minimal: collect any returned image paths for next round visibility/injection
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
            if new_image_paths:
                seen_paths = set()
                merged: List[str] = []
                for p in list(image_paths) + new_image_paths:
                    if p not in seen_paths:
                        seen_paths.add(p)
                        merged.append(p)
                image_paths = merged

            
            end_system = (
                f"If the task can be considered complete based on current tool results and question, Here is the original question: {last_user}, \n"
                "answer 'yes'. Otherwise 'no'.\n"
                "Answer strictly with 'yes' or 'no', No extra words."
            ).strip()
            end_messages = history + [{"role": "system", "content": end_system}]
            end_visible, _end_full = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model_driver.generate_once(end_messages)
            )
            end_visible = strip_think(end_visible).strip().lower()
            yn = "yes" if end_visible.startswith("y") else ("no" if end_visible.startswith("n") else "no")
            dialogues.append({"prepare": prep_visible, "work": {"tool_calls": tcs}, "end": yn})
            if yn == "yes":
                break

        return {
            "round_groups": round_groups,
            "results_flat": results_flat,
            "dialogues": dialogues,
        }

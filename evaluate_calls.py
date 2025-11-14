import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from models import create_model_driver  # reuse unified model loader/router

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
except Exception as e:
    raise RuntimeError("rich is required: pip install rich") from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Prediction JSON path (*_test_mcp_fuzzy.json)")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path (results/<model>/callanalysis.json)")
    parser.add_argument("--judge-model", type=str, required=True, help="Judge model name (e.g., gpt-5-nano)")
    parser.add_argument("--num_client", type=int, default=1, help="Number of worker threads")
    parser.add_argument("--max_new_tokens", type=int, default=32768, help="Max new tokens for judge model")
    return parser.parse_args()


def load_predictions(pred_path: str) -> List[Dict[str, Any]]:
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # If dict, try common container fields
    for key in ("data", "items", "results", "predictions"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    raise ValueError("Unable to parse prediction JSON: top-level is neither a list nor contains common list fields")


def count_illegal_calling(samples: List[Dict[str, Any]]) -> int:
    illegal = 0
    for item in samples:
        dialogue = item.get("dialogue")
        if not isinstance(dialogue, list):
            continue
        for turn in dialogue:
            if not isinstance(turn, dict):
                continue
            work = turn.get("work")
            if not isinstance(work, dict):
                continue
            tool_calls = work.get("tool_calls")
            if isinstance(tool_calls, list) and len(tool_calls) == 0:
                illegal += 1
                break  # count at most once per sample
    return illegal


def extract_all_calls(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for item in samples:
        steps = item.get("steps")
        if not isinstance(steps, list):
            continue
        for st in steps:
            if not isinstance(st, dict):
                continue
            call_list = st.get("calls")
            if not isinstance(call_list, list):
                continue
            for c in call_list:
                if not isinstance(c, dict):
                    continue
                # Keep only key info to avoid irrelevant fields affecting judging
                calls.append({
                    "id": item.get("id"),
                    "name": c.get("name"),
                    "arguments": c.get("arguments"),
                    "result": c.get("result"),
                })
    return calls


SYSTEM_PROMPT = (
"You are a strict auditor. You will be given an MCP call (including name/arguments/result).\n"
"Classify this call into exactly one of the following four categories and output only one number:\n"
"1 = Unknown Tool Invocation (the call targets a non-existent or non-exposed MCP tool)\n"
"2 = Invalid Invocation Arguments (the tool exists but the arguments are invalid/incomplete, HTTP 400, or messages like '[Tool error] Invalid arguments')\n"
"3 = Resource Not Found (the MCP server/tool backend could not find the requested resource or route, e.g. HTTP 404 or upstream 'not found')\n"
"4 = Successful call (the call succeeded and the result is a normal response, not an error message)\n\n"
"Requirements:\n"
"- Output only one number from 1/2/3/4. Do not output anything else.\n"
)


def build_messages_for_call(call_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    user_content = (
        "Please output only one number among 1/2/3/4. Here is the MCP call:\n\n"
        + json.dumps(call_obj, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_judge_output(text: str) -> Optional[int]:
    if text is None:
        return None
    s = (text or "").strip()
    # Take only the first line containing a pure number
    first_line = s.splitlines()[0].strip()
    try:
        val = int(first_line)
        if val in (1, 2, 3, 4):
            return val
    except Exception:
        return None
    return None


def classify_call(
    client: Any,
    call_obj: Dict[str, Any],
    max_new_tokens: int,
    max_retries: int = 10,
) -> int:
    messages = build_messages_for_call(call_obj)
    last_err: Optional[Exception] = None

    def _run_once_with_timeout(timeout_sec: float) -> Tuple[Optional[str], Optional[Exception], bool]:
        """Return (visible_text, error, timed_out)"""
        result: Dict[str, Any] = {"text": None, "err": None}

        def _target() -> None:
            try:
                visible, _ = client.generate_once(messages)
                result["text"] = visible
            except Exception as e:  # noqa: BLE001
                result["err"] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            # Timed out -> treat as failure
            return None, None, True
        return result["text"], result["err"], False

    for attempt in range(1, max_retries + 1):
        visible, err, timed_out = _run_once_with_timeout(timeout_sec=20.0)
        if timed_out:
            last_err = TimeoutError("judge model timeout > 20s")
        elif err is not None:
            last_err = err
        else:
            parsed = parse_judge_output(visible or "")
            if parsed is not None:
                return parsed

        # Back off slightly then retry
        time.sleep(min(0.2 * attempt, 2.0))

    # Exceeded retry limit, fail
    raise RuntimeError(f"Failed to parse judge output; exceeded retry limit. Last error: {last_err}")


def render_table(
    counts: Dict[str, int],
    processed_calls: int,
    total_calls: int,
    model_name: str,
) -> Panel:
    table = Table(title=f"Call classification (model: {model_name})", expand=True)
    table.add_column("Category", justify="left")
    table.add_column("Count", justify="right")
    table.add_row("Illegal calling", str(counts.get("illegal", 0)))
    table.add_row("1. Unknown Tool Invocation", str(counts.get("unknown_tool", 0)))
    table.add_row("2. Invalid Invocation Arguments", str(counts.get("invalid_arguments", 0)))
    table.add_row("3. Resource Not Found", str(counts.get("resource_not_found", 0)))
    table.add_row("4. Successful call", str(counts.get("success", 0)))
    table.add_section()
    table.add_row("Processed / Total calls", f"{processed_calls} / {total_calls}")
    return Panel(table, title="MCP Call Classification Live Stats", border_style="cyan")


def main() -> None:
    args = parse_args()

    pred_path = os.path.abspath(args.pred)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    samples = load_predictions(pred_path)

    # First count illegal calling (per sample: if any round has empty work.tool_calls, count once)
    illegal_cnt = count_illegal_calling(samples)

    # Aggregate all calls (classify per-call)
    all_calls = extract_all_calls(samples)
    total_calls = len(all_calls)

    # Counters
    counts = {
        "illegal": illegal_cnt,
        "unknown_tool": 0,
        "invalid_arguments": 0,
        "resource_not_found": 0,
        "success": 0,
    }

    # Judge client: reuse unified entry, compatible with various backends
    client = create_model_driver(args.judge_model, max_new_tokens=args.max_new_tokens)

    # Concurrent execution
    console = Console()
    lock = threading.Lock()
    processed = 0

    def worker(call_obj: Dict[str, Any]) -> Tuple[int, Optional[str]]:
        try:
            cls_id = classify_call(client, call_obj, args.max_new_tokens)
            return cls_id, None
        except Exception as e:
            return -1, str(e)

    with Live(render_table(counts, processed, total_calls, args.judge_model), console=console, refresh_per_second=8) as live:
        with ThreadPoolExecutor(max_workers=max(1, args.num_client)) as ex:
            futures = [ex.submit(worker, c) for c in all_calls]
            for fut in as_completed(futures):
                cls_id, err = fut.result()
                if cls_id == -1:
                    # Any single call exceeds retry limit: terminate as required
                    raise RuntimeError(f"Classification failed: {err}")

                with lock:
                    if cls_id == 1:
                        counts["unknown_tool"] += 1
                    elif cls_id == 2:
                        counts["invalid_arguments"] += 1
                    elif cls_id == 3:
                        counts["resource_not_found"] += 1
                    elif cls_id == 4:
                        counts["success"] += 1
                    processed += 1
                # Refresh live view
                live.update(render_table(counts, processed, total_calls, args.judge_model))

    # Output to JSON (five categories only)
    result = {
        "illegal_calling": counts["illegal"],
        "unknown_tool": counts["unknown_tool"],
        "invalid_arguments": counts["invalid_arguments"],
        "resource_not_found": counts["resource_not_found"],
        "success": counts["success"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Per requirement: on parse failure, terminate process (raise to exit)
        raise



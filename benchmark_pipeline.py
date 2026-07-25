#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, asyncio, textwrap, argparse, time
import inspect
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from models import create_model_driver
from mcp_host import MCPHost
from round_runner import RoundRunner, strip_think
import workspace as ws_mod

import tqdm

# Project-level media directory to store working images during benchmarking
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)



def _task_timeout(default: float = 1800.0) -> float:
    """Wall-clock budget for one task (env M3_TASK_TIMEOUT; 0 disables).

    Per-request LLM/tool timeouts bound a single call, but a task issues many of
    them, so without this a slow-but-alive task can run for hours.
    """
    try:
        return float(os.environ.get("M3_TASK_TIMEOUT", default))
    except (TypeError, ValueError):
        return default


def _max_task_requeue(default: int = 2) -> int:
    """How many times one task may be requeued (env M3_MAX_TASK_REQUEUE)."""
    try:
        return max(0, int(os.environ.get("M3_MAX_TASK_REQUEUE", default)))
    except (TypeError, ValueError):
        return default


def _executor_workers(num_client: int) -> int:
    """Size of the default ThreadPoolExecutor (env M3_EXECUTOR_WORKERS; 0 keeps
    the interpreter default). Each worker needs one thread per in-flight LLM
    call, plus headroom for threads leaked by cancelled (timed-out) calls."""
    raw = os.environ.get("M3_EXECUTOR_WORKERS")
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return max(32, num_client * 8)


def resolve_image_path(base_dir: Path, rel_path: str) -> Optional[str]:
    if not rel_path:
        return None
    cand1 = (base_dir / rel_path).resolve()
    if cand1.exists():
        return str(cand1)
    # Fallback: annotation 'image' fields are often prefixed (e.g. images/<id>.png)
    # while the actual files live flat under the image_dir as <id>.png. Try the
    # basename directly under base_dir so the real task image is still uploaded
    # (prevents the model from fabricating image paths for a missing upload).
    base_name = Path(rel_path).name
    cand2 = (base_dir / base_name).resolve()
    if cand2.exists():
        return str(cand2)
    return None


def copy_into_project_media(src_path: str, ws_dir: Optional[Path] = None) -> str:
    """Make a task's source image reachable by the MCP subprocesses.

    Delegates to ``workspace.materialize_input`` so the placement strategy is
    selected by ``M3_WORKSPACE_MODE``. The old unconditional copy-with-numbered-
    suffix behaviour is still available under ``legacy`` mode; the default
    ``dedup`` mode reuses an existing byte-identical copy so the returned path
    (and therefore the ``image_id`` the model sees) is the same on every run.
    """
    return ws_mod.materialize_input(src_path, ws_dir)


# Removed type_to_limits: max_step and max_concurrent are passed via CLI


async def run_single_task(host: MCPHost, model_driver, task: Dict[str, Any], top_tools: int, max_new_tokens: int, max_step: int, max_concurrent: int, image_base_dir: Optional[Path], fuzzy: bool, num_context_tools: int = 0, gt_tools: Optional[List[str]] = None, run_id: Optional[str] = None) -> Dict[str, Any]:

    question = (task.get("question") if fuzzy else task.get("prompt")) or ""
    image_rel = task.get("image") or ""
    task_dir = Path(task.get("__task_base_dir__", "."))
    base_for_images = image_base_dir if image_base_dir is not None else task_dir
    image_abs = resolve_image_path(base_for_images, image_rel)

    # Per-task workspace: a deterministic, freshly emptied directory so a rerun
    # of the same task sees exactly the same paths and none of the previous
    # attempt's (or any other task's) artifacts.
    ws_dir: Optional[Path] = None
    if ws_mod.workspace_mode() == ws_mod.MODE_ISOLATED:
        ws_dir = ws_mod.task_workspace(run_id or ws_mod.resolve_run_id(), str(task.get("id") or "task"))
    # Snapshot cwd artifacts so the post-task sweep can delete only what THIS
    # task created, instead of every .png in the repo root.
    cwd_before = ws_mod.snapshot_cwd_files()

    uploaded_manifest: List[Dict[str, str]] = []
    if image_abs:
        media_abs = copy_into_project_media(image_abs, ws_dir)
        uploaded_manifest.append({
            "image_id": Path(media_abs).name,
            "file_path": media_abs,
        })

    history: List[Dict[str, str]] = []
    if uploaded_manifest:
        hint = question + "\n\n[with images]\n" + "\n".join([f"- {u['image_id']}: file={u['file_path']}" for u in uploaded_manifest])
    else:
        hint = question
    history.append({"role": "user", "content": hint})

    # Use shared RoundRunner. ``ws_dir`` is passed only when the installed
    # RoundRunner accepts it, so this file stays importable against a runner
    # revision that predates the workspace work.
    runner_kwargs: Dict[str, Any] = dict(
        host=host, model_driver=model_driver, max_step=max_step,
        max_concurrent=max_concurrent, top_tools=top_tools,
        num_context_tools=num_context_tools, gt_tools=gt_tools,
    )
    if "ws_dir" in inspect.signature(RoundRunner.__init__).parameters:
        runner_kwargs["ws_dir"] = ws_dir
    runner = RoundRunner(**runner_kwargs)
    rr = await runner.run(history=history, last_user=hint, uploaded_file_paths=[u["file_path"] for u in uploaded_manifest])
    round_groups: List[List[Dict[str, Any]]] = rr.get("round_groups", [])
    dialogues: List[Dict[str, str]] = rr.get("dialogues", [])
    results_flat: List[Tuple[Dict[str, Any], str]] = rr.get("results_flat", [])

    summarize_system = textwrap.dedent("""
        You are finalizing the conversation. Produce ONLY the final answer in natural language.
        Do NOT include any <tool_call> tags or mention tools explicitly. Be concise and accurate,
        relying on the prior tool results contained in the conversation.
    """).strip()

    summary_messages = history + [
        {"role": "system", "content": summarize_system},
        {"role": "user", "content": "Now provide the final answer based on the above tool results."},
    ]
    final_visible, _ = await asyncio.get_event_loop().run_in_executor(
        None, lambda: model_driver.generate_once(summary_messages)
    )
    final_reply = strip_think(final_visible)

    result_steps: List[Dict[str, Any]] = []
    for idx, group in enumerate(round_groups, start=1):
        result_steps.append({
            "step": idx,
            "calls": group,
        })

    used_rounds = len(round_groups)
    used_concurrency = max((len(g) for g in round_groups), default=0)

    # Relocate/remove only the cwd artifacts this task produced. In isolated
    # mode they are moved into the task workspace so nothing is silently lost.
    if ws_mod.workspace_mode() != ws_mod.MODE_LEGACY:
        ws_mod.sweep_cwd_new_files(cwd_before, move_to=ws_dir)

    out: Dict[str, Any] = {
        "id": task.get("id"),
        "image": task.get("image"),
        "type": task.get("type"),
        "question": question,
        "steps": result_steps,
        "num_step": used_rounds,
        "max_num_concurrent": used_concurrency,
        "final_reply": final_reply,
        "dialogue": dialogues,
    }
    return out


def build_decision_messages(task: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are a strict task finisher. Read the provided interaction history and decide if the task is completed. \n"
        "Very important! If there is an incorrect causal order in multi-step MCP tools calls, or if the information returned by the MCP tools call is clearly wrong, but the call chain proceeds smoothly due to LLM hallucinations, it should be judged as an error.\n"
        "Output EXACTLY one of: success or fail. No other words." 
    )
    payload = {
        "id": result.get("id"),
        "question": result.get("question"),
        "final_reply": result.get("final_reply"),
        "round_groups": result.get("steps", []),
        "dialogue": result.get("dialogue", []),
        "results_flat": result.get("results_flat", []),
        "history": result.get("history", []),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_trajectory_messages(task: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You will summarize the optimal trajectory of useful MCP tool calls to solve the task.\n"
        "Rules:\n"
        "- Remove failed or unnecessary tool calls.\n"
        "- Group calls by round as steps; within a step, multiple calls are in parallel (array order arbitrary).\n"
        "- Operations that can invoke the MCP tools call in parallel were mistakenly executed across multiple steps — they should be merged into a single step.\n"
        "- Preserve only name and arguments for each call.\n"
        "- Output STRICT JSON only. No markdown, no comments, no extra text.\n"
        "- If you cannot determine a valid trajectory, output exactly the string fail (no JSON).\n"
    )
    skeleton = {
        "id": result.get("id"),
        "image": task.get("image"),
        "type": task.get("type"),
        "question": result.get("question"),
        "steps": [
            {"step": 1, "calls": [{"name": "server/tool", "arguments": {}}]}
        ],
    }
    context = {
        "question": result.get("question"),
        "final_reply": result.get("final_reply"),
        "round_groups": result.get("steps", []),
        "dialogue": result.get("dialogue", []),
    }
    user_prompt = (
        "Based on the following context, produce the optimal trajectory JSON with the exact keys as in the skeleton.\n"
        "Context:\n" + json.dumps(context, ensure_ascii=False) + "\n"
        "Skeleton (use these keys; fill with actual content):\n" + json.dumps(skeleton, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]


def iter_json_files(root: Path) -> List[Path]:
    """
    Find all .json files in the root directory, including its subdirectories.
    root.glob("*.json")
    root.rglob("*.json")
    """
    files: List[Path] = []
    for p in root.glob("*.json"):
        if p.is_file():
            files.append(p)
    return sorted(files)


def load_tasks_from_file(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    for obj in data:
        if isinstance(obj, dict):
            obj["__task_base_dir__"] = str(path.parent.resolve())
    return data


async def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--MODEL_PATH", required=True)
    parser.add_argument("--TOP_TOOLS", type=int, default=4)
    parser.add_argument("--max_step", type=int, default=4)
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--num_client", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--image_dir", required=True, help="Directory containing images referenced by tasks")
    parser.add_argument("--annotation_dir", required=True, help="Directory containing annotation JSON files, or a single .json file")
    parser.add_argument("--JUDGE_MODEL", required=False)
    parser.add_argument("--OUTPUT_DIR", required=False, help="Directory to write output JSON to, or a filename ending with .json to be created under ./results")
    parser.add_argument("--fuzzy", action="store_true", help="If set, read 'question' instead of 'prompt' from tasks")
    parser.add_argument("--num_context_tools", type=int, default=0, help="Number of context tools to show (GT + random)")
    parser.add_argument("--gt_file", required=False, help="Path to GT json file for tool selection")
    args, _ = parser.parse_known_args()

    model_path: str = args.MODEL_PATH
    top_tools: int = args.TOP_TOOLS
    max_new_tokens: int = args.max_new_tokens
    image_dir = Path(args.image_dir).resolve()
    annotation_dir_arg: str = args.annotation_dir
    judge_model: Optional[str] = args.JUDGE_MODEL
    max_step: int = args.max_step
    max_concurrent: int = args.max_concurrent
    num_client: int = max(1, int(args.num_client))
    num_context_tools: int = args.num_context_tools
    gt_file_arg: Optional[str] = args.gt_file

    # Load GT map if provided
    gt_map: Dict[str, List[str]] = {}
    if gt_file_arg:
        gt_path = Path(gt_file_arg).resolve()
        if gt_path.exists():
            try:
                data = json.loads(gt_path.read_text(encoding="utf-8"))
                for t in data:
                    tid = t.get("id")
                    tools = set()
                    if "steps" in t:
                        for step in t["steps"]:
                            if "calls" in step:
                                for call in step["calls"]:
                                    if "name" in call:
                                        tools.add(call["name"])
                    if tid:
                        gt_map[tid] = list(tools)
            except Exception as e:
                print(f"[WARN] Failed to load GT file {gt_path}: {e}")

    # Validate judge requirement based on fuzzy flag
    if not args.fuzzy and not judge_model:
        raise SystemExit("--JUDGE_MODEL is required unless --fuzzy is set")

    model_driver = create_model_driver(model_path, max_new_tokens=max_new_tokens)
    judge_driver = create_model_driver(judge_model, max_new_tokens=max_new_tokens) if not args.fuzzy else None

    def extract_model_name(mp: str) -> str:
        # Robustly derive a filename-safe model name
        try:
            name = Path(mp).name
        except Exception:
            name = str(mp)
        return name or "model"

    def ensure_parent_dir(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def compute_output_file(output_dir_arg: Optional[str]) -> Path:
        model_name = extract_model_name(model_path)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{model_name}_{date_str}.json"

        if output_dir_arg:
            if output_dir_arg.lower().endswith(".json") :
                # Use only the name, store under ./results
                target = Path("./results") / Path(output_dir_arg).name
                ensure_parent_dir(target)
                return target
            # Treat as a directory; place default-named file inside
            target_dir = Path(output_dir_arg)
            # Always allow existing directories; create if not exists
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir / default_filename
        # Not provided: place default-named file under ./results
        target = Path("./results") / default_filename
        ensure_parent_dir(target)
        return target

    def derive_success_fail_files(base_file: Path) -> Tuple[Path, Path]:
        base_dir = base_file.parent
        success = base_dir / f"{base_file.stem}_success{base_file.suffix}"
        fail = base_dir / f"{base_file.stem}_fail{base_file.suffix}"
        return success, fail

    def atomic_write_json(path: Path, data: List[Dict[str, Any]]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        ensure_parent_dir(path)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp_path), str(path))

    def load_existing_results(path: Path) -> Dict[Any, Dict[str, Any]]:
        """
        If output file already exists and contains a JSON list, load it and
        return a mapping from task id -> result object. Otherwise return empty.
        """
        id_to_result: Dict[Any, Dict[str, Any]] = {}
        try:
            if path.is_file() and path.suffix.lower() == ".json" and path.exists():
                content = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(content, list):
                    for obj in content:
                        if isinstance(obj, dict) and "id" in obj:
                            id_to_result[obj["id"]] = obj
        except Exception:
            # Ignore corrupted/partial file and treat as empty
            return {}
        return id_to_result

    run_id = ws_mod.resolve_run_id(extract_model_name(model_path))
    print(f"[workspace] {ws_mod.describe()} run_id={run_id}")

    # Cancelling a coroutine that awaits run_in_executor does NOT interrupt the
    # underlying thread, so every timed-out LLM call leaks one worker. The default
    # pool caps at min(32, cpu+4); once saturated, later run_in_executor calls
    # never get scheduled and the event loop stalls with no await to blame.
    # Sizing the pool to the worker count plus headroom keeps a leak from
    # starving the run. Set M3_EXECUTOR_WORKERS=0 to keep the default pool.
    _ex_workers = _executor_workers(num_client)
    if _ex_workers > 0:
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=_ex_workers, thread_name_prefix="m3-llm")
        )
        print(f"[executor] default ThreadPoolExecutor max_workers={_ex_workers}")

    host = MCPHost(Path("mcp_servers.json"))
    await host.start()
    try:
        # Prepare tasks (gather all first)
        annotation_path = Path(annotation_dir_arg).resolve()
        if annotation_path.is_file() and annotation_path.suffix.lower() == ".json":
            json_files = [annotation_path]
        else:
            json_files = iter_json_files(annotation_path)
        all_tasks: List[Dict[str, Any]] = []
        for jf in json_files:
            all_tasks.extend(load_tasks_from_file(jf))

        # Prepare output file and load existing results if any
        output_file = compute_output_file(args.OUTPUT_DIR)
        existing_id_to_result = load_existing_results(output_file)
        completed_ids = set(existing_id_to_result.keys())
        if not args.fuzzy:
            success_file, fail_file = derive_success_fail_files(output_file)
            # Seed success summaries from existing success file to avoid overwriting
            existing_success_id_to_summary = load_existing_results(success_file)
            # Always regenerate fail file at the start of each run (empty list)
            try:
                atomic_write_json(fail_file, [])
            except Exception:
                pass
        else:
            success_file, fail_file = None, None
            existing_success_id_to_summary = {}

        # Filter tasks to run
        tasks_to_run = [t for t in all_tasks if t.get("id") not in completed_ids]

        # Maintain mapping of id -> result; seed with existing results
        id_to_result: Dict[Any, Dict[str, Any]] = dict(existing_id_to_result)

        # Process tasks using a queue and worker clients; each worker retries once on failure,
        # then requeues the task for another client to take over.
        pbar = tqdm.tqdm(total=len(tasks_to_run), desc="Processing tasks")
        success_id_to_summary: Dict[Any, Dict[str, Any]] = dict(existing_success_id_to_summary)
        fail_id_to_result: Dict[Any, Dict[str, Any]] = {}
        write_lock = asyncio.Lock()

        task_queue: asyncio.Queue = asyncio.Queue()
        for t in tasks_to_run:
            task_queue.put_nowait(t)

        # A task that always fails used to be requeued forever (the requeue reset
        # ``attempt`` to 0), so ``task_queue.join()`` could never return and the
        # whole batch hung on it. Cap the total number of times any single task
        # may be handed back to the queue.
        requeue_counts: Dict[Any, int] = {}
        max_requeue = _max_task_requeue()
        task_budget = _task_timeout()

        async def _record_task_error(task: Dict[str, Any], exc: BaseException) -> None:
            """Write a minimal error record for a task that exhausted its budget."""
            tid = task.get("id")
            rec: Dict[str, Any] = {
                "id": tid,
                "image": task.get("image"),
                "type": task.get("type"),
                "question": (task.get("question") if args.fuzzy else task.get("prompt")) or "",
                "steps": [],
                "num_step": 0,
                "max_num_concurrent": 0,
                "final_reply": "",
                "dialogue": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[ERROR] task {tid} gave up after retries: {rec['error']}")
            async with write_lock:
                if tid is not None:
                    id_to_result[tid] = rec
                    if fail_file is not None:
                        fail_id_to_result[tid] = rec
                        atomic_write_json(fail_file, list(fail_id_to_result.values()))
                atomic_write_json(output_file, list(id_to_result.values()))
                pbar.update(1)

        async def worker(worker_idx: int) -> None:
            while True:
                task: Dict[str, Any] = await task_queue.get()
                try:
                    attempt = 0
                    while True:
                        try:
                            # Retrieve GT tools for this task
                            task_gt = gt_map.get(task.get("id"), [])
                            
                            _coro = run_single_task(
                                host,
                                model_driver,
                                task,
                                top_tools=top_tools,
                                max_new_tokens=max_new_tokens,
                                max_step=max_step,
                                max_concurrent=max_concurrent,
                                image_base_dir=image_dir,
                                fuzzy=bool(args.fuzzy),
                                num_context_tools=num_context_tools,
                                gt_tools=task_gt,
                                run_id=run_id,
                            )
                            # Outermost guard: the per-call LLM/tool timeouts bound
                            # one request each, but a task makes O(max_step) of them,
                            # so their sum is unbounded. This caps the whole task.
                            if task_budget > 0:
                                result = await asyncio.wait_for(_coro, timeout=task_budget)
                            else:
                                result = await _coro
                            task_id = result.get("id")

                            # Serialize updates and file writes
                            async with write_lock:
                                if task_id is not None:
                                    id_to_result[task_id] = result
                                atomic_write_json(output_file, list(id_to_result.values()))

                                if not args.fuzzy:
                                    # JUDGE_MODEL decision: success or fail
                                    try:
                                        decision_msgs = build_decision_messages(task, result)
                                        decision_visible, _ = await asyncio.get_event_loop().run_in_executor(
                                            None, lambda: judge_driver.generate_once(decision_msgs)
                                        )
                                        decision_text = strip_think(decision_visible).strip().lower()
                                        print(decision_text)
                                    except Exception:
                                        decision_text = "fail"

                                    if decision_text == "success":
                                        try:
                                            traj_msgs = build_trajectory_messages(task, result)
                                            traj_visible, _ = await asyncio.get_event_loop().run_in_executor(
                                                None, lambda: judge_driver.generate_once(traj_msgs)
                                            )
                                            traj_text = strip_think(traj_visible).strip()
                                            summary_obj = json.loads(traj_text)
                                            if isinstance(summary_obj, dict):
                                                summary_obj["id"] = result.get("id")
                                                summary_obj["image"] = task.get("image")
                                                summary_obj["type"] = task.get("type")
                                                summary_obj["question"] = result.get("question")
                                                summary_obj["final_reply"] = result.get("final_reply")
                                                success_id_to_summary[task_id] = summary_obj
                                                if success_file is not None:
                                                    atomic_write_json(success_file, list(success_id_to_summary.values()))
                                            else:
                                                fail_id_to_result[task_id] = result
                                                if fail_file is not None:
                                                    atomic_write_json(fail_file, list(fail_id_to_result.values()))
                                        except Exception:
                                            fail_id_to_result[task_id] = result
                                            if fail_file is not None:
                                                atomic_write_json(fail_file, list(fail_id_to_result.values()))
                                    else:
                                        fail_id_to_result[task_id] = result
                                        if fail_file is not None:
                                            atomic_write_json(fail_file, list(fail_id_to_result.values()))

                                pbar.update(1)
                                # Legacy cleanup: wipe every .png in cwd. This is
                                # destructive across tasks and workers (it deleted
                                # git-tracked figures), so outside legacy mode the
                                # sweep is scoped per task inside run_single_task.
                                if ws_mod.workspace_mode() == ws_mod.MODE_LEGACY:
                                    try:
                                        for p in Path(".").iterdir():
                                            if (
                                                p.is_file()
                                                and p.suffix.lower() == ".png"
                                                and p.name not in ws_mod._CWD_PROTECTED
                                            ):
                                                try:
                                                    p.unlink()
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass
                            break  # success
                        except Exception as exc:
                            attempt += 1
                            if attempt <= 1:
                                await asyncio.sleep(2.0)
                                continue
                            tid = task.get("id")
                            seen = requeue_counts.get(tid, 0)
                            if seen < max_requeue:
                                requeue_counts[tid] = seen + 1
                                task_queue.put_nowait(task)
                                break
                            # Budget exhausted: persist an error record so the task
                            # is accounted for instead of silently stalling the run.
                            await _record_task_error(task, exc)
                            break
                finally:
                    task_queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(num_client)]
        await task_queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # Do not clear media directory contents anymore
    finally:
        await host.stop()


if __name__ == "__main__":
    asyncio.run(main())



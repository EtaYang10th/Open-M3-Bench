#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import create_model_driver

import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import re
from multiprocessing import get_context


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def index_by_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for obj in items:
        if not isinstance(obj, dict):
            continue
        k = str(obj.get("id", ""))
        if k:
            idx[k] = obj
    return idx


def build_completion_messages(question: str, pred_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are a strict and professional academic reviewer tasked with evaluating task completion when a model uses MCP tools.\n"
        "Judge solely based on the visible prediction content; do not use or allude to any ground truth.\n"
        "Scoring rule: output a single scalar in the form \\boxed{S}, and nothing else. S is a float in [0,10].\n"
        "Rubric:\n"
        "1) Planning (0-3): Clearly presents the MCP call process and plan; complete and organized.\n"
        "2) Process (0-3): MCP calls obtain desirable tool feedback and intermediate results; tool use is effective.\n"
        "3) Final result (0-4): Degree of task completion (partial credit allowed); final output addresses the task.\n"
        "Please provide an overall score from 0 to 10 combining the three parts, and output only \\boxed{S}."
    )
    payload = {
        "question": question,
        "prediction": {
            "id": pred_obj.get("id"),
            "steps": pred_obj.get("steps"),
            "final_reply": pred_obj.get("final_reply"),
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_information_grounding_messages(question: str, gt_obj: Dict[str, Any], pred_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are a strict and professional academic reviewer responsible for evaluating information grounding.\n"
        "Task: compare the reference steps (ground truth steps) with the predicted steps and determine whether the prediction includes all key steps from the reference.\n"
        "Rules:\n"
        "- Coverage-only criterion: equivalence/paraphrase/minor order changes count as covered;\n"
        "- Extra steps are not penalized;\n"
        "- Missing or clearly deviating key steps are penalized linearly by the proportion of missing steps relative to the total reference steps;\n"
        "Output format: output only a scalar in the form \\boxed{G}, where G is a float in [0,1]."
    )
    payload = {
        "question": question,
        "reference": {
            "id": gt_obj.get("id"),
            "steps": gt_obj.get("steps"),
        },
        "prediction": {
            "id": pred_obj.get("id"),
            "steps": pred_obj.get("steps"),
            "final_reply": pred_obj.get("final_reply"),
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_boxed_float(text: Optional[str], min_value: float, max_value: float) -> float:
    if text is None:
        raise ValueError("Judge returned None; expected \\boxed{number}.")
    s = text.strip()
    m = re.search(r"\\boxed\{\s*([^}]+)\s*\}", s)
    if m:
        raw = m.group(1).strip()
    else:
        # Fallback: accept a bare number if not wrapped in \boxed{}
        m_num = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m_num:
            raise ValueError(f"Invalid judge output: {repr(s)}; expected \\boxed{{number}} or a bare number.")
        raw = m_num.group(0).strip()
    try:
        val = float(raw)
    except Exception:
        raise ValueError(f"Invalid number in boxed: {repr(raw)}")
    if not (min_value <= val <= max_value):
        raise ValueError(f"Number {val} out of range [{min_value}, {max_value}].")
    return val


# =============== Multiprocessing scheduling (each process holds its own drivers) ===============
_WORKER_STATE: Dict[str, Any] = {
    "drivers": None,
    "judge_specs": None,
}


def _init_worker(judge_specs: List[str], max_new_tokens: int) -> None:
    # Worker process init: create independent drivers to avoid cross-process shared state
    _WORKER_STATE["judge_specs"] = judge_specs
    _WORKER_STATE["drivers"] = [create_model_driver(spec, max_new_tokens=max_new_tokens) for spec in judge_specs]


def _judge_once_float_worker(model_idx: int, messages: List[Dict[str, str]], *, min_value: float, max_value: float) -> float:
    drivers = _WORKER_STATE.get("drivers") or []
    judge_specs = _WORKER_STATE.get("judge_specs") or []
    max_attempts = 5
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        visible, _ = drivers[model_idx].generate_once(messages)
        try:
            return parse_boxed_float(visible, min_value, max_value)
        except ValueError as e:
            last_error = e
            print(
                f"[WARN][worker] judge={judge_specs[model_idx]} attempt {attempt} returned unparseable output; expected only \\boxed{{number}}. raw_output={repr(visible)}",
                flush=True,
            )
            if attempt < max_attempts:
                print("[INFO][worker] Retrying in 10 seconds...", flush=True)
                time.sleep(10)
    raise ValueError(f"Judge {judge_specs[model_idx]} returned unparseable outputs multiple times; last error: {last_error}")


def _process_one_task(task: Dict[str, Any]) -> Dict[str, Any]:
    sid: str = task["sid"]
    question: str = task["question"]
    gt: Dict[str, Any] = task["gt"]
    pred: Dict[str, Any] = task["pred"]

    judge_specs: List[str] = _WORKER_STATE.get("judge_specs") or []
    num_models: int = len(judge_specs)

    # completion (0-10 -> normalized to 0-1)
    comp_messages = build_completion_messages(question, pred)
    try:
        with ThreadPoolExecutor(max_workers=num_models) as ex:
            comp_futs = [
                ex.submit(_judge_once_float_worker, mi, comp_messages, min_value=0.0, max_value=10.0)
                for mi in range(num_models)
            ]
            comp_raw = [float(f.result()) for f in comp_futs]
    except Exception as e:
        return {"sid": sid, "error": f"completion_failed: {e.__class__.__name__}: {e}"}
    comp_norm = [s / 10.0 for s in comp_raw]
    comp_sorted = sorted(comp_norm)
    comp_trimmed = comp_sorted[1:-1] if len(comp_sorted) > 2 else comp_sorted
    comp_final = sum(comp_trimmed) / max(1, len(comp_trimmed))

    # information grounding（0-1）
    ig_messages = build_information_grounding_messages(question, gt, pred)
    try:
        with ThreadPoolExecutor(max_workers=num_models) as ex:
            ig_futs = [
                ex.submit(_judge_once_float_worker, mi, ig_messages, min_value=0.0, max_value=1.0)
                for mi in range(num_models)
            ]
            ig_scores = [float(f.result()) for f in ig_futs]
    except Exception as e:
        return {"sid": sid, "error": f"info_grounding_failed: {e.__class__.__name__}: {e}"}
    ig_sorted = sorted(ig_scores)
    ig_trimmed = ig_sorted[1:-1] if len(ig_sorted) > 2 else ig_sorted
    ig_final = sum(ig_trimmed) / max(1, len(ig_trimmed))

    detail = {
        "final": float(comp_final),
        "judges": [
            {"name": judge_specs[i], "score": float(comp_norm[i])} for i in range(num_models)
        ],
        "info_grounding": {
            "final": float(ig_final),
            "judges": [
                {"name": judge_specs[i], "score": float(ig_scores[i])} for i in range(num_models)
            ],
        },
    }

    return {
        "sid": sid,
        "detail": detail,
        "comp_norm": [float(x) for x in comp_norm],
        "ig_scores": [float(x) for x in ig_scores],
        "comp_final": float(comp_final),
        "ig_final": float(ig_final),
        "judge_specs": judge_specs,
    }

def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON file")
    parser.add_argument("--judge-models", dest="judge_models", default=None, help="Comma-separated model names for parallel judging (>=3)")
    parser.add_argument("--out", dest="out_path", default=None, help="Output JSON path for taskcompletion.json")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--num_client", type=int, default=1, help="Number of worker processes for multiprocessing")
    args = parser.parse_args()

    gt_path = Path(args.gt).resolve()
    pred_path = Path(args.pred).resolve()
    gt_items = load_json_array(gt_path)
    pred_items = load_json_array(pred_path)
    pred_index = index_by_id(pred_items)

    # Three-model parallel judging and output to --out (required)
    if not args.judge_models:
        raise SystemExit("You must provide --judge-models as a comma-separated list of at least 3 models")
    if not args.out_path:
        raise SystemExit("You must specify --out output file path, e.g., results/<exp>/taskcompletion.json")

    judge_specs = [m.strip() for m in str(args.judge_models).split(",") if m.strip()]
    if len(judge_specs) < 3:
        raise SystemExit("--judge-models requires at least 3 models (comma-separated)")
    out_path = Path(args.out_path).resolve()

    # Load existing results, skip completed ids (prefer pre_id, compatible with per_id)
    # existing_per_id_scores is used for skipping logic (stores final score float[0,1])
    # existing_pre_id_detail stores full structure (final and judges and info_grounding)
    existing_per_id_scores: Dict[str, float] = {}
    existing_pre_id_detail: Dict[str, Any] = {}
    if out_path.exists():
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                m = old.get("pre_id") or old.get("per_id") or {}
                if isinstance(m, dict):
                    for k, v in m.items():
                        sid = str(k)
                        # Legacy format: raw 0/1 or any numeric (treated as completion [0,1])
                        if isinstance(v, (int, float)) or v in (0, 1, "0", "1"):
                            try:
                                final_score = float(v)
                            except Exception:
                                continue
                            existing_per_id_scores[sid] = final_score
                            existing_pre_id_detail[sid] = {"final": final_score, "judges": []}
                        # New format: contains final and judges (may be float)
                        elif isinstance(v, dict):
                            final_v = v.get("final")
                            try:
                                final_score = float(final_v)
                            except Exception:
                                continue
                            existing_per_id_scores[sid] = final_score
                            judges = v.get("judges") if isinstance(v.get("judges"), list) else []
                            entry: Dict[str, Any] = {"final": final_score, "judges": judges}
                            info_g = v.get("info_grounding")
                            if isinstance(info_g, dict):
                                entry["info_grounding"] = info_g
                            existing_pre_id_detail[sid] = entry
        except Exception:
            pass

    # Number of models (per-sample still judged in parallel)
    num_models = len(judge_specs)

    # Full list of GT ids
    gt_ids: List[str] = []
    for gt in gt_items:
        if isinstance(gt, dict):
            sid = str(gt.get("id", ""))
            if sid:
                gt_ids.append(sid)

    # Incremental processing: skip existing ids; filter out samples missing in pred
    todo_ids = [sid for sid in gt_ids if (sid not in existing_per_id_scores) and (sid in pred_index)]

    # Build id → gt map for quick access
    gt_index: Dict[str, Dict[str, Any]] = {str(x.get("id")): x for x in gt_items if isinstance(x, dict) and x.get("id")}

    # Result details container (preserving existing)
    per_id_detail: Dict[str, Any] = dict(existing_pre_id_detail)

    # Task prep: pass only necessary fields to workers
    tasks: List[Dict[str, Any]] = []
    for sid in todo_ids:
        gt = gt_index.get(sid)
        pred = pred_index.get(sid)
        if (gt is None) or (pred is None):
            continue
        question = str(gt.get("question", "") or "")
        tasks.append({
            "sid": sid,
            "question": question,
            "gt": gt,
            "pred": pred,
        })

    def _write_checkpoint() -> None:
        num_scored = len(per_id_detail)
        sum_scored = sum(float(v.get("final", 0.0)) for v in per_id_detail.values())
        ig_vals = [float(v.get("info_grounding", {}).get("final", 0.0)) for v in per_id_detail.values() if isinstance(v.get("info_grounding"), dict)]
        ig_rate = (sum(ig_vals) / len(ig_vals)) if ig_vals else 0.0
        aggregate = {
            "num_total": len(gt_ids),
            "num_scored": num_scored,
            "completion_rate": (float(sum_scored) / float(num_scored)) if num_scored else 0.0,
            "information_grounding_rate": ig_rate,
        }
        out_obj = {"aggregate": aggregate, "pre_id": per_id_detail}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(tasks) == 0:
        # Nothing to process; write once to update aggregate (preserve behavior)
        _write_checkpoint()
    elif args.num_client <= 1:
        # Single-process path: synchronous processing
        for task in tqdm.tqdm(tasks):
            res = _process_one_task(task)
            sid = res["sid"]
            if "error" in res:
                print(f"[SKIP] id={sid} judging failed: {res['error']}")
                continue
            per_id_detail[sid] = res["detail"]
            _write_checkpoint()
            judge_specs_local = res["judge_specs"]
            comp_norm = res["comp_norm"]
            ig_scores = res["ig_scores"]
            judge_detail_comp = ", ".join([f"{judge_specs_local[i]}:{comp_norm[i]:.3f}" for i in range(len(judge_specs_local))])
            judge_detail_ig = ", ".join([f"{judge_specs_local[i]}:{ig_scores[i]:.3f}" for i in range(len(judge_specs_local))])
            print(f"completion_judges= {judge_detail_comp}")
            print(f"information_grounding_judges= {judge_detail_ig}")
            print(f"id={sid}\tcompletion={res['comp_final']:.3f}\tinfo_grounding={res['ig_final']:.3f}")
    else:
        # Multiprocess: main process writes files; workers compute only
        ctx = get_context("fork")
        with ctx.Pool(processes=args.num_client, initializer=_init_worker, initargs=(judge_specs, args.max_new_tokens)) as pool:
            for res in tqdm.tqdm(pool.imap_unordered(_process_one_task, tasks), total=len(tasks)):
                sid = res["sid"]
                if "error" in res:
                print(f"[SKIP] id={sid} judging failed: {res['error']}")
                    continue
                per_id_detail[sid] = res["detail"]
                _write_checkpoint()
                judge_specs_local = res["judge_specs"]
                comp_norm = res["comp_norm"]
                ig_scores = res["ig_scores"]
                judge_detail_comp = ", ".join([f"{judge_specs_local[i]}:{comp_norm[i]:.3f}" for i in range(len(judge_specs_local))])
                judge_detail_ig = ", ".join([f"{judge_specs_local[i]}:{ig_scores[i]:.3f}" for i in range(len(judge_specs_local))])
                print(f"completion_judges= {judge_detail_comp}")
                print(f"information_grounding_judges= {judge_detail_ig}")
                print(f"id={sid}\tcompletion={res['comp_final']:.3f}\tinfo_grounding={res['ig_final']:.3f}")

    # Print summary at the end
    final_num_scored = len(per_id_detail)
    final_sum_scored = sum(float(v.get("final", 0.0)) for v in per_id_detail.values())
    final_completion_rate = (final_sum_scored / max(1, final_num_scored))
    ig_vals_final = [float(v.get("info_grounding", {}).get("final", 0.0)) for v in per_id_detail.values() if isinstance(v.get("info_grounding"), dict)]
    final_ig_rate = (sum(ig_vals_final) / len(ig_vals_final)) if ig_vals_final else 0.0
    print(json.dumps({
        "wrote": str(out_path),
        "num_total": len(gt_ids),
        "num_scored": final_num_scored,
        "completion_rate": final_completion_rate,
        "information_grounding_rate": final_ig_rate
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()



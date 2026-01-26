#!/usr/bin/env python3
"""
Evaluate MCP trajectories with a focus on Vision Specific Technical Issues.

Metrics:
- Perception Recall: Measure if OCR/Detection tools are correctly triggered.
- Grounding Similarity: Measure argument similarity for Grounding tools.
- Visual Reasoning Ability: Identify if vision steps (Perception/Grounding) perform better than or equal to Planning.

Usage:
  python evaluate_cv_issues.py \
    --gt /path/to/gt.json \
    --pred /path/to/pred.json \
    --out-json results_cv_eval.json \
    --out-csv results_cv_eval.csv \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --tau-strong 0.80 \
    --tau-weak 0.60 \
    --cache-path .emb_cache.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    SentenceTransformer = None

# ----------------------------- Data structures -----------------------------

@dataclass
class Call:
    tool_name: str
    arguments: Any
    step_index: int
    call_index_in_step: int

@dataclass
class Step:
    step_index: int
    calls: List[Call]

@dataclass
class Trajectory:
    task_id: str
    steps: List[Step]

@dataclass
class MatchPair:
    gt_idx: int
    pred_idx: int
    similarity: float
    gt_step_idx: int
    pred_step_idx: int

# ----------------------------- Tool Categorization -----------------------------

PERCEPTION_TOOLS = {
    "ocr/perform_ocr",
    "dinox-mcp/detect-all-objects",
    "dinox-mcp/detect-objects-by-text",
    "pyzbar-mcp/scan_barcode",
}

GROUNDING_TOOLS = {
    "amazon/search_products",
    "google-maps/places_text_search",
    "wiki/search",
}

PERCEPTION_PREFIXES = ["imagesorcery-mcp/"]
GROUNDING_PREFIXES = ["car-price/", "food_nutrition_mcp/"]

def is_perception_tool(tool_name: str) -> bool:
    if tool_name in PERCEPTION_TOOLS:
        return True
    for prefix in PERCEPTION_PREFIXES:
        if tool_name.startswith(prefix):
            return True
    return False

def is_grounding_tool(tool_name: str) -> bool:
    if tool_name in GROUNDING_TOOLS:
        return True
    for prefix in GROUNDING_PREFIXES:
        if tool_name.startswith(prefix):
            return True
    return False

def is_vision_tool(tool_name: str) -> bool:
    return is_perception_tool(tool_name) or is_grounding_tool(tool_name)

# ----------------------------- Utilities -----------------------------

def read_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_tasks(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        tasks: List[Dict[str, Any]] = []
        for k, v in obj.items():
            if isinstance(v, dict):
                if "id" not in v:
                    v = {**v, "id": k}
                tasks.append(v)
        return tasks
    raise ValueError("Unsupported JSON root; expected list or dict of tasks")

def to_trajectory(task: Dict[str, Any]) -> Trajectory:
    task_id = str(task.get("id", ""))
    raw_steps = task.get("steps", []) or []
    steps: List[Step] = []
    for s_idx, step_obj in enumerate(raw_steps):
        raw_calls = step_obj.get("calls", []) or []
        calls: List[Call] = []
        for c_idx, call in enumerate(raw_calls):
            name = str(call.get("name", "")).strip()
            args = call.get("arguments", None)
            calls.append(Call(tool_name=name, arguments=args, step_index=s_idx, call_index_in_step=c_idx))
        steps.append(Step(step_index=s_idx, calls=calls))
    return Trajectory(task_id=task_id, steps=steps)

def flatten_arguments_to_text(arguments: Any) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(arguments)

class EmbeddingEncoder:
    def __init__(self, model_name: str, cache_path: Optional[str] = None) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed.")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if isinstance(cached, dict):
                        for k, v in cached.items():
                            if isinstance(k, str) and isinstance(v, list):
                                self.cache[k] = v
            except Exception:
                pass

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        indices: List[int] = []
        miss_texts: List[str] = []
        embeddings: List[Optional[np.ndarray]] = [None] * len(texts)
        for i, t in enumerate(texts):
            cached = self.cache.get(t)
            if cached is not None:
                embeddings[i] = np.asarray(cached, dtype=np.float32)
            else:
                indices.append(i)
                miss_texts.append(t)

        if miss_texts:
            new_embs = self.model.encode(miss_texts, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=False)
            for idx, emb in zip(indices, new_embs):
                arr = np.asarray(emb, dtype=np.float32)
                embeddings[idx] = arr
                self.cache[texts[idx]] = arr.tolist()

        if not embeddings:
            return np.array([])
            
        final = np.vstack([e for e in embeddings if e is not None])
        return final

    def persist_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except Exception:
            pass

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.clip(a_norm @ b_norm.T, -1.0, 1.0)

# ----------------------------- Matching logic -----------------------------

def extract_calls(trajectory: Trajectory) -> List[Call]:
    calls: List[Call] = []
    for step in trajectory.steps:
        calls.extend(step.calls)
    return calls

def bucket_indices_by_tool_name(calls: List[Call]) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = {}
    for idx, c in enumerate(calls):
        buckets.setdefault(c.tool_name, []).append(idx)
    return buckets

def hungarian_bucketed_match(
    gt_calls: List[Call],
    pred_calls: List[Call],
    sims: np.ndarray,
    tau_weak: float,
) -> Tuple[List[MatchPair], List[int], List[int]]:
    matched_pairs: List[MatchPair] = []
    used_pred: Set[int] = set()
    used_gt: Set[int] = set()

    gt_buckets = bucket_indices_by_tool_name(gt_calls)
    pred_buckets = bucket_indices_by_tool_name(pred_calls)

    for tool_name, gt_indices in gt_buckets.items():
        pred_indices = pred_buckets.get(tool_name, [])
        if not pred_indices:
            continue
        sub = sims[np.array(gt_indices)[:, None], np.array(pred_indices)[None, :]]
        large_cost = 1e6
        cost = 1.0 - sub
        mask_low = sub < tau_weak
        cost = cost.copy()
        cost[mask_low] = large_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            gt_global = gt_indices[r]
            pred_global = pred_indices[c]
            if gt_global in used_gt or pred_global in used_pred:
                continue
            sim = float(sub[r, c])
            if sim < tau_weak:
                continue
            matched_pairs.append(
                MatchPair(
                    gt_idx=gt_global,
                    pred_idx=pred_global,
                    similarity=sim,
                    gt_step_idx=gt_calls[gt_global].step_index,
                    pred_step_idx=pred_calls[pred_global].step_index,
                )
            )
            used_gt.add(gt_global)
            used_pred.add(pred_global)

    unmatched_gt = [i for i in range(len(gt_calls)) if i not in used_gt]
    unmatched_pred = [j for j in range(len(pred_calls)) if j not in used_pred]
    return matched_pairs, unmatched_gt, unmatched_pred

def safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0

# ----------------------------- New Metrics -----------------------------

def compute_cv_metrics(
    gt_calls: List[Call],
    matched_pairs: List[MatchPair],
    unmatched_gt: List[int],
) -> Dict[str, Any]:
    
    # 1. Perception Recall
    perception_gt_indices = [i for i, c in enumerate(gt_calls) if is_perception_tool(c.tool_name)]
    total_perception_gt = len(perception_gt_indices)
    matched_perception_gt_indices = [p.gt_idx for p in matched_pairs if p.gt_idx in perception_gt_indices]
    matched_perception_count = len(matched_perception_gt_indices)
    
    perception_recall = safe_div(matched_perception_count, total_perception_gt) if total_perception_gt > 0 else 1.0 # Default to 1 if no perception tools needed

    # 2. Grounding Similarity
    grounding_gt_indices = [i for i, c in enumerate(gt_calls) if is_grounding_tool(c.tool_name)]
    total_grounding_gt = len(grounding_gt_indices)
    
    grounding_matches = [p for p in matched_pairs if p.gt_idx in grounding_gt_indices]
    
    if total_grounding_gt > 0:
        if grounding_matches:
            grounding_sim_sum = sum(p.similarity for p in grounding_matches)
            grounding_arg_sim = grounding_sim_sum / total_grounding_gt
        else:
            grounding_arg_sim = 0.0
    else:
        grounding_arg_sim = 1.0 # Not applicable, so perfect score? Or None? 1.0 seems safer for aggregation.

    # 3. Visual Reasoning Ability
    # "If these steps [Perception/Grounding] similarity < threshold (or planning similarity), or missing -> bottleneck"
    # Definition:
    #   For each task, we look at Vision steps (Perception + Grounding).
    #   We also look at Planning steps (everything else).
    #   Visual Reasoning Ability = 1 if (Vision Performance >= Planning Performance) and (Vision Present), else 0.
    
    # Let's define "Performance" as average similarity (with 0 for missing).
    
    vision_gt_indices = perception_gt_indices + grounding_gt_indices
    planning_gt_indices = [i for i in range(len(gt_calls)) if i not in vision_gt_indices]
    
    # Calculate Vision Score
    visual_reasoning_ability = None
    if not vision_gt_indices:
        visual_reasoning_ability = 1.0 # No vision steps, so no bottleneck (pass)
    else:
        vision_scores = []
        for idx in vision_gt_indices:
            # Find match
            match = next((p for p in matched_pairs if p.gt_idx == idx), None)
            if match:
                vision_scores.append(match.similarity)
            else:
                vision_scores.append(0.0) # Missing = 0
        avg_vision_score = sum(vision_scores) / len(vision_scores)
        
        # Calculate Planning Score
        if not planning_gt_indices:
             visual_reasoning_ability = None # No planning steps to compare against -> N/A
        else:
            planning_scores = []
            for idx in planning_gt_indices:
                match = next((p for p in matched_pairs if p.gt_idx == idx), None)
                if match:
                    planning_scores.append(match.similarity)
                else:
                    planning_scores.append(0.0)
            avg_planning_score = sum(planning_scores) / len(planning_scores)
            
            # Bottleneck condition
            # To pass, vision must be >= planning AND vision must not be zero (recall > 0).
            # This prevents low-performance models (e.g. 0.0 vs 0.0) from getting a pass.
            is_vision_pass = (avg_vision_score >= avg_planning_score) and (avg_vision_score > 0.4)
            visual_reasoning_ability = 1.0 if is_vision_pass else 0.0

    return {
        "perception_recall": perception_recall,
        "grounding_arg_sim": grounding_arg_sim,
        "visual_reasoning_ability": visual_reasoning_ability,
        "has_vision_steps": len(vision_gt_indices) > 0,
        "has_perception_steps": len(perception_gt_indices) > 0,
        "has_grounding_steps": len(grounding_gt_indices) > 0
    }

def compute_metrics_for_id(
    gt: Trajectory,
    pred: Trajectory,
    encoder: EmbeddingEncoder,
    tau_strong: float,
    tau_weak: float,
) -> Dict[str, Any]:
    gt_calls = extract_calls(gt)
    pred_calls = extract_calls(pred)
    
    gt_texts = [flatten_arguments_to_text(c.arguments) for c in gt_calls]
    pred_texts = [flatten_arguments_to_text(c.arguments) for c in pred_calls]
    
    if len(gt_texts) == 0:
         return {
            "id": gt.task_id,
            "perception_recall": 1.0,
            "grounding_arg_sim": 1.0,
            "visual_reasoning_ability": 1.0,
            "has_vision_steps": False,
            "has_perception_steps": False,
            "has_grounding_steps": False,
            "details": {}
        }

    all_texts = gt_texts + pred_texts
    all_embs = encoder.encode_texts(all_texts)
    gt_embs = all_embs[: len(gt_texts)]
    pred_embs = all_embs[len(gt_texts) :]
    sim_matrix = cosine_similarity_matrix(gt_embs, pred_embs)

    matched_pairs, unmatched_gt, unmatched_pred = hungarian_bucketed_match(
        gt_calls, pred_calls, sim_matrix, tau_weak
    )
    
    cv_metrics = compute_cv_metrics(gt_calls, matched_pairs, unmatched_gt)

    match_details = [
        {
            "gt_call_index": p.gt_idx,
            "pred_call_index": p.pred_idx,
            "similarity": p.similarity,
            "gt_tool": gt_calls[p.gt_idx].tool_name,
            "pred_tool": pred_calls[p.pred_idx].tool_name,
        }
        for p in matched_pairs
    ]

    result = {
        "id": gt.task_id,
        "perception_recall": cv_metrics["perception_recall"],
        "grounding_arg_sim": cv_metrics["grounding_arg_sim"],
        "visual_reasoning_ability": cv_metrics["visual_reasoning_ability"],
        "has_vision_steps": cv_metrics["has_vision_steps"],
        "has_perception_steps": cv_metrics["has_perception_steps"],
        "has_grounding_steps": cv_metrics["has_grounding_steps"],
        "details": {
            "matches": match_details,
            "unmatched_gt": unmatched_gt
        },
    }
    return result

def aggregate_metrics(per_id: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregation needs to filter for relevant tasks (e.g. only tasks with perception steps for Perception Recall)
    
    # Perception Recall
    perception_tasks = [x for x in per_id if x.get("has_perception_steps")]
    if perception_tasks:
        avg_perception_recall = sum(x["perception_recall"] for x in perception_tasks) / len(perception_tasks)
    else:
        avg_perception_recall = 0.0 # Or 1.0? Use 0 to indicate N/A or no performance.
        
    # Grounding Arg Sim
    grounding_tasks = [x for x in per_id if x.get("has_grounding_steps")]
    if grounding_tasks:
        avg_grounding_arg_sim = sum(x["grounding_arg_sim"] for x in grounding_tasks) / len(grounding_tasks)
    else:
        avg_grounding_arg_sim = 0.0

    # Visual Reasoning Ability
    vision_tasks = [
        x for x in per_id if x.get("has_vision_steps") and x.get("visual_reasoning_ability") is not None
    ]
    if vision_tasks:
        visual_reasoning_ability = sum(x["visual_reasoning_ability"] for x in vision_tasks) / len(vision_tasks)
    else:
        visual_reasoning_ability = None

    return {
        "num_ids": len(per_id),
        "perception_recall_avg": avg_perception_recall,
        "grounding_arg_sim_avg": avg_grounding_arg_sim,
        "visual_reasoning_ability": visual_reasoning_ability,
        "num_perception_tasks": len(perception_tasks),
        "num_grounding_tasks": len(grounding_tasks),
        "num_vision_tasks": len(vision_tasks),
    }

def write_csv(path: str, per_id: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "perception_recall",
        "grounding_arg_sim",
        "visual_reasoning_ability",
        "has_vision_steps",
        "has_perception_steps",
        "has_grounding_steps"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for x in per_id:
            row = {k: x.get(k) for k in fieldnames}
            writer.writerow(row)

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MCP trajectories for Vision/Grounding issues")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON file")
    parser.add_argument("--output-dir", dest="output_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--tau-strong", type=float, default=0.80, help="Strong similarity threshold")
    parser.add_argument("--tau-weak", type=float, default=0.60, help="Weak similarity threshold")
    parser.add_argument("--experiment-name", help="Name of the experiment/model")
    parser.add_argument("--save-metrics-to", help="Path to JSON file to append metrics")
    parser.add_argument("--cache-path", help="Path to embedding cache")

    args = parser.parse_args(argv)
    
    gt_raw = read_json_any(args.gt)
    pred_raw = read_json_any(args.pred)
    
    gt_tasks = normalize_tasks(gt_raw)
    pred_tasks = normalize_tasks(pred_raw)
    
    gt_map = {t["id"]: to_trajectory(t) for t in gt_tasks if "id" in t}
    pred_map = {t["id"]: to_trajectory(t) for t in pred_tasks if "id" in t}
    
    common_ids = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cache_path = args.cache_path or os.path.join(output_dir, ".emb_cache.json")
    
    encoder = EmbeddingEncoder(args.model, cache_path=cache_path)
    
    per_id_results = []
    for task_id in common_ids:
        gt_traj = gt_map[task_id]
        pred_traj = pred_map[task_id]
        res = compute_metrics_for_id(gt_traj, pred_traj, encoder, args.tau_strong, args.tau_weak)
        per_id_results.append(res)
        
    encoder.persist_cache()
    
    aggregate = aggregate_metrics(per_id_results)
    
    out_json_path = os.path.join(output_dir, "results_cv_eval.json")
    out_csv_path = os.path.join(output_dir, "results_cv_eval.csv")
    
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "per_id": per_id_results}, f, indent=2)
        
    write_csv(out_csv_path, per_id_results)
    
    if args.save_metrics_to and args.experiment_name:
        metrics_entry = {
            "Model": args.experiment_name,
            "tau_weak": args.tau_weak,
            "Perception Recall": aggregate.get("perception_recall_avg", 0.0),
            "Grounding Similarity": aggregate.get("grounding_arg_sim_avg", 0.0),
            "Visual Reasoning Ability": aggregate.get("visual_reasoning_ability", 0.0),
        }
        
        existing_data = []
        if os.path.exists(args.save_metrics_to):
            try:
                with open(args.save_metrics_to, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        existing_data = json.loads(content)
                        if not isinstance(existing_data, list):
                            existing_data = []
            except Exception:
                existing_data = []
                
        existing_data.append(metrics_entry)
        os.makedirs(os.path.dirname(args.save_metrics_to), exist_ok=True)
        with open(args.save_metrics_to, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)
            
    print(f"Perception Recall: {aggregate.get('perception_recall_avg', 0.0):.4f}")
    print(f"Grounding Similarity: {aggregate.get('grounding_arg_sim_avg', 0.0):.4f}")
    
    vra = aggregate.get("visual_reasoning_ability")
    if vra is not None:
        print(f"Visual Reasoning Ability: {vra:.4f}")
    else:
        print("Visual Reasoning Ability: None")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

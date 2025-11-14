#!/usr/bin/env python3
"""
Evaluate MCP trajectories against ground truth with multi-turn and concurrency-aware matching.

Features
- Argument flattening and text-embedding via SentenceTransformers
- Tool-name bucketed Hungarian matching with strong/weak thresholds
- Step-level correspondence matrix, fragmentation and order error
- Per-id metrics and aggregated metrics; JSON/CSV output

Usage
  python tools/evaluate_trajectories.py \
    --gt /path/to/gt.json \
    --pred /path/to/pred.json \
    --out-json results_eval.json \
    --out-csv results_eval.csv \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --tau-strong 0.80 \
    --tau-weak 0.60 \
    --cache-path .emb_cache.json

Input JSON formats supported
- List of task objects, each with: {"id": str, "steps": [{"step": int, "calls": [{"name": str, "arguments": object}]}]}
- Dict of id -> task object (will be converted to list)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    # Lazy import so the script can show a helpful error if not installed
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - import guidance
    SentenceTransformer = None  # type: ignore


# ----------------------------- Data structures -----------------------------


@dataclass
class Call:
    tool_name: str
    arguments: Any
    step_index: int  # zero-based index of the step this call belongs to
    call_index_in_step: int  # zero-based index within the step


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


# ----------------------------- Utilities -----------------------------


def read_json_any(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_tasks(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Dict of id -> task
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
    """Flatten arbitrary arguments into a deterministic text string.

    - Use JSON with sorted keys for dicts, compact separators
    - Preserve primitives and lists/tuples recursively
    """
    try:
        return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(arguments)


class EmbeddingEncoder:
    def __init__(self, model_name: str, cache_path: Optional[str] = None) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install via 'pip install sentence-transformers'"
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if isinstance(cached, dict):
                        # Only accept str->list[float]
                        for k, v in cached.items():
                            if isinstance(k, str) and isinstance(v, list):
                                self.cache[k] = v
            except Exception:
                # Ignore cache read errors
                pass

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        # Lookup cache and collect misses
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

        # All must be filled now
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
    """Perform tool-name bucketed matching using Hungarian algorithm.

    Args
      gt_calls: list of GT calls
      pred_calls: list of Pred calls
      sims: cosine similarity matrix for all calls (shape [len(gt_calls), len(pred_calls)])
      tau_weak: threshold for accepting a match

    Returns
      matched_pairs: list of accepted matches
      unmatched_gt_indices: indices (in gt_calls) that did not match
      unmatched_pred_indices: indices (in pred_calls) that did not match
    """
    matched_pairs: List[MatchPair] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()

    gt_buckets = bucket_indices_by_tool_name(gt_calls)
    pred_buckets = bucket_indices_by_tool_name(pred_calls)

    for tool_name, gt_indices in gt_buckets.items():
        pred_indices = pred_buckets.get(tool_name, [])
        if not pred_indices:
            continue
        # Build submatrix
        sub = sims[np.array(gt_indices)[:, None], np.array(pred_indices)[None, :]]
        # Convert to cost (minimize). Use large cost for below-threshold to discourage matching.
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


# ----------------------------- Step metrics -----------------------------


def build_step_matrix(
    matched_pairs: List[MatchPair],
    num_gt_steps: int,
    num_pred_steps: int,
) -> np.ndarray:
    W = np.zeros((num_gt_steps, num_pred_steps), dtype=np.float32)
    for p in matched_pairs:
        W[p.gt_step_idx, p.pred_step_idx] += p.similarity
    return W


def compute_fragmentation(
    matched_pairs: List[MatchPair],
    num_gt_steps: int,
    num_pred_steps: int,
    gt_calls_per_step: List[int],
) -> Tuple[float, List[float]]:
    """Compute weighted average fragmentation and per-step fragmentation.

    For a GT step, if its matched calls fall into k distinct Pred steps,
    fragmentation = (k - 1) / k. Weighted by number of matched calls in the step.
    """
    per_step_frag: List[float] = [0.0] * num_gt_steps
    weights: List[int] = [0] * num_gt_steps
    by_gt_step: Dict[int, List[int]] = {}
    for p in matched_pairs:
        by_gt_step.setdefault(p.gt_step_idx, []).append(p.pred_step_idx)

    for s in range(num_gt_steps):
        pred_steps = by_gt_step.get(s, [])
        k = len(set(pred_steps)) if pred_steps else 0
        if k <= 1:
            per_step_frag[s] = 0.0
        else:
            per_step_frag[s] = float(k - 1) / float(k)
        weights[s] = len(pred_steps)

    total_weight = sum(weights) or 1
    weighted_avg = sum(f * w for f, w in zip(per_step_frag, weights)) / float(total_weight)
    return weighted_avg, per_step_frag


def compute_order_error(matched_pairs: List[MatchPair]) -> float:
    """Compute inversion rate over matched pairs based on step indices.

    Consider pairs (i, j). If gt_step_i < gt_step_j but pred_step_i > pred_step_j, count an inversion.
    Pairs with equal step indices in GT or Pred are ignored for inversion counting.
    """
    if len(matched_pairs) <= 1:
        return 0.0
    pairs = matched_pairs
    inversions = 0
    total = 0
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            a = pairs[i]
            b = pairs[j]
            if a.gt_step_idx == b.gt_step_idx or a.pred_step_idx == b.pred_step_idx:
                # Ignore same-step comparisons
                continue
            total += 1
            if (a.gt_step_idx < b.gt_step_idx and a.pred_step_idx > b.pred_step_idx) or (
                a.gt_step_idx > b.gt_step_idx and a.pred_step_idx < b.pred_step_idx
            ):
                inversions += 1
    if total == 0:
        return 0.0
    return inversions / float(total)


# ----------------------------- Metrics -----------------------------


def safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def compute_merge_entropy(W: np.ndarray) -> float:
    """Compute normalized conditional entropy MergeEntropy in [0, 1].

    Definition (based on step-weight matrix W where W[g, p] >= 0):
      - Let S_p = sum_g W[g, p], S = sum_p S_p, P(p) = S_p / S.
      - For each pred step p with S_p > 0, define q_{g|p} = W[g, p] / S_p.
      - Conditional entropy H(G|P) = sum_p P(p) * ( - sum_g q_{g|p} log q_{g|p} ).
      - Normalize by log(G_active), where G_active is the number of GT steps with any mass.

    Edge cases return 0.0 (no mass, or <=1 active GT step).
    """
    if W.size == 0:
        return 0.0
    # Active GT steps (rows with any positive mass)
    row_mass = W.sum(axis=1)
    G_active = int((row_mass > 0).sum())
    if G_active <= 1:
        return 0.0
    # Column masses and active columns
    col_mass = W.sum(axis=0)
    active_cols_mask = col_mass > 0
    if not bool(active_cols_mask.any()):
        return 0.0
    col_mass_active = col_mass[active_cols_mask]
    W_active = W[:, active_cols_mask]
    S = float(col_mass_active.sum())
    if S <= 0.0:
        return 0.0
    P_col = col_mass_active / S
    # Compute conditional entropy
    eps = 1e-12
    H = 0.0
    for j in range(W_active.shape[1]):
        Sj = float(col_mass_active[j])
        if Sj <= 0.0:
            continue
        q = W_active[:, j] / Sj
        mask = q > 0
        if not bool(mask.any()):
            continue
        H_j = -float((q[mask] * np.log(q[mask] + eps)).sum())
        H += float(P_col[j] * H_j)
    H_max = math.log(G_active)
    return float(H / H_max) if H_max > 0 else 0.0


def compute_metrics_for_id(
    gt: Trajectory,
    pred: Trajectory,
    encoder: EmbeddingEncoder,
    tau_strong: float,
    tau_weak: float,
) -> Dict[str, Any]:
    gt_calls = extract_calls(gt)
    pred_calls = extract_calls(pred)
    
    # Prepare embeddings for arguments
    gt_texts = [flatten_arguments_to_text(c.arguments) for c in gt_calls]
    pred_texts = [flatten_arguments_to_text(c.arguments) for c in pred_calls]
    if len(gt_texts) == 0 and len(pred_texts) == 0:
        # Degenerate case
        return {
            "id": gt.task_id,
            "total_gt_calls": 0,
            "total_pred_calls": len(pred_calls),
            "matched": 0,
            "recall": 1.0,
            "precision": 0.0 if len(pred_calls) else 1.0,
            "avg_sim_all_cov": 0.0,
            "avg_sim_strong_cov": 0.0,
            "step_coherence_cov": 1.0,
            "order_consistency_cov": 1.0,
            "merge_purity_cov": 1.0,
            "flags": {"merged_steps": False, "split_steps": False},
            "details": {},
        }

    all_texts = gt_texts + pred_texts
    all_embs = encoder.encode_texts(all_texts)
    gt_embs = all_embs[: len(gt_texts)]
    pred_embs = all_embs[len(gt_texts) :]
    sim_matrix = cosine_similarity_matrix(gt_embs, pred_embs)

    matched_pairs, unmatched_gt, unmatched_pred = hungarian_bucketed_match(
        gt_calls, pred_calls, sim_matrix, tau_weak
    )

    # Metrics
    total_gt = len(gt_calls)
    total_pred = len(pred_calls)
    matched = len(matched_pairs)
    used_pred = len({p.pred_idx for p in matched_pairs})
    recall = safe_div(matched, total_gt)
    precision = safe_div(used_pred, total_pred)
    # removed extra/missing rate metrics

    if matched:
        sims_all = [p.similarity for p in matched_pairs]
        avg_sim_all = float(sum(sims_all) / len(sims_all))
        sims_strong = [s for s in sims_all if s >= tau_strong]
        avg_sim_strong = float(sum(sims_strong) / len(sims_strong)) if sims_strong else 0.0
    else:
        sims_all = []
        sims_strong = []
        avg_sim_all = 0.0
        avg_sim_strong = 0.0

    # Step metrics
    num_gt_steps = len(gt.steps)
    num_pred_steps = len(pred.steps)
    W = build_step_matrix(matched_pairs, num_gt_steps, num_pred_steps)

    # Fragmentation
    gt_calls_per_step = [len(s.calls) for s in gt.steps]
    fragmentation, per_step_frag = compute_fragmentation(
        matched_pairs, num_gt_steps, num_pred_steps, gt_calls_per_step
    )

    # Determine primary pred step per gt step (argmax on W row)
    primary_pred_for_gt: List[Optional[int]] = []
    for i in range(num_gt_steps):
        if W.shape[1] == 0:
            primary_pred_for_gt.append(None)
            continue
        row = W[i]
        if np.all(row == 0.0):
            primary_pred_for_gt.append(None)
        else:
            primary_pred_for_gt.append(int(np.argmax(row)))

    order_error = compute_order_error(matched_pairs)
    merge_entropy = compute_merge_entropy(W)
    # Convert to higher-is-better scores in [0,1] (internal only)
    order_consistency = max(0.0, min(1.0, 1.0 - float(order_error)))
    merge_purity = max(0.0, min(1.0, 1.0 - float(merge_entropy)))

    # Coverage-weighted metrics (Scheme B)
    # AvgSimAll_cov = Recall * AvgSimAll
    avg_sim_all_cov = float(recall * avg_sim_all)
    # AvgSimStrong_cov = StrongRecall * AvgSimStrong
    strong_matches_count = int(len(sims_strong))
    strong_recall = safe_div(float(strong_matches_count), float(total_gt))
    avg_sim_strong_cov = float(strong_recall * avg_sim_strong)

    # StepCoherence_cov: per-step coverage weighted by GT calls
    matched_counts_by_step: List[int] = [0] * num_gt_steps
    for p in matched_pairs:
        if 0 <= p.gt_step_idx < num_gt_steps:
            matched_counts_by_step[p.gt_step_idx] += 1
    step_scores: List[float] = []
    for s in range(num_gt_steps):
        gt_cnt = int(gt_calls_per_step[s]) if s < len(gt_calls_per_step) else 0
        if gt_cnt <= 0:
            step_scores.append(0.0)
            continue
        coverage_s = safe_div(float(matched_counts_by_step[s]), float(gt_cnt))
        frag_s = float(per_step_frag[s]) if s < len(per_step_frag) else 0.0
        step_scores.append(coverage_s * (1.0 - frag_s))
    # Weighted by GT calls per step
    total_gt_calls_across_steps = float(sum(gt_calls_per_step)) or 1.0
    step_coherence_cov = float(
        sum(step_scores[s] * float(gt_calls_per_step[s]) for s in range(num_gt_steps)) / total_gt_calls_across_steps
    )

    # OrderConsistency_cov: pair coverage c_pair * order_consistency
    # Total GT cross-step pairs
    total_gt_pairs = 0
    for i in range(num_gt_steps):
        for j in range(i + 1, num_gt_steps):
            total_gt_pairs += int(gt_calls_per_step[i]) * int(gt_calls_per_step[j])
    # Matched cross-step pairs (by counts)
    matched_pairs_count_cross = 0
    for i in range(num_gt_steps):
        for j in range(i + 1, num_gt_steps):
            matched_pairs_count_cross += matched_counts_by_step[i] * matched_counts_by_step[j]
    c_pair = safe_div(float(matched_pairs_count_cross), float(total_gt_pairs))
    order_consistency_cov = float(order_consistency * c_pair)

    # MergePurity_cov = MergePurity * Recall
    merge_purity_cov = float(merge_purity * recall)

    # Build details for debugging/inspection
    match_details = [
        {
            "gt_call_index": p.gt_idx,
            "pred_call_index": p.pred_idx,
            "gt_step": p.gt_step_idx,
            "pred_step": p.pred_step_idx,
            "similarity": p.similarity,
            "gt_tool": gt_calls[p.gt_idx].tool_name,
            "pred_tool": pred_calls[p.pred_idx].tool_name,
        }
        for p in matched_pairs
    ]

    result = {
        "id": gt.task_id,
        "total_gt_calls": total_gt,
        "total_pred_calls": total_pred,
        "matched": matched,
        "recall": recall,
        "precision": precision,
        "avg_sim_all_cov": avg_sim_all_cov,
        "avg_sim_strong_cov": avg_sim_strong_cov,
        "step_coherence_cov": step_coherence_cov,
        "order_consistency_cov": order_consistency_cov,
        "merge_purity_cov": merge_purity_cov,
        "details": {
            "unmatched_gt_indices": unmatched_gt,
            "unmatched_pred_indices": unmatched_pred,
            "matches": match_details,
            "W": W.tolist(),
            "primary_pred_for_gt": primary_pred_for_gt,
            "per_step_fragmentation": per_step_frag,
        },
    }
    return result


def aggregate_metrics(per_id: List[Dict[str, Any]], tau_strong: float) -> Dict[str, Any]:
    total_gt_calls = sum(x.get("total_gt_calls", 0) for x in per_id)
    total_pred_calls = sum(x.get("total_pred_calls", 0) for x in per_id)
    total_matched = sum(x.get("matched", 0) for x in per_id)

    # Weighted recalls/precisions across all ids
    recall = safe_div(total_matched, total_gt_calls)
    precision = safe_div(
        sum(len(set([m["pred_call_index"] for m in x.get("details", {}).get("matches", [])])) for x in per_id),
        total_pred_calls,
    )

    # Coverage-weighted aggregated metrics (weighted by gt calls per id)
    avg_sim_all_cov_num = sum(x.get("avg_sim_all_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)
    avg_sim_strong_cov_num = sum(x.get("avg_sim_strong_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)
    step_coherence_cov_num = sum(x.get("step_coherence_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)
    order_consistency_cov_num = sum(x.get("order_consistency_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)
    merge_purity_cov_num = sum(x.get("merge_purity_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)

    avg_sim_all_cov = safe_div(avg_sim_all_cov_num, total_gt_calls)
    avg_sim_strong_cov = safe_div(avg_sim_strong_cov_num, total_gt_calls)
    step_coherence_cov = safe_div(step_coherence_cov_num, total_gt_calls)
    order_consistency_cov = safe_div(order_consistency_cov_num, total_gt_calls)
    merge_purity_cov = safe_div(merge_purity_cov_num, total_gt_calls)

    # Coverage-weighted aggregated metrics (weighted by gt calls)
    avg_sim_all_cov_num = sum(x.get("avg_sim_all_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)
    avg_sim_strong_cov_num = sum(x.get("avg_sim_strong_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)
    step_coherence_cov_num = sum(x.get("step_coherence_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)
    order_consistency_cov_num = sum(x.get("order_consistency_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)
    merge_purity_cov_num = sum(x.get("merge_purity_cov", 0.0) * x.get("total_gt_calls", 0) for x in per_id)

    avg_sim_all_cov = safe_div(avg_sim_all_cov_num, total_gt_calls)
    avg_sim_strong_cov = safe_div(avg_sim_strong_cov_num, total_gt_calls)
    step_coherence_cov = safe_div(step_coherence_cov_num, total_gt_calls)
    order_consistency_cov = safe_div(order_consistency_cov_num, total_gt_calls)
    merge_purity_cov = safe_div(merge_purity_cov_num, total_gt_calls)

    return {
        "num_ids": len(per_id),
        "total_gt_calls": total_gt_calls,
        "total_pred_calls": total_pred_calls,
        "matched": total_matched,
        "recall": recall,
        "precision": precision,
        "avg_sim_all_cov": avg_sim_all_cov,
        "avg_sim_strong_cov": avg_sim_strong_cov,
        "step_coherence_cov": step_coherence_cov,
        "order_consistency_cov": order_consistency_cov,
        "merge_purity_cov": merge_purity_cov,
    }


def write_csv(path: str, per_id: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "total_gt_calls",
        "total_pred_calls",
        "matched",
        "recall",
        "precision",
        "avg_sim_all_cov",
        "avg_sim_strong_cov",
        "step_coherence_cov",
        "merge_purity_cov",
        "order_consistency_cov",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for x in per_id:
            row = {
                "id": x.get("id"),
                "total_gt_calls": x.get("total_gt_calls"),
                "total_pred_calls": x.get("total_pred_calls"),
                "matched": x.get("matched"),
                "recall": x.get("recall"),
                "precision": x.get("precision"),
                "avg_sim_all_cov": x.get("avg_sim_all_cov"),
                "avg_sim_strong_cov": x.get("avg_sim_strong_cov"),
                "step_coherence_cov": x.get("step_coherence_cov"),
                "merge_purity_cov": x.get("merge_purity_cov"),
                "order_consistency_cov": x.get("order_consistency_cov"),
            }
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MCP trajectories against ground truth")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON file")
    parser.add_argument("--output-dir", "--output_file", dest="output_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--tau-strong", type=float, default=0.80, help="Strong similarity threshold")
    parser.add_argument("--tau-weak", type=float, default=0.60, help="Weak similarity threshold")

    args = parser.parse_args(argv)

    gt_raw = read_json_any(args.gt)
    pred_raw = read_json_any(args.pred)

    gt_tasks = normalize_tasks(gt_raw)
    pred_tasks = normalize_tasks(pred_raw)

    # Build id -> type map from GT tasks
    id_to_type: Dict[str, str] = {}
    for t in gt_tasks:
        tid = str(t.get("id", ""))
        if not tid:
            continue
        typ = t.get("type", None)
        if isinstance(typ, str) and typ.strip():
            id_to_type[tid] = typ.strip()
        else:
            id_to_type[tid] = "UNKNOWN"

    # Build id -> trajectory for pred (GT ids guaranteed to exist in pred; pred may have extras)
    gt_map: Dict[str, Trajectory] = {}
    for t in gt_tasks:
        traj = to_trajectory(t)
        if not traj.task_id:
            continue
        gt_map[traj.task_id] = traj

    pred_map: Dict[str, Trajectory] = {}
    for t in pred_tasks:
        traj = to_trajectory(t)
        if not traj.task_id:
            continue
        pred_map[traj.task_id] = traj

    # Warn about id mismatches and only evaluate intersection ids
    missing_pred_ids = [i for i in gt_map.keys() if i not in pred_map]
    missing_gt_ids = [i for i in pred_map.keys() if i not in gt_map]
    if missing_pred_ids:
        print(
            f"Warning: {len(missing_pred_ids)} GT ids are missing in predictions; excluded from evaluation.",
            file=sys.stderr,
        )
    if missing_gt_ids:
        print(
            f"Warning: {len(missing_gt_ids)} prediction ids are missing in ground truth; excluded from evaluation.",
            file=sys.stderr,
        )
    common_ids = sorted(set(gt_map.keys()) & set(pred_map.keys()))

    # Prepare output directory and default cache path
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, ".emb_cache.json")

    encoder = EmbeddingEncoder(args.model, cache_path=cache_path)

    per_id_results: List[Dict[str, Any]] = []
    for task_id in common_ids:
        gt_traj = gt_map[task_id]
        pred_traj = pred_map[task_id]
        res = compute_metrics_for_id(gt_traj, pred_traj, encoder, args.tau_strong, args.tau_weak)
        per_id_results.append(res)

    encoder.persist_cache()

    aggregate = aggregate_metrics(per_id_results, args.tau_strong)

    # Additional aggregate metrics unrelated to bucketed matching
    num_eval_ids = len(common_ids)
    total_rounds = sum(len(pred_map[tid].steps) for tid in common_ids)
    avg_rounds = safe_div(float(total_rounds), float(num_eval_ids))
    avg_tool_calls = safe_div(float(aggregate.get("total_pred_calls", 0.0)), float(aggregate.get("num_ids", 0)))
    aggregate["avg_Rounds"] = avg_rounds
    aggregate["avg_tool_calls"] = avg_tool_calls

    # Derive standard output paths
    out_json_path = os.path.join(output_dir, "results_eval.json")
    out_csv_path = os.path.join(output_dir, "results_eval.csv")

    out_json_obj = {"aggregate": aggregate, "per_id": per_id_results}
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json_obj, f, ensure_ascii=False, indent=2)

    write_csv(out_csv_path, per_id_results)

    # Build strong/weak/unmatched outputs using per-id details and original maps
    strong_matches: List[Dict[str, Any]] = []
    weak_matches: List[Dict[str, Any]] = []
    unmatched_predictions: List[Dict[str, Any]] = []

    for res in per_id_results:
        task_id = str(res.get("id"))
        gt_traj = gt_map.get(task_id)
        pred_traj = pred_map.get(task_id, Trajectory(task_id=task_id, steps=[]))
        if gt_traj is None:
            continue
        gt_calls = extract_calls(gt_traj)
        pred_calls = extract_calls(pred_traj)

        for m in res.get("details", {}).get("matches", []):
            sim = float(m.get("similarity", 0.0))
            gt_idx = int(m.get("gt_call_index", -1))
            pred_idx = int(m.get("pred_call_index", -1))
            if gt_idx < 0 or pred_idx < 0 or gt_idx >= len(gt_calls) or pred_idx >= len(pred_calls):
                continue
            gt_call = gt_calls[gt_idx]
            pred_call = pred_calls[pred_idx]
            record = {
                "id": task_id,
                "similarity": sim,
                "gt": {
                    "tool": gt_call.tool_name,
                    "arguments": gt_call.arguments,
                    "step_index": gt_call.step_index,
                    "call_index_in_step": gt_call.call_index_in_step,
                },
                "pred": {
                    "tool": pred_call.tool_name,
                    "arguments": pred_call.arguments,
                    "step_index": pred_call.step_index,
                    "call_index_in_step": pred_call.call_index_in_step,
                },
            }
            if sim >= args.tau_strong:
                strong_matches.append(record)
            elif sim >= args.tau_weak:
                weak_matches.append(record)

        for pred_idx in res.get("details", {}).get("unmatched_pred_indices", []):
            if 0 <= pred_idx < len(pred_calls):
                c = pred_calls[pred_idx]
                unmatched_predictions.append(
                    {
                        "id": task_id,
                        "pred": {
                            "tool": c.tool_name,
                            "arguments": c.arguments,
                            "step_index": c.step_index,
                            "call_index_in_step": c.call_index_in_step,
                        },
                    }
                )

    strong_path = os.path.join(output_dir, "strong_matches.json")
    weak_path = os.path.join(output_dir, "weak_matches.json")
    unmatched_path = os.path.join(output_dir, "unmatched_predictions.json")

    with open(strong_path, "w", encoding="utf-8") as f:
        json.dump(strong_matches, f, ensure_ascii=False, indent=2)
    with open(weak_path, "w", encoding="utf-8") as f:
        json.dump(weak_matches, f, ensure_ascii=False, indent=2)
    with open(unmatched_path, "w", encoding="utf-8") as f:
        json.dump(unmatched_predictions, f, ensure_ascii=False, indent=2)

    # Pretty console summary (human-readable)
    # Prepare lines with aligned labels and percentage values
    percent_keys_and_labels = [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("avg_sim_all_cov", "Avg Sim All"),
        ("avg_sim_strong_cov", "Avg Sim Strong"),
        ("step_coherence_cov", "Step Coherence"),
        ("merge_purity_cov", "Merge Purity"),
        ("order_consistency_cov", "Order Consistency"),
    ]

    label_width = max(
        20,
        max(len(lbl) for _, lbl in percent_keys_and_labels),
    )

    lines: List[str] = []
    lines.append(f"{'Output Dir':<{label_width}} : {output_dir}")
    lines.append(f"{'Num Ids':<{label_width}} : {aggregate.get('num_ids', 0)}")
    lines.append("")

    for key, label in percent_keys_and_labels:
        val = float(aggregate.get(key, 0.0))
        lines.append(f"{label:<{label_width}} : {val * 100:6.2f}%")

    # Non-percentage aggregates
    lines.append(f"{'Avg Rounds':<{label_width}} : {float(aggregate.get('avg_Rounds', 0.0)):6.2f}")
    lines.append(f"{'Avg Tool Calls':<{label_width}} : {float(aggregate.get('avg_tool_calls', 0.0)):6.2f}")

    # Type-level Top-5 by GT-weighted averages
    # Aggregate per type (coverage-weighted metrics)
    type_acc: Dict[str, Dict[str, float]] = {}
    for res in per_id_results:
        tid = str(res.get("id", ""))
        typ = id_to_type.get(tid, "UNKNOWN")
        w = int(res.get("total_gt_calls", 0))
        coh = float(res.get("step_coherence_cov", 0.0))
        orderc = float(res.get("order_consistency_cov", 0.0))
        mergep = float(res.get("merge_purity_cov", 0.0))
        acc = type_acc.setdefault(typ, {"w": 0.0, "coh_cov": 0.0, "orderc_cov": 0.0, "mergep_cov": 0.0})
        acc["w"] += float(w)
        acc["coh_cov"] += coh * float(w)
        acc["orderc_cov"] += orderc * float(w)
        acc["mergep_cov"] += mergep * float(w)

    def _avg(val_key: str, s: Dict[str, float]) -> float:
        return safe_div(s.get(val_key, 0.0), s.get("w", 0.0))

    # Select the smallest N (to keep the underlying logic of surfacing the worst types)
    coh_top = sorted(((typ, _avg("coh_cov", s)) for typ, s in type_acc.items()), key=lambda x: x[1])
    order_top = sorted(((typ, _avg("orderc_cov", s)) for typ, s in type_acc.items()), key=lambda x: x[1])
    merge_top = sorted(((typ, _avg("mergep_cov", s)) for typ, s in type_acc.items()), key=lambda x: x[1])

    lines.append("")
    n = 10
    lines.append(f"{'Top N Types - Step Coherence':<{label_width}} :")
    for typ, val in coh_top[:n]:
        lines.append(f"  {typ:<{label_width - 2}} : {val * 100:6.2f}%")
    lines.append(f"{'Top N Types - Order Consistency':<{label_width}} :")
    for typ, val in order_top[:n]:
        lines.append(f"  {typ:<{label_width - 2}} : {val * 100:6.2f}%")
    lines.append(f"{'Top N Types - Merge Purity':<{label_width}} :")
    for typ, val in merge_top[:n]:
        lines.append(f"  {typ:<{label_width - 2}} : {val * 100:6.2f}%")

    # Type-level Recall and Precision (sorted descending)
    type_stats: Dict[str, Dict[str, float]] = {}
    for res in per_id_results:
        tid = str(res.get("id", ""))
        typ = id_to_type.get(tid, "UNKNOWN")
        matched_sum = float(res.get("matched", 0))
        gt_sum = float(res.get("total_gt_calls", 0))
        pred_sum = float(res.get("total_pred_calls", 0))
        used_pred = float(len(set(
            m.get("pred_call_index")
            for m in res.get("details", {}).get("matches", [])
        )))
        acc = type_stats.setdefault(typ, {"matched": 0.0, "gt": 0.0, "used_pred": 0.0, "pred": 0.0})
        acc["matched"] += matched_sum
        acc["gt"] += gt_sum
        acc["used_pred"] += used_pred
        acc["pred"] += pred_sum

    recall_sorted = sorted(((typ, safe_div(s["matched"], s["gt"])) for typ, s in type_stats.items()), key=lambda x: x[1], reverse=True)
    precision_sorted = sorted(((typ, safe_div(s["used_pred"], s["pred"])) for typ, s in type_stats.items()), key=lambda x: x[1], reverse=True)

    # lines.append("")
    # lines.append(f"{'Types by Recall (desc)':<{label_width}} :")
    # for typ, val in recall_sorted:
    #     lines.append(f"  {typ:<{label_width - 2}} : {val * 100:6.2f}%")
    # lines.append(f"{'Types by Precision (desc)':<{label_width}} :")
    # for typ, val in precision_sorted:
    #     lines.append(f"  {typ:<{label_width - 2}} : {val * 100:6.2f}%")

    # No merged/split flags printing

    print("\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



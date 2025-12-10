#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# ========= Fill JSON list to process here (or pass via --files) =========
JSON_FILES = []  # default empty; pass via --files or customize relative paths
# ==========================================================================


def should_drop_sample(sample: Dict[str, Any]) -> bool:
    """Drop sample iff it has a "steps" field equal to an empty list."""
    if not isinstance(sample, dict):
        return False
    steps = sample.get("steps", None)
    return isinstance(steps, list) and len(steps) == 0


def filter_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept = []
    for s in samples:
        if should_drop_sample(s):
            continue
        kept.append(s)
    return kept


def process_file(path_str: str):
    p = Path(path_str)
    if not p.exists():
        print(f"[SKIP] Not found: {p}")
        return

    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read: {p} -> {e}")
        return

    changed = False
    removed = 0

    if isinstance(data, list):
        before = len(data)
        data = filter_samples(data)
        after = len(data)
        removed = before - after
        changed = removed > 0

    elif isinstance(data, dict) and isinstance(data.get("samples"), list):
        before = len(data["samples"])
        data["samples"] = filter_samples(data["samples"])
        after = len(data["samples"])
        removed = before - after
        changed = removed > 0

    else:
        print(f"[WARN] Unrecognized structure (no changes): {p}")
        return

    if changed:
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Saved over: {p}, removed {removed} samples (empty steps)")
        except Exception as e:
            print(f"[ERROR] Failed to write back: {p} -> {e}")
    else:
        print(f"[OK] No changes needed: {p} (removed 0)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and delete samples where steps is an empty array (in-place overwrite)"
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="List of JSON files to process (empty uses script's JSON_FILES)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    targets = args.files if args.files else JSON_FILES
    if not targets:
        print("[INFO] No files to process. Use --files or set JSON_FILES in script.")
        return
    for fp in targets:
        process_file(fp)


if __name__ == "__main__":
    main()



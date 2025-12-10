#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# ========= Fill your lists here =========
RESULT_FILES = []

JSON_FILES = []

PREFIX_IDS_4 = [
]

EXACT_IDS_8 = [
]
# =======================================

# Normalization: 4-char prefixes; exact IDs to 8 chars (preserve leading zeros)
PREFIX_SET = {str(p).strip()[:4].zfill(4) for p in PREFIX_IDS_4}
EXACT_SET = {str(e).strip()[:8].zfill(8) for e in EXACT_IDS_8}

def id_to_8str(any_id) -> str:
    """Normalize sample id to 8-char string (preserve leading zeros)"""
    return str(any_id).strip().zfill(8)

def should_drop(id8: str) -> bool:
    """Decide drop by 4-char prefix or exact 8-char id"""
    if id8 in EXACT_SET:
        return True
    if id8[:4] in PREFIX_SET:
        return True
    return False

def filter_list(samples: list) -> list:
    """Filter structures where top-level is a list of samples"""
    kept = []
    for s in samples:
        # Keep samples without id directly (typically should have id)
        if not isinstance(s, dict) or "id" not in s:
            kept.append(s)
            continue
        id8 = id_to_8str(s["id"])
        if not should_drop(id8):
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
        data = filter_list(data)
        after = len(data)
        removed = before - after
        changed = removed > 0

    elif isinstance(data, dict) and isinstance(data.get("samples"), list):
        before = len(data["samples"])
        data["samples"] = filter_list(data["samples"])
        after = len(data["samples"])
        removed = before - after
        changed = removed > 0

    else:
        # Basic helper: supports only top-level list or dict with `samples` list
        print(f"[WARN] Unrecognized structure (no changes): {p}")
        return

    if changed:
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Saved over: {p}, removed {removed} samples")
        except Exception as e:
            print(f"[ERROR] Failed to write back: {p} -> {e}")
    else:
        print(f"[OK] No changes needed: {p} (removed 0)")

def process_result_file(path_str: str):
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

    if not isinstance(data, dict) or not isinstance(data.get("pre_id"), dict):
        print(f"[WARN] Unrecognized structure (no changes): {p}")
        return

    removed = 0
    for k in list(data["pre_id"].keys()):
        id8 = id_to_8str(k)
        if should_drop(id8):
            del data["pre_id"][k]
            removed += 1

    if removed > 0:
        if "aggregate" in data:
            del data["aggregate"]
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Saved over: {p}, removed {removed} pre_id entries and removed aggregate")
        except Exception as e:
            print(f"[ERROR] Failed to write back: {p} -> {e}")
    else:
        print(f"[OK] No changes needed: {p} (removed 0 pre_id)")

def main():
    if not JSON_FILES and not RESULT_FILES:
        print("[INFO] JSON_FILES and RESULT_FILES are both empty. Please fill the lists first.")
        return
    print(f"[INFO] Prefix set (4 chars): {sorted(PREFIX_SET)}")
    print(f"[INFO] Exact set (8 chars): {sorted(EXACT_SET)}")
    for fp in RESULT_FILES:
        process_result_file(fp)
    for fp in JSON_FILES:
        process_file(fp)

if __name__ == "__main__":
    main()

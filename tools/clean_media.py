#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify and (optionally) prune the accumulated ``media/`` directory.

Default is a dry run: nothing is ever deleted without both ``--apply`` and
``--yes``. Deletion is restricted to files this script has *proven* redundant,
i.e. numbered duplicates whose bytes are identical (md5) to the GT original they
were copied from.

Usage
-----
    python tools/clean_media.py --dry-run
    python tools/clean_media.py --apply --yes
    python tools/clean_media.py --apply --yes --include-probes
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Set

REPO = Path(__file__).resolve().parent.parent
MEDIA = REPO / "media"
GT_FILE = REPO / "json" / "test_mcp_GT.json"
REPORT_DIR = REPO / "results" / "workspace_redesign"

DUP_RE = re.compile(r"^(.+?)_(\d+)$")
PROBE_HINTS = ("_audit", "healthcheck", "mcp_probe", "_verify")
# Hand-made QR fixture used by the pyzbar health check; nothing regenerates it.
PROBE_KEEP = {"_qr_healthcheck.png"}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def gt_image_names() -> Set[str]:
    data = json.loads(GT_FILE.read_text(encoding="utf-8"))
    return {Path(t["image"]).name for t in data if t.get("image")}


def referenced_names() -> Set[str]:
    """Filenames mentioned as ``media/<name>`` anywhere in delivered results."""
    out: Set[str] = set()
    for pattern in ("results/**/*.json", "results/*.json"):
        for f in glob.glob(str(REPO / pattern), recursive=True):
            try:
                txt = Path(f).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            out.update(re.findall(r"media/([A-Za-z0-9_.\-]+)", txt))
    return out


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def classify() -> Dict[str, List[Path]]:
    gt = gt_image_names()
    buckets: Dict[str, List[Path]] = {
        "gt_original": [],
        "verified_duplicate": [],
        "suffix_but_different": [],
        "probe": [],
        "artifact": [],
    }
    files = sorted(p for p in MEDIA.iterdir() if p.is_file())
    for p in files:
        name = p.name
        if name in gt:
            buckets["gt_original"].append(p)
            continue
        if any(h in name for h in PROBE_HINTS):
            buckets["probe"].append(p)
            continue
        m = DUP_RE.match(p.stem)
        if m and (m.group(1) + p.suffix) in gt:
            orig = MEDIA / (m.group(1) + p.suffix)
            if orig.exists() and orig.stat().st_size == p.stat().st_size and md5(orig) == md5(p):
                buckets["verified_duplicate"].append(p)
            else:
                buckets["suffix_but_different"].append(p)
            continue
        buckets["artifact"].append(p)
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="default; classify and report only")
    ap.add_argument("--apply", action="store_true", help="actually delete (requires --yes)")
    ap.add_argument("--yes", action="store_true", help="confirm destructive deletion")
    ap.add_argument("--include-probes", action="store_true",
                    help="also delete probe/healthcheck artifacts (keeps _qr_healthcheck.png)")
    ap.add_argument("--quiet", action="store_true", help="suppress per-file listing")
    args = ap.parse_args()

    if not MEDIA.is_dir():
        raise SystemExit(f"no media dir at {MEDIA}")

    buckets = classify()
    refs = referenced_names()

    print(f"media/ = {MEDIA}")
    for k, ps in buckets.items():
        size = sum(p.stat().st_size for p in ps)
        n_ref = sum(1 for p in ps if p.name in refs)
        print(f"  {k:24s} {len(ps):5d} files  {human(size):>9s}  (referenced in results/: {n_ref})")

    delete: List[Path] = list(buckets["verified_duplicate"])
    if args.include_probes:
        delete += [p for p in buckets["probe"] if p.name not in PROBE_KEEP]

    freed = sum(p.stat().st_size for p in delete)
    print(f"\nwould delete: {len(delete)} files, freeing {human(freed)}")
    print(f"would keep  : gt_original={len(buckets['gt_original'])} "
          f"artifact={len(buckets['artifact'])} "
          f"probe={len(buckets['probe']) - (len(delete) - len(buckets['verified_duplicate']))} "
          f"suffix_but_different={len(buckets['suffix_but_different'])}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = REPORT_DIR / "media_cleanup_manifest.json"
    manifest.write_text(json.dumps({
        "counts": {k: len(v) for k, v in buckets.items()},
        "bytes": {k: sum(p.stat().st_size for p in v) for k, v in buckets.items()},
        "delete_candidates": [p.name for p in delete],
        "delete_bytes": freed,
        "referenced_in_results": sorted(n for n in refs if (MEDIA / n).exists()),
        "keep_artifacts": [p.name for p in buckets["artifact"]],
        "keep_probes": sorted({p.name for p in buckets["probe"]} - {p.name for p in delete}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest}")

    if not args.quiet:
        print("\n-- artifacts kept (tool outputs / office files) --")
        for p in buckets["artifact"]:
            print(f"   KEEP {p.name}{'  [referenced]' if p.name in refs else ''}")
        print("\n-- probes --")
        for p in buckets["probe"]:
            act = "DELETE" if p in delete else "KEEP"
            print(f"   {act} {p.name}")

    if not (args.apply and args.yes):
        print("\n[dry-run] nothing deleted. Pass --apply --yes to execute.")
        return

    removed = 0
    for p in delete:
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}")
    print(f"\ndeleted {removed}/{len(delete)} files")


if __name__ == "__main__":
    main()

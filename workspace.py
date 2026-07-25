#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-task workspace management for the MCP benchmark.

Three modes, selected via ``M3_WORKSPACE_MODE``:

``legacy``
    Byte-for-byte the historical behaviour: every task copies its input image
    into the flat ``media/`` directory, appending ``_1``/``_2``/... on name
    collision. Kept purely as an escape hatch.

``dedup`` (default)
    Still a flat ``media/``, but an existing copy with identical content is
    reused instead of creating yet another numbered duplicate. This makes the
    per-task ``image_id`` deterministic (``00110000.png`` every run) and stops
    ``media/`` from growing by one file per task per run.

``isolated``
    Each task gets ``media/runs/<run_id>/<task_id>/`` containing a symlink to
    the read-only source image. Tool artifacts land there because the image
    tools derive output paths from the input path, and because the host rewrites
    explicit output-path arguments into this directory.

The workspace root deliberately lives *under* ``media/``: the PowerPoint server
runs with ``ENFORCE_OUTPUT_DIR=true`` and ``OUTPUT_DIR=./media``, so any
absolute path it is asked to write must be inside that subtree.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

MEDIA_DIR = Path("media")
RUNS_DIRNAME = "runs"

MODE_LEGACY = "legacy"
MODE_DEDUP = "dedup"
MODE_ISOLATED = "isolated"
_VALID_MODES = (MODE_LEGACY, MODE_DEDUP, MODE_ISOLATED)

# Files in the repo root that the post-task cwd sweep must never touch. The
# historical sweep deleted every *.png in cwd, which removed git-tracked figures
# (mcp_tools_per_server.png, metrics_mllm_step_eval.png).
_CWD_PROTECTED = {
    "m3_logo.jpg",
    "mcp_tools_per_server.png",
    "metrics_mllm_step_eval.png",
}


def workspace_mode() -> str:
    raw = (os.environ.get("M3_WORKSPACE_MODE") or MODE_DEDUP).strip().lower()
    return raw if raw in _VALID_MODES else MODE_DEDUP


def keep_workspace() -> bool:
    raw = os.environ.get("M3_KEEP_WORKSPACE")
    if raw is None:
        return True
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _md5(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def resolve_run_id(model_name: str = "run") -> str:
    """Stable per-run identifier. Set ``M3_RUN_ID`` to make paths reproducible."""
    env = (os.environ.get("M3_RUN_ID") or "").strip()
    if env:
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in env)
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in (model_name or "run"))
    return f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def runs_root() -> Path:
    return MEDIA_DIR / RUNS_DIRNAME


def task_workspace(run_id: str, task_id: str, fresh: bool = True) -> Path:
    """Return (and create) the deterministic workspace dir for one task.

    ``fresh=True`` wipes the directory first so a rerun of the same task starts
    from the same clean state instead of inheriting the previous attempt's
    artifacts. Only this one directory is ever removed.
    """
    safe_task = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(task_id or "task"))
    ws = runs_root() / run_id / safe_task
    if fresh and ws.exists():
        try:
            shutil.rmtree(ws)
        except Exception:
            pass
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def link_or_copy(src: Path, dst: Path) -> Path:
    """Symlink ``src`` at ``dst``, falling back to a copy where links fail.

    MCP servers are same-host stdio subprocesses reading the real filesystem, so
    a symlink is resolved transparently by the kernel; nothing needs to know.
    """
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except Exception:
            return dst
    try:
        dst.symlink_to(src.resolve())
        if dst.exists():
            return dst
        dst.unlink()
    except Exception:
        pass
    try:
        shutil.copy2(str(src), str(dst))
    except Exception:
        return src
    return dst


def _legacy_copy(src: Path) -> Path:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    target = MEDIA_DIR / src.name
    if target.exists():
        stem, suf = target.stem, target.suffix
        k = 1
        while True:
            cand = MEDIA_DIR / f"{stem}_{k}{suf}"
            if not cand.exists():
                target = cand
                break
            k += 1
    shutil.copy2(str(src), str(target))
    return target


def _dedup_copy(src: Path) -> Path:
    """Reuse an existing identical copy in ``media/`` instead of adding another.

    Only falls back to a numbered name when a *different* file already occupies
    the basename, which keeps the deterministic name in the common case.
    """
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    target = MEDIA_DIR / src.name
    if not target.exists():
        shutil.copy2(str(src), str(target))
        return target
    try:
        if target.samefile(src):
            return target
    except Exception:
        pass
    src_size, tgt_size = src.stat().st_size, target.stat().st_size
    if src_size == tgt_size and _md5(src) == _md5(target):
        return target
    # Genuinely different content under the same basename: keep both, but pick a
    # name derived from content so it stays stable across runs.
    digest = (_md5(src) or "0")[:8]
    cand = MEDIA_DIR / f"{target.stem}_{digest}{target.suffix}"
    if not cand.exists():
        shutil.copy2(str(src), str(cand))
    return cand


def materialize_input(src_path: str, ws_dir: Optional[Path], mode: Optional[str] = None) -> str:
    """Place a task's source image where the tools can reach it.

    Returns an absolute path. Never mutates the source file.
    """
    m = mode or workspace_mode()
    try:
        src = Path(src_path).resolve()
    except Exception:
        return src_path
    if not src.exists() or not src.is_file():
        return src_path
    try:
        if m == MODE_LEGACY:
            return str(_legacy_copy(src).resolve())
        if m == MODE_ISOLATED and ws_dir is not None:
            ws_dir.mkdir(parents=True, exist_ok=True)
            placed = link_or_copy(src, ws_dir / src.name)
            # Deliberately NOT .resolve()d: resolving a symlink would hand back
            # the read-only source path, and the image tools derive their default
            # output path from the path string they are given -- artifacts would
            # then land next to the pristine source instead of in the workspace.
            return str(Path(os.path.abspath(str(placed))))
        return str(_dedup_copy(src).resolve())
    except Exception:
        return str(src)


def snapshot_cwd_files(exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".xlsx", ".pptx")) -> Set[str]:
    """Record cwd artifact filenames so only *new* ones get swept later."""
    wanted = tuple(e.lower() for e in exts)
    out: Set[str] = set()
    try:
        for p in Path(".").iterdir():
            if p.is_file() and p.suffix.lower() in wanted:
                out.add(p.name)
    except Exception:
        pass
    return out


def sweep_cwd_new_files(
    before: Set[str],
    exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".xlsx", ".pptx"),
    move_to: Optional[Path] = None,
) -> List[str]:
    """Remove (or relocate) cwd artifacts that appeared after ``before``.

    Concurrency-safe in the sense that a worker only ever touches files that did
    not exist when *it* started, and never touches the protected git-tracked
    figures. ``move_to`` relocates instead of deleting, which keeps a task's
    stray output inspectable inside its workspace.
    """
    wanted = tuple(e.lower() for e in exts)
    handled: List[str] = []
    try:
        entries = list(Path(".").iterdir())
    except Exception:
        return handled
    for p in entries:
        try:
            if not p.is_file() or p.suffix.lower() not in wanted:
                continue
            if p.name in _CWD_PROTECTED or p.name in before:
                continue
            if move_to is not None:
                move_to.mkdir(parents=True, exist_ok=True)
                dest = move_to / p.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(p), str(dest))
            else:
                p.unlink()
            handled.append(p.name)
        except Exception:
            continue
    return handled


def cleanup_workspace(ws_dir: Optional[Path]) -> None:
    if ws_dir is None or keep_workspace():
        return
    try:
        if ws_dir.exists() and runs_root().resolve() in ws_dir.resolve().parents:
            shutil.rmtree(ws_dir)
    except Exception:
        pass


def describe() -> Dict[str, str]:
    return {
        "mode": workspace_mode(),
        "runs_root": str(runs_root()),
        "keep_workspace": str(keep_workspace()),
    }

"""Reproducibility manifest written next to every results CSV.

It records exactly what produced the numbers (item-set hash and parameters, models, decoding, library
versions, hardware, git commit) and lets `analyze` rebuild the identical item set automatically.
Re-running into the same CSV with a different item set is refused.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from importlib import metadata
from pathlib import Path


def _ver(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return None


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _git():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def build_manifest(args: dict, items_hash: str, n_items: int, extra: dict | None = None) -> dict:
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        ram = None
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "item_params": {k: args[k] for k in
            ("per_family", "perms", "seed", "framing", "families") if k in args},
            "items_hash": items_hash, "n_items": n_items,
            "run_params": {k: args[k] for k in ("backend", "models", "styles", "max_new_tokens", "batch_size", "temperatures",
                                                 "decode_seeds") if k in args},
            "python": platform.python_version(), "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(),
            "ram_gb": ram, "gpu": _gpu(), "git_commit": _git(),
            "packages": {p: _ver(p) for p in ("numpy", "pandas", "scipy", "matplotlib", "requests", "transformers", "torch")},
            **(extra or {})}


def manifest_path(csv_path) -> Path:
    p = Path(csv_path); return p.with_name(p.stem + ".manifest.json")


def write_manifest(csv_path, manifest: dict):
    mp = manifest_path(csv_path)
    if mp.exists():
        old = json.loads(mp.read_text())
        if old["items_hash"] != manifest["items_hash"]:
            raise ValueError(f"{mp} was made with a different item set (hash {old['items_hash']} != "
                             f"{manifest['items_hash']}). Use a new --out file.")
        manifest["runs"] = old.get("runs", [old.get("run_params")]) + [manifest["run_params"]]
        manifest["created_first"] = old.get("created_first", old["created"])
    mp.write_text(json.dumps(manifest, indent=2))


def read_manifest(csv_path) -> dict | None:
    mp = manifest_path(csv_path)
    return json.loads(mp.read_text()) if mp.exists() else None


def merge_csvs(csv_paths, out_csv):
    """Combine result files made by separate runs (e.g. one Kaggle notebook per model).
    Refuses if they were generated from different item sets. Returns number of rows."""
    import pandas as pd
    mans = [read_manifest(p) for p in csv_paths]
    if any(m is None for m in mans):
        raise ValueError("every CSV needs its .manifest.json next to it")
    base = mans[0]
    for p, m in zip(csv_paths, mans):
        if m["items_hash"] != base["items_hash"] or m["item_params"] != base["item_params"]:
            raise ValueError(f"{p} used a different item set (hash {m['items_hash']} vs {base['items_hash']}); cannot merge")
    df = pd.concat([pd.read_csv(p, keep_default_na=False) for p in csv_paths], ignore_index=True)
    df = df.drop_duplicates(["item_id", "model", "style", "temperature", "decode_seed"], keep="first")
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    merged = dict(base); merged["merged_from"] = [str(p) for p in csv_paths]
    merged["run_params"] = {**base["run_params"], "models": sorted(df.model.unique().tolist())}
    manifest_path(out).write_text(json.dumps(merged, indent=2))
    return len(df)

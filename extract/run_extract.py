#!/usr/bin/env python3
"""Extract every-layer readouts for one backbone, driven by a config.

For each (size, geometry) in the config, writes to
  <out>/<backbone>/<size>px/<geometry>/
    keys.txt                    record keys, in row order
    cls.npy                     [N, layers, width]  (absent if no CLS)
    patch_mean.npy              [N, layers, width]
    patch_mean_normed.npy       [N, layers, width]
    pooled_<name>.npy           [N, d]   e.g. pre_projection, post_projection
    manifest.json               what was run, on what, and the checks
as float16. Unnormalised: any L2 or other normalisation is an evaluation
choice, not an extraction one.

Checks recorded in the manifest (the run fails loudly on the first two):
  - the adapter's reconstruction of the model's own pooled output, on the
    first batch, matches the model's forward;
  - no NaN or infinite value anywhere;
  - row count equals the dev-shard index count (or --limit).

    python -m extract.run_extract configs\\extract\\wp8_clip_laion2b.yaml
    python -m extract.run_extract <config> --limit 512      # quick test
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .backbones import get_adapter
from .dataset import DevShardDataset
from .preprocess import SHARD_SET


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True,
                                         stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL)
    except Exception:                                  # noqa: BLE001
        return {"commit": None, "dirty": None}
    # A dirty tree is recorded WITH its diff, so the run stays reproducible
    # (the WP7 lesson: a flag alone records that something differed, not what).
    return {"commit": commit, "dirty": bool(diff.strip()),
            "diff": diff if diff.strip() else None}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions():
    import importlib
    out = {"torch": torch.__version__, "numpy": np.__version__}
    for m in ("open_clip", "transformers", "birder", "PIL"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:                              # noqa: BLE001
            pass
    return out


def run_one(adapter, cfg, size, geometry, out_dir, args, dev):
    ds = DevShardDataset(cfg["dev_shards"], SHARD_SET[geometry], geometry,
                         size, adapter.mean, adapter.std, limit=args.limit,
                         input_mode=adapter.input_mode,
                         patch_size=adapter.patch_size)
    n, L, D = len(ds), adapter.n_layers, adapter.width
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 128),
                    num_workers=cfg.get("num_workers", 8), shuffle=False,
                    pin_memory=dev.type == "cuda",
                    persistent_workers=False)
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "keys.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(ds.keys) + "\n")
    mm, nonfinite, check, t0 = {}, 0, None, time.time()

    def arr(name, shape):
        if name not in mm:
            mm[name] = np.lib.format.open_memmap(
                os.path.join(out_dir, f"{name}.npy"), mode="w+",
                dtype=np.float16, shape=shape)
        return mm[name]

    amp = torch.autocast(dev.type, dtype=torch.float16) \
        if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    done = 0
    for idx, *x in dl:
        x = [t.to(dev, non_blocking=True) for t in x]
        with amp:
            if check is None:
                check = adapter.self_check(*x)
                if not check.get("ok", True):
                    raise RuntimeError(f"self-check failed: {check}")
            r = adapter.forward(*x)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        parts = {"patch_mean": r.patch_mean,
                 "patch_mean_normed": r.patch_mean_normed}
        if r.cls is not None:
            parts["cls"] = r.cls
        for name, t in parts.items():
            a = t.float().cpu().numpy()
            nonfinite += int((~np.isfinite(a)).sum())
            arr(name, (n, L, D))[lo:hi] = a
        for name, t in r.pooled.items():
            a = t.float().cpu().numpy()
            nonfinite += int((~np.isfinite(a)).sum())
            arr(f"pooled_{name}", (n, a.shape[1]))[lo:hi] = a
        done += hi - lo
        if done % (50 * dl.batch_size) < dl.batch_size or done == n:
            el = time.time() - t0
            print(f"    {done:,}/{n:,}  {done / el:,.0f} img/s", flush=True)
    for a in mm.values():
        a.flush()
    secs = time.time() - t0
    if nonfinite:
        raise RuntimeError(f"{nonfinite} non-finite values written")
    man = {
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "backbone": adapter.describe(), "size": size, "geometry": geometry,
        "shard_set": SHARD_SET[geometry], "rows": n, "limit": args.limit,
        "arrays": {k: list(v.shape) for k, v in mm.items()},
        "dtype": "float16", "normalised": False,
        "self_check": check, "nonfinite": nonfinite,
        "seconds": round(secs, 1), "images_per_s": round(n / secs, 1),
        "dev_shards_manifest_sha256": sha256_file(
            os.path.join(cfg["dev_shards"], "dev_shards_manifest.json")),
        "config": cfg, "git": git_state(), "versions": versions(),
        "device": str(dev),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=None,
                    help="test run on N records per geometry")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-weights", action="store_true",
                    help="random-init model; plumbing tests only")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    dev = torch.device(args.device or cfg.get("device", "cuda:0"))
    root = os.path.join(cfg["out"] + ("_test" if args.limit else ""),
                        cfg["backbone"])
    for size in cfg["sizes"]:
        adapter = get_adapter(cfg["backbone"]).load(dev, size, args.no_weights)
        for geometry in cfg["geometries"]:
            out_dir = os.path.join(root, f"{size}px", geometry)
            if os.path.exists(os.path.join(out_dir, "manifest.json")):
                print(f"skip {out_dir} (complete; delete it to redo)")
                continue
            if os.path.exists(out_dir):
                # The manifest is written last, so a folder without one is a
                # run that did not finish. Never skip it silently.
                sys.exit(f"{out_dir} exists but has no manifest.json: an "
                         "unfinished run. Delete the folder and re-run.")
            print(f"\n{cfg['backbone']}  {size}px  {geometry}"
                  f"{'  [interpolated]' if adapter.interpolated else ''}",
                  flush=True)
            m = run_one(adapter, cfg, size, geometry, out_dir, args, dev)
            print(f"  {m['rows']:,} rows, {m['images_per_s']} img/s, "
                  f"self-check {m['self_check']}")
        del adapter
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())

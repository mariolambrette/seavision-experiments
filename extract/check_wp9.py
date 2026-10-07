#!/usr/bin/env python3
"""Does WP9's extraction reproduce WP8's? Run before the main extraction.

WP8's embedding arrays are deleted, so this compares the two code paths now,
on the same records:

  old  the WP8 path: DevShardDataset over the dev tars, adapter.forward,
       every layer -- then the WP9 arrays picked out of it;
  new  the WP9 path: extract_tar() over the MAIN shard tar the records sit
       in, writing to disk exactly as the real run will -- then read back.

Same keys, same order, same batch size, so the GPU sees identical batches.
Pass criterion, fixed before the test was run (7 October 2026): for every
array, max over rows of  max|old - new| / max|old|  <= 1e-3.

    python -m extract.check_wp9 configs\\extract\\wp9.yaml --backbone clip_laion2b
    python -m extract.check_wp9 <cfg> --backbone siglip2_naflex --geometry native

Writes nothing outside a temporary folder, which it deletes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .backbones import get_adapter
from .dataset import DevShardDataset
from .preprocess import SHARD_SET
from .readout_spec import check_against, load_spec, readout_arrays, select
from .run_wp9 import extract_tar, n_workers, tars_of
from .tar_dataset import index_tar

TOL = 1e-3


def old_path(adapter, dev_dir, set_name, geometry, size, keys, names, batch,
             dev):
    ds = DevShardDataset(dev_dir, set_name, geometry, size, adapter.mean,
                         adapter.std, input_mode=adapter.input_mode,
                         patch_size=adapter.patch_size)
    ds.keys = list(keys)                     # same records, same order
    dl = DataLoader(ds, batch_size=batch, num_workers=0, shuffle=False)
    amp = torch.autocast(dev.type, dtype=torch.float16) \
        if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    out = {}
    for idx, *x in dl:
        x = [t.to(dev) for t in x]
        with amp:
            r = adapter.forward(*x)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        for n, t in select(r, names).items():
            a = t.float().cpu().numpy()
            if n not in out:
                out[n] = np.empty((len(keys), a.shape[1]), np.float16)
            out[n][lo:hi] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--geometry", default=None,
                    help="default: the first geometry in the spec")
    ap.add_argument("--dev-shards",
                    default="D:/marineai/scratch/wp8_sweep/dev_shards")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-weights", action="store_true",
                    help="random-init model; plumbing tests only")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    spec = load_spec(cfg["spec"], args.backbone)
    g = args.geometry or spec["geometries"][0]
    set_name = SHARD_SET[g]
    dev = torch.device(args.device or cfg.get("device", "cuda:0"))
    batch = cfg.get("batch_size_overrides", {}).get(
        args.backbone, cfg.get("batch_size", 128))
    cfg["_batch"], cfg["_workers"] = batch, n_workers(cfg)
    names = readout_arrays(spec["readouts"])

    with open(os.path.join(args.dev_shards, f"dev-{set_name}.index.json"),
              encoding="utf-8") as fh:
        dev_keys = set(json.load(fh)["keys"])
    tar, keys = None, []
    for t in tars_of(cfg["shards"], set_name):          # first tar with dev
        k, _, _ = index_tar(t)                          # records in it
        hit = [x for x in k if x in dev_keys]
        if len(hit) >= min(args.n, 32):
            tar, keys = t, hit[:args.n]
            break
    if tar is None:
        sys.exit(f"no main {set_name} tar holds dev records")
    print(f"{args.backbone} {spec['size']} {g}: {len(keys)} dev records from "
          f"{os.path.basename(tar)}; arrays {names}")

    adapter = get_adapter(args.backbone).load(dev, spec["size"],
                                              args.no_weights)
    check_against(adapter, names)
    old = old_path(adapter, args.dev_shards, set_name, g, spec["size"], keys,
                   names, batch, dev)
    tmp = tempfile.mkdtemp(prefix="check_wp9_")
    try:
        out_dir = os.path.join(tmp, "t")
        rec = extract_tar(adapter, tar, g, spec["size"], names, out_dir, cfg,
                          dev, {"commit": "test", "dirty": True},
                          only_keys=keys)
        with open(os.path.join(out_dir, "keys.txt"), encoding="utf-8") as fh:
            new_keys = fh.read().split()
        if new_keys != sorted(keys):
            sys.exit("FAIL: key order differs between the two paths")
        ok = True
        for n in names:
            a = old[n].astype(np.float32)
            b = np.load(os.path.join(out_dir, f"{n}.npy")).astype(np.float32)
            if a.shape != b.shape:
                print(f"  {n:<28} FAIL shape {a.shape} v {b.shape}")
                ok = False
                continue
            scale = np.maximum(np.abs(a).max(1), 1e-12)
            rel = float((np.abs(a - b).max(1) / scale).max())
            exact = float((a == b).all(1).mean())
            p = rel <= TOL
            ok &= p
            print(f"  {n:<28} max rel diff {rel:.2e}  rows identical "
                  f"{exact:6.1%}  {'PASS' if p else 'FAIL'}")
        print(f"  self-check (new path): {rec['self_check']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

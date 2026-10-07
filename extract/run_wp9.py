#!/usr/bin/env python3
"""WP9: extract the readouts WP8 chose, for every record in the main shards.

What is extracted is fixed by `configs/extract/wp9_readouts.yaml` (the WP8
decision; checked against the sweep by `evaluate/check_readout_spec.py`).
How is the WP8 code path, unchanged: the same adapters, preprocessing, fp16
autocast, self-check and float16 storage. Only two things differ from
`run_extract.py`: records come from the main shard tars instead of the dev
tars, and only the chosen readouts are written instead of every layer.

Output, one folder per input tar so a crash costs at most one tar:

  <out>/<backbone>/<size>px/<geometry>/
      run_manifest.json                what is being run, written once
      <set>/<tar stem>/
          keys.txt                     record keys, in row order
          <token>@<layer>.npy          [rows, d] float16, unnormalised
          done.json                    checks for this tar; written LAST

A tar is complete only when its folder exists under its final name, which
happens by renaming `<tar stem>.part` after done.json is written. Re-running
skips complete tars and redoes any `.part` folder (a run that was
interrupted), so resuming after a crash is just running the same command.

Sets per geometry: the animal set the geometry reads (crops, square-m00,
square-m10), plus the background set for the crops-type geometries listed
in the config (background crops are box regions, so they have no square
version).

Checks (any failure stops the run): the self-check on the first batch of
EVERY tar reproduces the model's own output; no non-finite value; every
requested readout exists; a resumed geometry is being run with the same
readouts, size and adapter as when it started; the git tree is clean (pass
--allow-dirty for test runs only -- those outputs are labelled dirty).

    python -m extract.run_wp9 configs\\extract\\wp9.yaml --backbone clip_laion2b
    python -m extract.run_wp9 configs\\extract\\wp9.yaml --backbone clip_laion2b --status
    python -m extract.run_wp9 <cfg> --backbone X --max-tars 1 --allow-dirty   # test

Test runs (--max-tars or --allow-dirty) write to <out>_test, never <out>.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .backbones import get_adapter
from .preprocess import SHARD_SET
from .readout_spec import check_against, load_spec, readout_arrays, select
from .run_extract import git_state, sha256_file, versions
from .tar_dataset import TarDataset, index_tar

TAR_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{6})\.tar$")


def uncommitted():
    """Lines of `git status --porcelain`: modified AND untracked files. The
    WP8 git_state() looks at `git diff HEAD`, which cannot see a new file
    nobody has added -- such as this module before its first commit."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      text=True, stderr=subprocess.DEVNULL)
    except Exception:                                  # noqa: BLE001
        return ["(git status failed)"]
    return [ln for ln in out.splitlines() if ln.strip()]


def n_workers(cfg):
    w = cfg.get("num_workers", "auto")
    if w == "auto":
        return max(1, ((os.cpu_count() or 2) - 1) // int(cfg.get(
            "concurrent_runs", 1)))
    return int(w)


def sets_for(geometry, cfg):
    sets = [SHARD_SET[geometry]]
    if geometry in cfg.get("background_geometries", []):
        sets.append("background")
    return sets


def tars_of(shards_root, set_name):
    d = os.path.join(shards_root, set_name)
    if not os.path.isdir(d):
        sys.exit(f"shard set folder missing: {d}")
    tars = sorted(f for f in os.listdir(d) if TAR_RE.match(f))
    if not tars:
        sys.exit(f"no tars in {d}")
    return [os.path.join(d, f) for f in tars]


def extract_tar(adapter, tar_path, geometry, size, names, out_dir, cfg, dev,
                git, only_keys=None):
    """Extract one tar into out_dir (which must not exist). Returns the
    done.json content. `only_keys` restricts to a subset (tests only)."""
    t0 = time.time()
    keys, members, no_image = index_tar(tar_path)
    if only_keys is not None:
        want = set(only_keys)
        keys = [k for k in keys if k in want]
    part = out_dir + ".part"
    os.makedirs(part)
    ds = TarDataset(tar_path, keys, members, geometry, size, adapter.mean,
                    adapter.std, input_mode=adapter.input_mode,
                    patch_size=adapter.patch_size)
    dl = DataLoader(ds, batch_size=cfg["_batch"], num_workers=cfg["_workers"],
                    shuffle=False, pin_memory=dev.type == "cuda",
                    persistent_workers=False)
    amp = torch.autocast(dev.type, dtype=torch.float16) \
        if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    arrs, check, nonfinite, done = {}, None, 0, 0
    for idx, *x in dl:
        x = [t.to(dev, non_blocking=True) for t in x]
        with amp:
            if check is None:
                check = adapter.self_check(*x)
                if not check.get("ok", True):
                    raise RuntimeError(f"{tar_path}: self-check failed {check}")
            r = adapter.forward(*x)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        if hi - lo != len(idx):
            raise RuntimeError("batch indices not contiguous")
        for n, t in select(r, names).items():
            a = t.float().cpu().numpy()          # as run_extract.py
            nonfinite += int((~np.isfinite(a)).sum())
            if n not in arrs:
                arrs[n] = np.empty((len(keys), a.shape[1]), np.float16)
            arrs[n][lo:hi] = a
        done = hi
    if done != len(keys):
        raise RuntimeError(f"{tar_path}: wrote {done} of {len(keys)} rows")
    if nonfinite:
        raise RuntimeError(f"{tar_path}: {nonfinite} non-finite values")
    for n, a in arrs.items():
        np.save(os.path.join(part, f"{n}.npy"), a)
    with open(os.path.join(part, "keys.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(keys) + "\n")
    st = os.stat(tar_path)
    secs = time.time() - t0
    rec = {"tar": os.path.basename(tar_path), "tar_bytes": st.st_size,
           "subset": only_keys is not None,
           "tar_mtime": st.st_mtime, "rows": len(keys),
           "keys_without_image": no_image,
           "arrays": {n: list(a.shape) for n, a in arrs.items()},
           # checksums of the files as written, so a later copy (the NAS
           # push) or a disk fault can be detected by recomputing them
           "sha256": {f"{n}.npy": sha256_file(os.path.join(part, f"{n}.npy"))
                      for n in arrs} | {"keys.txt": sha256_file(
                          os.path.join(part, "keys.txt"))},
           "dtype": "float16", "normalised": False,
           "self_check": check, "nonfinite": nonfinite,
           "seconds": round(secs, 1),
           "images_per_s": round(len(keys) / max(secs, 1e-9), 1),
           "git_commit": git["commit"], "git_dirty": git["dirty"],
           "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(part, "done.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    os.replace(part, out_dir)                    # complete only from here
    return rec


def run_manifest(path, want):
    """Write on first run; on resume, refuse if what is being run changed."""
    # Compare as JSON would store it (tuples become lists, and so on), so a
    # resume is judged on content, never on Python types.
    want = json.loads(json.dumps(want, default=str))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            have = json.load(fh)
        for k in ("backbone", "size", "geometry", "arrays", "readouts"):
            if have.get(k) != want.get(k):
                sys.exit(f"{path}: resumed run differs in {k!r}:\n  was "
                         f"{have.get(k)}\n  now {want.get(k)}\nMove the old "
                         "output aside rather than mixing the two.")
        return
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(want, fh, indent=1, default=str)


def status(cfg, backbone, spec):
    root = os.path.join(cfg["out"], backbone, f"{spec['size']}px")
    print(f"{backbone} {spec['size']}px")
    for g in spec["geometries"]:
        for s in sets_for(g, cfg):
            tars = tars_of(cfg["shards"], s)
            rows, n_done, commits, partial = 0, 0, set(), 0
            for t in tars:
                d = os.path.join(root, g, s, os.path.splitext(
                    os.path.basename(t))[0])
                if os.path.exists(d + ".part"):
                    partial += 1
                f = os.path.join(d, "done.json")
                if os.path.exists(f):
                    with open(f, encoding="utf-8") as fh:
                        rec = json.load(fh)
                    n_done += 1
                    rows += rec["rows"]
                    commits.add((rec["git_commit"] or "?")[:8] +
                                ("-dirty" if rec["git_dirty"] else ""))
            print(f"  {g:<11} {s:<11} {n_done:>3}/{len(tars)} tars  "
                  f"{rows:>10,} rows  partial {partial}  commits "
                  f"{sorted(commits) or '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--geometry", nargs="*", default=None,
                    help="only these geometries (default: all in the spec)")
    ap.add_argument("--max-tars", type=int, default=None,
                    help="test: at most N tars per set")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="test runs only; outputs are labelled dirty")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-weights", action="store_true",
                    help="random-init model; plumbing tests only")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    spec = load_spec(cfg["spec"], args.backbone)
    if args.status:
        status(cfg, args.backbone, spec)
        return
    git = git_state()
    if git["commit"] is None:
        sys.exit("not a git checkout: the run could not be tied to its code")
    pending = uncommitted()
    git["dirty"] = bool(pending)
    git["status"] = pending
    if pending and not args.allow_dirty:
        sys.exit("uncommitted changes (including untracked files):\n  " +
                 "\n  ".join(pending[:20]) + "\nCommit first, so every "
                 "embedding is tied to a commit (or --allow-dirty for a test "
                 "run).")
    dev = torch.device(args.device or cfg.get("device", "cuda:0"))
    cfg["_batch"] = cfg.get("batch_size_overrides", {}).get(
        args.backbone, cfg.get("batch_size", 128))
    cfg["_workers"] = n_workers(cfg)
    names = readout_arrays(spec["readouts"])
    geoms = args.geometry or spec["geometries"]
    unknown = set(geoms) - set(spec["geometries"])
    if unknown:
        sys.exit(f"geometries not in the spec for {args.backbone}: {unknown}")
    size = spec["size"]
    adapter = get_adapter(args.backbone).load(dev, size, args.no_weights)
    check_against(adapter, names)
    print(f"{args.backbone} {size}px  arrays {names}  batch {cfg['_batch']}  "
          f"workers {cfg['_workers']}  {dev}")
    # Test runs never write into the real tree: a dirty or partial tar left
    # there would be skipped as "done" by the real run.
    test = args.allow_dirty or args.max_tars is not None or args.no_weights
    out = cfg["out"] + ("_test" if test else "")
    root = os.path.join(out, args.backbone, f"{size}px")
    print(f"  writing to {root}{'  [TEST]' if test else ''}")
    for g in geoms:
        gdir = os.path.join(root, g)
        os.makedirs(gdir, exist_ok=True)
        run_manifest(os.path.join(gdir, "run_manifest.json"), {
            "backbone": adapter.describe(), "size": size, "geometry": g,
            "arrays": names, "readouts": spec["readouts"],
            "sets": sets_for(g, cfg), "config": {k: v for k, v in cfg.items()
                                                 if not k.startswith("_")},
            "spec_file": cfg["spec"], "git_at_start": git,
            "versions": versions(),
            "started": datetime.datetime.now().isoformat(timespec="seconds")})
        for s in sets_for(g, cfg):
            tars = tars_of(cfg["shards"], s)[:args.max_tars]
            for i, t in enumerate(tars, 1):
                stem = os.path.splitext(os.path.basename(t))[0]
                out_dir = os.path.join(gdir, s, stem)
                if os.path.exists(os.path.join(out_dir, "done.json")):
                    print(f"  skip {g}/{s}/{stem} (done)")
                    continue
                if os.path.exists(out_dir):
                    sys.exit(f"{out_dir} exists without done.json -- should "
                             "be impossible (folders are renamed into place "
                             "complete). Inspect it before doing anything.")
                if os.path.exists(out_dir + ".part"):
                    print(f"  removing interrupted {stem}.part, redoing it")
                    shutil.rmtree(out_dir + ".part")
                print(f"  {g}/{s}  [{i}/{len(tars)}] {stem} ...", flush=True)
                rec = extract_tar(adapter, t, g, size, names, out_dir, cfg,
                                  dev, git)
                print(f"    {rec['rows']:,} rows, {rec['images_per_s']} img/s,"
                      f" self-check {rec['self_check']}", flush=True)
    del adapter
    if dev.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())

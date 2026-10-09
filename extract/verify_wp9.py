#!/usr/bin/env python3
"""Verify the WP9 embeddings before anything is built on them.

`run_wp9.py --status` counts done.json files. That says the run finished; it
does not say the output is right. This script re-checks every tar's output
against things outside the run's own bookkeeping:

  1. completeness  every tar of every set the spec implies has a finished
                   folder; no `.part` folder or unexpected folder is left;
  2. keys          keys.txt equals the keys read afresh from the shard tar,
                   in the same order (so row i really is record i), and the
                   tar has not changed since it was embedded (size, mtime);
  3. arrays        exactly the arrays the spec asks for; each loads, is
                   float16, has the row count of keys.txt and one width per
                   backbone and array across every tar;
  4. content       (skipped with --quick) every file's sha256 matches the one
                   written at extraction; no non-finite value; no all-zero
                   row, which a silently failed batch would leave;
  5. provenance    self-check passed on every tar; no test or subset output;
                   no dirty tree; the commits used are listed;
  6. totals        records per set match the shard builds (WP7).

Read-only. Exits non-zero if anything fails, and writes a JSON report.

    python -m extract.verify_wp9 configs\\extract\\wp9.yaml
    python -m extract.verify_wp9 configs\\extract\\wp9.yaml --backbone bio_dino --quick
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

from .readout_spec import readout_arrays
from .run_extract import sha256_file
from .run_wp9 import sets_for, tars_of
from .tar_dataset import index_tar

# Records per shard set as built and verified in WP7 (schedule WP7). Keys
# with an image; a mismatch means the shards changed or a tar was missed.
EXPECTED_RECORDS = {"crops": 1_485_215, "square-m00": 1_370_528,
                    "square-m10": 1_370_528, "background": 106_039}

CHUNK = 65_536   # rows read at a time for the content scan


def scan_array(path, rows, width):
    """-> (n non-finite values, n all-zero rows). Reads in chunks."""
    a = np.load(path, mmap_mode="r")
    nonfinite = zero = 0
    for lo in range(0, rows, CHUNK):
        x = np.asarray(a[lo:lo + CHUNK], dtype=np.float32)
        nonfinite += int((~np.isfinite(x)).sum())
        zero += int((np.abs(x).max(axis=1) == 0).sum())
    return nonfinite, zero


def verify_tar(d, tar, tar_index, names, quick, widths, problems):
    """Check one finished tar folder. Returns rows (0 if unusable)."""
    def bad(msg):
        problems.append(f"{d}: {msg}")

    try:
        with open(os.path.join(d, "done.json"), encoding="utf-8") as fh:
            rec = json.load(fh)
    except Exception as e:                              # noqa: BLE001
        bad(f"done.json unreadable ({type(e).__name__})")
        return 0
    keys_tar = tar_index
    try:
        with open(os.path.join(d, "keys.txt"), encoding="utf-8") as fh:
            keys = fh.read().splitlines()
    except OSError:
        bad("keys.txt missing")
        return 0
    if keys != keys_tar:
        n_common = len(set(keys) & set(keys_tar))
        bad(f"keys.txt differs from the tar ({len(keys)} vs {len(keys_tar)} "
            f"keys, {n_common} in common)")
    if rec.get("rows") != len(keys):
        bad(f"done.json rows {rec.get('rows')} != keys.txt {len(keys)}")
    st = os.stat(tar)
    if rec.get("tar_bytes") != st.st_size:
        bad(f"tar size changed since extraction ({rec.get('tar_bytes')} -> "
            f"{st.st_size})")
    if rec.get("tar_mtime") is not None and \
            abs(float(rec["tar_mtime"]) - st.st_mtime) > 2:
        bad("tar modified since extraction (mtime differs)")
    if rec.get("subset"):
        bad("output of a subset (test) run")
    if rec.get("git_dirty"):
        bad("extracted from a dirty tree")
    sc = rec.get("self_check") or {}
    if not sc or not sc.get("ok", False):
        bad(f"self-check missing or failed: {sc}")
    if rec.get("nonfinite", 1) != 0:
        bad(f"done.json records {rec.get('nonfinite')} non-finite values")

    present = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npy"))
    if sorted(names) != present:
        bad(f"arrays {present}, spec asks for {sorted(names)}")
    for n in names:
        p = os.path.join(d, f"{n}.npy")
        if not os.path.exists(p):
            continue
        try:
            a = np.load(p, mmap_mode="r")
        except Exception as e:                          # noqa: BLE001
            bad(f"{n}.npy does not load ({type(e).__name__})")
            continue
        if a.dtype != np.float16 or a.ndim != 2:
            bad(f"{n}.npy is {a.dtype} with {a.ndim} dims")
            continue
        if a.shape[0] != len(keys):
            bad(f"{n}.npy has {a.shape[0]} rows, keys.txt {len(keys)}")
        widths[n].add(int(a.shape[1]))
        if list(a.shape) != rec.get("arrays", {}).get(n):
            bad(f"{n}.npy shape {list(a.shape)} != done.json "
                f"{rec.get('arrays', {}).get(n)}")
        if not quick:
            want = rec.get("sha256", {}).get(f"{n}.npy")
            if sha256_file(p) != want:
                bad(f"{n}.npy sha256 does not match done.json")
            nf, zero = scan_array(p, a.shape[0], a.shape[1])
            if nf:
                bad(f"{n}.npy holds {nf} non-finite values")
            if zero:
                bad(f"{n}.npy holds {zero} all-zero rows")
        del a
    if not quick:
        want = rec.get("sha256", {}).get("keys.txt")
        if sha256_file(os.path.join(d, "keys.txt")) != want:
            bad("keys.txt sha256 does not match done.json")
    return len(keys)


def verify_backbone(cfg, bb, spec, quick, tar_cache, problems, report):
    names = readout_arrays(spec["readouts"])
    root = os.path.join(cfg["out"], bb, f"{spec['size']}px")
    widths = defaultdict(set)
    commits = set()
    if not os.path.isdir(root):
        problems.append(f"{root}: missing")
        return
    extra = sorted(set(os.listdir(root)) - set(spec["geometries"]))
    if extra:
        problems.append(f"{root}: unexpected folders {extra}")
    for g in spec["geometries"]:
        gdir = os.path.join(root, g)
        man = os.path.join(gdir, "run_manifest.json")
        if not os.path.exists(man):
            problems.append(f"{gdir}: run_manifest.json missing")
        else:
            with open(man, encoding="utf-8") as fh:
                m = json.load(fh)
            if m.get("arrays") != names or m.get("size") != spec["size"]:
                problems.append(f"{man}: arrays/size {m.get('arrays')} "
                                f"{m.get('size')} differ from the spec")
        for s in sets_for(g, cfg):
            tars = tars_of(cfg["shards"], s)
            stems = {os.path.splitext(os.path.basename(t))[0]: t for t in tars}
            sdir = os.path.join(gdir, s)
            found = set(os.listdir(sdir)) if os.path.isdir(sdir) else set()
            leftovers = sorted(f for f in found if f.endswith(".part"))
            stray = sorted(found - set(stems) - set(leftovers))
            if leftovers:
                problems.append(f"{sdir}: interrupted folders {leftovers}")
            if stray:
                problems.append(f"{sdir}: unexpected folders {stray}")
            rows = n_ok = 0
            for stem, t in stems.items():
                d = os.path.join(sdir, stem)
                if not os.path.exists(os.path.join(d, "done.json")):
                    problems.append(f"{d}: not extracted")
                    continue
                if t not in tar_cache:
                    tar_cache[t] = index_tar(t)[0]
                before = len(problems)
                rows += verify_tar(d, t, tar_cache[t], names, quick, widths,
                                   problems)
                n_ok += len(problems) == before
                with open(os.path.join(d, "done.json"), encoding="utf-8") as fh:
                    commits.add(json.load(fh).get("git_commit"))
            exp = cfg.get("expected_records", EXPECTED_RECORDS).get(s)
            if exp is not None and rows != exp:
                problems.append(f"{bb}/{g}/{s}: {rows:,} rows, the {s} "
                                f"shard set holds {exp:,}")
            line = (f"  {g:<11} {s:<11} {n_ok:>3}/{len(tars)} tars clean  "
                    f"{rows:>10,} rows")
            print(line, flush=True)
            report.setdefault(bb, {}).setdefault(g, {})[s] = {
                "tars": len(tars), "clean": n_ok, "rows": rows}
    for n, w in widths.items():
        if len(w) != 1:
            problems.append(f"{bb}: {n} has several widths {sorted(w)}")
    report.setdefault(bb, {})["_widths"] = {n: sorted(w)
                                             for n, w in widths.items()}
    report[bb]["_commits"] = sorted(c or "?" for c in commits)
    print(f"  widths {dict((n, sorted(w)) for n, w in widths.items())}  "
          f"commits {sorted(c[:8] if c else '?' for c in commits)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--backbone", nargs="*", default=None,
                    help="default: every backbone in the spec")
    ap.add_argument("--quick", action="store_true",
                    help="skip checksums and the content scan")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with open(cfg["spec"], encoding="utf-8") as fh:
        spec_all = yaml.safe_load(fh)["backbones"]
    bbs = args.backbone or sorted(spec_all)
    unknown = set(bbs) - set(spec_all)
    if unknown:
        sys.exit(f"not in the spec: {sorted(unknown)}")
    problems, report, tar_cache = [], {}, {}
    for bb in bbs:
        print(f"{bb} {spec_all[bb]['size']}px "
              f"{readout_arrays(spec_all[bb]['readouts'])}", flush=True)
        verify_backbone(cfg, bb, spec_all[bb], args.quick, tar_cache,
                        problems, report)
    out = {"checked": datetime.datetime.now().isoformat(timespec="seconds"),
           "config": args.config, "quick": args.quick, "backbones": bbs,
           "problems": problems, "summary": report}
    path = os.path.join(cfg["out"], "verify_wp9_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nreport: {path}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:50]:
            print("  " + p)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more (see the report)")
        return 1
    print(f"\nALL CLEAN: {len(bbs)} backbone(s)"
          f"{' (quick: no checksums or content scan)' if args.quick else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

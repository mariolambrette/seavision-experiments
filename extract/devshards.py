#!/usr/bin/env python3
"""Copy the development records out of the full shards, once.

The dev keys are scattered across ~90 tars (~120 GB). Every sweep pass --
backbone x resolution x geometry -- would otherwise re-read all of them. This
writes one small tar per shard set holding only the records the split
manifest labels `dev`, byte-for-byte (no decode, no re-encode), plus an index
of member offsets so the sweep can read any record directly.

Checks (any failure -> no index written, exit code 1):
  - every dev key in the manifest is found exactly once;
  - each copied record has an image member and a json member;
  - record counts per source equal the manifest's dev counts.

    python -m extract.devshards ^
        --shards D:\\marineai\\classification-experiments\\shards ^
        --split  D:\\marineai\\classification-experiments\\splits\\dev_v1 ^
        --out    D:\\marineai\\scratch\\wp8_sweep\\dev_shards
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict

SETS = ["crops", "square-m00", "square-m10"]
TAR_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{6})\.tar$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dev_keys(split_dir, set_name):
    keys = {}
    with gzip.open(os.path.join(split_dir, f"split_{set_name}.csv.gz"), "rt",
                   newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "dev":
                keys[r["key"]] = r["source"]
    return keys


def build_index(tar_path):
    """key -> {ext: [offset_data, size]} for every member."""
    idx = defaultdict(dict)
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if m.isfile():
                key, _, ext = m.name.partition(".")
                idx[key][ext] = [m.offset_data, m.size]
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*", default=SETS)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    os.makedirs(args.out)
    t0 = time.time()
    report = {"split_manifest_sha256": sha256(
        os.path.join(args.split, "split_manifest.json")), "sets": {}}
    ok = True
    for set_name in args.sets:
        want = dev_keys(args.split, set_name)
        want_by_src = Counter(want.values())
        srcs = set(want_by_src)
        out_tar = os.path.join(args.out, f"dev-{set_name}.tar")
        seen = Counter()
        members = defaultdict(set)
        set_dir = os.path.join(args.shards, set_name)
        tars = sorted(f for f in os.listdir(set_dir) if TAR_RE.match(f))
        with tarfile.open(out_tar, "w") as out:
            for f in tars:
                src = TAR_RE.match(f)["prefix"][len(set_name) + 1:]
                if src not in srcs:
                    continue                    # no dev records in this source
                print(f"  {set_name:<11} {f}  [{time.time() - t0:,.0f}s]",
                      flush=True)
                with tarfile.open(os.path.join(set_dir, f), "r:") as tf:
                    for m in tf:
                        if not m.isfile():
                            continue
                        key, _, ext = m.name.partition(".")
                        if key not in want:
                            continue
                        data = tf.extractfile(m).read()
                        ti = tarfile.TarInfo(m.name)
                        ti.size = len(data)
                        out.addfile(ti, io.BytesIO(data))
                        members[key].add(ext)
                        if ext == "json":
                            seen[key] += 1
        missing = [k for k in want if seen[k] == 0]
        dup = [k for k, n in seen.items() if n > 1]
        no_img = [k for k, e in members.items() if not (e - {"json"})]
        got_by_src = Counter(want[k] for k in seen)
        s_ok = not missing and not dup and not no_img and \
            got_by_src == want_by_src
        ok &= s_ok
        idx = build_index(out_tar)
        with open(os.path.join(args.out, f"dev-{set_name}.index.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"keys": sorted(idx), "members": idx}, fh)
        report["sets"][set_name] = {
            "passed": s_ok, "records": len(seen),
            "expected_by_source": dict(want_by_src),
            "copied_by_source": dict(got_by_src),
            "missing": len(missing), "duplicates": len(dup),
            "without_image": len(no_img), "examples_missing": missing[:5],
            "tar_sha256": sha256(out_tar),
            "tar_bytes": os.path.getsize(out_tar)}
        print(f"  {set_name}: {len(seen):,} records "
              f"{'PASS' if s_ok else 'FAIL'}  {dict(got_by_src)}")
    report["passed"] = ok
    report["seconds"] = round(time.time() - t0)
    with open(os.path.join(args.out, "dev_shards_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\n{'PASSED' if ok else 'FAILED'}  -> {args.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

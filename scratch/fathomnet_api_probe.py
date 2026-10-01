#!/usr/bin/env python3
"""Probe what the FathomNet API actually returns per image. READ-ONLY, ~20 calls.

Before writing the external check on the URL-derived groups, look at what
the API exposes rather than constructing calls from memory -- the same rule
that the OzFish URLs and the .MP4 token taught. Three parts:

1. Every public function in fathomnet.api's image, upload and DarwinCore
   modules, with its signature.
2. The full image record for 2 images from each major institution, chosen
   deterministically from the groups CSV.
3. Any upload / DarwinCore lookup that takes an image uuid, tried on the
   same images. Failures are printed, not raised.

    python scratch/fathomnet_api_probe.py ^
        --groups D:\\marineai\\scratch\\fathomnet_groups\\fathomnet_groups.csv.gz ^
        --out    D:\\marineai\\scratch\\fathomnet_api_probe.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import inspect
import json
import os
import sys

MODULES = ["images", "imagesetuploads", "darwincore", "boundingboxes"]
INSTITUTIONS = ["NOAA NMFS SEFSC", "MBARI", "NOAA Ocean Exploration",
                "Schmidt Ocean Institute"]
PER_INST = 2


def as_dict(obj):
    for m in ("to_dict", "dict", "model_dump"):
        f = getattr(obj, m, None)
        if callable(f):
            try:
                return f()
            except Exception:                            # noqa: BLE001
                pass
    if isinstance(obj, (list, tuple)):
        return [as_dict(o) for o in obj]
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")

    out = {"functions": {}, "images": []}

    print("PART 1  available functions")
    mods = {}
    for name in MODULES:
        try:
            mod = importlib.import_module(f"fathomnet.api.{name}")
        except ImportError as exc:
            print(f"  fathomnet.api.{name}: NOT AVAILABLE ({exc})")
            continue
        mods[name] = mod
        fns = {}
        for fn_name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn_name.startswith("_") or fn.__module__ != mod.__name__:
                continue
            try:
                sig = str(inspect.signature(fn))
            except (TypeError, ValueError):
                sig = "(?)"
            fns[fn_name] = sig
            print(f"  {name}.{fn_name}{sig}")
        out["functions"][name] = fns

    print("\nPART 2/3  sample images")
    picked = {i: [] for i in INSTITUTIONS}
    with gzip.open(args.groups, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            inst = row["owner_institution"]
            if inst in picked and len(picked[inst]) < PER_INST and \
                    row["group"] and row["group_level"] in ("dive",
                                                            "deployment",
                                                            "rig_day"):
                picked[inst].append(row)
            if all(len(v) >= PER_INST for v in picked.values()):
                break

    img_mod = mods.get("images")
    up_mod = mods.get("imagesetuploads")
    # Upload lookups that plausibly take an image uuid, found by name rather
    # than assumed to exist.
    up_fns = []
    if up_mod:
        up_fns = [f for f in out["functions"].get("imagesetuploads", {})
                  if "image" in f.lower() and "uuid" in f.lower()]
        print(f"  upload lookups by image uuid found: {up_fns or 'NONE'}")

    for inst, rows in picked.items():
        for row in rows:
            uu = row["fathomnet_image_uuid"]
            rec = {"institution": inst, "uuid": uu, "our_group": row["group"],
                   "our_level": row["group_level"]}
            print(f"\n  {inst}  {uu}  ours: {row['group']}")
            if img_mod and hasattr(img_mod, "find_by_uuid"):
                try:
                    d = as_dict(img_mod.find_by_uuid(uu))
                    d.pop("boundingBoxes", None)        # long and not needed
                    rec["image"] = d
                    print("    image:", json.dumps(d, default=str)[:600])
                except Exception as exc:                 # noqa: BLE001
                    rec["image_error"] = repr(exc)
                    print(f"    image lookup failed: {exc!r}")
            for fn_name in up_fns:
                try:
                    d = as_dict(getattr(up_mod, fn_name)(uu))
                    rec[f"upload:{fn_name}"] = d
                    print(f"    {fn_name}:", json.dumps(d, default=str)[:600])
                except Exception as exc:                 # noqa: BLE001
                    rec[f"upload_error:{fn_name}"] = repr(exc)
                    print(f"    {fn_name} failed: {exc!r}")
            out["images"].append(rec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

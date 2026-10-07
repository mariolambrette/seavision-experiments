#!/usr/bin/env python3
"""Build the label table for the development records, once.

One row per dev record (crops set): key, source, deployment, category_id,
rank, and the species/genus/family/order names from the category's WoRMS
lineage. Square-set keys are the same keys as crops-set keys (C5 in the split
checks), so one table serves every geometry.

Taxonomy comes from `lineage`, never from the name: WoRMS can accept a
binomial whose genus is itself a synonym (WP6a), so splitting the name would
give the wrong genus.

    python -m evaluate.labels ^
        --dev-shards D:\\marineai\\scratch\\wp8_sweep\\dev_shards ^
        --split      D:\\marineai\\classification-experiments\\splits\\dev_v1 ^
        --coco       D:\\marineai\\dataset\\collated\\seavision.json ^
                     D:\\marineai\\dataset\\collated\\seavision_fathomnet.json ^
        --out        D:\\marineai\\scratch\\wp8_sweep\\labels.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter

FIELDS = ["key", "source", "deployment", "category_id", "rank", "species",
          "genus", "family", "order"]


def load_categories(paths):
    """category_id -> (rank, species, genus, family, order). Streams nothing
    clever: the COCO files are loaded once and only categories are kept."""
    cats = {}
    for p in paths:
        print(f"  loading categories from {os.path.basename(p)} ...",
              flush=True)
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        for c in doc["categories"]:
            lin = c.get("lineage") or {}
            rank = str(c.get("rank", "")).lower()
            cats[c["id"]] = (rank,
                             c.get("name") if rank == "species" else "",
                             lin.get("genus") or "", lin.get("family") or "",
                             lin.get("order") or "")
        del doc
    return cats


def read_dev_meta(dev_dir):
    """key -> record json, from the crops dev tar via its index."""
    with open(os.path.join(dev_dir, "dev-crops.index.json"),
              encoding="utf-8") as fh:
        ix = json.load(fh)
    out = {}
    with open(os.path.join(dev_dir, "dev-crops.tar"), "rb") as fh:
        for key in ix["keys"]:
            off, n = ix["members"][key]["json"]
            fh.seek(off)
            out[key] = json.loads(fh.read(n))
    return out


def read_groups(split_dir):
    g = {}
    with gzip.open(os.path.join(split_dir, "split_crops.csv.gz"), "rt",
                   newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "dev":
                g[r["key"]] = (r["source"], r["group"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-shards", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--coco", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    cats = load_categories(args.coco)
    groups = read_groups(args.split)
    meta = read_dev_meta(args.dev_shards)
    rows, no_cat, unknown = [], 0, Counter()
    for key in sorted(meta):
        src, dep = groups[key]
        cid = meta[key].get("category_id")
        if cid is None:
            no_cat += 1            # FishWIO's shipped background crops
            continue
        if cid not in cats:
            unknown[src] += 1
            continue
        rank, sp, ge, fa, od = cats[cid]
        rows.append([key, src, dep, cid, rank, sp, ge, fa, od])
    if unknown:
        sys.exit(f"category ids missing from the COCO files: {dict(unknown)}")
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(rows)
    by_src = Counter(r[1] for r in rows)
    print(f"wrote {len(rows):,} labelled dev records {dict(by_src)}; "
          f"{no_cat} without a category (excluded)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only peek at record formats, for designing the WP8 development split.

Writes ONE json file and changes nothing else. Three parts:

1. SHARDS   the first record's .json from the first tar of every
            (shard set, source) pair, found by walking --shards. Fields
            differ by source as well as by set, so one per source, not
            one per set.
2. COCO     one full image record per source from seavision.json, and
            one per owner_institution from seavision_fathomnet.json,
            each with its annotation(s), category and dataset entry.
3. FATHOMNET GROUPING
            for every source_meta key: how many FathomNet images carry
            it (non-empty), per institution. For keys whose name looks
            like a collection/dive/cruise field: distinct-value counts
            and the 10 commonest values per institution. That answers
            "can we split by dive code?" directly, instead of from the
            documentation.

Loading seavision_fathomnet.json takes a minute or two and several GB of
RAM. Nothing is written to shards or collated.

    python scratch/peek_records.py ^
        --shards   D:\\marineai\\classification-experiments\\shards ^
        --collated D:\\marineai\\dataset\\collated ^
        --out      D:\\marineai\\scratch\\peek_records.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict

TAR_IDX = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{6})\.tar$")
GROUPISH = re.compile(r"collect|dive|cruise|expedition|deploy|station|"
                      r"mission|survey|platform|vehicle|upload|imageset",
                      re.I)


# ----------------------------------------------------------------- shards

def peek_shards(root):
    """-> {set_dir: {prefix: {"tar": name, "key": key, "meta": {...}}}}

    Groups tars by filename prefix (everything before -NNNNNN.tar), which
    is '<set>-<source>' in every builder, and reads only the first json
    member of the lowest-numbered tar. A tar is sequential, so that is a
    read of a few KB, not the whole shard.
    """
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        tars = [f for f in files if f.endswith(".tar")]
        if not tars:
            continue
        by_prefix = defaultdict(list)
        for f in tars:
            m = TAR_IDX.match(f)
            if m:
                by_prefix[m["prefix"]].append((int(m["idx"]), f))
        set_name = os.path.relpath(dirpath, root)
        out[set_name] = {}
        for prefix, lst in sorted(by_prefix.items()):
            first = sorted(lst)[0][1]
            entry = {"tar": first, "n_tars": len(lst)}
            with tarfile.open(os.path.join(dirpath, first)) as tf:
                for info in tf:
                    key, _, ext = info.name.partition(".")
                    if ext == "json":
                        entry["key"] = key
                        entry["meta"] = json.loads(tf.extractfile(info).read())
                        break
            out[set_name][prefix] = entry
            print(f"  {set_name:<14} {prefix:<32} key={entry.get('key')}")
        mf = os.path.join(dirpath, "build_manifest.json")
        if os.path.exists(mf):
            with open(mf, encoding="utf-8") as fh:
                man = json.load(fh)
            man.get("git", {}).pop("diff", None)       # can be huge
            out[set_name]["_manifest_per_source"] = man.get("per_source")
    return out


# ------------------------------------------------------------------- coco

def load(path):
    print(f"  loading {os.path.basename(path)} ...", flush=True)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sample_images(doc, group_fn, per_group=1):
    """One image per group, preferring images that carry an annotation, with
    its annotations, their categories and its dataset entry attached."""
    anns = defaultdict(list)
    for a in doc.get("annotations", []):
        anns[a["image_id"]].append(a)
    cats = {c["id"]: c for c in doc.get("categories", [])}
    dsets = {d["id"]: d for d in doc.get("datasets", [])}

    picked = defaultdict(list)
    for im in doc.get("images", []):
        g = group_fn(im, dsets)
        if len(picked[g]) >= per_group or not anns.get(im["id"]):
            continue
        a = anns[im["id"]][:3]
        picked[g].append({
            "image": im,
            "annotations": a,
            "categories": [cats.get(x.get("category_id")) for x in a],
            "dataset": dsets.get(im.get("dataset_id")),
        })
    return dict(picked)


def fathomnet_grouping(doc):
    """Which source_meta fields exist per institution, and what the
    grouping-like ones contain."""
    present = defaultdict(Counter)            # inst -> key -> n non-empty
    n_inst = Counter()
    values = defaultdict(lambda: defaultdict(Counter))  # key -> inst -> val
    groups_nonempty = 0
    for im in doc.get("images", []):
        sm = im.get("source_meta") or {}
        inst = str(sm.get("owner_institution") or "UNKNOWN")
        n_inst[inst] += 1
        if im.get("groups"):
            groups_nonempty += 1
        for k, v in sm.items():
            if v in (None, "", [], {}):
                continue
            present[inst][k] += 1
            if GROUPISH.search(k) and isinstance(v, (str, int, float)):
                values[k][inst][str(v)] += 1

    grouping = {}
    for k, per_inst in values.items():
        grouping[k] = {
            inst: {"images_with_value": sum(c.values()),
                   "distinct_values": len(c),
                   "top10": c.most_common(10)}
            for inst, c in sorted(per_inst.items(), key=lambda kv: -n_inst[kv[0]])
        }
    return {
        "images_per_institution": dict(n_inst.most_common()),
        "images_with_nonempty_groups": groups_nonempty,
        "source_meta_keys_present": {
            inst: dict(present[inst].most_common())
            for inst, _n in n_inst.most_common()},
        "grouping_like_keys": grouping,
    }


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--collated", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")

    result = {}
    print("SHARDS")
    result["shards"] = peek_shards(args.shards)

    print("\nCOCO")
    main_doc = load(os.path.join(args.collated, "seavision.json"))
    result["seavision_json"] = sample_images(
        main_doc, lambda im, d: (d.get(im.get("dataset_id")) or {})
        .get("name", "unknown"))
    del main_doc

    fn_doc = load(os.path.join(args.collated, "seavision_fathomnet.json"))
    result["fathomnet_json"] = sample_images(
        fn_doc, lambda im, d: str((im.get("source_meta") or {})
                                  .get("owner_institution") or "UNKNOWN"))
    print("  profiling FathomNet source_meta ...", flush=True)
    result["fathomnet_grouping"] = fathomnet_grouping(fn_doc)
    del fn_doc

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)

    g = result["fathomnet_grouping"]
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out) / 1e3:.0f} KB)")
    print(f"FathomNet images with non-empty groups: "
          f"{g['images_with_nonempty_groups']:,}")
    print("grouping-like source_meta keys found:",
          sorted(g["grouping_like_keys"]) or "NONE")


if __name__ == "__main__":
    main()

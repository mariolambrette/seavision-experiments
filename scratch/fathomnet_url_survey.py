#!/usr/bin/env python3
"""Read-only survey of FathomNet frame_url paths, per institution.

Question: does the URL path carry dive / deployment identity that the
ingest never parsed out? Before writing a parser, look at what the paths
actually look like -- every one of them, not a sample.

For each institution it reports the commonest path TEMPLATES (digit runs
-> '#', UUIDs -> 'U'), with crop count, distinct frames and three real
example URLs per template. Also, per path segment position, how many
distinct values occur -- a segment with a few hundred values across a
large institution is a dive/deployment candidate; one value is a
constant; one value per frame is a filename.

    python scratch/fathomnet_url_survey.py ^
        --coco D:\\marineai\\dataset\\collated\\seavision_fathomnet.json ^
        --out  D:\\marineai\\scratch\\fathomnet_url_survey.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse, unquote

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
                  r"[0-9a-f]{12}", re.I)
DIGITS = re.compile(r"\d+")


def template(path):
    return DIGITS.sub("#", UUID.sub("U", path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")

    print(f"loading {os.path.basename(args.coco)} ...", flush=True)
    with open(args.coco, encoding="utf-8") as fh:
        doc = json.load(fh)

    tmpl = defaultdict(Counter)                   # inst -> template -> crops
    tmpl_frames = defaultdict(lambda: defaultdict(set))
    examples = defaultdict(lambda: defaultdict(list))
    hosts = defaultdict(Counter)
    seg_vals = defaultdict(lambda: defaultdict(set))  # (inst, depth) -> vals
    no_url = Counter()
    frames = defaultdict(set)

    for im in doc["images"]:
        sm = im.get("source_meta") or {}
        inst = str(sm.get("owner_institution") or "UNKNOWN")
        url = sm.get("frame_url")
        uu = sm.get("fathomnet_image_uuid")
        if not url:
            no_url[inst] += 1
            continue
        p = urlparse(url)
        path = unquote(p.path)
        hosts[inst][p.netloc] += 1
        t = template(path)
        tmpl[inst][t] += 1
        tmpl_frames[inst][t].add(uu)
        frames[inst].add(uu)
        if len(examples[inst][t]) < 3 and url not in examples[inst][t]:
            examples[inst][t].append(url)
        segs = [s for s in path.split("/") if s]
        for d, s in enumerate(segs[:-1]):         # last segment = filename
            seg_vals[inst][(p.netloc, len(segs), d)].add(s)
    del doc

    out = {}
    for inst, c in sorted(tmpl.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        print(f"\n{'=' * 78}\n{inst}: {n:,} crops, {len(frames[inst]):,} "
              f"frames, {len(c):,} templates, missing url {no_url[inst]:,}")
        print("  hosts:", dict(hosts[inst].most_common(5)))
        rows = []
        for t, k in c.most_common(args.top):
            nf = len(tmpl_frames[inst][t])
            print(f"  {k:>9,} crops {nf:>8,} frames  {t}")
            for e in examples[inst][t]:
                print(f"        e.g. {e}")
            rows.append({"template": t, "crops": k, "frames": nf,
                         "examples": examples[inst][t]})
        segs = []
        for (host, depth, pos), vals in sorted(seg_vals[inst].items()):
            segs.append({"host": host, "path_depth": depth, "segment": pos,
                         "distinct": len(vals),
                         "sample": sorted(vals)[:8]})
        out[inst] = {"crops": n, "frames": len(frames[inst]),
                     "n_templates": len(c), "missing_url": no_url[inst],
                     "hosts": dict(hosts[inst]), "top_templates": rows,
                     "segment_cardinality": segs}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out) / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
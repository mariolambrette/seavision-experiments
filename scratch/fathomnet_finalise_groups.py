#!/usr/bin/env python3
"""Final FathomNet `inferred_deployment` assignment. READ-ONLY with respect
to the collation; writes one CSV for the split (and, later, a migration).

PURPOSE, stated once so it does not drift: keep near-duplicates -- the same
animal in consecutive frames, the other camera on the same drop, the same
scene minutes later -- on ONE side of any split. That is all these groups
are for. They say nothing about how hard a hold-out is.

THREE RULES, applied to the URL groups (fathomnet_url_groups.py):

1. Unsafe URL groups stay ungrouped (unmatched, unverified, MBARI stills,
   NOAA s3) -- as in fathomnet_sites.py.

2. SEFSC: co-located URL groups are MERGED. API positions are one point
   per drop, and the site run showed every same-day SC/BSC pair at one
   position (SC2+SC4 on 15, 19 and 21 March; BSC1+SC1 on 13 and 14
   March ...), consecutive station IDs at one position (761901358-362),
   and one station spelled two ways (2021_NCO-004 / 2021_NCO_004). Those
   are cameras of one drop, or repeat drops at one station: near-duplicate
   territory either way. The merge uses the 1 km position clusters from
   fathomnet_sites.py. A revisit years apart is merged too -- over-merging
   is the safe direction.

3. MBARI: dive groups whose positions spread more than --max-spread-km
   (p95 from the median) are demoted. A bad coordinate and a folder that
   is not a dive cannot be told apart, and demoting is the safe side.

Positions are NOT used to merge MBARI or NOAA dives: their dive identity is
in the URL, and nearby dives are different recordings.

    python scratch/fathomnet_finalise_groups.py ^
        --groups    D:\\marineai\\scratch\\fathomnet_groups\\fathomnet_groups.csv.gz ^
        --sites     D:\\marineai\\scratch\\fathomnet_sites\\fathomnet_sites.csv.gz ^
        --positions D:\\marineai\\scratch\\fathomnet_positions.jsonl ^
        --out       D:\\marineai\\scratch\\fathomnet_inferred_deployment.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

SEFSC = "NOAA NMFS SEFSC"


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = (math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2)
         * math.sin((lo2 - lo1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def p95(vals):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * 0.95))] if s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-spread-km", type=float, default=5.0)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")

    pos = {}
    with open(args.positions, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("latitude") is not None and d.get("longitude") is not None:
                lat, lon = float(d["latitude"]), float(d["longitude"])
                if not (lat == 0 and lon == 0):
                    pos[d["uuid"]] = (lat, lon)

    with gzip.open(args.groups, "rt", encoding="utf-8") as fh:
        grows = {r["uid"]: r for r in csv.DictReader(fh)}
    with gzip.open(args.sites, "rt", encoding="utf-8") as fh:
        srows = list(csv.DictReader(fh))
    if len(srows) != len(grows):
        sys.exit(f"row counts differ: groups {len(grows):,} vs sites "
                 f"{len(srows):,} -- were they built from the same run?")

    # rule 3: MBARI spread
    gpts = defaultdict(list)
    for s in srows:
        if s["owner_institution"] == "MBARI" and s["deployment"]:
            p = pos.get(s["fathomnet_image_uuid"])
            if p:
                gpts[s["deployment"]].append(p)
    demoted = {}
    for g, pts in gpts.items():
        if len(set(pts)) < 2:
            continue
        c = (statistics.median(p[0] for p in pts),
             statistics.median(p[1] for p in pts))
        r = p95([km(p, c) for p in pts])
        if r > args.max_spread_km:
            demoted[g] = r

    # name each SEFSC cluster after its smallest SEFSC URL group, so a
    # cluster that also caught a stray singleton from another institution
    # is still named for what it is
    site_members = defaultdict(set)
    for s in srows:
        if s["owner_institution"] == SEFSC and s["deployment"] and s["site"]:
            site_members[s["site"]].add(s["deployment"])
    site_label = {k: min(v) + (f"+{len(v) - 1}" if len(v) > 1 else "")
                  for k, v in site_members.items()}

    out_rows, before, after = [], defaultdict(set), defaultdict(set)
    crops, tel, tel_groups = Counter(), Counter(), defaultdict(set)
    for s in srows:
        inst = s["owner_institution"]
        g = grows[s["uid"]]
        dep = s["deployment"]                 # "" when unsafe (rule 1)
        source = "frame_url"
        level = s["deployment_level"]
        if dep:
            before[inst].add(dep)
        if dep and dep in demoted:
            dep, level = "", ""
        elif dep and inst == SEFSC:
            # rule 2: replace by the position cluster
            site = s["site"]
            if site:
                dep = site_label[site]
                source, level = "frame_url+api_position", "position_merged"
        if dep:
            after[inst].add(dep)
            crops[inst] += 1
            if g["teleost"] == "1":
                tel[inst] += 1
                tel_groups[inst].add(dep)
        out_rows.append([s["uid"], s["fathomnet_image_uuid"], inst, dep,
                         level, source if dep else "", g["teleost"]])

    with gzip.open(args.out, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "fathomnet_image_uuid", "owner_institution",
                    "inferred_deployment", "derived_level", "group_source",
                    "teleost"])
        w.writerows(out_rows)

    total = Counter(r[2] for r in out_rows)
    print(f"{'institution':<28}{'crops':>10}{'grouped':>10}{'groups':>8}"
          f"{'(before)':>9}{'teleost':>10}{'tel.groups':>11}")
    for inst, n in total.most_common():
        print(f"{inst[:27]:<28}{n:>10,}{crops[inst]:>10,}"
              f"{len(after[inst]):>8,}{len(before[inst]):>9,}"
              f"{tel[inst]:>10,}{len(tel_groups[inst]):>11,}")
    g_all = sum(crops.values())
    print(f"\ngrouped {g_all:,} of {len(out_rows):,} crops "
          f"({100 * g_all / len(out_rows):.1f}%)")
    print(f"\nMBARI dives demoted for spread > {args.max_spread_km} km: "
          f"{len(demoted)}")
    for g, r in sorted(demoted.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {r:>9.1f} km  {g}")
    print(f"\nwrote {args.out}\nNothing in the collation was changed.")


if __name__ == "__main__":
    main()

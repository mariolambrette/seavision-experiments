#!/usr/bin/env python3
"""Check the URL groups against API positions, and derive a SITE level.
READ-ONLY with respect to the collation.

The URL groups (fathomnet_url_groups.py) are a `deployment`-level guess.
The spatial check could only show that a group was not over-merged. The
dangerous direction -- one deployment split across two groups -- needs
evidence from outside the URLs, and per-image positions are the only such
evidence the API has (tags carry no dive id; uploads are bulk batches).

WHAT THIS DOES
  1. Joins API positions to every FathomNet frame.
  2. Within each deployment group: spread of its positions. A BRUV drop
     sits still; an ROV dive moves a few km. Large spread = suspect group.
  3. SITES: groups are linked when ANY of their positions lie within R km
     of each other, transitively. Linking on every position rather than on
     a group's median matters: a rig-day that visited stations A and B must
     share a site with any other group that visited B.
     Frames with no usable deployment group (unmatched, unverified, MBARI
     stills, NOAA s3) join as singleton nodes, so wherever they have a
     position they get a site even without a deployment.
  4. R is measured, not chosen: clusters are reported at several radii so
     the choice can be made on the table. --site-km sets the one written.

WHY A SITE LEVEL, NOT A REPAIR OF THE DEPLOYMENT LEVEL
  Two groups at one position may be one drop split in two (a leak), or the
  same station revisited a year later (not a leak at deployment level, but
  it is at site level). Positions cannot tell those apart. Writing both
  levels -- `deployment` from URLs, `site` from positions -- is what every
  other source in the collation already does (schema groups dict, plan
  4.3), and lets an experiment hold out at whichever rung it declares.

    python scratch/fathomnet_sites.py ^
        --groups    D:\\marineai\\scratch\\fathomnet_groups\\fathomnet_groups.csv.gz ^
        --positions D:\\marineai\\scratch\\fathomnet_positions.jsonl ^
        --out       D:\\marineai\\scratch\\fathomnet_sites
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

RADII = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
UNSAFE_RULES = {"mbari_stills_day", "noaa_cruise_day", "mbari_unpadded"}


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = (math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2)
         * math.sin((lo2 - lo1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))] if s else 0


def cluster(node_pts, radius):
    """node_pts: {node: set((lat, lon))}. Single linkage on positions,
    grid-hashed so it is not O(n^2) on every position."""
    dsu = DSU()
    cell = radius / 111.0                     # degrees of latitude
    grid = defaultdict(list)
    for node, pts in node_pts.items():
        dsu.find(node)
        for lat, lon in pts:
            # longitude cells scaled by latitude so a cell is ~radius wide
            c = max(math.cos(math.radians(lat)), 0.05)
            grid[(int(lat // cell), int(lon * c // cell))].append(
                (node, (lat, lon)))
    for (ci, cj), members in grid.items():
        neigh = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                neigh.extend(grid.get((ci + di, cj + dj), ()))
        for node, p in members:
            for node2, p2 in neigh:
                if node2 != node and dsu.find(node) != dsu.find(node2) \
                        and km(p, p2) <= radius:
                    dsu.union(node, node2)
    return dsu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--out", required=True, help="output DIRECTORY")
    ap.add_argument("--site-km", type=float, default=1.0)
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
                if -90 <= lat <= 90 and -180 <= lon <= 180 and \
                        not (lat == 0 and lon == 0):
                    pos[d["uuid"]] = (round(lat, 4), round(lon, 4))
    print(f"{len(pos):,} frames with a usable position")

    with gzip.open(args.groups, "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    group_rules = defaultdict(set)
    for r in rows:
        if r["group"]:
            group_rules[r["group"]].add(r["rule"])

    def deployment_safe(r):
        g = r["group"]
        if not g or r["group_level"] == "unverified":
            return False
        # a group made ONLY of weak members is unsafe; one where weak
        # members were attached to a real dive is not
        return not group_rules[g] <= UNSAFE_RULES

    for r in rows:
        r["safe"] = deployment_safe(r)
        r["node"] = r["group"] if r["safe"] else f"IMG:{r['fathomnet_image_uuid']}"
        r["pos"] = pos.get(r["fathomnet_image_uuid"])

    # ---- 1. coverage
    print(f"\n{'=' * 78}\nPOSITION COVERAGE")
    by_inst = defaultdict(list)
    for r in rows:
        by_inst[r["owner_institution"]].append(r)
    report = {"coverage": {}}
    for inst, rs in sorted(by_inst.items(), key=lambda kv: -len(kv[1])):
        n = len(rs)
        wp = sum(1 for r in rs if r["pos"])
        print(f"  {inst:<26} {n:>9,} crops  with position {100 * wp / n:5.1f}%")
        report["coverage"][inst] = [n, wp]

    # ---- 2. within-group spread
    print(f"\n{'=' * 78}\nWITHIN-GROUP SPREAD (deployment groups, >=2 distinct "
          f"positions; p95 distance from median)")
    gpts = defaultdict(list)
    grule = {}
    for r in rows:
        if r["safe"] and r["pos"]:
            gpts[r["group"]].append(r["pos"])
            grule[r["group"]] = r["rule"]
    spread = defaultdict(list)
    for g, pts in gpts.items():
        if len(set(pts)) < 2:
            spread[grule[g]].append((0.0, g))
            continue
        c = (statistics.median(p[0] for p in pts),
             statistics.median(p[1] for p in pts))
        spread[grule[g]].append((pct([km(p, c) for p in pts], .95), g))
    report["spread"] = {}
    for rule, lst in sorted(spread.items()):
        lst.sort(reverse=True)
        rads = [x[0] for x in lst]
        print(f"  {rule:<22} {len(lst):>5} groups  p50 {pct(rads, .5):6.2f} "
              f"km  p90 {pct(rads, .9):6.2f}  max {rads[0]:7.1f}  "
              f">5 km: {sum(1 for x in rads if x > 5)}")
        for rad, g in lst[:3]:
            if rad > 5:
                print(f"        {rad:7.1f} km  {g}")
        report["spread"][rule] = [[round(a, 2), b] for a, b in lst[:25]]

    # ---- 3. sites at several radii
    node_pts = defaultdict(set)
    node_inst = {}
    for r in rows:
        if r["pos"]:
            node_pts[r["node"]].add(r["pos"])
            node_inst[r["node"]] = r["owner_institution"]
    node_crops = Counter(r["node"] for r in rows)
    safe_nodes = {r["node"] for r in rows if r["safe"]}

    print(f"\n{'=' * 78}\nSITES BY RADIUS (single linkage over every position; "
          f"deployment groups with no position stay their own site)")
    print(f"  {'R km':>6} {'sites':>8} {'dep groups':>10} "
          f"{'groups sharing a site':>22} {'largest site':>14}")
    report["radius_sweep"] = []
    chosen = None
    for R in RADII:
        dsu = cluster(node_pts, R)
        site_of = {n: dsu.find(n) for n in node_pts}
        site_groups = defaultdict(set)
        site_crops = Counter()
        for n, s in site_of.items():
            if n in safe_nodes:
                site_groups[s].add(n)
            site_crops[s] += node_crops[n]
        shared = sum(len(v) for v in site_groups.values() if len(v) > 1)
        big = max(site_crops.values(), default=0)
        total = sum(site_crops.values())
        print(f"  {R:>6.2f} {len(set(site_of.values())):>8,} "
              f"{len(safe_nodes & set(node_pts)):>10,} {shared:>22,} "
              f"{100 * big / max(total, 1):>12.1f}%")
        report["radius_sweep"].append([R, len(set(site_of.values())),
                                       shared, big])
        if abs(R - args.site_km) < 1e-9:
            chosen = (dsu, site_of, site_groups)
    if chosen is None:
        dsu = cluster(node_pts, args.site_km)
        site_of = {n: dsu.find(n) for n in node_pts}
        site_groups = defaultdict(set)
        for n, s in site_of.items():
            if n in safe_nodes:
                site_groups[s].add(n)
        chosen = (dsu, site_of, site_groups)
    dsu, site_of, site_groups = chosen

    # deterministic site names: smallest member node + count
    members = defaultdict(set)
    for n, s in site_of.items():
        members[s].add(n)
    site_name = {s: f"SITE:{min(ms)}" + (f"+{len(ms) - 1}" if len(ms) > 1
                                         else "")
                 for s, ms in members.items()}

    # ---- 4. what the site level buys, per institution, at the chosen R
    print(f"\n{'=' * 78}\nAT R = {args.site_km} km, per institution")
    print(f"  {'institution':<26} {'dep-safe crops':>14} {'site crops':>11} "
          f"{'recovered':>10} {'dep groups':>10} {'sites':>7} "
          f"{'teleost sites':>13}")
    report["per_inst"] = {}
    for inst, rs in sorted(by_inst.items(), key=lambda kv: -len(kv[1])):
        dep = sum(1 for r in rs if r["safe"])
        has_site = [r for r in rs if (r["node"] in site_of) or r["safe"]]
        rec = sum(1 for r in has_site if not r["safe"])
        dgroups = {r["group"] for r in rs if r["safe"]}
        sites = {site_of.get(r["node"], r["node"]) for r in has_site}
        tsites = {site_of.get(r["node"], r["node"]) for r in has_site
                  if r["teleost"] == "1"}
        print(f"  {inst[:26]:<26} {dep:>14,} {len(has_site):>11,} "
              f"{rec:>10,} {len(dgroups):>10,} {len(sites):>7,} "
              f"{len(tsites):>13,}")
        report["per_inst"][inst] = {"deployment_safe": dep,
                                    "with_site": len(has_site),
                                    "recovered_by_site": rec,
                                    "deployment_groups": len(dgroups),
                                    "sites": len(sites),
                                    "teleost_sites": len(tsites)}

    multi = sorted(((len(v), site_name[s], sorted(v)[:6])
                    for s, v in site_groups.items() if len(v) > 1),
                   reverse=True)
    print(f"\n  sites holding more than one deployment group: {len(multi):,}")
    for n, name, eg in multi[:12]:
        print(f"    {n:>4} groups  {name}\n           e.g. {', '.join(eg)}")
    report["multi_group_sites"] = multi[:200]

    # ---- write
    os.makedirs(args.out)
    path = os.path.join(args.out, "fathomnet_sites.csv.gz")
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "fathomnet_image_uuid", "owner_institution",
                    "deployment", "deployment_level", "site", "teleost"])
        for r in rows:
            s = site_of.get(r["node"])
            if s is None and r["safe"]:
                # a safe group with no position anywhere is its own site
                s_name = f"SITE:{r['node']}"
            else:
                s_name = site_name.get(s, "") if s else ""
            w.writerow([r["uid"], r["fathomnet_image_uuid"],
                        r["owner_institution"],
                        r["group"] if r["safe"] else "",
                        r["group_level"] if r["safe"] else "",
                        s_name, r["teleost"]])
    with open(os.path.join(args.out, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"\nwrote {path}\n      {os.path.join(args.out, 'report.json')}")
    print("Nothing in the collation was changed.")


if __name__ == "__main__":
    main()

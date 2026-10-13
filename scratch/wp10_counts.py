#!/usr/bin/env python3
"""WP10: what the locked test set can support, before anything is scored.

READ-ONLY. Reads configs/evaluate/wp10.yaml and applies its rules to the
locked test records (lock + heldout), counting only -- no embedding is read.

For every pair, direction (queries from A, references from B), level, arm and
size matching (matched / unmatched), and every k: how many taxa are testable,
and the chance level that implies. Plus the fixed taxon sets (testable at
k = 50 and at k = 100), per-taxon detail (units, deployments, matched share,
query units), and OzFish survey composition per taxon (decision 11), flagging
taxa whose units come mostly from survey E.

Units (decision 13): one representative crop per unit -- OzFish one camera per
deployment x category (more crops; left if tied), FathomNet one crop per
collapsed unit (SEFSC video-second, else frame; first by key), FishWIO and
yolo-bruv every crop.

A taxon is testable for queries from A, references from B, at k when
  A: units >= k + q_min and units outside its largest deployment >= q_min
  B: units >= k in >= ref_min_groups deployments
(decision 4). Size matching (decision 6, corrected 13 October): per taxon,
both sources are subsampled to the same SHAPE of short-side distribution --
per bin the smaller of the two sources' shares, renormalised -- each keeping
as many units as that shape allows, chosen by a seeded hash. Counts are not
equalised; the larger source keeps its larger total.

    python scratch\\wp10_counts.py --config configs\\evaluate\\wp10.yaml ^
        --out D:\\marineai\\scratch\\wp10_counts_v1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import yaml

SEFSC = "NOAA NMFS SEFSC"
SEFSC_SECOND = re.compile(
    r"/([^/]+)/[^/]+\.(?:mp4|avi)\.(\d\d)\.(\d\d)\.(\d\d)\.\d+\.jpg$", re.I)
ORDER = {"order": 0, "family": 1, "genus": 2, "species": 3}


def h(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def taxa_of(cat):
    """level -> name, from the lineage (never the name), only for levels at
    or above the category's own rank. Species uses the accepted name."""
    lin = cat.get("lineage") or {}
    rank = str(cat.get("rank", "")).lower()
    out = {lv: lin[lv] for lv in ("order", "family", "genus") if lin.get(lv)}
    if rank == "species":
        out["species"] = cat.get("name")
    if rank in ORDER:
        out = {lv: v for lv, v in out.items() if ORDER[lv] <= ORDER[rank]}
    return out


def read_records(cfg):
    """-> list of dicts, one per locked-test annotation record."""
    keep = {}
    with gzip.open(os.path.join(cfg["split_dir"], "split_crops.csv.gz"),
                   "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in cfg["use_splits"]:
                keep[r["key"]] = (r["source"], r["group"], r["split"])
    recs = []
    for path in cfg["coco"]:
        print(f"  loading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        cats = {c["id"]: c for c in doc["categories"]}
        by_image = defaultdict(list)
        for an in doc["annotations"]:
            by_image[an["image_id"]].append(an)
        for im in doc["images"]:
            uid = os.path.splitext(im["file_name"])[0]
            sm = im.get("source_meta") or {}
            g = im.get("groups") or {}
            for an in by_image.get(im["id"], []):
                c = cats.get(an.get("category_id"))
                if not c:
                    continue
                key = f"{uid}-a{an['id']}" \
                    if im.get("crop_provenance") == "frame" else uid
                if key not in keep:
                    if im.get("crop_provenance") != "frame":
                        break
                    continue
                src, grp, split = keep[key]
                w, hgt = an["bbox"][2], an["bbox"][3]
                if src == "fathomnet":
                    m = SEFSC_SECOND.search(sm.get("frame_url") or "") \
                        if sm.get("owner_institution") == SEFSC else None
                    unit = (m[1], m[2], m[3], m[4]) if m \
                        else sm.get("fathomnet_image_uuid")
                else:
                    unit = None
                recs.append({
                    "key": key, "src": src, "grp": grp, "split": split,
                    "cid": an["category_id"], "taxa": taxa_of(c),
                    "short": min(w, hgt),
                    "camera": "L" if str(sm.get("camera") or "").upper()
                    == "L" else "R",
                    "survey": g.get("survey") or "", "unit": unit})
                if im.get("crop_provenance") != "frame":
                    break
        del doc, by_image
    missing = len(keep) - len({r["key"] for r in recs})
    print(f"  locked-test records with a category: {len(recs):,} "
          f"(of {len(keep):,} keys; {missing:,} have no category)")
    return recs


def representatives(recs):
    """Decision 13: one representative crop per unit."""
    cam = defaultdict(Counter)                 # (dep, cid) -> camera counts
    for r in recs:
        if r["src"] == "ozfish":
            cam[(r["grp"], r["cid"])][r["camera"]] += 1
    chosen = {k: ("L" if c["L"] >= c["R"] else "R") for k, c in cam.items()}
    seen_unit = set()
    out = []
    for r in sorted(recs, key=lambda r: r["key"]):
        if r["src"] == "ozfish":
            if r["camera"] != chosen[(r["grp"], r["cid"])]:
                continue
        elif r["src"] == "fathomnet":
            u = (r["grp"], r["unit"], r["cid"])
            if u in seen_unit:
                continue
            seen_unit.add(u)
        out.append(r)
    return out


def ok_query(groups, k, q):
    tot = sum(groups.values())
    return bool(groups) and tot >= k + q and tot - max(groups.values()) >= q


def ok_ref(groups, k, min_groups):
    return sum(groups.values()) >= k and \
        sum(1 for v in groups.values() if v > 0) >= min_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    os.makedirs(args.out)
    ks, q, mg = cfg["k"], cfg["q_min"], cfg["ref_min_groups"]
    bins = cfg["size_matching"]["short_side_bins"]
    seed = cfg["seed"]
    report = {"config": cfg, "inputs": {
        "split_crops": sha256(os.path.join(cfg["split_dir"],
                                           "split_crops.csv.gz")),
        **{os.path.basename(p): sha256(p) for p in cfg["coco"]}}}

    recs = read_records(cfg)
    units = representatives(recs)
    print(f"  representative units: "
          f"{dict(Counter(r['src'] for r in units))} "
          f"(from {dict(Counter(r['src'] for r in recs))} records)")

    def bin_of(s):
        for i in range(len(bins) - 1):
            if bins[i] <= s < bins[i + 1]:
                return i
        return len(bins) - 2

    # pools: (src, level, taxon) -> list of units; species-only variant too
    pool = defaultdict(list)
    sp_pool = defaultdict(list)
    for r in units:
        for lv, t in r["taxa"].items():
            pool[(r["src"], lv, t)].append(r)
            if "species" in r["taxa"]:
                sp_pool[(r["src"], lv, t)].append(r)

    def groups_of(us):
        return Counter(u["grp"] for u in us)

    def match(ua, ub):
        """Decision 6 (corrected): match the SHAPE of the short-side
        distribution, not the counts. Common shape p_i = min(share_A_i,
        share_B_i), renormalised; each source keeps as many units as that
        shape allows (its binding bin limits it), chosen per bin by hash. The
        larger source keeps its larger total."""
        ba, bb = defaultdict(list), defaultdict(list)
        for u in ua:
            ba[bin_of(u["short"])].append(u)
        for u in ub:
            bb[bin_of(u["short"])].append(u)
        if not ua or not ub:
            return [], []
        na, nb = len(ua), len(ub)
        m = {i: min(len(ba[i]) / na, len(bb[i]) / nb)
             for i in set(ba) | set(bb)}
        tot = sum(m.values())
        if tot == 0:
            return [], []
        p = {i: v / tot for i, v in m.items() if v > 0}
        out = []
        for bins_s in (ba, bb):
            n = min(len(bins_s[i]) / p[i] for i in p)
            kept = []
            for i in p:
                take = min(len(bins_s[i]), int(n * p[i] + 1e-9))
                kept += sorted(bins_s[i], key=lambda u: h(seed, u["key"]))[:take]
            out.append(kept)
        return out[0], out[1]

    rows, detail = [], []
    pairs = [tuple(p) for p in cfg["pairs"]]
    shared_pairs = [tuple(p) for p in cfg["arms"]["shared_children"]["pairs"]]
    for a, b in pairs:
        for qsrc, rsrc in ((a, b), (b, a)):           # queries, references
            for lv in cfg["levels"]:
                arms = [("whole_taxon", pool)]
                if (a, b) in shared_pairs or (b, a) in shared_pairs:
                    if lv != "species":
                        arms.append(("shared_children", None))
                for arm, src_pool in arms:
                    taxa = {t for (s, l, t) in pool if s == qsrc and l == lv} \
                        & {t for (s, l, t) in pool if s == rsrc and l == lv}
                    for matching in ("matched", "unmatched"):
                        per_k = Counter()
                        sets = defaultdict(set)
                        for t in sorted(taxa):
                            if arm == "whole_taxon":
                                uq, ur = pool[(qsrc, lv, t)], pool[(rsrc, lv, t)]
                            else:
                                spq = {u["taxa"]["species"]
                                       for u in sp_pool[(qsrc, lv, t)]}
                                spr = {u["taxa"]["species"]
                                       for u in sp_pool[(rsrc, lv, t)]}
                                both = spq & spr
                                uq = [u for u in sp_pool[(qsrc, lv, t)]
                                      if u["taxa"]["species"] in both]
                                ur = [u for u in sp_pool[(rsrc, lv, t)]
                                      if u["taxa"]["species"] in both]
                            n_q0 = len(uq)
                            if matching == "matched":
                                uq, ur = match(uq, ur)
                            gq, gr = groups_of(uq), groups_of(ur)
                            for k in ks:
                                if ok_query(gq, k, q) and ok_ref(gr, k, mg):
                                    per_k[k] += 1
                                    sets[k].add(t)
                            if arm == "whole_taxon" and ok_query(gq, ks[0], q) \
                                    and ok_ref(gr, ks[0], mg):
                                detail.append([qsrc, rsrc, lv, matching, t,
                                               len(uq), len(gq), len(ur),
                                               len(gr),
                                               round(len(uq) / n_q0, 3)
                                               if n_q0 else "",
                                               max((kk for kk in ks if t in
                                                    sets[kk]), default="")])
                        for k in ks:
                            n = per_k[k]
                            rows.append([qsrc, rsrc, lv, arm, matching,
                                         "per_k", k, n,
                                         round(1 / n, 4) if n else ""])
                        for fixed in (50, 100):
                            if fixed in ks:
                                fs = sets[fixed]
                                rows.append([qsrc, rsrc, lv, arm, matching,
                                             f"fixed_k{fixed}", fixed,
                                             len(fs),
                                             round(1 / len(fs), 4)
                                             if fs else ""])

    with open(os.path.join(args.out, "wp10_counts.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_source", "reference_source", "level", "arm",
                    "matching", "taxon_set", "k", "taxa", "chance"])
        w.writerows(rows)
    with open(os.path.join(args.out, "wp10_taxa.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_source", "reference_source", "level", "matching",
                    "taxon", "query_units", "query_deployments", "ref_units",
                    "ref_deployments", "matched_share_of_query",
                    "largest_k_testable"])
        w.writerows(detail)

    # OzFish survey composition per taxon (decision 11)
    comp = defaultdict(Counter)
    for r in units:
        if r["src"] == "ozfish":
            for lv, t in r["taxa"].items():
                comp[(lv, t)][r["survey"]] += 1
    tested = {(d[2], d[4]) for d in detail
              if "ozfish" in (d[0], d[1])}
    flag = []
    with open(os.path.join(args.out, "ozfish_survey_composition.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "taxon", "A", "B", "E", "G", "E_share",
                    "testable_somewhere"])
        for (lv, t), c in sorted(comp.items()):
            tot = sum(c.values())
            e = round(c["E"] / tot, 3) if tot else 0
            w.writerow([lv, t, c["A"], c["B"], c["E"], c["G"], e,
                        (lv, t) in tested])
            if (lv, t) in tested and e > 0.5:
                flag.append((lv, t, e))

    # ---- printed summary
    print(f"\n{'=' * 78}\nTESTABLE TAXA, whole-taxon arm, per-k sets "
          f"(matched / unmatched); queries <- references")
    idx = defaultdict(dict)
    for r in rows:
        if r[3] == "whole_taxon" and r[5] == "per_k":
            idx[(r[0], r[1], r[2])][(r[4], r[6])] = r[7]
    hdr = "".join(f"{'k=' + str(k):>12}" for k in ks)
    print(f"  {'queries <- refs':<28}{'level':<9}{hdr}")
    for (qs, rs, lv), d in idx.items():
        cells = "".join(f"{str(d[('matched', k)]) + '/' + str(d[('unmatched', k)]):>12}"
                        for k in ks)
        print(f"  {qs + ' <- ' + rs:<28}{lv:<9}{cells}")
    print(f"\n  shared-children control (OzFish x FishWIO), matched, per-k:")
    for r in rows:
        if r[3] == "shared_children" and r[5] == "per_k" \
                and r[4] == "matched" and r[6] in (20, 100):
            print(f"    {r[0]} <- {r[1]:<10}{r[2]:<9}k={r[6]:<5}{r[7]} taxa")
    print(f"\n  OzFish taxa testable somewhere with > 50% of units from "
          f"survey E: {len(flag)}")
    for lv, t, e in flag[:20]:
        print(f"    {lv:<9}{t:<40}E share {e}")
    report["survey_E_flags"] = flag
    with open(os.path.join(args.out, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

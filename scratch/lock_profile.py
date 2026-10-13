#!/usr/bin/env python3
"""Profile WP10's shared taxa and a candidate locked test set. READ-ONLY.

Chooses nothing and freezes nothing. It answers two questions before any
main-set result is read, so that the answers can be decided blind:

  A. WP10 -- which taxa can be compared ACROSS sources? For every pair of
     sources and every level (order, family, genus, species): how many taxa
     are testable in BOTH sources, i.e. hold >= N units in >= 2 groups in
     each (two groups, so the within-source cross-deployment baseline exists
     too). Swept over N. Plus, per shared taxon, how many crops survive
     matching crop size between the two sources (WP10 is size-matched).
  B. The locked test set -- what would it cost and what would it hold?
     OzFish split by deployment (random, stratified by survey, several
     seeds) at a grid of fractions; FishWIO and yolo-bruv whole (test only);
     FathomNet split by inferred group, stratified by institution; and a
     grid of genera held out of training entirely (unseen-species test).

Everything runs on the MAIN set: the 93 dev deployments and the 43 dev
FathomNet groups (configs/splits/) are removed first.

UNITS (as wp8_profile.py, so figures are comparable)
  OzFish     individuals: per deployment x taxon, max(left, right camera
             crops). Stereo pairs merged.
  FishWIO    crops. Tracks not collapsed, so units OVERSTATE individuals.
  yolo-bruv  crops.
  FathomNet  SEFSC: distinct (video, whole second); elsewhere distinct
             frame images. Groups are INFERRED; always reported separately.
  Size matching counts crops, not units: it measures overlap of the two
  sources' crop-size distributions, which is a property of crops.

USABLE (testable within one partition) at k: units >= k + qmin and units
outside the largest group >= qmin -- the WP8 rule, so references and
queries can come from different groups.

SELF-CHECKS, printed first; if either fails, nothing below is trustworthy:
  0a. the WP6 gate from the COCO file: 454/78, 304/57, 243/48;
  0b. the same gate after removing the dev deployments: 436/74, 304/57,
      243/48 (dev_selection.json's gate_remainder).

    python scratch\\lock_profile.py ^
      --coco-main D:\\marineai\\dataset\\collated\\seavision.json ^
      --coco-fn   D:\\marineai\\dataset\\collated\\seavision_fathomnet.json ^
      --fn-groups D:\\marineai\\dataset\\collated\\fathomnet_inferred_deployment.csv.gz ^
      --dev-dir   configs\\splits ^
      --out       D:\\marineai\\scratch\\lock_profile_v1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

LEVELS = ["order", "family", "genus", "species"]
GATE_CELLS = [(10, 2), (50, 2), (100, 3)]
WP6_EXPECTED = {(10, 2): (454, 78), (50, 2): (304, 57), (100, 3): (243, 48)}
DEV_EXPECTED = {(10, 2): (436, 74), (50, 2): (304, 57), (100, 3): (243, 48)}
SEFSC = "NOAA NMFS SEFSC"
SEFSC_SECOND = re.compile(
    r"/([^/]+)/[^/]+\.(?:mp4|avi)\.(\d\d)\.(\d\d)\.(\d\d)\.\d+\.jpg$", re.I)
# crop short side, px; log-spaced so a bin is a similar relative size range
SIZE_EDGES = [0, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 10 ** 9]
SOURCES = ["ozfish", "fishwio", "yolo-bruv", "fathomnet"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    print(f"  loading {os.path.basename(path)} ...", flush=True)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def size_bin(s):
    for i in range(len(SIZE_EDGES) - 1):
        if SIZE_EDGES[i] <= s < SIZE_EDGES[i + 1]:
            return i
    return len(SIZE_EDGES) - 2


def usable(by_group, k, qmin):
    if not by_group:
        return False
    tot = sum(by_group.values())
    return tot >= k + qmin and tot - max(by_group.values()) >= qmin


def taxa_of(cat):
    """level -> name for every level the category resolves to. From the
    lineage, never the name (WP6a), except the species itself."""
    lin = cat.get("lineage") or {}
    rank = str(cat.get("rank", "")).lower()
    out = {}
    for lv in ("order", "family", "genus"):
        if lin.get(lv):
            out[lv] = lin[lv]
    if rank == "species":
        out["species"] = cat.get("name")
    # a category above a level must not claim that level
    order_rank = {"order": 0, "family": 1, "genus": 2, "species": 3}
    if rank in order_rank:
        out = {lv: v for lv, v in out.items()
               if order_rank[lv] <= order_rank[rank]}
    return out


# --------------------------------------------------------------- records

class Rec:
    """One annotation: source, group, survey/institution, taxa, short side,
    and a unit key (what counts as one independent observation)."""
    __slots__ = ("src", "grp", "stratum", "taxa", "short", "unit", "cid",
                 "rank", "camera")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def read_main(path):
    doc = load(path)
    cats = {c["id"]: c for c in doc["categories"]}
    src_of = {d["id"]: d["name"] for d in doc["datasets"]}
    img = {i["id"]: i for i in doc["images"]}
    recs, names = [], Counter()
    for an in doc["annotations"]:
        c = cats.get(an.get("category_id"))
        if not c:
            continue
        im = img[an["image_id"]]
        src = src_of.get(im["dataset_id"], "?")
        names[src] += 1
        g = im.get("groups") or {}
        sm = im.get("source_meta") or {}
        w, h = an["bbox"][2], an["bbox"][3]
        recs.append(Rec(src=src, grp=g.get("deployment"),
                        stratum=g.get("survey") or "",
                        taxa=taxa_of(c), short=min(w, h), unit=None,
                        cid=an["category_id"],
                        rank=str(c.get("rank", "")).lower(),
                        camera=str(sm.get("camera") or "").upper()))
    print(f"    sources in {os.path.basename(path)}: {dict(names)}")
    del doc, img
    return recs, cats


def read_fn(path, fn_groups):
    g_of, inst_of = {}, {}
    with gzip.open(fn_groups, "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["inferred_deployment"]:
                g_of[r["uid"]] = r["inferred_deployment"]
                inst_of[r["inferred_deployment"]] = r["owner_institution"]
    doc = load(path)
    cats = {c["id"]: c for c in doc["categories"]}
    img = {i["id"]: i for i in doc["images"]}
    recs = []
    for an in doc["annotations"]:
        c = cats.get(an.get("category_id"))
        if not c:
            continue
        im = img[an["image_id"]]
        g = g_of.get(os.path.splitext(im["file_name"])[0])
        if not g:
            continue
        sm = im.get("source_meta") or {}
        m = SEFSC_SECOND.search(sm.get("frame_url") or "") \
            if sm.get("owner_institution") == SEFSC else None
        unit = (m[1], m[2], m[3], m[4]) if m \
            else sm.get("fathomnet_image_uuid")
        w, h = an["bbox"][2], an["bbox"][3]
        recs.append(Rec(src="fathomnet", grp=g, stratum=inst_of[g],
                        taxa=taxa_of(c), short=min(w, h), unit=unit,
                        cid=an["category_id"],
                        rank=str(c.get("rank", "")).lower(), camera=""))
    del doc, img
    return recs


def units_by(recs, level):
    """-> {(src, taxon): Counter(group -> units)}, using each source's unit
    rule. OzFish: max(L, R) per group x taxon. FathomNet: distinct unit
    keys. Others: crops."""
    oz = defaultdict(lambda: [Counter(), Counter()])     # (taxon, grp) -> L, R
    fn = defaultdict(set)                                # (taxon, grp) -> units
    out = defaultdict(Counter)
    for r in recs:
        t = r.taxa.get(level)
        if not t or not r.grp:
            continue
        if r.src == "ozfish":
            oz[(t, r.grp)][0 if r.camera == "L" else 1][0] += 1
        elif r.src == "fathomnet":
            fn[(t, r.grp)].add(r.unit)
        else:
            out[(r.src, t)][r.grp] += 1
    for (t, g), (L, R) in oz.items():
        out[("ozfish", t)][g] = max(L[0], R[0])
    for (t, g), s in fn.items():
        out[("fathomnet", t)][g] = len(s)
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-main", required=True)
    ap.add_argument("--coco-fn", required=True)
    ap.add_argument("--fn-groups", required=True)
    ap.add_argument("--dev-dir", required=True,
                    help="folder holding dev_deployments.csv and "
                         "fn_dev_groups.csv (configs/splits)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared-n", default="10,20,50,100",
                    help="WP10: units per source a shared taxon needs")
    ap.add_argument("--lock-fractions", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--holdout-genera", default="0,3,5,10")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", default="20,100",
                    help="reference sizes for 'usable'")
    ap.add_argument("--query-min", type=int, default=10)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    os.makedirs(args.out)
    shared_n = [int(x) for x in args.shared_n.split(",")]
    fracs = [float(x) for x in args.lock_fractions.split(",")]
    n_hold = [int(x) for x in args.holdout_genera.split(",")]
    ks = [int(x) for x in args.k.split(",")]
    qmin = args.query_min
    report = {"args": vars(args), "inputs": {}}
    for p in (args.coco_main, args.coco_fn, args.fn_groups):
        report["inputs"][os.path.basename(p)] = sha256(p)
        print(f"sha256 {os.path.basename(p)} {report['inputs'][os.path.basename(p)]}")

    dev = set()
    with open(os.path.join(args.dev_dir, "dev_deployments.csv"),
              newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            dev.add((r["source"], r["deployment"]))
    with open(os.path.join(args.dev_dir, "fn_dev_groups.csv"),
              newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            dev.add(("fathomnet", r["deployment"]))
    print(f"dev groups to remove: {Counter(s for s, _ in dev)}")

    main_recs, cats = read_main(args.coco_main)
    unknown = set(r.src for r in main_recs) - set(SOURCES)
    if unknown:
        sys.exit(f"unexpected source names {unknown}; expected {SOURCES}")

    # ---- 0. self-checks
    print(f"\n{'=' * 78}\n0. SELF-CHECKS")
    genus_of = {cid: (c.get("lineage") or {}).get("genus")
                for cid, c in cats.items()}

    def gate(recs):
        sg = defaultdict(Counter)
        for r in recs:
            if r.grp and r.rank == "species":
                sg[r.cid][(r.src, r.grp)] += 1
        res = {}
        for mc, mg in GATE_CELLS:
            q = [s for s, g in sg.items()
                 if sum(g.values()) >= mc and len(g) >= mg]
            pg = Counter(genus_of[s] for s in q if genus_of.get(s))
            res[(mc, mg)] = (len(q), sum(1 for v in pg.values() if v >= 2))
        return res

    ok = True
    for label, recs, exp in (
            ("0a whole collation", main_recs, WP6_EXPECTED),
            ("0b dev removed", [r for r in main_recs
                                if (r.src, r.grp) not in dev], DEV_EXPECTED)):
        got = gate(recs)
        for cell in GATE_CELLS:
            flag = "ok" if got[cell] == exp[cell] else "MISMATCH"
            ok &= flag == "ok"
            print(f"  {label}: >={cell[0]}/{cell[1]} -> {got[cell]} "
                  f"(expected {exp[cell]}) {flag}")
    report["self_check_ok"] = ok
    if not ok:
        print("  ! self-check failed: stop and resolve before reading below")

    main_recs = [r for r in main_recs if (r.src, r.grp) not in dev]
    fn_recs = [r for r in read_fn(args.coco_fn, args.fn_groups)
               if ("fathomnet", r.grp) not in dev]
    recs = main_recs + fn_recs
    print(f"  main-set annotations: "
          f"{dict(Counter(r.src for r in recs))} (grouped only for FathomNet)")

    # ---- A. shared taxa for WP10
    print(f"\n{'=' * 78}\nA. WP10: TAXA TESTABLE IN BOTH SOURCES "
          f"(>= N units in >= 2 groups, each source)")
    U = {lv: units_by(recs, lv) for lv in LEVELS}
    pairs = [(a, b) for i, a in enumerate(SOURCES) for b in SOURCES[i + 1:]]
    rows_a = []
    print(f"  {'pair':<24}{'level':<9}" +
          "".join(f"{'N=' + str(n):>8}" for n in shared_n))
    for a, b in pairs:
        for lv in LEVELS:
            taxa_a = {t for (s, t) in U[lv] if s == a}
            taxa_b = {t for (s, t) in U[lv] if s == b}
            cnt = []
            for n in shared_n:
                ok_t = [t for t in taxa_a & taxa_b
                        if all(sum(U[lv][(s, t)].values()) >= n
                               and len(U[lv][(s, t)]) >= 2 for s in (a, b))]
                cnt.append(len(ok_t))
            rows_a.append([a, b, lv] + cnt)
            print(f"  {a + ' x ' + b:<24}{lv:<9}" +
                  "".join(f"{c:>8}" for c in cnt))
    with open(os.path.join(args.out, "wp10_shared_counts.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_a", "source_b", "level"] +
                   [f"n_ge_{n}" for n in shared_n])
        w.writerows(rows_a)

    # per shared taxon detail, at the smallest N, with size matching
    nmin = min(shared_n)
    size_bins = defaultdict(Counter)            # (src, lv, taxon) -> bin
    for r in recs:
        if r.grp:
            for lv, t in r.taxa.items():
                size_bins[(r.src, lv, t)][size_bin(r.short)] += 1
    detail = []
    for a, b in pairs:
        for lv in LEVELS:
            for t in sorted({t for (s, t) in U[lv] if s == a}
                            & {t for (s, t) in U[lv] if s == b}):
                ua, ub = U[lv][(a, t)], U[lv][(b, t)]
                if not all(sum(u.values()) >= nmin and len(u) >= 2
                           for u in (ua, ub)):
                    continue
                ba, bb = size_bins[(a, lv, t)], size_bins[(b, lv, t)]
                matched = sum(min(ba[i], bb[i]) for i in set(ba) | set(bb))
                detail.append([a, b, lv, t, sum(ua.values()), len(ua),
                               sum(ub.values()), len(ub),
                               sum(ba.values()), sum(bb.values()), matched,
                               round(matched / min(sum(ba.values()),
                                                   sum(bb.values())), 3)])
    with open(os.path.join(args.out, "wp10_shared_taxa.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_a", "source_b", "level", "taxon",
                    "units_a", "groups_a", "units_b", "groups_b",
                    "crops_a", "crops_b", "size_matched_crops",
                    "matched_share_of_smaller"])
        w.writerows(detail)
    print(f"\n  size matching (per taxon, crops in common short-side bins; "
          f"taxa at N={nmin}): median share of the smaller source kept")
    for a, b in pairs:
        for lv in LEVELS:
            v = sorted(d[11] for d in detail
                       if d[0] == a and d[1] == b and d[2] == lv)
            if v:
                print(f"  {a + ' x ' + b:<24}{lv:<9}{len(v):>5} taxa  "
                      f"median {v[len(v) // 2]:.2f}  "
                      f"min {v[0]:.2f}")
    report["A"] = {"pairs": rows_a}

    # ---- B. locked test set
    print(f"\n{'=' * 78}\nB. LOCKED TEST SET (main set; nothing frozen)")
    sp = U["species"]

    def summarise(src, groups, holdout=frozenset()):
        """usable species/genera in `groups` of `src` at each k, plus the
        units left for training outside them (held-out genera excluded)."""
        out = {}
        for k in ks:
            us = [t for (s, t) in sp if s == src
                  and usable(Counter({g: n for g, n in sp[(s, t)].items()
                                      if g in groups}), k, qmin)]
            gen = Counter(genus_lookup.get(t) for t in us
                          if genus_lookup.get(t))
            out[f"species_k{k}"] = len(us)
            out[f"genera2_k{k}"] = sum(1 for v in gen.values() if v >= 2)
        train = 0
        for (s, t), d in sp.items():
            if s != src or genus_lookup.get(t) in holdout:
                continue
            train += sum(n for g, n in d.items() if g not in groups)
        out["train_species_units"] = train
        return out

    genus_lookup = {}
    for r in recs:
        if "species" in r.taxa and "genus" in r.taxa:
            genus_lookup[r.taxa["species"]] = r.taxa["genus"]

    # whole sources, test only
    for src in ("fishwio", "yolo-bruv"):
        groups = {g for (s, t), d in sp.items() if s == src for g in d}
        res = summarise(src, groups)
        print(f"  {src} (whole, test only): {len(groups)} groups  " +
              "  ".join(f"{k}={v}" for k, v in res.items()
                        if k != "train_species_units"))
        report.setdefault("B_whole", {})[src] = {"groups": len(groups), **res}

    # OzFish: random deployments, stratified by survey
    survey_of = {}
    for r in main_recs:
        if r.src == "ozfish" and r.grp:
            survey_of[r.grp] = r.stratum
    oz_deps = sorted(survey_of)
    by_survey = defaultdict(list)
    for d in oz_deps:
        by_survey[survey_of[d]].append(d)
    print(f"  OzFish main deployments by survey: "
          f"{ {s: len(v) for s, v in sorted(by_survey.items())} }")

    def draw(strata, frac, seed):
        rng = random.Random(seed)
        pick = set()
        for s in sorted(strata):
            v = sorted(strata[s])
            rng.shuffle(v)
            pick.update(v[:round(frac * len(v))])
        return pick

    rows_b = []
    print(f"\n  OzFish locked test by fraction of deployments "
          f"(median [min] over {args.seeds} seeds)")
    for frac in fracs:
        res = [summarise("ozfish", draw(by_survey, frac, s))
               for s in range(args.seeds)]
        keys = list(res[0])
        med = {k: sorted(r[k] for r in res)[len(res) // 2] for k in keys}
        mn = {k: min(r[k] for r in res) for k in keys}
        rows_b.append(["ozfish", frac, 0] + [med[k] for k in keys]
                      + [mn[k] for k in keys])
        print(f"   frac {frac:.1f}: " + "  ".join(
            f"{k}={med[k]} [{mn[k]}]" for k in keys))

    # held-out genera: genera with >= 2 species usable at the smallest k
    # across ALL OzFish main deployments (so they are testable as unseen)
    kmin = min(ks)
    all_oz = set(oz_deps)
    cand_sp = [t for (s, t) in sp if s == "ozfish"
               and usable(sp[(s, t)], kmin, qmin)]
    gc = Counter(genus_lookup.get(t) for t in cand_sp if genus_lookup.get(t))
    cand_gen = sorted(g for g, n in gc.items() if n >= 2)
    print(f"\n  genera eligible to hold out of training (>= 2 species "
          f"usable at k={kmin} over all OzFish main): {len(cand_gen)}")
    oz_total = sum(sum(d.values()) for (s, t), d in sp.items()
                   if s == "ozfish")
    fw_species = {t for (s, t) in sp if s == "fishwio"}
    print(f"  holding out H genera entirely (independent of the deployment "
          f"fraction; median over seeds): unseen species usable, OzFish "
          f"species units moved from training to test")
    for h in n_hold:
        unseen, lost, fw_unseen = [], [], []
        for s in range(args.seeds):
            rng = random.Random(1000 + s)
            hold = set(rng.sample(cand_gen, min(h, len(cand_gen))))
            unseen.append(sum(1 for t in cand_sp
                              if genus_lookup.get(t) in hold))
            fw_unseen.append(sum(1 for t in fw_species
                                 if genus_lookup.get(t) in hold))
            lost.append(sum(sum(d.values()) for (src, t), d in sp.items()
                            if src == "ozfish"
                            and genus_lookup.get(t) in hold))
        m = lambda v: sorted(v)[len(v) // 2]
        print(f"   H={h:>3}: unseen OzFish species {m(unseen)}, "
              f"of which also in FishWIO {m(fw_unseen)}, units moved out of "
              f"training {m(lost):,} of {oz_total:,} "
              f"({m(lost) / oz_total:.1%})")
        rows_b.append(["ozfish_holdout_genera", "", h, m(unseen),
                       m(fw_unseen), m(lost), oz_total])

    # FathomNet: inferred groups, stratified by institution
    fn_inst = defaultdict(set)
    for r in fn_recs:
        fn_inst[r.stratum].add(r.grp)
    print(f"\n  FathomNet locked test by fraction of inferred groups, "
          f"stratified by institution (median [min]; collapsed units; "
          f"reported separately)")
    for frac in fracs:
        res = [summarise("fathomnet", draw(fn_inst, frac, s))
               for s in range(args.seeds)]
        keys = list(res[0])
        med = {k: sorted(r[k] for r in res)[len(res) // 2] for k in keys}
        mn = {k: min(r[k] for r in res) for k in keys}
        rows_b.append(["fathomnet", frac, 0] + [med[k] for k in keys]
                      + [mn[k] for k in keys])
        print(f"   frac {frac:.1f}: " + "  ".join(
            f"{k}={med[k]} [{mn[k]}]" for k in keys))

    with open(os.path.join(args.out, "lock_profile.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["what", "fraction", "holdout_genera", "values..."])
        w.writerows(rows_b)
    report["B_rows"] = rows_b
    with open(os.path.join(args.out, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cut the locked test set and the retraining pool from the main set.

Decided 13 October 2026 (schedule WP10 prerequisites; plan to follow):

  dev        unchanged from dev_v1. Every choice is made here, including
             retraining settings (it is the validation set).
  train      the retraining pool. Used ONLY to fit retrained weights.
  lock       the locked test set. Frozen and retrained models are both
             scored here, paired. Never trained on, never chosen on.
  heldout    records of the held-out genera that sit in training-pool
             deployments: kept out of training, available to test (unseen
             genera). Every record of a held-out genus, in every source and
             partition, also carries heldout_genus = 1.
  ungrouped  unchanged: ineligible on both sides.

  OzFish     30% of main-set deployments to lock, drawn at random within
             each survey (A, B, E, G); the rest is the training pool.
  FishWIO    whole main set -> lock (test only).
  yolo-bruv  whole -> lock (test only).
  FathomNet  30% of main-set inferred groups to lock, drawn at random within
             each institution; the rest is the training pool. Inferred
             groups; always reported separately.
  Genera     5 held out of training entirely, drawn at random from genera
             with >= 2 species usable at k = 20 over all OzFish main
             deployments. Applies to every record whose LINEAGE genus is
             held out (species- or genus-labelled), in every source.

REQUIREMENTS, fixed before the draw (defaults below; set from the
lock_profile_v1 minima with a margin). A draw that misses any of them is
rejected and the next seed in sequence is tried, up to --max-attempts.
Every attempt and its check results are written, so the number of
rejections is on the record. This selects on the COMPOSITION of the test
set only -- no embedding or result is read.

  R1  OzFish lock: species usable at k=20 >= 60; genera with >= 2 usable
      species at k=20 >= 12; species usable at k=100 >= 15
  R2  held-out genera: unseen species usable at k=20 over all OzFish main
      records of those genera >= 10; of those, present in FishWIO and
      usable there at k=20 >= 5
  R3  FathomNet lock: species usable at k=20 >= 100
  R4  OzFish training pool: species-level units >= 19,000
  R5  OzFish lock holds 20-40% of OzFish main species-level units (the
      deployment draw did not land on unusually large or small ones)

INTEGRITY CHECKS (any failure -> passed: false, exit 1, nothing usable):
  C1  every record of every set gets exactly one label; dev and ungrouped
      counts identical to dev_v1
  C2  no group is in both train and lock
  C3  no record of a held-out genus is labelled train
  C4  a key present in several sets has the same label in all of them
  C5  FishWIO and yolo-bruv have no train records
  C6  per stratum, lock deployments == round(0.3 x stratum size)
  C7  every OzFish background record's group is a known OzFish deployment,
      and sits in the same partition (lock / training pool) as that
      deployment's crops; heldout counts as training pool
  C8  sha256 of the inputs, and of the dev_v1 split files, recorded

UNITS for "usable" are wp8_profile.py's: OzFish max(left, right) per
deployment x species; FishWIO and yolo-bruv crops; FathomNet SEFSC
(video, whole second), elsewhere distinct frames. Usable at k: units >=
k + 10 and units outside the largest group >= 10.

Nothing in the shards, the collation or dev_v1 is changed. Loaders filter
by the files written here.

    python scratch\\lock_cut_split.py ^
      --dev-split D:\\marineai\\classification-experiments\\splits\\dev_v1 ^
      --coco-main D:\\marineai\\dataset\\collated\\seavision.json ^
      --coco-fn   D:\\marineai\\dataset\\collated\\seavision_fathomnet.json ^
      --out       D:\\marineai\\classification-experiments\\splits\\lock_v1
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

SETS = ["crops", "square-m00", "square-m10", "background"]
TAXON_SETS = {"crops", "square-m00", "square-m10"}
SEFSC = "NOAA NMFS SEFSC"
SEFSC_SECOND = re.compile(
    r"/([^/]+)/[^/]+\.(?:mp4|avi)\.(\d\d)\.(\d\d)\.(\d\d)\.\d+\.jpg$", re.I)
QMIN = 10


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def usable(by_group, k, qmin=QMIN):
    if not by_group:
        return False
    tot = sum(by_group.values())
    return tot >= k + qmin and tot - max(by_group.values()) >= qmin


def split_path(d, s):
    return os.path.join(d, f"split_{s}.csv.gz")


def read_split(d, s):
    with gzip.open(split_path(d, s), "rt", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------- taxonomy

def key_info(coco_paths):
    """key -> (rank, species, genus, unit_tag, stratum) for animal records,
    by the shard builder's key rule: '{uid}-a{ann id}' for frame sources,
    '{uid}' where the image is the crop. unit_tag: OzFish camera; FathomNet
    collapse unit. stratum: OzFish survey; FathomNet institution."""
    info = {}
    for path in coco_paths:
        print(f"  loading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        cats = {c["id"]: c for c in doc["categories"]}
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        by_image = defaultdict(list)
        for an in doc["annotations"]:
            by_image[an["image_id"]].append(an)
        for im in doc["images"]:
            uid = os.path.splitext(im["file_name"])[0]
            sm = im.get("source_meta") or {}
            g = im.get("groups") or {}
            src = src_of.get(im.get("dataset_id"), "fathomnet")
            for an in by_image.get(im["id"], []):
                c = cats.get(an.get("category_id"))
                if not c:
                    continue
                lin = c.get("lineage") or {}
                rank = str(c.get("rank", "")).lower()
                key = f"{uid}-a{an['id']}" \
                    if im.get("crop_provenance") == "frame" else uid
                if src == "fathomnet":
                    m = SEFSC_SECOND.search(sm.get("frame_url") or "") \
                        if sm.get("owner_institution") == SEFSC else None
                    tag = (m[1], m[2], m[3], m[4]) if m \
                        else sm.get("fathomnet_image_uuid")
                    stratum = sm.get("owner_institution") or ""
                elif src == "ozfish":
                    tag = "L" if str(sm.get("camera") or "").upper() == "L" \
                        else "R"
                    stratum = g.get("survey") or ""
                else:
                    tag, stratum = None, ""
                info[key] = (rank, c.get("name") if rank == "species" else "",
                             lin.get("genus") or "", tag, stratum)
                if im.get("crop_provenance") != "frame":
                    break                       # the image IS one crop
        del doc, by_image
    return info


def species_units(rows, info, keep):
    """rows: crops-set rows; keep(row) -> bool. -> {(src, sp): Counter(grp)}"""
    oz = defaultdict(lambda: [0, 0])
    fn = defaultdict(set)
    out = defaultdict(Counter)
    for r in rows:
        if not keep(r):
            continue
        i = info.get(r["key"])
        if not i or i[0] != "species":
            continue
        sp, g, src = i[1], r["group"], r["source"]
        if src == "ozfish":
            oz[(sp, g)][0 if i[3] == "L" else 1] += 1
        elif src == "fathomnet":
            fn[(sp, g)].add(i[3])
        else:
            out[(src, sp)][g] += 1
    for (sp, g), (L, R) in oz.items():
        out[("ozfish", sp)][g] = max(L, R)
    for (sp, g), s in fn.items():
        out[("fathomnet", sp)][g] = len(s)
    return out


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-split", required=True,
                    help="dev_v1 folder holding split_<set>.csv.gz")
    ap.add_argument("--coco-main", required=True)
    ap.add_argument("--coco-fn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lock-frac", type=float, default=0.3)
    ap.add_argument("--holdout-genera", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--max-attempts", type=int, default=10)
    ap.add_argument("--r1", default="60,12,15",
                    help="OzFish lock: species k20, genera>=2 k20, species k100")
    ap.add_argument("--r2", default="10,5",
                    help="held-out genera: unseen species k20, of which in FishWIO")
    ap.add_argument("--r3", type=int, default=100)
    ap.add_argument("--r4", type=int, default=19000)
    ap.add_argument("--r5", default="0.20,0.40")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    r1 = [int(x) for x in args.r1.split(",")]
    r2 = [int(x) for x in args.r2.split(",")]
    r5 = [float(x) for x in args.r5.split(",")]

    hashes = {os.path.basename(p): sha256(p)
              for p in (args.coco_main, args.coco_fn)}
    for s in SETS:
        hashes[f"dev_v1/split_{s}.csv.gz"] = sha256(split_path(args.dev_split, s))
    for k, v in hashes.items():
        print(f"sha256 {k} {v}")

    rows = {s: read_split(args.dev_split, s) for s in SETS}
    for s in SETS:
        print(f"  {s}: {len(rows[s]):,} records "
              f"{dict(Counter(r['split'] for r in rows[s]))}")
    info = key_info([args.coco_main, args.coco_fn])
    crops = rows["crops"]
    missing = sum(1 for r in crops if r["split"] == "main"
                  and r["source"] != "fishwio" and r["key"] not in info)
    print(f"  main crops records with no taxonomy (no category): {missing:,}")

    # strata of main groups
    strata = defaultdict(lambda: defaultdict(set))   # src -> stratum -> groups
    for r in crops:
        if r["split"] != "main" or r["source"] not in ("ozfish", "fathomnet"):
            continue
        i = info.get(r["key"])
        if i:
            strata[r["source"]][i[4]].add(r["group"])
    for src in ("ozfish", "fathomnet"):
        print(f"  {src} main groups by stratum: "
              f"{ {k: len(v) for k, v in sorted(strata[src].items())} }")
    # a group must sit in one stratum
    for src in strata:
        seen = Counter(g for v in strata[src].values() for g in v)
        multi = [g for g, n in seen.items() if n > 1]
        if multi:
            sys.exit(f"{src}: groups in more than one stratum: {multi[:5]}")

    genus_of_sp = {i[1]: i[2] for i in info.values() if i[1] and i[2]}
    main_oz = species_units(crops, info, lambda r: r["split"] == "main"
                            and r["source"] == "ozfish")
    elig_sp = [sp for (s, sp), d in main_oz.items() if usable(d, 20)]
    gc = Counter(genus_of_sp.get(sp) for sp in elig_sp
                 if genus_of_sp.get(sp))
    eligible = sorted(g for g, n in gc.items() if n >= 2)
    print(f"  genera eligible to hold out: {len(eligible)}")

    def draw(src_strata, frac, rng):
        pick = set()
        for st in sorted(src_strata):
            v = sorted(src_strata[st])
            rng.shuffle(v)
            pick.update(v[:round(frac * len(v))])
        return pick

    attempts = []
    chosen = None
    for a in range(args.max_attempts):
        seed = args.seed + a
        lock_oz = draw(strata["ozfish"], args.lock_frac,
                       random.Random(f"{seed}-ozfish"))
        lock_fn = draw(strata["fathomnet"], args.lock_frac,
                       random.Random(f"{seed}-fathomnet"))
        hold = set(random.Random(f"{seed}-genera").sample(
            eligible, min(args.holdout_genera, len(eligible))))

        def is_lock(r):
            return r["split"] == "main" and (
                r["source"] in ("fishwio", "yolo-bruv")
                or (r["source"] == "ozfish" and r["group"] in lock_oz)
                or (r["source"] == "fathomnet" and r["group"] in lock_fn))

        def genus(r):
            i = info.get(r["key"])
            return i[2] if i else ""

        res, fails = {}, []
        u = species_units(crops, info, lambda r: is_lock(r)
                          and r["source"] == "ozfish")
        sp20 = [sp for (s, sp), d in u.items() if usable(d, 20)]
        g2 = sum(1 for n in Counter(genus_of_sp.get(sp) for sp in sp20
                                    if genus_of_sp.get(sp)).values() if n >= 2)
        sp100 = sum(1 for d in u.values() if usable(d, 100))
        res["R1"] = [len(sp20), g2, sp100]
        if len(sp20) < r1[0] or g2 < r1[1] or sp100 < r1[2]:
            fails.append("R1")
        unseen = [sp for sp in elig_sp if genus_of_sp.get(sp) in hold]
        fw = species_units(crops, info, lambda r: r["split"] == "main"
                           and r["source"] == "fishwio")
        unseen_fw = [sp for sp in unseen if usable(fw.get(("fishwio", sp)), 20)]
        res["R2"] = [len(unseen), len(unseen_fw)]
        if len(unseen) < r2[0] or len(unseen_fw) < r2[1]:
            fails.append("R2")
        ufn = species_units(crops, info, lambda r: is_lock(r)
                            and r["source"] == "fathomnet")
        res["R3"] = sum(1 for d in ufn.values() if usable(d, 20))
        if res["R3"] < args.r3:
            fails.append("R3")
        train_units = sum(n for (s, sp), d in main_oz.items()
                          if genus_of_sp.get(sp) not in hold
                          for g, n in d.items() if g not in lock_oz)
        lock_units = sum(n for d in main_oz.values()
                         for g, n in d.items() if g in lock_oz)
        total = sum(n for d in main_oz.values() for n in d.values())
        res["R4"] = train_units
        res["R5"] = round(lock_units / total, 4) if total else 0
        if train_units < args.r4:
            fails.append("R4")
        if not r5[0] <= res["R5"] <= r5[1]:
            fails.append("R5")
        attempts.append({"attempt": a, "seed": seed, "results": res,
                         "failed": fails, "held_out_genera": sorted(hold)})
        print(f"  attempt {a} (seed {seed}): {res}  "
              f"{'PASS' if not fails else 'fails ' + ','.join(fails)}")
        if not fails:
            chosen = (seed, lock_oz, lock_fn, hold, is_lock, genus)
            break

    report = {"decided": "2026-10-13", "args": vars(args), "hashes": hashes,
              "attempts": attempts, "eligible_genera": eligible}
    os.makedirs(args.out)
    if chosen is None:
        report["passed"] = False
        with open(os.path.join(args.out, "lock_selection.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        sys.exit(f"no draw met the requirements in {args.max_attempts} "
                 f"attempts; nothing written but the attempt log")
    seed, lock_oz, lock_fn, hold, is_lock, genus = chosen

    # ---- label every record
    oz_deps = {r["group"] for r in crops if r["source"] == "ozfish"}
    labels = {}
    checks = {}
    out_rows = {}
    for s in SETS:
        rs = []
        for r in rows[s]:
            g_hold = int(s in TAXON_SETS and genus(r) in hold)
            if r["split"] in ("dev", "ungrouped"):
                lab = r["split"]
            elif r["split"] != "main":
                sys.exit(f"{s}: unknown split label {r['split']!r}")
            elif is_lock(r):
                lab = "lock"
            elif g_hold:
                lab = "heldout"
            else:
                lab = "train"
            rs.append([r["key"], r["source"], lab, r["group"], g_hold])
        out_rows[s] = rs

    # ---- integrity checks
    def put(name, ok, detail):
        checks[name] = {"ok": bool(ok), "detail": detail}
        print(f"  {name} {'ok' if ok else 'FAIL'}  {detail}")

    print("\nINTEGRITY")
    c1 = all(len(out_rows[s]) == len(rows[s]) for s in SETS) and all(
        Counter(x[2] for x in out_rows[s] if x[2] in ("dev", "ungrouped"))
        == Counter(r["split"] for r in rows[s]
                   if r["split"] in ("dev", "ungrouped")) for s in SETS)
    put("C1", c1, "record counts and dev/ungrouped counts unchanged")
    both = [(src, g) for s in SETS for src in ("ozfish", "fathomnet")
            for g in ({x[3] for x in out_rows[s] if x[1] == src
                       and x[2] == "train"}
                      & {x[3] for x in out_rows[s] if x[1] == src
                         and x[2] == "lock"})]
    put("C2", not both, f"{len(both)} groups in both train and lock")
    bad = sum(1 for s in TAXON_SETS for x in out_rows[s]
              if x[4] and x[2] == "train")
    put("C3", bad == 0, f"{bad} held-out-genus records labelled train")
    lab_of = defaultdict(set)
    for s in TAXON_SETS:
        for x in out_rows[s]:
            lab_of[x[0]].add(x[2])
    incon = sum(1 for v in lab_of.values() if len(v) > 1)
    put("C4", incon == 0, f"{incon} keys with different labels across sets")
    tr = sum(1 for s in SETS for x in out_rows[s]
             if x[1] in ("fishwio", "yolo-bruv") and x[2] == "train")
    put("C5", tr == 0, f"{tr} FishWIO/yolo-bruv train records")
    c6 = []
    for src, picked in (("ozfish", lock_oz), ("fathomnet", lock_fn)):
        for st, v in strata[src].items():
            want = round(args.lock_frac * len(v))
            if len(picked & v) != want:
                c6.append((src, st, len(picked & v), want))
    put("C6", not c6, f"{len(c6)} strata off the {args.lock_frac:.0%} rule")
    # background groups come from frame_key (dev_v1 derived them); the key
    # itself is the record key, so check the group, and that the record's
    # label matches its deployment's label in crops
    # Compare PARTITIONS, not labels: in a training-pool deployment crops
    # are "train" or "heldout" (held-out genus) while background, which has
    # no genus, is "train". A deployment whose crops are all held-out genera
    # is therefore legitimate and is counted separately, not failed.
    part = {"train": "pool", "heldout": "pool", "lock": "lock"}
    crop_part = defaultdict(set)
    all_held = defaultdict(lambda: True)
    for x in out_rows["crops"]:
        if x[1] == "ozfish" and x[2] in part:
            crop_part[x[3]].add(part[x[2]])
            all_held[x[3]] &= x[2] == "heldout"
    unmapped = mismatch = in_all_held = 0
    for x in out_rows["background"]:
        if x[1] != "ozfish" or x[2] in ("dev", "ungrouped"):
            continue
        if x[3] not in oz_deps:
            unmapped += 1
        elif crop_part[x[3]] != {part[x[2]]}:
            mismatch += 1
        elif x[3] in crop_part and all_held[x[3]]:
            in_all_held += 1
    put("C7", unmapped == 0 and mismatch == 0,
        f"{unmapped} OzFish background records with an unknown deployment, "
        f"{mismatch} in a different partition from their deployment's crops "
        f"({in_all_held} sit in training-pool deployments whose crops are "
        f"all held-out genera: legitimate, reported)")
    put("C8", True, "hashes recorded in lock_selection.json")

    passed = all(c["ok"] for c in checks.values())
    print("\nCOUNTS (set, source, label)")
    summary = {}
    for s in SETS:
        c = Counter((x[1], x[2]) for x in out_rows[s])
        summary[s] = {f"{a}/{b}": n for (a, b), n in sorted(c.items())}
        print(f"  {s}: " + "  ".join(f"{k} {v:,}"
                                     for k, v in summary[s].items()))

    for s in SETS:
        with gzip.open(os.path.join(args.out, f"split_{s}.csv.gz"), "wt",
                       newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["key", "source", "split", "group", "heldout_genus"])
            w.writerows(out_rows[s])
    with open(os.path.join(args.out, "lock_groups.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "stratum", "group"])
        for src, picked in (("ozfish", lock_oz), ("fathomnet", lock_fn)):
            for st, v in sorted(strata[src].items()):
                for g in sorted(picked & v):
                    w.writerow([src, st, g])
    with open(os.path.join(args.out, "heldout_genera.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["genus"])
        w.writerows([g] for g in sorted(hold))
    report.update({"passed": passed, "seed_used": seed,
                   "held_out_genera": sorted(hold), "checks": checks,
                   "counts": summary,
                   "lock_groups": {"ozfish": len(lock_oz),
                                   "fathomnet": len(lock_fn)}})
    with open(os.path.join(args.out, "lock_selection.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\n{'PASSED' if passed else 'FAILED'}: wrote {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

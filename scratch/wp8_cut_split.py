#!/usr/bin/env python3
"""WP8: cut the development split into key manifests, one per shard set.

Reads the frozen selection (dev_deployments.csv + dev_selection.json from
`wp8_profile.py --fix`) and every tar under --shards, and labels every
record in every shard set:

    dev         its deployment is in the frozen selection
    main        grouped, and its deployment is not
    ungrouped   no deployment (or no inferred_deployment, for FathomNet):
                ineligible on BOTH sides, per the WP8 decision

Nothing in the shards or the collation is changed. Shards stay as built;
loaders filter by these manifests.

Where each record's group comes from -- read from the shard, not assumed:
    crops / square, OzFish FishWIO yolo-bruv   record json groups.deployment
    crops / square, FathomNet                  key (= uid) -> fn-groups CSV
    background, OzFish                         frame_key ozfish-{video}_{cam}_{frame};
                                               video must be a known OzFish
                                               deployment, else the run fails
    background, FathomNet                      frame_key fathomnet-{image uuid}
                                               -> fn-groups CSV
    background, yolo-bruv                      frame_key yolo-bruv-{uid}
                                               -> that uid's deployment in crops

Checks (any failure -> manifest says passed: false, exit code 1):
    C1  per dev deployment, crops records == dev_deployments.csv crops
    C2  total dev crops == dev_selection.json result.crops
    C3  no duplicate key within a set
    C4  a key present in several sets has the same group in all of them
    C5  OzFish keys identical across crops, square-m00, square-m10, both
        all keys and dev keys; FathomNet dev keys likewise
        (FishWIO has no frames, so no square: reported, not failed)
    C6  every OzFish background frame maps to a known deployment
    C7  WP6 gate recomputed from the crops SHARDS reproduces WP6
        (454/78, 304/57, 243/48) -- a different input path from the
        profiler, which read the COCO
    C8  the gate on the remainder matches the profiler's gate_remainder
    C10 no yolo-bruv background record is ungrouped (empty frames map
        through the COCO images, not through `crops`)
    C9  sha256 of seavision.json and the fn-groups CSV match the ones the
        selection was made on

    python scratch\\wp8_cut_split.py ^
        --selection D:\\marineai\\scratch\\wp8_dev_v1 ^
        --shards    D:\\marineai\\classification-experiments\\shards ^
        --coco-main D:\\marineai\\dataset\\collated\\seavision.json ^
        --fn-groups D:\\marineai\\scratch\\fathomnet_inferred_deployment.csv.gz ^
        --out       D:\\marineai\\scratch\\wp8_split_v1
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

SETS = ["crops", "square-m00", "square-m10", "background"]
TAR_RE = re.compile(r"^(?P<prefix>.+)-(?P<idx>\d{6})\.tar$")
OZ_BG = re.compile(r"^ozfish-(?P<video>.+)_(?P<cam>[A-Za-z]+)_(?P<frame>\d+)$")
WP6_EXPECTED = {(10, 2): (454, 78), (50, 2): (304, 57), (100, 3): (243, 48)}
JSON_SOURCES = {"ozfish", "fishwio", "yolo-bruv"}   # read record json
SHA_CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(SHA_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def gate(species_groups, genus_of):
    out = {}
    for mc, mg in WP6_EXPECTED:
        q = {s for s, g in species_groups.items()
             if sum(g.values()) >= mc and len(g) >= mg}
        pg = Counter(genus_of[s] for s in q if genus_of.get(s))
        out[f"{mc}/{mg}"] = [len(q), sum(1 for v in pg.values() if v >= 2),
                             sum(1 for v in pg.values() if v >= 3)]
    return out


def tars_by_source(set_dir, set_name):
    """-> {source: [tar paths in index order]}"""
    out = defaultdict(list)
    for f in sorted(os.listdir(set_dir)):
        m = TAR_RE.match(f)
        if not m:
            continue
        prefix = m["prefix"]
        if not prefix.startswith(set_name + "-"):
            sys.exit(f"unexpected tar name {f} in {set_dir}")
        out[prefix[len(set_name) + 1:]].append(
            (int(m["idx"]), os.path.join(set_dir, f)))
    return {s: [p for _i, p in sorted(v)] for s, v in out.items()}


def iter_records(tar_path, want_json):
    """Yield (key, meta-or-None) once per record. Reads only json members,
    and only when asked; other members are skipped by header."""
    with tarfile.open(tar_path, mode="r:") as tf:
        for info in tf:
            if not info.isfile():
                continue
            key, _, ext = info.name.partition(".")
            if ext == "json":
                meta = json.loads(tf.extractfile(info).read()) \
                    if want_json else None
                yield key, meta
    # a record without a json member would be invisible here; the builders'
    # level-1 verify already guarantees one per record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True,
                    help="folder holding dev_deployments.csv and "
                         "dev_selection.json")
    ap.add_argument("--fn-selection", default=None,
                    help="folder holding fn_dev_groups.csv and "
                         "fn_dev_selection.json (wp8_fathomnet_slice.py)")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--coco-main", required=True)
    ap.add_argument("--fn-groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*", default=SETS)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    t0 = time.time()

    # ---- inputs
    sel_json = os.path.join(args.selection, "dev_selection.json")
    sel_csv = os.path.join(args.selection, "dev_deployments.csv")
    with open(sel_json, encoding="utf-8") as fh:
        sel = json.load(fh)
    with open(sel_csv, newline="", encoding="utf-8") as fh:
        dep_rows = list(csv.DictReader(fh))
    dev = {(r["source"], r["deployment"]): int(r["crops"]) for r in dep_rows}
    expected_dev_crops = sel["result"]["crops"]
    fn_sel, fn_sel_files = None, {}
    if args.fn_selection:
        fj = os.path.join(args.fn_selection, "fn_dev_selection.json")
        fc = os.path.join(args.fn_selection, "fn_dev_groups.csv")
        with open(fj, encoding="utf-8") as fh:
            fn_sel = json.load(fh)
        with open(fc, newline="", encoding="utf-8") as fh:
            fn_rows = list(csv.DictReader(fh))
        for r in fn_rows:      # crops = crop records in the group = shard records
            dev[("fathomnet", r["deployment"])] = int(r["crops"])
        expected_dev_crops += sum(int(r["crops"]) for r in fn_rows)
        fn_sel_files = {"fn_dev_selection.json": sha256(fj),
                        "fn_dev_groups.csv": sha256(fc)}
    print(f"selection: {len(dev)} dev groups "
          f"({Counter(s for s, _d in dev)})")

    checks = {}
    print("hashing inputs ...", flush=True)
    coco_sha = sha256(args.coco_main)
    fn_sha = sha256(args.fn_groups)
    checks["C9_inputs_match_selection"] = (
        coco_sha == sel.get("coco_main_sha256") and
        fn_sha == sel.get("fn_groups_sha256") and
        (fn_sel is None or fn_sha == fn_sel.get("fn_groups_sha256")))

    print("loading categories ...", flush=True)
    with open(args.coco_main, encoding="utf-8") as fh:
        doc = json.load(fh)
    cats = doc["categories"]
    # yolo-bruv frames by uid -> deployment, from the COCO images, so that
    # empty frames (no annotation, so absent from `crops`) still map
    yb_dep_of_uid = {}
    for im in doc["images"]:
        # the uid is not stored on the image; it is the file_name stem
        # (yolo-bruv-0cd68f1494.jpg -> yolo-bruv-0cd68f1494), as in crops
        u = str(im.get("uid") or os.path.splitext(im.get("file_name") or "")[0])
        if u.startswith("yolo-bruv-"):
            yb_dep_of_uid[u] = (im.get("groups") or {}).get("deployment") or ""
    print(f"  yolo-bruv frames found in the COCO: {len(yb_dep_of_uid):,} "
          f"(expected 2,667)")
    del doc
    genus_of = {c["id"]: (c.get("lineage") or {}).get("genus") for c in cats}
    is_species = {c["id"] for c in cats
                  if str(c.get("rank", "")).lower() == "species"}
    del cats

    print("loading fn-groups ...", flush=True)
    fn_by_uid, fn_by_uuid = {}, {}
    with gzip.open(args.fn_groups, "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = r.get("inferred_deployment") or ""
            fn_by_uid[r["uid"]] = g
            if g and r.get("fathomnet_image_uuid"):
                fn_by_uuid[r["fathomnet_image_uuid"]] = g

    def split_of(src, grp):
        if not grp:
            return "ungrouped"
        return "dev" if (src, grp) in dev else "main"

    os.makedirs(args.out)
    counts = defaultdict(Counter)            # set -> (source, split) -> n
    group_of_key = {}                        # non-FathomNet: key -> group
    dev_keys = defaultdict(lambda: defaultdict(set))   # set -> src -> keys
    dev_crops_per_dep = Counter()
    sg_all, sg_rem = defaultdict(Counter), defaultdict(Counter)
    unannotated_dev = Counter()
    c3_dups, c4_conflicts, c6_unmatched = Counter(), [], []
    oz_deps = set()
    oz_keys = defaultdict(set)               # set -> all OzFish keys
    file_sha = {}

    # crops first: it defines OzFish deployments and yolo-bruv uid -> dep
    order = [s for s in SETS if s in args.sets]
    for set_name in order:
        set_dir = os.path.join(args.shards, set_name)
        if not os.path.isdir(set_dir):
            sys.exit(f"missing shard set {set_dir}")
        out_path = os.path.join(args.out, f"split_{set_name}.csv.gz")
        raw = open(out_path, "wb")
        gz = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
        txt = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        w = csv.writer(txt)
        w.writerow(["key", "source", "split", "group"])
        seen = set()
        for src, tars in sorted(tars_by_source(set_dir, set_name).items()):
            want_json = src in JSON_SOURCES or set_name == "background"
            for tp in tars:
                print(f"  {set_name:<11} {os.path.basename(tp)} "
                      f"[{time.time() - t0:,.0f}s]", flush=True)
                for key, meta in iter_records(tp, want_json):
                    if key in seen:
                        c3_dups[set_name] += 1
                    seen.add(key)
                    # ---- the record's group
                    if set_name == "background":
                        fk = meta["frame_key"]
                        if src == "ozfish":
                            m = OZ_BG.match(fk)
                            grp = m["video"] if m else ""
                            if not m or grp not in oz_deps:
                                c6_unmatched.append(fk)
                        elif src == "fathomnet":
                            grp = fn_by_uuid.get(fk[len("fathomnet-"):], "")
                        else:
                            grp = yb_dep_of_uid.get(fk, "")
                    elif src == "fathomnet":
                        grp = fn_by_uid.get(key, "")
                    else:
                        grp = ((meta or {}).get("groups") or {}) \
                            .get("deployment") or ""
                    sp = split_of(src, grp)
                    counts[set_name][(src, sp)] += 1
                    w.writerow([key, src, sp, grp])
                    # ---- bookkeeping for checks
                    if src != "fathomnet" and set_name != "background":
                        prev = group_of_key.get(key)
                        if prev is not None and prev != grp:
                            c4_conflicts.append((set_name, key, prev, grp))
                        group_of_key.setdefault(key, grp)
                    if sp == "dev":
                        dev_keys[set_name][src].add(key)
                    if src == "ozfish" and set_name != "background":
                        oz_keys[set_name].add(key)
                    if set_name == "crops":
                        if src == "ozfish" and grp:
                            oz_deps.add(grp)
                        if src == "yolo-bruv":
                            yb_dep_of_uid.setdefault(meta["uid"], grp)
                        # C1/C2 count annotated records, as the profiler
                        # does; unannotated ones (FishWIO's shipped
                        # background crops) stay dev but are counted apart
                        if sp == "dev":
                            if meta is not None and \
                                    meta.get("category_id") is None:
                                unannotated_dev[(src, grp)] += 1
                            else:
                                dev_crops_per_dep[(src, grp)] += 1
                        if src in JSON_SOURCES and grp and \
                                meta.get("category_id") in is_species:
                            k = (src, grp)
                            sg_all[meta["category_id"]][k] += 1
                            if sp != "dev":
                                sg_rem[meta["category_id"]][k] += 1
        txt.close()          # closes gz too
        raw.close()
        file_sha[os.path.basename(out_path)] = sha256(out_path)

    # ---- checks
    c1 = {f"{s}:{d}": [n, dev_crops_per_dep.get((s, d), 0)]
          for (s, d), n in dev.items()
          if dev_crops_per_dep.get((s, d), 0) != n}
    checks["C1_dev_crops_per_deployment"] = not c1
    tot = sum(dev_crops_per_dep.values())
    checks["C2_dev_crops_total"] = tot == expected_dev_crops
    checks["C3_no_duplicate_keys"] = not c3_dups
    checks["C4_group_consistent_across_sets"] = not c4_conflicts
    c5 = {}
    if "crops" in order:
        for s in ("square-m00", "square-m10"):
            if s in order:
                c5[s] = {}
                for lab, ref, got in (
                        ("all", oz_keys["crops"], oz_keys[s]),
                        ("dev", dev_keys["crops"]["ozfish"],
                         dev_keys[s]["ozfish"]),
                        ("fathomnet_dev", dev_keys["crops"]["fathomnet"],
                         dev_keys[s]["fathomnet"])):
                    c5[s][lab] = {"missing": len(ref - got),
                                  "extra": len(got - ref)}
    checks["C5_ozfish_inner_join"] = all(
        v["missing"] == 0 and v["extra"] == 0
        for per in c5.values() for v in per.values())
    checks["C6_ozfish_background_mapped"] = not c6_unmatched
    g_all = gate(sg_all, genus_of)
    checks["C7_wp6_gate_from_shards"] = all(
        tuple(g_all[f"{mc}/{mg}"][:2]) == exp
        for (mc, mg), exp in WP6_EXPECTED.items())
    g_rem = gate(sg_rem, genus_of)
    exp_rem = sel["result"]["gate_remainder"]
    checks["C8_remainder_gate_matches_profiler"] = all(
        list(g_rem[c]) == list(exp_rem[c]) for c in exp_rem)
    yb_bg_ung = counts["background"][("yolo-bruv", "ungrouped")] \
        if "background" in order else 0
    checks["C10_yolo_bruv_background_grouped"] = yb_bg_ung == 0
    passed = all(checks.values())

    # ---- report
    print(f"\n{'=' * 78}\nSPLIT  (records per set, source and split)")
    print(f"  {'set':<12}{'source':<11}{'dev':>9}{'main':>11}"
          f"{'ungrouped':>11}")
    for s in order:
        for src in sorted({k[0] for k in counts[s]}):
            c = counts[s]
            print(f"  {s:<12}{src:<11}{c[(src, 'dev')]:>9,}"
                  f"{c[(src, 'main')]:>11,}{c[(src, 'ungrouped')]:>11,}")
    print(f"\nCHECKS")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if c1:
        print(f"  C1 mismatches (expected, found): {list(c1.items())[:10]}")
    if c3_dups:
        print(f"  C3 duplicates: {dict(c3_dups)}")
    if c4_conflicts:
        print(f"  C4 conflicts (first 5): {c4_conflicts[:5]}")
    print(f"  C2 dev annotated crops {tot:,} against selection "
          f"{expected_dev_crops:,}; unannotated dev records (kept as dev, "
          f"not counted): {sum(unannotated_dev.values()):,} "
          f"{ {f'{a}:{b}': n for (a, b), n in unannotated_dev.items()} }")
    print(f"  C5 OzFish keys vs crops: {c5}")
    fw_sq = {s: len(dev_keys[s]["fishwio"]) for s in order if s != "crops"}
    print(f"     FishWIO dev keys outside crops (expected 0, no frames): "
          f"{fw_sq}")
    if c6_unmatched:
        print(f"  C6 unmatched OzFish background frames: "
              f"{len(c6_unmatched)}, e.g. {c6_unmatched[:5]}")
    print(f"  C7 gate from shards: {g_all}  (WP6: "
          f"{ {f'{a}/{b}': v for (a, b), v in WP6_EXPECTED.items()} })")
    print(f"  C8 remainder gate: {g_rem}  (profiler: {exp_rem})")
    if not checks["C9_inputs_match_selection"]:
        print(f"  C9 coco {coco_sha[:12]} vs {sel.get('coco_main_sha256', '')[:12]}"
              f", fn {fn_sha[:12]} vs {sel.get('fn_groups_sha256', '')[:12]}")

    manifest = {
        "passed": passed, "checks": checks,
        "decided": sel.get("decided"), "size": sel.get("size"),
        "inputs": {"dev_selection.json": sha256(sel_json),
                   "dev_deployments.csv": sha256(sel_csv),
                   "coco_main": coco_sha, "fn_groups": fn_sha,
                   **fn_sel_files},
        "outputs": file_sha,
        "counts": {s: {f"{a}:{b}": n for (a, b), n in sorted(c.items())}
                   for s, c in counts.items()},
        "dev_crops": tot,
        "dev_unannotated": {f"{a}:{b}": n
                            for (a, b), n in unannotated_dev.items()},
        "gate_all_from_shards": g_all, "gate_remainder": g_rem,
        "c5": c5, "fishwio_dev_outside_crops": fw_sq,
        "c6_unmatched_examples": c6_unmatched[:20],
        "fathomnet_note": ("FathomNet slice from non-SEFSC dive groups; "
                           "inferred groups; report separately, never pool")
        if fn_sel else "no FathomNet slice: FathomNet records are main or "
                       "ungrouped only",
        "seconds": round(time.time() - t0),
    }
    with open(os.path.join(args.out, "split_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\n{'PASSED' if passed else 'FAILED'}: wrote "
          f"{len(file_sha)} split files and split_manifest.json to {args.out}"
          f"  [{time.time() - t0:,.0f}s]")
    print("Nothing in the shards or the collation was changed.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

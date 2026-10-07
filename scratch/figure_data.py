#!/usr/bin/env python3
"""Write the small, tidy CSVs (and a few PNGs) the methods figures are drawn
from. READ-ONLY with respect to every input. The R scripts in the SharePoint
`figures/` folder read only what this writes -- no JSON, no tars -- so a
figure can be restyled without touching Python, and every number in a figure
can be traced to one file here.

Five parts, each its own subcommand so a slow one need not be re-run:

  collation   COCO files (+ FathomNet inferred groups) ->
                sources.csv, category_source.csv, size_hist.csv, geo.csv,
                depth_hist.csv, group_sizes.csv
  split       dev split + profiler reports ->
                split_counts.csv, dev_order.csv, pools.csv, random_pools.csv
  testable    labels.csv + draws -> testable.csv
  wp8         WP8 results archive -> readouts_avg.csv, readouts_geom.csv,
                leakage_all.csv, backbones.csv, decision.csv, stability.csv,
                fathomnet_check.csv
  images      dev + main shards -> images/*.png, examples.csv, geometry.csv

    python scratch\\figure_data.py collation --out D:\\marineai\\classification-experiments\\results\\figure_data ^
        --coco D:\\marineai\\dataset\\collated\\seavision.json ^
        --coco D:\\marineai\\dataset\\collated\\seavision_fathomnet.json ^
        --fn-groups D:\\marineai\\dataset\\collated\\fathomnet_inferred_deployment.csv.gz
    (the other parts: see each subcommand's --help)
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import io
import json
import math
import os
import sys
import tarfile
from collections import Counter, defaultdict

# ---------------------------------------------------------------- helpers


def write_csv(path, header, rows):
    tmp = path + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)
    print(f"  wrote {os.path.basename(path)} ({len(rows):,} rows)")


def lin_of(cat):
    lin = cat.get("lineage") or {}
    return {str(k).strip().lower(): v for k, v in lin.items()} \
        if isinstance(lin, dict) else {}


# size bins: 1/8 octave from 4 px to 8192 px, so the ECDF is smooth on a log
# axis without writing 1.47M rows
SIZE_EDGES = [4 * 2 ** (i / 8) for i in range(0, 8 * 11 + 1)]
DEPTH_EDGES = list(range(0, 4100, 100))


def bin_index(x, edges):
    if x < edges[0]:
        return 0
    if x >= edges[-1]:
        return len(edges) - 2
    lo, hi = 0, len(edges) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if edges[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


def short_side(img, ann):
    """As scratch/dataset_summary.py: bbox for full frames, image size for
    pre-cropped or cut-from-frame records."""
    if img.get("crop_provenance") == "frame":
        b = ann.get("bbox") or []
        if len(b) < 4:
            return None
        w, h = b[2], b[3]
    else:
        w, h = img.get("width"), img.get("height")
    if not w or not h or w <= 0 or h <= 0:
        return None
    return float(min(w, h))


# ---------------------------------------------------------------- collation


def cmd_collation(a):
    fn_dep = {}
    if a.fn_groups:
        with gzip.open(a.fn_groups, "rt", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["inferred_deployment"]:
                    fn_dep[r["uid"]] = (r["inferred_deployment"],
                                        r["owner_institution"])
        print(f"  {len(fn_dep):,} FathomNet crops with an inferred deployment")
    cats = {}
    # (cid, src) -> crops ; groups (recorded or inferred), namespaced by source
    n_cs = Counter()
    g_cs = defaultdict(set)
    kind_cs = {}
    src = defaultdict(lambda: Counter())
    src_frames = defaultdict(set)
    size_h = Counter()                    # (src, stratum, bin) -> n
    geo = Counter()                       # (src, lat, lon) -> images
    depth = Counter()                     # (src, institution, bin) -> images
    depth_missing = Counter()
    grp = defaultdict(lambda: [0, set()])  # (src, kind, group) -> [crops, cids]
    for path in a.coco:
        print(f"  loading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        ds = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        for c in doc.get("categories", []):
            cats[c["id"]] = c
        imgs = {}
        for im in doc.get("images", []):
            imgs[im["id"]] = im
            s = ds.get(im.get("dataset_id"), "unknown")
            src[s]["images"] += 1
            if im.get("lat") is not None and im.get("lon") is not None:
                src[s]["images_with_latlon"] += 1
                geo[(s, math.floor(im["lat"]), math.floor(im["lon"]))] += 1
            sm = im.get("source_meta") or {}
            if s == "fathomnet":
                inst = sm.get("owner_institution") or "unknown"
                if im.get("depth_m") is None:
                    depth_missing[(s, inst)] += 1
                else:
                    depth[(s, inst, bin_index(float(im["depth_m"]),
                                              DEPTH_EDGES))] += 1
                fk = sm.get("fathomnet_image_uuid") or sm.get("frame_sha256")
                if fk:
                    src_frames[s].add(fk)
            elif im.get("crop_provenance") == "frame":
                src_frames[s].add(im["id"])
        for an in doc.get("annotations", []):
            im = imgs.get(an.get("image_id"))
            if im is None:
                continue
            s = ds.get(im.get("dataset_id"), "unknown")
            src[s]["annotations"] += 1
            groups = im.get("groups") or {}
            stratum = groups.get("survey", "all") if s == "ozfish" else "all"
            ss = short_side(im, an)
            if ss is not None:
                size_h[(s, stratum, bin_index(ss, SIZE_EDGES))] += 1
            if "deployment" in groups:
                kind, g = "recorded", groups["deployment"]
            elif s == "fathomnet":
                uid = os.path.splitext(im.get("file_name") or "")[0]
                hit = fn_dep.get(uid)
                kind, g = ("inferred", hit[0]) if hit else ("none", None)
            else:
                kind, g = "none", None
            cid = an.get("category_id")
            if g is not None:
                gg = grp[(s, kind, g)]
                gg[0] += 1
                if cid is not None:
                    gg[1].add(cid)
            if cid is None:
                continue
            n_cs[(cid, s)] += 1
            if g is not None:
                g_cs[(cid, s)].add(g)
                kind_cs[(cid, s)] = kind
        del doc, imgs
    out = a.out
    os.makedirs(out, exist_ok=True)
    rows = []
    for s in sorted(src):
        cats_here = {c for (c, s2) in n_cs if s2 == s}
        n_groups = len({g for (s2, k, g) in grp if s2 == s})
        kinds = sorted({k for (s2, k, g) in grp if s2 == s})
        rows.append([s, src[s]["images"], src[s]["annotations"],
                     len(src_frames[s]) or "", len(cats_here),
                     src[s]["images_with_latlon"], n_groups, "|".join(kinds)])
    write_csv(os.path.join(out, "sources.csv"),
              ["source", "images", "annotations", "frames", "categories",
               "images_with_latlon", "groups", "group_kind"], rows)
    rows = []
    for (cid, s), n in sorted(n_cs.items()):
        c = cats.get(cid, {})
        ln = lin_of(c)
        rows.append([cid, c.get("name"), c.get("rank"), ln.get("kingdom", ""),
                     ln.get("phylum", ""), ln.get("class", ""),
                     ln.get("order", ""), ln.get("family", ""),
                     ln.get("genus", ""), s, n, len(g_cs[(cid, s)]),
                     kind_cs.get((cid, s), "none")])
    write_csv(os.path.join(out, "category_source.csv"),
              ["category_id", "name", "rank", "kingdom", "phylum", "class",
               "order", "family", "genus", "source", "n_crops", "n_groups",
               "group_kind"], rows)
    write_csv(os.path.join(out, "size_hist.csv"),
              ["source", "stratum", "lo", "hi", "n"],
              [[s, st, round(SIZE_EDGES[b], 3), round(SIZE_EDGES[b + 1], 3), n]
               for (s, st, b), n in sorted(size_h.items())])
    write_csv(os.path.join(out, "geo.csv"),
              ["source", "lat_bin", "lon_bin", "n_images"],
              [[s, la, lo, n] for (s, la, lo), n in sorted(geo.items())])
    write_csv(os.path.join(out, "depth_hist.csv"),
              ["source", "institution", "lo", "hi", "n"],
              [[s, i, DEPTH_EDGES[b], DEPTH_EDGES[b + 1], n]
               for (s, i, b), n in sorted(depth.items())] +
              [[s, i, "", "", n] for (s, i), n in sorted(depth_missing.items())])
    write_csv(os.path.join(out, "group_sizes.csv"),
              ["source", "group_kind", "group", "n_crops", "n_categories"],
              [[s, k, g, v[0], len(v[1])]
               for (s, k, g), v in sorted(grp.items())])


# ---------------------------------------------------------------- split


def cmd_split(a):
    out = a.out
    os.makedirs(out, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(a.split_dir, "split_*.csv.gz"))):
        set_name = os.path.basename(p)[len("split_"):-len(".csv.gz")]
        c = Counter()
        with gzip.open(p, "rt", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                c[(r["source"], r["split"])] += 1
        rows += [[set_name, s, sp, n] for (s, sp), n in sorted(c.items())]
    write_csv(os.path.join(out, "split_counts.csv"),
              ["set", "source", "split", "n"], rows)
    rows = []
    for p in (a.dev_deployments, a.fn_dev_groups):
        if not p:
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append([r.get("source"), int(r["order"]),
                             r.get("deployment"),
                             r.get("survey") or r.get("institution") or "",
                             r.get("crops", ""), r.get("units", "")])
    write_csv(os.path.join(out, "dev_order.csv"),
              ["source", "order", "deployment", "stratum", "crops", "units"],
              rows)
    if a.profile_report:
        with open(a.profile_report, encoding="utf-8") as fh:
            rep = json.load(fh)
        rows = []
        for key in ("pools", "smallest_search"):
            for r in rep.get(key) or []:
                for part in ("all", "oz_alone"):
                    lv = r.get(part) or {}
                    rows.append([key, r.get("mode", "all"), r["oz"], r["fw"],
                                 r.get("crops"), r.get("units"), part,
                                 lv.get("species"), lv.get("genera_2sp"),
                                 lv.get("families_2gen"),
                                 lv.get("orders_2fam"), r.get("lost_50_2"),
                                 r.get("lost_100_3")])
        if not rows:
            print("  !! profiler report has no 'pools' or 'smallest_search': "
                  "it was written by a --skip-grid run. Re-run wp8_profile.py "
                  "without --skip-grid into a new --out for these figures.")
        write_csv(os.path.join(out, "pools.csv"),
                  ["table", "mode", "oz", "fw", "crops", "units", "part",
                   "species", "genera_2sp", "families_2gen", "orders_2fam",
                   "lost_50_2", "lost_100_3"], rows)
        rows = []
        for r in rep.get("random_baseline") or []:
            for d in range(len(r["genera"])):
                rows.append([r["oz"], d, r["species"][d], r["genera"][d],
                             r["families"][d], r["lost_50_2"][d],
                             r["lost_100_3"][d]])
        write_csv(os.path.join(out, "random_pools.csv"),
                  ["oz", "draw", "species", "genera_2sp", "families_2gen",
                   "lost_50_2", "lost_100_3"], rows)


# ---------------------------------------------------------------- testable


def cmd_testable(a):
    with open(a.labels, newline="", encoding="utf-8") as fh:
        labels = {r["key"]: r for r in csv.DictReader(fh)}
    rows = []
    for p in sorted(glob.glob(os.path.join(a.draws_dir, "draws_*.json"))):
        pool = os.path.basename(p)[len("draws_"):-len(".json")]
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        draws = d["draws"] if isinstance(d, dict) else d
        failed = set(d.get("failed", [])) if isinstance(d, dict) else set()
        first = draws[0]
        for sp, v in sorted(first.items()):
            if sp in failed:
                continue
            keys = v["ref"] + v["query"]
            r0 = labels.get(keys[0], {})
            deps = {labels[k]["source"] + "|" + labels[k]["deployment"]
                    for k in keys if k in labels}
            srcs = sorted({labels[k]["source"] for k in keys if k in labels})
            rows.append([pool, sp, r0.get("genus", ""), r0.get("family", ""),
                         r0.get("order", ""), r0.get("class", ""),
                         "|".join(srcs), len(keys), len(deps)])
    write_csv(os.path.join(a.out, "testable.csv"),
              ["pool", "species", "genus", "family", "order", "class",
               "sources", "n_crops", "n_deployments"], rows)


# ---------------------------------------------------------------- wp8


def cmd_wp8(a):
    import numpy as np
    import pandas as pd
    out = a.out
    os.makedirs(out, exist_ok=True)
    summ = os.path.join(a.archive, "summary")
    for f in ("decision.csv", "stability.csv", "fathomnet_check.csv"):
        pd.read_csv(os.path.join(summ, f)).to_csv(os.path.join(out, f),
                                                  index=False)
        print(f"  copied {f}")
    df = pd.read_csv(os.path.join(summ, "results_all.csv.gz"),
                     dtype={"layer": str})
    crops = {"letterbox", "distort", "native"}
    key = ["backbone", "size", "token", "layer", "pool", "rule", "k",
           "level", "mode"]
    # (a) the decision view: per draw, mean over the backbone's crops
    # geometries (paired: same draws), then mean/sd/min/max over draws
    d = df[df.geometry.isin(crops)]
    per = d.groupby(key + ["draw"], dropna=False).macro_recall.mean()
    agg = per.groupby(key, dropna=False).agg(["mean", "std", "min", "max"])
    agg.reset_index().to_csv(os.path.join(out, "readouts_avg.csv"),
                             index=False)
    print(f"  wrote readouts_avg.csv ({len(agg):,} rows)")
    # (b) every geometry on its own (ozfish pool, the only one in all)
    g = df[(df.pool == "ozfish") & (df.rule == "prototype")]
    agg = g.groupby(key + ["geometry"], dropna=False).macro_recall.agg(
        ["mean", "std", "min", "max"])
    agg.reset_index().to_csv(os.path.join(out, "readouts_geom.csv"),
                             index=False)
    print(f"  wrote readouts_geom.csv ({len(agg):,} rows)")
    emb = os.path.join(a.archive, "emb")
    leaks = [pd.read_csv(p, dtype={"layer": str}) for p in
             sorted(glob.glob(os.path.join(emb, "*", "*", "*", "leakage.csv")))]
    pd.concat(leaks).to_csv(os.path.join(out, "leakage_all.csv"), index=False)
    print(f"  wrote leakage_all.csv ({sum(len(x) for x in leaks):,} rows)")
    seen, rows = set(), []
    for p in sorted(glob.glob(os.path.join(emb, "*", "*", "*",
                                           "manifest.json"))):
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        b = m["backbone"]
        k = (b["name"], m["size"])
        if k in seen:
            continue
        seen.add(k)
        rows.append([b["name"], m["size"], b.get("library"),
                     b.get("checkpoint"), b.get("n_layers"), b.get("width"),
                     b.get("patch_size"), b.get("prefix_tokens"),
                     b.get("has_cls"), b.get("cls_index"),
                     b.get("native_size"), b.get("interpolated"),
                     b.get("input_mode")])
    if a.spec:
        import yaml
        with open(a.spec, encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)["backbones"]
        dec = pd.read_csv(os.path.join(summ, "decision.csv"),
                          dtype={"layer": str})
        won = {(r.backbone, str(r.token), str(r.layer)) for r in
               dec.itertuples()}
        write_csv(os.path.join(out, "kept.csv"),
                  ["backbone", "size", "token", "layer", "why"],
                  [[b, v["size"], r["token"], str(r["layer"]),
                    "decision" if (b, r["token"], str(r["layer"])) in won
                    else "stability re-plan"]
                   for b, v in spec.items() for r in v["readouts"]])
    write_csv(os.path.join(out, "backbones.csv"),
              ["backbone", "size", "library", "checkpoint", "n_layers",
               "width", "patch_size", "prefix_tokens", "has_cls",
               "cls_index", "native_size", "interpolated", "input_mode"],
              rows)


# ---------------------------------------------------------------- images


def read_member(tar_path, members, key):
    m = members[key]
    ext = next(e for e in m if e != "json")
    with open(tar_path, "rb") as fh:
        fh.seek(m[ext][0])
        return fh.read(m[ext][1])


def cmd_images(a):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))))
    from PIL import Image
    from extract.preprocess import apply_geometry, naflex_size
    img_dir = os.path.join(a.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    with open(a.labels, newline="", encoding="utf-8") as fh:
        labels = {r["key"]: r for r in csv.DictReader(fh)}

    def dev(set_name):
        with open(os.path.join(a.dev_shards, f"dev-{set_name}.index.json"),
                  encoding="utf-8") as fh:
            ix = json.load(fh)
        return os.path.join(a.dev_shards, f"dev-{set_name}.tar"), \
            ix["members"]
    tar, mem = dev("crops")
    # a species well represented in both OzFish and FishWIO dev records
    by = defaultdict(lambda: defaultdict(list))
    for k in mem:
        r = labels.get(k)
        if r and r.get("species"):
            by[r["species"]][r["source"]].append(k)
    shared = sorted((sp for sp, v in by.items()
                     if len(v.get("ozfish", [])) >= 3 and
                     len(v.get("fishwio", [])) >= 3),
                    key=lambda sp: -min(len(by[sp]["ozfish"]),
                                        len(by[sp]["fishwio"])))
    ex = []
    pick = []
    if shared:
        sp = a.species or shared[0]
        for s in ("ozfish", "fishwio"):
            pick += [(k, s, sp) for k in sorted(by[sp][s])[:: max(1, len(
                by[sp][s]) // 3)][:3]]
    fn_sp = sorted(((sp, v["fathomnet"]) for sp, v in by.items()
                    if len(v.get("fathomnet", [])) >= 3),
                   key=lambda t: -len(t[1]))
    if fn_sp:
        sp, ks = fn_sp[0]
        pick += [(k, "fathomnet", sp) for k in sorted(ks)[:3]]
    for k, s, sp in pick:
        im = Image.open(io.BytesIO(read_member(tar, mem, k))).convert("RGB")
        f = f"example_{s}_{k}.png"
        im.save(os.path.join(img_dir, f))
        ex.append([f, s, sp, min(im.size), "dev"])
    # YOLO-BRUV has no dev records: first records of its main crops tar
    yb = sorted(glob.glob(os.path.join(a.shards, "crops",
                                       "crops-yolo-bruv-*.tar")))
    if yb:
        with tarfile.open(yb[0]) as tf:
            n = 0
            for m in tf:
                if m.name.endswith((".jpg", ".png")) and n < 3:
                    im = Image.open(tf.extractfile(m)).convert("RGB")
                    f = f"example_yolo-bruv_{m.name.rsplit('.', 1)[0]}.png"
                    im.save(os.path.join(img_dir, f))
                    ex.append([f, "yolo-bruv", "", min(im.size), "main"])
                    n += 1
    write_csv(os.path.join(a.out, "examples.csv"),
              ["file", "source", "species", "short_side", "from"], ex)
    # one OzFish animal under every geometry, through the real pipeline
    ozk = a.geometry_key or next((k for k, s, _ in pick if s == "ozfish"),
                                 None)
    rows = []
    if ozk:
        mean = (0.485, 0.456, 0.406)       # display only; fill colour
        raw = Image.open(io.BytesIO(read_member(tar, mem, ozk)))
        for g, set_name in (("letterbox", "crops"), ("distort", "crops"),
                            ("square_m00", "square-m00"),
                            ("square_m10", "square-m10"),
                            ("native", "crops")):
            t, m = (tar, mem) if set_name == "crops" else dev(set_name)
            im = Image.open(io.BytesIO(read_member(t, m, ozk)))
            gi = apply_geometry(im, g, mean)
            if g == "native":
                h, w = naflex_size(gi.height, gi.width, 16, 256)
                gi = gi.resize((w, h), Image.BICUBIC)
            else:
                gi = gi.resize((224, 224), Image.BICUBIC)
            f = f"geometry_{g}.png"
            gi.save(os.path.join(img_dir, f))
            rows.append([g, f, gi.width, gi.height, raw.width, raw.height])
        raw.convert("RGB").save(os.path.join(img_dir, "geometry_raw.png"))
        rows.append(["raw", "geometry_raw.png", raw.width, raw.height,
                     raw.width, raw.height])
    write_csv(os.path.join(a.out, "geometry.csv"),
              ["geometry", "file", "width", "height", "raw_width",
               "raw_height"], rows)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collation")
    p.add_argument("--coco", action="append", required=True)
    p.add_argument("--fn-groups")
    p = sub.add_parser("split")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--dev-deployments")
    p.add_argument("--fn-dev-groups")
    p.add_argument("--profile-report")
    p = sub.add_parser("testable")
    p.add_argument("--labels", required=True)
    p.add_argument("--draws-dir", required=True)
    p = sub.add_parser("wp8")
    p.add_argument("--archive", required=True,
                   help="results/wp8_readout_sweep")
    p.add_argument("--spec", default="configs/extract/wp9_readouts.yaml",
                   help="the readouts WP9 keeps (kept.csv)")
    p = sub.add_parser("images")
    p.add_argument("--labels", required=True)
    p.add_argument("--dev-shards", required=True)
    p.add_argument("--shards", required=True)
    p.add_argument("--species", default=None)
    p.add_argument("--geometry-key", default=None)
    for p in sub.choices.values():
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"collation": cmd_collation, "split": cmd_split, "testable": cmd_testable,
     "wp8": cmd_wp8, "images": cmd_images}[a.cmd](a)


if __name__ == "__main__":
    main()

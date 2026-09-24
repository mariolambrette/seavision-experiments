#!/usr/bin/env python3
"""Settle the FathomNet crop-size disagreement.

THE DISAGREEMENT
    `dataset_summary.py` reports FathomNet at median short side 101 px, 34.4%
    under 64 px and 15.9% under 32 px (plan §3.3b, §5.5, WP6).

    A uniform random sample of 200 FathomNet records drawn from the built
    `crops` shards puts 57% under 64 px and 27.5% under 32 px. At n = 200 the
    standard error is 3.4 points, so that is about 6.7 SE out, and the two are
    internally incompatible anyway: 57% of crops below 64 px cannot sit
    underneath a median of 101 px.

    The build is not the culprit. Only 2.4% of FathomNet crops were resized,
    and `min_short_side` means a resize can never push a crop BELOW 64 px, so
    nothing in the shard build can move records into the bins where the
    disagreement lives.

WHAT THIS DOES, AND WHY IT IS NOT JUST "RECOMPUTE IT"
    Recomputing gives one number and no explanation, and an unexplained
    correction is indistinguishable from a second mistake. So this computes
    the distribution under every definition of "crop size" the data admits,
    and prints which of them reproduces the recorded figures. The definition
    that matches is, by elimination, the one `dataset_summary.py` used -- which
    says what to fix rather than merely that something is wrong.

    image_dims        the stored image's width x height, every image
    image_dims_lab    same, but only images carrying an annotation
    ann_bbox          every annotation's bbox, one row per annotation
    crop_correct      the honest definition: for `crop_provenance == frame`
                      the image is a FRAME, so the crop is the annotation
                      bbox; everywhere else the image IS the crop. One row per
                      crop as the shard builder defines a crop
    frame_bbox        source_meta.frame_bbox, the box on the parent frame
                      before clamping (FathomNet)
    frame_bbox_used   source_meta.frame_bbox_used, after clamping

    `crop_correct` is the one that should be quoted. The rest exist to locate
    the fault.

    Note the trap `crop_correct` exists for: yolo-bruv's images are frames of
    around 1920x1080, so `image_dims` for that source reports the FRAME size,
    not the crop size, and would put its median short side near 1080 instead
    of 47.

    --shards additionally reads native_width/native_height back out of the
    built tars. The shards are derived from the COCO, so they must agree with
    crop_correct; checking costs an I/O pass and turns "must" into "does".

    python scratch/crop_size_audit.py --config configs/shards_crops.yaml
    python scratch/crop_size_audit.py --config configs/shards_crops.yaml --shards
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from array import array
from collections import defaultdict

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required")

# Quoted from plan §5.5 / WP6 so the comparison is against what is written
# down, not against memory. under_64 / under_32 are percentages.
RECORDED = {
    "yolo-bruv": {"p10": 25, "median": 47, "p90": 112,
                  "under_64": 67.9, "under_32": 22.9},
    "fishwio":   {"p10": 52, "median": 88, "p90": 165,
                  "under_64": 22.5, "under_32": 0.3},
    "fathomnet": {"p10": 23, "median": 101, "p90": 341,
                  "under_64": 34.4, "under_32": 15.9},
    "ozfish":    {"p10": 67, "median": 121, "p90": 283,
                  "under_64": 7.8, "under_32": 0.4},
}

DEFS = ["crop_correct", "image_dims", "image_dims_lab", "ann_bbox",
        "frame_bbox", "frame_bbox_used"]


def pct(sorted_arr, q):
    if not len(sorted_arr):
        return 0
    i = min(len(sorted_arr) - 1, max(0, int(round(q * (len(sorted_arr) - 1)))))
    return sorted_arr[i]


def summarise(vals):
    """-> dict of the five figures the documents quote."""
    if not len(vals):
        return None
    s = sorted(vals)
    n = len(s)
    return {
        "n": n,
        "p10": pct(s, 0.10), "median": pct(s, 0.50), "p90": pct(s, 0.90),
        "under_64": 100.0 * sum(1 for v in s if v < 64) / n,
        "under_32": 100.0 * sum(1 for v in s if v < 32) / n,
    }


def collect(coco_paths):
    """-> {(source, definition): array('i') of short sides}"""
    out = defaultdict(lambda: array("i"))
    for path in coco_paths:
        print(f"  reading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}

        ann_by_image = defaultdict(list)
        for an in doc.get("annotations", []):
            ann_by_image[an["image_id"]].append(an)

        for im in doc.get("images", []):
            src = src_of.get(im.get("dataset_id"), "unknown")
            anns = ann_by_image.get(im["id"], [])
            w, h = im.get("width"), im.get("height")
            prov = im.get("crop_provenance")

            if w and h:
                out[(src, "image_dims")].append(min(int(w), int(h)))
                if anns:
                    out[(src, "image_dims_lab")].append(min(int(w), int(h)))

            for an in anns:
                b = an.get("bbox") or []
                if len(b) >= 4 and b[2] >= 1 and b[3] >= 1:
                    out[(src, "ann_bbox")].append(int(min(b[2], b[3])))

            # the honest one: what the shard builder calls a crop
            if prov == "frame":
                for an in anns:
                    b = an.get("bbox") or []
                    if len(b) >= 4 and b[2] >= 1 and b[3] >= 1:
                        out[(src, "crop_correct")].append(int(min(b[2], b[3])))
            elif w and h:
                out[(src, "crop_correct")].append(min(int(w), int(h)))

            sm = im.get("source_meta") or {}
            for key, name in (("frame_bbox", "frame_bbox"),
                              ("frame_bbox_used", "frame_bbox_used")):
                b = sm.get(key)
                if b and len(b) >= 4 and b[2] >= 1 and b[3] >= 1:
                    out[(src, name)].append(int(min(b[2], b[3])))

        del doc, ann_by_image
    return out


def from_shards(out_dir, set_name):
    """Read native_width/native_height back out of the tars. Only the .json
    members are parsed; the images are skipped without being read."""
    out = defaultdict(lambda: array("i"))
    tars = sorted(f for f in os.listdir(out_dir) if f.endswith(".tar"))
    for t in tars:
        stem = t[:-4]
        if stem.startswith(set_name + "-"):
            stem = stem[len(set_name) + 1:]
        src = stem.rsplit("-", 1)[0]
        n = 0
        with tarfile.open(os.path.join(out_dir, t)) as tf:
            for info in tf:
                if not info.name.endswith(".json"):
                    continue
                m = json.loads(tf.extractfile(info).read())
                nw, nh = m.get("native_width"), m.get("native_height")
                if nw and nh:
                    out[(src, "shard_native")].append(min(int(nw), int(nh)))
                    n += 1
        print(f"  {t}: {n:,} records", flush=True)
    return out


def row(label, s, width=26):
    return (f"{label:<{width}} {s['n']:>10,} {s['p10']:>6} {s['median']:>7} "
            f"{s['p90']:>6} {s['under_64']:>8.1f} {s['under_32']:>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", action="store_true",
                    help="also read the built tars back (slow, I/O bound)")
    ap.add_argument("--csv", help="write every definition to this CSV")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    print("reading the collation")
    data = collect(cfg["coco"])
    if args.shards:
        print("\nreading the shards")
        data.update(from_shards(cfg["out_dir"], cfg.get("name", "crops")))

    defs = DEFS + (["shard_native"] if args.shards else [])
    sources = sorted({s for s, _d in data})

    hdr = (f"{'definition':<26} {'n':>10} {'p10':>6} {'median':>7} {'p90':>6} "
           f"{'<64px%':>8} {'<32px%':>8}")
    results = {}
    for src in sources:
        print(f"\n{src}")
        print(hdr)
        print("-" * 76)
        rec = RECORDED.get(src)
        if rec:
            print(f"{'RECORDED (docs)':<26} {'--':>10} {rec['p10']:>6} "
                  f"{rec['median']:>7} {rec['p90']:>6} "
                  f"{rec['under_64']:>8.1f} {rec['under_32']:>8.1f}")
        for d in defs:
            s = summarise(data.get((src, d), array("i")))
            if s is None:
                continue
            results[(src, d)] = s
            print(row(d, s))

    # ---- which definition reproduces the documents? --------------------
    print("\n\nWHICH DEFINITION MATCHES THE RECORDED FIGURES")
    print("-" * 76)
    print("distance = mean absolute difference on <64px% and <32px%.")
    print("A definition at ~0 is the one dataset_summary.py must have used.\n")
    for src in sources:
        rec = RECORDED.get(src)
        if not rec:
            continue
        scored = []
        for d in defs:
            s = results.get((src, d))
            if not s:
                continue
            dist = (abs(s["under_64"] - rec["under_64"])
                    + abs(s["under_32"] - rec["under_32"])) / 2
            scored.append((dist, d))
        scored.sort()
        best_d, best = scored[0][1], scored[0][0]
        print(f"{src:<12} closest: {best_d:<18} distance {best:6.1f} pts"
              + ("   <-- matches" if best < 1.0 else
                 "   <-- NOTHING MATCHES"))
        for dist, d in scored[1:3]:
            print(f"{'':<12}   then:  {d:<18} distance {dist:6.1f} pts")

    print("\nREAD IT LIKE THIS")
    print("-" * 76)
    print("  crop_correct matches the documents  -> the documents are right")
    print("     and the shard sample was the fluke. Say so and move on.")
    print("  a DIFFERENT definition matches      -> that is the bug. The")
    print("     documents quote that definition; crop_correct is the truth,")
    print("     so correct §3.3b, §5.5, WP5 and WP6 to crop_correct and fix")
    print("     dataset_summary.py to compute it.")
    print("  nothing matches                     -> neither figure describes")
    print("     the collation as it now stands. Most likely the documented")
    print("     numbers predate a rebuild. Do not patch a number you cannot")
    print("     reproduce; recompute the whole table.")
    print("\n  --shards disagreeing with crop_correct would be a separate and")
    print("  worse fault: the built set would not match the COCO it was")
    print("  built from, which level 2 of build_shards --verify denies.")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["source", "definition", "n", "p10", "median", "p90",
                         "under_64_pct", "under_32_pct"])
            for (src, d), s in sorted(results.items()):
                wr.writerow([src, d, s["n"], s["p10"], s["median"], s["p90"],
                             f"{s['under_64']:.2f}", f"{s['under_32']:.2f}"])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

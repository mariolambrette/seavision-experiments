#!/usr/bin/env python3
"""Re-derive FishWIO's track structure and its design effect.

WHY THIS EXISTS. WP2 and plan §7.4 record 43,062 tracks across 113,087 crops,
mean 2.63 crops per track, max 131, giving DEFF 1.81 at rho = 0.5. WP19 is told
to reuse that figure rather than re-derive it. But the code that produced it was
never committed -- `git log --all -S "43,062"` returns nothing on any branch, and
the one pickaxe hit on the plain form is a coincidental substring inside a
FathomNet download manifest. So the figure currently has no reproducible source,
which is the WP7 dirty-manifest problem one step worse: there the build was
sound and only the record was wrong; here there is no record.

This reproduces it, or shows that it cannot be reproduced.

WHAT A TRACK IS HERE. FishWIO annotates fish across consecutive frames of one
video, so crops of one individual are not independent observations. A track is
built by linking crops that share a video and a taxon, sit within `window`
frames of each other, and whose boxes overlap by at least `iou`. Linking is
transitive (union-find), so a fish that drifts across the frame still forms one
track even where its first and last boxes do not overlap.

TWO PARAMETERS WE DO NOT KNOW. The original `window` and `iou` were never
recorded, so a single run cannot be compared against 43,062 honestly. This
sweeps both and prints the surface. If a cell reproduces the recorded figures
the setting is recovered; if the counts move sharply across neighbouring cells,
the recorded figure was never robust and that is worth knowing before it reaches
a methods section.

THE OZFISH LESSON, APPLIED. That measurement gave 1.001 and then 1.806 before
either was trustworthy. What fixed it was a check needing no threshold at all,
plus a null comparison. Both are here:

  * a NULL -- box coordinates permuted among crops within each (video, taxon)
    group, frames kept. Real tracks must beat it by a wide margin, or the
    "tracks" are just boxes that happen to overlap.
  * an ASSUMPTION-FREE check -- the mean IoU of crop pairs exactly one frame
    apart, real against null, which needs no window and no threshold.

A clean-looking number is not evidence until something has tried to produce it
by accident.

USAGE
    python scratch/fishwio_tracks.py --selftest
    python scratch/fishwio_tracks.py --coco D:/marineai/dataset/collated/seavision.json
    python scratch/fishwio_tracks.py --coco ... --fix 5,0.3 --out fishwio_tracks.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

SOURCE = "fishwio"
WINDOWS = [1, 2, 5, 10, 30]
IOUS = [0.0, 0.1, 0.3, 0.5]
RECORDED = {"tracks": 43062, "crops": 113087, "mean": 2.63, "max": 131}


# ----------------------------------------------------------------- union-find
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# -------------------------------------------------------------------- loading
BOX_KEYSETS = [("crop_x", "crop_y", "crop_w", "crop_h"),
               ("x", "y", "w", "h"),
               ("box_x", "box_y", "box_w", "box_h"),
               ("offset_x", "offset_y", "width", "height")]
VIDEO_KEYS = ["video_name", "video", "source_video", "clip"]
FRAME_KEYS = ["frame", "frame_index", "frame_number", "frame_id"]


def pick_key(sm, candidates):
    for k in candidates:
        if k in sm and sm[k] is not None:
            return k
    return None


LIST_BOX_KEYS = ["frame_bbox", "crop_box", "box", "bbox", "geometry"]


def extract_box(sm, forced=None):
    """-> ((x, y, w, h), key_used) or (None, None).

    `frame_bbox` is tried first because it is the collation's own name for the
    box on the parent frame -- the OzFish and FathomNet converters use it too,
    and the FishWIO one follows suit. Returns None rather than guessing, so a
    wrong key stops the run instead of producing a plausible number from
    nothing, which is the failure this whole script exists to correct.
    """
    if forced:
        v = sm.get(forced)
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            return tuple(float(t) for t in v[:4]), forced
        return None, None
    for k in LIST_BOX_KEYS:
        v = sm.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            return tuple(float(t) for t in v[:4]), k
    for ks in BOX_KEYSETS:
        if all(k in sm and sm[k] is not None for k in ks):
            return tuple(float(sm[k]) for k in ks), "+".join(ks)
    return None, None


def load(coco_path, verbose=True, box_key=None):
    """-> list of (video, category_id, frame, box), plus a diagnostic dict.

    Only crops that CARRY AN ANNOTATION are returned. FishWIO's 1,600 shipped
    background crops sit in the collation as images with no annotation (WP7's
    audit found them leaking into an annotation pool once already); they are
    not fish and must not enter a track count.
    """
    with open(coco_path, encoding="utf-8") as fh:
        doc = json.load(fh)

    ds_id = None
    for d in doc.get("datasets", []):
        if d.get("name") == SOURCE:
            ds_id = d["id"]
    if ds_id is None:
        sys.exit(f"no dataset named {SOURCE!r} in {coco_path}; found "
                 f"{[d.get('name') for d in doc.get('datasets', [])]}")

    cat_of = {}
    for an in doc.get("annotations", []):
        cat_of.setdefault(an["image_id"], an.get("category_id"))

    rows, no_ann, no_fields = [], 0, 0
    sample_sm = None
    vk = fk = bk = None
    for im in doc.get("images", []):
        if im.get("dataset_id") != ds_id:
            continue
        sm = im.get("source_meta") or {}
        if sample_sm is None:
            sample_sm = sm
            vk, fk = pick_key(sm, VIDEO_KEYS), pick_key(sm, FRAME_KEYS)
        if im["id"] not in cat_of:
            no_ann += 1
            continue
        box, used = extract_box(sm, box_key)
        if used and bk is None:
            bk = used
        v = sm.get(vk) if vk else None
        f = sm.get(fk) if fk else None
        if box is None or v is None or f is None:
            no_fields += 1
            continue
        rows.append((str(v), cat_of[im["id"]], int(f), box))

    diag = {"video_key": vk, "frame_key": fk, "box_key": bk,
            "crops_with_annotation": len(rows),
            "images_without_annotation": no_ann,
            "images_missing_fields": no_fields,
            "sample_source_meta_keys": sorted(sample_sm or {})}
    if verbose:
        print(f"  video key   : {vk}")
        print(f"  frame key   : {fk}")
        print(f"  box key     : {bk}")
        print(f"  crops       : {len(rows):,} with an annotation")
        print(f"  skipped     : {no_ann:,} with none (FishWIO ships ~1,600 "
              f"background crops), {no_fields:,} missing video/frame/box")
        if no_fields or not rows:
            print(f"  ! source_meta keys seen: {sorted(sample_sm or {})}")
            print("  ! if the box/video/frame keys above are wrong, pass them "
                  "explicitly rather than letting this guess")
    if not rows:
        sys.exit("no usable FishWIO crops -- fix the key detection above "
                 "before trusting anything below")
    return rows, diag


# ------------------------------------------------------------------- tracking
def group(rows):
    g = defaultdict(list)
    for v, c, f, b in rows:
        g[(v, c)].append((f, b))
    for k in g:
        g[k].sort(key=lambda t: t[0])
    return g


def link(groups, window, thr):
    """-> list of track sizes."""
    sizes = []
    for items in groups.values():
        n = len(items)
        if n == 1:
            sizes.append(1)
            continue
        d = DSU(n)
        for i in range(n):
            fi, bi = items[i]
            for j in range(i + 1, n):
                fj, bj = items[j]
                if fj - fi > window:
                    break            # sorted by frame, so no later j qualifies
                if fj == fi:
                    continue         # two fish in one frame are not one track
                v = iou(bi, bj)
                if (v >= thr) if thr > 0 else (v > 0):
                    d.union(i, j)
        sizes.extend(Counter(d.find(i) for i in range(n)).values())
    return sizes


def shuffled(groups, rng):
    """Null: permute boxes among crops within each group, keep frames."""
    out = {}
    for k, items in groups.items():
        boxes = [b for _f, b in items]
        rng.shuffle(boxes)
        out[k] = [(f, boxes[i]) for i, (f, _b) in enumerate(items)]
    return out


def summarise(sizes, rho):
    n_tracks = len(sizes)
    n_crops = sum(sizes)
    mean = n_crops / n_tracks if n_tracks else 0.0
    kish = sum(m * m for m in sizes) / n_crops if n_crops else 0.0
    return {"tracks": n_tracks, "crops": n_crops, "mean": round(mean, 3),
            "max": max(sizes) if sizes else 0,
            "deff_mean": round(1 + (mean - 1) * rho, 3),
            "deff_kish": round(1 + (kish - 1) * rho, 3),
            "mean_weighted": round(kish, 3),
            "effective_n": int(round(n_crops / (1 + (kish - 1) * rho)))
            if n_crops else 0}


def gap1_iou(groups):
    """Assumption-free: mean IoU of pairs exactly one frame apart. No window,
    no threshold, so it cannot be tuned into agreement."""
    vals = []
    for items in groups.values():
        for i in range(len(items) - 1):
            if items[i + 1][0] - items[i][0] == 1:
                vals.append(iou(items[i][1], items[i + 1][1]))
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


# ----------------------------------------------------------------------- main
def run(rows, rho, seed, fix=None):
    groups = group(rows)
    rng = random.Random(seed)
    null_groups = shuffled(groups, rng)

    real_iou, n_pairs = gap1_iou(groups)
    null_iou, _ = gap1_iou(null_groups)
    print(f"\nASSUMPTION-FREE CHECK  (no window, no threshold)")
    print(f"  mean IoU of crop pairs one frame apart, over {n_pairs:,} pairs")
    ratio = f"   ratio {real_iou / null_iou:.1f}x" if null_iou > 0 else ""
    print(f"    real {real_iou:.3f}   null {null_iou:.3f}{ratio}")
    if real_iou < 2 * max(null_iou, 1e-9):
        print("  ! real barely beats the null. Boxes one frame apart overlap "
              "about as much by\n    chance as they do for real, so there may "
              "be no track structure to find and\n    every number below is "
              "an artefact of the thresholds.")

    cells = [(w, t) for w in WINDOWS for t in IOUS] if not fix else [fix]
    print(f"\nSWEEP  (rho = {rho})")
    print(f"  {'win':>4} {'iou':>5} {'tracks':>9} {'mean':>6} {'max':>5} "
          f"{'DEFF(mean)':>11} {'DEFF(Kish)':>11} {'null tracks':>12}")
    print("  " + "-" * 72)
    results = []
    for w, t in cells:
        r = summarise(link(groups, w, t), rho)
        nr = summarise(link(null_groups, w, t), rho)
        r["window"], r["iou"] = w, t
        r["null_tracks"], r["null_mean"] = nr["tracks"], nr["mean"]
        results.append(r)
        hit = ("  <== reproduces the recorded figure"
               if abs(r["tracks"] - RECORDED["tracks"]) <= 50
               and abs(r["mean"] - RECORDED["mean"]) <= 0.02 else "")
        print(f"  {w:>4} {t:>5.2f} {r['tracks']:>9,} {r['mean']:>6.2f} "
              f"{r['max']:>5,} {r['deff_mean']:>11.3f} {r['deff_kish']:>11.3f} "
              f"{nr['tracks']:>12,}{hit}")

    print(f"\n  recorded: {RECORDED['tracks']:,} tracks over "
          f"{RECORDED['crops']:,} crops, mean {RECORDED['mean']}, "
          f"max {RECORDED['max']}, DEFF 1.81")
    if not any(abs(r["tracks"] - RECORDED["tracks"]) <= 50 for r in results):
        print("  ! no cell reproduces it. Either the original used settings "
              "outside this sweep,\n    a different linking rule, or a "
              "different crop set. Do not quote 1.81 until\n    this is "
              "resolved -- state the rule and the setting alongside whatever "
              "is used.")

    print("""
  READING DEFF. The recorded 1.81 is 1 + (mean - 1) * rho on the UNWEIGHTED
  mean track length. With clusters this unequal -- max 131 against a mean near
  2.6 -- Kish's estimator, 1 + (sum m^2 / sum m - 1) * rho, is the right one,
  and it is larger. If the two columns differ materially then the recorded
  figure UNDERSTATES the design effect, every interval computed from it is too
  narrow, and plan §7.4 and WP19 need the Kish column instead.""")
    return results


# ------------------------------------------------------------------- selftest
def selftest():
    """Fixture with a known answer, including a case that must FAIL to link.
    A test that only ever passes proves nothing (WP6a)."""
    rows = [
        # V1/T1: one fish drifting over three frames -> one track of 3
        ("V1", 1, 10, (100, 100, 50, 50)),
        ("V1", 1, 11, (102, 101, 50, 50)),
        ("V1", 1, 12, (104, 103, 50, 50)),
        # V1/T1: a second fish elsewhere over two frames -> one track of 2
        ("V1", 1, 10, (500, 400, 40, 40)),
        ("V1", 1, 11, (503, 402, 40, 40)),
        # V1/T2: a single crop -> one track of 1
        ("V1", 2, 50, (10, 10, 20, 20)),
        # V2/T1: same place, frames 1 and 5 -> separate at window<4, one at >=4
        ("V2", 1, 1, (0, 0, 30, 30)),
        ("V2", 1, 5, (1, 1, 30, 30)),
    ]
    g = group(rows)
    a = summarise(link(g, 2, 0.3), 0.5)
    b = summarise(link(g, 5, 0.3), 0.5)
    c = summarise(link(g, 2, 0.99), 0.5)     # threshold too high to link
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {got} "
              f"({'expected ' + str(want)})")

    chk("window 2, tracks", a["tracks"], 5)
    chk("window 2, crops", a["crops"], 8)
    chk("window 2, max", a["max"], 3)
    chk("window 5 merges the gap-4 pair", b["tracks"], 4)
    chk("iou 0.99 links nothing", c["tracks"], 8)
    chk("same frame never links", link(group([
        ("V", 1, 7, (0, 0, 10, 10)), ("V", 1, 7, (0, 0, 10, 10))]), 5, 0.3),
        [1, 1])
    # Kish must exceed the plain mean whenever clusters are unequal
    k = summarise([1] * 9 + [11], 0.5)
    chk("Kish > unweighted mean on unequal clusters",
        k["mean_weighted"] > k["mean"], True)
    # Stage 2: plant a known track structure, write it as a COCO, and read it
    # back through load(). The hand-built fixture above tests the linker only;
    # this tests the key detection and the background-crop exclusion, which is
    # where a silent failure would actually come from.
    import tempfile
    rng = random.Random(3)
    images, anns, iid, aid, planted = [], [], 0, 0, []
    def add(video, frame, box, cat):
        nonlocal iid, aid
        iid += 1
        images.append({"id": iid, "file_name": f"f{iid}.jpg", "dataset_id": 7,
                       "source_meta": {"video_name": video, "frame": frame,
                                       "crop_x": box[0], "crop_y": box[1],
                                       "crop_w": box[2], "crop_h": box[3]}})
        if cat is not None:
            aid += 1
            anns.append({"id": aid, "image_id": iid, "category_id": cat,
                         "bbox": [0, 0, box[2], box[3]]})
    for v in range(20):
        for t in range(20):
            L = rng.choice([1, 1, 1, 2, 2, 3, 5, 9])
            planted.append(L)
            x, y = rng.randrange(0, 900), rng.randrange(0, 900)
            f0 = rng.randrange(0, 5000)
            for k in range(L):
                add(f"V{v}", f0 + k, (x + 2 * k, y + k, 60, 60), 100 + (t % 3))
    for i in range(50):                       # background: no annotation
        add("VBG", i, (0, 0, 30, 30), None)
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"datasets": [{"id": 7, "name": SOURCE}],
                   "images": images, "annotations": anns}, fh)
    try:
        rows, diag = load(path, verbose=False)
        got = summarise(link(group(rows), 2, 0.3), 0.5)
    finally:
        os.unlink(path)
    print()
    chk("loader finds the video key", diag["video_key"], "video_name")
    chk("loader finds the frame key", diag["frame_key"], "frame")
    chk("frame_bbox wins over scalar keys",
        extract_box({"frame_bbox": [1, 2, 3, 4], "crop_x": 9, "crop_y": 9,
                     "crop_w": 9, "crop_h": 9})[1], "frame_bbox")
    chk("a wrong forced key returns nothing, not a guess",
        extract_box({"frame_bbox": [1, 2, 3, 4]}, forced="nope")[0], None)
    chk("background crops excluded", diag["images_without_annotation"], 50)
    chk("no crop silently dropped", diag["images_missing_fields"], 0)
    chk("all fish crops read", got["crops"], sum(planted))
    chk("longest planted track recovered", got["max"], max(planted))
    # Two planted tracks can coincide in space and time within one
    # (video, taxon); merging them is correct, so allow a small shortfall.
    near = abs(got["tracks"] - len(planted)) <= 0.02 * len(planted)
    chk(f"planted tracks recovered ({got['tracks']} of {len(planted)})",
        near, True)

    print("\n  " + ("all self-tests pass" if ok else "SELF-TEST FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", help="collated/seavision.json")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fix", help="freeze one setting, e.g. 5,0.3")
    ap.add_argument("--box-key", default=None,
                    help="source_meta key holding [x, y, w, h] on the parent "
                         "frame; auto-detected, override if a converter "
                         "renames it")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.coco:
        sys.exit("--coco is required (or --selftest)")

    print(f"reading {args.coco}")
    rows, diag = load(args.coco, box_key=args.box_key)
    fix = None
    if args.fix:
        w, t = args.fix.split(",")
        fix = (int(w), float(t))
    results = run(rows, args.rho, args.seed, fix)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"source": SOURCE, "coco": os.path.abspath(args.coco),
                       "rho": args.rho, "seed": args.seed,
                       "diagnostics": diag, "recorded": RECORDED,
                       "results": results}, fh, indent=2)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()

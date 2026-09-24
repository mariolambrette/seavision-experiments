#!/usr/bin/env python3
"""How often does a square crop shift, shrink, or have to be padded?

Pure arithmetic over the boxes and their frame sizes -- no image is decoded
except to read a header. The point is to choose the stored margin from a
measurement rather than a feeling, the same way `crop_size_profile.py` chose
the long-side cap.

THE RULE BEING PROFILED
    side  S = round(max(w, h) * (1 + margin))        margin 0.10 -> x 1.1
    ideal the square centred on the box's centre
    fit   shift the origin into the frame; the SIDE never changes

    Shifting is safe in a way worth stating: provided S is no larger than the
    frame's short side, a square clamped into the frame still contains the
    whole box. Proof by the two cases -- unclamped, the square is centred on
    the box and S >= max(w,h); clamped to an edge, the box was within S/2 of
    that edge to begin with. So shifting moves the framing and never clips the
    animal.

    Three tiers, and the profile counts each:
      FITS            S <= min(fw, fh). Shift if needed; nothing is lost
      MARGIN_REDUCED  the box fits a square but the margin tips it over the
                      frame's short side. Shrink S to min(fw, fh) and keep the
                      whole animal at a smaller margin than asked for
      PADDED          max(w, h) itself exceeds the frame's short side -- a fish
                      spanning the full frame height. No square contains it, so
                      pad rather than clip, because clipping loses the animal
                      and padding only loses honesty, which a flag recovers

WHERE THE FRAME SIZE COMES FROM, PER SOURCE
    PrePARED   the image IS the frame, so width/height in the COCO
    FathomNet  source_meta.frame_size, recorded at ingest
    OzFish     nowhere in the collation -- so it is READ FROM THE FRAME
               HEADERS on disk. Guessing 1920x1080 because a sample looked
               that way is exactly the assumption that has cost time twice
               today; --assume-frame-size exists but says so loudly.
    FishWIO    no frames, ever. Excluded, and counted as excluded.

    python scratch/square_profile.py --config configs/shards_crops.yaml \\
        --ozfish-frames "D:/marineai/dataset/raw/ozfish/frames"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required")

from PIL import Image

SUFFIX = re.compile(r"^(?P<base>.+?\.png)-\d+-\d+\.png$", re.I)
FRAME_PAT = re.compile(
    r"^(?P<vid>[A-Za-z]+\d+)_(?P<cam>[LR])\.(?P<ext>avi|mp4|mpeg)\."
    r"(?P<frame>\d+)\.png$", re.I)


def ozfish_frame_sizes(root, assume=None):
    """-> {(video, CAMERA, frame): (w, h)}

    Image.open parses the header and stops; .load() is never called, so this
    is a stat-and-read-a-few-bytes per file rather than a decode.
    """
    if not root:
        return {}, "no --ozfish-frames given"
    out, bad = {}, 0
    for dirpath, _d, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".png"):
                continue
            m = SUFFIX.match(f)
            fm = FRAME_PAT.match(m.group("base") if m else f)
            if not fm:
                continue
            key = (fm["vid"], fm["cam"].upper(), int(fm["frame"]))
            if key in out:
                continue
            if assume:
                out[key] = assume
                continue
            try:
                with Image.open(os.path.join(dirpath, f)) as im:
                    out[key] = im.size
            except Exception:                            # noqa: BLE001
                bad += 1
    return out, (f"{bad:,} frame headers unreadable" if bad else None)


def tier(w, h, fw, fh, margin):
    """-> (tier, side, shift_x, shift_y, pad_fraction)"""
    long_side = max(w, h)
    short_frame = min(fw, fh)
    S = int(round(long_side * (1.0 + margin)))

    if long_side > short_frame:
        # nothing contains the animal; pad the shortfall
        S_fit = short_frame
        pad = 1.0 - (S_fit * S_fit) / float(S * S) if S else 0.0
        return "PADDED", S, 0, 0, max(0.0, pad)

    reduced = False
    if S > short_frame:
        S, reduced = short_frame, True

    cx, cy = w / 2.0, h / 2.0            # relative to the box origin
    return ("MARGIN_REDUCED" if reduced else "FITS"), S, cx, cy, 0.0


def shift_of(x0, y0, w, h, fw, fh, S):
    """-> (dx, dy) the square's origin moves from the box-centred ideal."""
    ix = (x0 + w / 2.0) - S / 2.0
    iy = (y0 + h / 2.0) - S / 2.0
    x = min(max(ix, 0.0), max(0.0, fw - S))
    y = min(max(iy, 0.0), max(0.0, fh - S))
    return x - ix, y - iy


def collect(cfg, oz_sizes):
    """-> list of (source, x0, y0, w, h, fw, fh); one per square-able crop."""
    rows, skipped = [], Counter()
    for path in cfg["coco"]:
        print(f"  reading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh_:
            doc = json.load(fh_)
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        by_image = defaultdict(list)
        for an in doc.get("annotations", []):
            by_image[an["image_id"]].append(an)

        for im in doc.get("images", []):
            src = src_of.get(im.get("dataset_id"), "unknown")
            prov = im.get("crop_provenance")
            sm = im.get("source_meta") or {}

            if prov == "frame":                      # PrePARED
                fw, fh = im.get("width"), im.get("height")
                if not fw or not fh:
                    skipped[f"{src}/no frame size"] += 1
                    continue
                for an in by_image.get(im["id"], []):
                    b = an.get("bbox") or []
                    if len(b) < 4 or b[2] < 2 or b[3] < 2:
                        continue
                    rows.append((src, float(b[0]), float(b[1]),
                                 float(b[2]), float(b[3]), int(fw), int(fh)))

            elif prov == "cut_from_frame":           # FathomNet
                fb = sm.get("frame_bbox_used") or sm.get("frame_bbox")
                fs = sm.get("frame_size")
                if not fb or not fs or len(fb) < 4 or len(fs) < 2:
                    skipped[f"{src}/no frame geometry"] += 1
                    continue
                rows.append((src, float(fb[0]), float(fb[1]), float(fb[2]),
                             float(fb[3]), int(fs[0]), int(fs[1])))

            elif sm.get("ozfish_uid") is not None:   # OzFish
                fb = sm.get("frame_bbox")
                key = (str(sm.get("video")), str(sm.get("camera") or "").upper(),
                       int(sm.get("frame", -1)))
                fs = oz_sizes.get(key)
                if not fb or len(fb) < 4:
                    skipped[f"{src}/no box"] += 1
                elif not fs:
                    skipped[f"{src}/frame not on disk"] += 1
                else:
                    rows.append((src, float(fb[0]), float(fb[1]), float(fb[2]),
                                 float(fb[3]), int(fs[0]), int(fs[1])))
            else:
                skipped[f"{src}/no frames exist"] += 1
        del doc, by_image
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ozfish-frames",
                    default="D:/marineai/dataset/raw/ozfish/frames")
    ap.add_argument("--assume-frame-size",
                    help="e.g. 1920x1080 -- skips reading OzFish headers. "
                         "Reported as an assumption, because it is one")
    ap.add_argument("--margins", nargs="*", type=float,
                    default=[0.0, 0.10, 0.25])
    ap.add_argument("--cap", type=int, default=1024,
                    help="the crops set's long-side cap, to report how many "
                         "squares would be resampled")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    assume = None
    if args.assume_frame_size:
        w, h = args.assume_frame_size.lower().split("x")
        assume = (int(w), int(h))
        print(f"! ASSUMING every OzFish frame is {assume[0]}x{assume[1]}. "
              f"Nothing checked this.")

    print("reading OzFish frame headers")
    oz_sizes, note = ozfish_frame_sizes(args.ozfish_frames, assume)
    print(f"  {len(oz_sizes):,} frames sized"
          + (f"   ! {note}" if note else ""))
    if oz_sizes and not assume:
        shapes = Counter(oz_sizes.values())
        print("  frame shapes: "
              + ", ".join(f"{w}x{h} ({n:,})" for (w, h), n in
                          shapes.most_common(4)))

    print("\nreading the collation")
    rows, skipped = collect(cfg, oz_sizes)
    print(f"  {len(rows):,} crops can produce a square")
    for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  excluded  {k:<34} {v:>10,}")

    per_src = Counter(r[0] for r in rows)
    print(f"\n  {'source':<12} {'square-able':>12}")
    for s, n in sorted(per_src.items()):
        print(f"  {s:<12} {n:>12,}")

    for margin in args.margins:
        print(f"\n\nMARGIN {margin:.0%}   side = max(w, h) x {1 + margin:.2f}")
        print("=" * 78)
        hdr = (f"{'source':<12} {'n':>10} {'fits':>8} {'shifted':>8} "
               f"{'reduced':>8} {'padded':>7} {'>cap':>7} {'med side':>9}")
        print(hdr)
        print("-" * 78)
        grand = Counter()
        for src in sorted(per_src):
            c, sides, shifts, pads = Counter(), [], [], []
            for (s, x0, y0, w, h, fw, fh) in rows:
                if s != src:
                    continue
                t, S, _cx, _cy, pad = tier(w, h, fw, fh, margin)
                c[t] += 1
                sides.append(S)
                if t == "PADDED":
                    pads.append(pad)
                else:
                    dx, dy = shift_of(x0, y0, w, h, fw, fh, S)
                    if abs(dx) > 0.5 or abs(dy) > 0.5:
                        c["shifted"] += 1
                        shifts.append(max(abs(dx), abs(dy)) / max(S, 1))
                if S > args.cap:
                    c["over_cap"] += 1
            n = sum(c[t] for t in ("FITS", "MARGIN_REDUCED", "PADDED"))
            sides.sort()
            med = sides[len(sides) // 2] if sides else 0
            print(f"{src:<12} {n:>10,} {c['FITS'] / max(1, n):>7.1%} "
                  f"{c['shifted'] / max(1, n):>7.1%} "
                  f"{c['MARGIN_REDUCED'] / max(1, n):>7.1%} "
                  f"{c['PADDED'] / max(1, n):>6.2%} "
                  f"{c['over_cap'] / max(1, n):>6.1%} {med:>9,}")
            for k in ("FITS", "MARGIN_REDUCED", "PADDED", "shifted",
                      "over_cap"):
                grand[k] += c[k]
            if shifts:
                shifts.sort()
                p50 = shifts[len(shifts) // 2]
                p90 = shifts[int(len(shifts) * 0.9)]
                print(f"{'':<12}   shift as a fraction of the side: "
                      f"median {p50:.1%}, p90 {p90:.1%}, "
                      f"max {shifts[-1]:.1%}")
            if pads:
                pads.sort()
                print(f"{'':<12}   ! padded area: median "
                      f"{pads[len(pads) // 2]:.1%}, max {pads[-1]:.1%}")
        n = sum(grand[t] for t in ("FITS", "MARGIN_REDUCED", "PADDED"))
        print("-" * 78)
        print(f"{'ALL':<12} {n:>10,} {grand['FITS'] / max(1, n):>7.1%} "
              f"{grand['shifted'] / max(1, n):>7.1%} "
              f"{grand['MARGIN_REDUCED'] / max(1, n):>7.1%} "
              f"{grand['PADDED'] / max(1, n):>6.2%} "
              f"{grand['over_cap'] / max(1, n):>6.1%}")

    print("\n\nHOW TO CHOOSE FROM THIS")
    print("-" * 78)
    print("  `shifted` is not a cost -- the animal is still whole, only the")
    print("  framing moved, and a deployment produces off-centre animals")
    print("  anyway. `reduced` costs margin but no animal. `padded` is the")
    print("  only tier that invents pixels, so it is the one to keep small.")
    print("  Pick the largest margin whose padded column is still negligible.")
    print("  `>cap` is how many squares the 1024 long-side cap would resample,")
    print("  which is the square set's only lossy step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

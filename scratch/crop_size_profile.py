#!/usr/bin/env python3
"""
Long-side distribution of the collation, to set the shard cap from data.

The shard build copies a crop's bytes verbatim unless its LONG side exceeds a
cap, in which case it is decoded, downscaled and re-encoded. So the cap is the
only lossy decision in the whole build, and it should be measured rather than
chosen by feel.

Capping the long side does resample the SHORT side on an elongated crop, and
the short side is where the species information lives -- so this reports what
each candidate cap would actually cost, in crops touched and in short-side
pixels lost, rather than just how many files exceed it.

    python crop_size_profile.py `
        --coco "D:/marineai/dataset/collated/seavision.json" `
        --coco "D:/marineai/dataset/collated/seavision_fathomnet.json"
"""

import argparse
import json
import os
from collections import Counter, defaultdict


def short_side(img, ann):
    if img.get("crop_provenance") == "frame":
        b = ann.get("bbox") or []
        if len(b) < 4:
            return None, None
        w, h = b[2], b[3]
    else:
        w, h = img.get("width"), img.get("height")
    if not w or not h or w <= 0 or h <= 0:
        return None, None
    return float(min(w, h)), float(max(w, h))


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", action="append", required=True)
    ap.add_argument("--caps", default="512,768,1024,1536,2048")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",")]

    per_src = defaultdict(lambda: {"long": [], "aspect": [], "pairs": []})
    for path in args.coco:
        print(f"  loading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        names = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        imgs = {i["id"]: i for i in doc.get("images", [])}

        # ONE ROW PER CROP, which is not one row per image.
        # Where crop_provenance is 'frame' the image IS a frame and every
        # annotation on it becomes a separate crop -- yolo-bruv is 15,945
        # crops across 2,667 frames, not 2,667. Deduplicating by image (as an
        # earlier version did) undercounted it sevenfold.
        seen_img = set()
        for an in doc.get("annotations", []):
            im = imgs.get(an.get("image_id"))
            if im is None:
                continue
            frame_src = im.get("crop_provenance") == "frame"
            if not frame_src:
                if im["id"] in seen_img:
                    continue
                seen_img.add(im["id"])
            sh, lo = short_side(im, an)
            if sh is None:
                continue
            d = per_src[names.get(im.get("dataset_id"), "unknown")]
            d["long"].append(lo)
            d["aspect"].append(lo / sh)
            d["pairs"].append((sh, lo))

        # An image with no annotation is background, and still occupies a
        # shard slot -- but only where the image is itself the crop.
        annotated = {an.get("image_id") for an in doc.get("annotations", [])}
        for im in doc.get("images", []):
            if im["id"] in annotated or im.get("crop_provenance") == "frame":
                continue
            sh, lo = short_side(im, {})
            if sh is None:
                continue
            d = per_src[names.get(im.get("dataset_id"), "unknown")]
            d["long"].append(lo)
            d["aspect"].append(lo / sh)
            d["pairs"].append((sh, lo))
        del doc, imgs

    print("\nLONG SIDE (px)")
    print("-" * 78)
    print(f"{'source':<14}{'n':>10}{'p50':>8}{'p90':>8}{'p99':>8}"
          f"{'max':>9}{'asp p50':>9}{'asp p99':>9}")
    allpairs = []
    for src in sorted(per_src):
        d = per_src[src]
        L = sorted(d["long"]); A = sorted(d["aspect"])
        allpairs += d["pairs"]
        print(f"{src:<14}{len(L):>10,}{pct(L,.5):>8.0f}{pct(L,.9):>8.0f}"
              f"{pct(L,.99):>8.0f}{max(L):>9.0f}{pct(A,.5):>9.2f}{pct(A,.99):>9.2f}")

    print("\nWHAT EACH CAP WOULD COST")
    print("-" * 78)
    print("  'touched'  = crops decoded, downscaled and re-encoded (the rest are")
    print("               copied byte-for-byte, losslessly)")
    print("  'short<64' = of those, how many end up with a short side under 64 px")
    print("               AFTER the downscale -- that is information destroyed,")
    print("               not merely re-packed")
    print()
    print(f"{'cap':>7}{'touched':>12}{'% of all':>10}{'short<64 after':>17}"
          f"{'worst short':>13}{'exempt@64':>11}")
    n = len(allpairs)
    for cap in caps:
        touched = [(sh, lo) for sh, lo in allpairs if lo > cap]
        after = [sh * cap / lo for sh, lo in touched]
        tiny = sum(1 for a in after if a < 64)
        worst = min(after) if after else float("nan")
        # With a short-side FLOOR, a crop is left oversized rather than
        # resampled below the floor -- so the lossy column becomes zero by
        # construction and the cost is a handful of large files instead.
        exempt = tiny
        print(f"{cap:>7}{len(touched):>12,}{100*len(touched)/n:>9.2f}%"
              f"{tiny:>17,}{worst:>13.0f}{exempt:>11,}")

    print()
    print("  'exempt@64' is what a SHORT-SIDE FLOOR would leave untouched: rather")
    print("  than resample a crop below 64 px on its short side, leave it")
    print("  oversized. That makes the lossy column zero at any cap, and costs a")
    print("  handful of large files instead of destroyed signal.")

    print("\nRead it this way: pick the smallest cap whose 'touched' count you are")
    print("content to re-encode AND whose 'short<64 after' is zero or negligible.")
    print("A cap that pushes crops under 64 px is destroying the signal that the")
    print("whole size analysis (plan 5.5) exists to measure.")


if __name__ == "__main__":
    main()

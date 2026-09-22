#!/usr/bin/env python
"""
Does OzFish contain the same fish twice, once per stereo camera?

Standard stereo-BRUVS practice is to annotate one camera and use the other
only for measurement. If that holds here, stereo duplication is zero and the
design effect of 1.04 stands. If it does not, every per-crop confidence
interval built on OzFish is too narrow.

The test runs in four steps, each only reached if the previous one leaves the
question open:

  1. Overall  L vs R crop counts.
     Overwhelmingly one camera -> done, no duplication.

  2. Per DEPLOYMENT.  A 50/50 overall split is NOT evidence of duplication:
     it is equally consistent with half the deployments annotated on the left
     camera and half on the right, which is still one view per fish. Only
     deployments carrying BOTH cameras can duplicate anything.

  3. Per FRAME within those deployments -- AFTER aligning the cameras.
     The two cameras are NOT frame-synchronised: plan section 3.3 records
     that they are offset by a constant frame count per deployment. Matching
     on raw frame index therefore compares different moments in time and
     finds almost nothing, which is exactly the false negative this script
     returned on its first run. The offset is estimated per deployment from
     the mode of the pairwise frame differences, then applied.

  4. Per TAXON within those frames. Same deployment, same frame, same taxon,
     both cameras is the signature of one animal recorded twice. This is an
     UPPER bound: a school of twenty gives the same signature whether or not
     the same individuals were boxed.

Read-only. Usage:

    python ozfish_stereo.py --coco "N:/marineai/dataset/collated/seavision.json"
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict

# A000001_L.avi.5107.806.371.922.448.png
FN = re.compile(r"^([A-Za-z]\d+)_([LR])\.(?:avi|mp4|mpeg)\.(\d+)\.", re.I)


def parse_name(s):
    m = FN.match(str(s))
    if not m:
        return None
    return m.group(1).upper(), m.group(2).upper(), int(m.group(3))


CAM_KEYS   = ("camera", "cam", "stereo_camera")
FRAME_KEYS = ("frame", "frame_index", "frame_no")
DEP_KEYS   = ("video", "deployment", "ozfish_uid")


def find_fields(im):
    """Deployment, camera and frame for one image.

    The OzFish converter already parsed these out of the filename into
    source_meta, so read them rather than re-parsing. Falls back to parsing a
    filename only if the structured fields are absent.
    """
    sm = im.get("source_meta") or {}

    cam = next((str(sm[k]).strip().upper() for k in CAM_KEYS
                if sm.get(k) not in (None, "")), None)
    frame = next((sm[k] for k in FRAME_KEYS
                  if sm.get(k) not in (None, "")), None)
    dep = (im.get("groups") or {}).get("deployment")
    if dep in (None, ""):
        dep = next((sm[k] for k in DEP_KEYS
                    if sm.get(k) not in (None, "")), None)

    if cam and frame is not None and dep:
        cam = cam[0] if cam[0] in ("L", "R") else cam
        try:
            return str(dep).upper(), cam, int(frame)
        except (TypeError, ValueError):
            return None

    for v in list(sm.values()) + [im.get("file_name")]:
        if isinstance(v, str):
            got = parse_name(v)
            if got:
                return got
    return None


def main():
    ap = argparse.ArgumentParser(description="OzFish stereo duplication test")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--source", default="ozfish")
    ap.add_argument("--null-trials", type=int, default=3,
                    help="randomised comparisons per deployment")
    ap.add_argument("--max-offset", type=int, default=2000,
                    help="largest plausible L/R frame offset to search")
    args = ap.parse_args()

    print(f"loading {args.coco} ...", flush=True)
    with open(args.coco, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    ds = {d["id"]: d["name"] for d in doc.get("datasets", [])}
    want = {i for i, n in ds.items() if n == args.source}
    if not want:
        sys.exit(f"no dataset named {args.source!r}; found: {sorted(ds.values())}")

    cats = {c["id"]: c.get("name") for c in doc.get("categories", [])}
    imgs = {im["id"]: im for im in doc["images"] if im.get("dataset_id") in want}
    print(f"  {len(imgs):,} {args.source} images")

    # stereo_note is written by the converter and may settle this outright
    notes = Counter()
    for im in imgs.values():
        notes[str((im.get("source_meta") or {}).get("stereo_note"))] += 1
    if len(notes) > 1 or "None" not in notes:
        print("\n  stereo_note values recorded at ingest:")
        for v, n in notes.most_common(8):
            print(f"    {n:>8,}  {v}")

    parsed, unparsed, sample_keys = {}, 0, None
    for i, im in imgs.items():
        got = find_fields(im)
        if got is None:
            unparsed += 1
            if sample_keys is None:
                sample_keys = sorted((im.get("source_meta") or {}).keys())
        else:
            parsed[i] = got

    if not parsed:
        print("\nCould not find deployment/camera/frame in any record.")
        print("source_meta keys on the first image:", sample_keys)
        sys.exit("nothing to test -- tell me those keys and I will adjust the parser")
    if unparsed:
        print(f"  !! {unparsed:,} images did not parse and are excluded")

    # taxon per image, via its annotations
    taxa = defaultdict(set)
    for a in doc.get("annotations", []):
        if a.get("image_id") in parsed and a.get("category_id") is not None:
            taxa[a["image_id"]].add(a["category_id"])

    # ---- step 0: count balance -------------------------------------------
    # The strongest evidence available without any frame alignment. If a
    # deployment holds 14 of a taxon on the left and 14 on the right, and
    # that holds across the board, the pairing is systematic. Overall balance
    # can hide per-deployment imbalance, so it is checked at that level.
    pair = defaultdict(Counter)
    for i, (dep, c, f) in parsed.items():
        for cat in taxa.get(i, ()):
            pair[(dep, cat)][c] += 1
    both_sides = [(k, v) for k, v in pair.items() if v.get("L") and v.get("R")]
    exact = sum(1 for _, v in both_sides if v["L"] == v["R"])
    close = sum(1 for _, v in both_sides
                if abs(v["L"] - v["R"]) <= max(1, 0.1 * max(v["L"], v["R"])))
    one_side = len(pair) - len(both_sides)
    print("\n" + "=" * 66)
    print("STEP 0  count balance per deployment and taxon (no alignment)")
    print("=" * 66)
    print(f"  deployment x taxon combinations       {len(pair):>8,}")
    print(f"  present in one camera only            {one_side:>8,} "
          f"({100*one_side/len(pair):.1f}%)" if pair else "")
    print(f"  present in both                       {len(both_sides):>8,}")
    if both_sides:
        print(f"  ... with EXACTLY equal counts         {exact:>8,} "
              f"({100*exact/len(both_sides):.1f}% of those)")
        print(f"  ... within 10%                        {close:>8,} "
              f"({100*close/len(both_sides):.1f}% of those)")
        print("\n  A high share of exact matches means an annotator worked both")
        print("  cameras on the same animals. A high share present in only one")
        print("  camera means they did not.")

    # ---- step 1: overall -------------------------------------------------
    cam = Counter(c for _, c, _ in parsed.values())
    tot = sum(cam.values())
    print("\n" + "=" * 66)
    print("STEP 1  overall camera split")
    print("=" * 66)
    for c in ("L", "R"):
        print(f"  {c}: {cam.get(c,0):>8,}  ({100*cam.get(c,0)/tot:5.1f}%)")
    minor = min(cam.get("L", 0), cam.get("R", 0)) / tot if tot else 0
    if minor < 0.01:
        print("\n  VERDICT: effectively one camera. No stereo duplication is")
        print("  possible. The design effect of 1.04 stands and §8 needs no")
        print("  stereo treatment.")
        return

    # ---- step 2: per deployment -----------------------------------------
    by_dep = defaultdict(Counter)
    for dep, c, _ in parsed.values():
        by_dep[dep][c] += 1
    both = {d for d, k in by_dep.items() if k.get("L") and k.get("R")}
    print("\n" + "=" * 66)
    print("STEP 2  per deployment")
    print("=" * 66)
    print(f"  deployments               {len(by_dep):>8,}")
    print(f"  left camera only          {sum(1 for d,k in by_dep.items() if k.get('L') and not k.get('R')):>8,}")
    print(f"  right camera only         {sum(1 for d,k in by_dep.items() if k.get('R') and not k.get('L')):>8,}")
    print(f"  BOTH cameras              {len(both):>8,}")
    crops_in_both = sum(sum(by_dep[d].values()) for d in both)
    print(f"  crops in both-camera deployments  {crops_in_both:>8,} "
          f"({100*crops_in_both/tot:.1f}% of OzFish)")
    if not both:
        print("\n  VERDICT: no deployment carries both cameras. Annotators worked")
        print("  one camera per deployment, so no fish can appear twice.")
        return

    # ---- step 3: align the cameras, then match frames --------------------
    # The rig's two cameras are offset by a constant frame count per
    # deployment (plan 3.3). Estimate that offset as the mode of the pairwise
    # differences between annotated L and R frames, then match on the aligned
    # index. Without this the test compares different moments and reports no
    # duplication however much there is.
    L = defaultdict(list)
    R = defaultdict(list)
    for i, (dep, c, f) in parsed.items():
        (L if c == "L" else R)[dep].append(f)

    def best_offset(lset, rset):
        """Offset maximising frame overlap, and the overlap it achieves.
        Both arguments are SETS of distinct frames -- counting crops here
        and frames in the denominator is how the first version reported
        113% alignment."""
        diffs = Counter()
        for l in lset:
            for r in rset:
                d = l - r
                if abs(d) <= args.max_offset:
                    diffs[d] += 1
        if not diffs:
            return None, 0
        off = diffs.most_common(1)[0][0]
        return off, sum(1 for l in lset if (l - off) in rset)

    rng = random.Random(0)
    offsets, aligned, null_aligned = {}, {}, {}
    for dep in both:
        lset, rset = set(L[dep]), set(R[dep])
        if not lset or not rset:
            continue
        off, hits = best_offset(lset, rset)
        if off is None:
            continue
        offsets[dep] = off
        aligned[dep] = hits
        # NULL: the same search against randomised R frames drawn from the
        # observed span. If a real deployment aligns no better than this,
        # the offset is fitted noise rather than rig sync.
        lo, hi = min(rset), max(rset)
        if hi > lo:
            trials = []
            for _ in range(args.null_trials):
                fake = {rng.randint(lo, hi) for _ in range(len(rset))}
                trials.append(best_offset(lset, fake)[1])
            null_aligned[dep] = sum(trials) / len(trials)
        else:
            null_aligned[dep] = 0.0

    n_frames = sum(len(set(L[d])) for d in offsets)
    matched = sum(aligned.values())
    null_matched = sum(null_aligned.values())
    off_vals = Counter(offsets.values())
    print("\n" + "=" * 66)
    print("STEP 3  per frame, after aligning the two cameras")
    print("=" * 66)
    print(f"  deployments with an estimated offset  {len(offsets):>8,}")
    print(f"  offset == 0 (cameras already in sync) {off_vals.get(0,0):>8,}")
    print(f"  most common non-zero offsets: " +
          ", ".join(f"{o:+d} ({n})" for o, n in off_vals.most_common(6)
                    if o != 0) or "  none")
    print(f"  distinct L frames                     {n_frames:>8,}")
    print(f"  ... with an aligned R partner         {matched:>8,} "
          f"({100*matched/n_frames:.1f}%)" if n_frames else "")
    print(f"  ... expected from RANDOM R frames     {null_matched:>8,.0f} "
          f"({100*null_matched/n_frames:.1f}%)" if n_frames else "")
    if n_frames:
        ratio = matched / null_matched if null_matched else float("inf")
        print(f"  real / null                           {ratio:>8.2f}")
        if ratio < 1.5:
            print("\n  VERDICT: real alignment is no better than chance. The")
            print("  per-deployment offsets are fitted noise, not rig sync, and")
            print("  this test cannot demonstrate stereo duplication. Step 4")
            print("  below would be measuring the same artefact -- ignore it.")
    if matched == 0:
        print("\n  VERDICT: no aligned frame pairs. The cameras were never")
        print("  annotated at the same moment, so no fish is recorded twice.")
        return

    shared = set()
    for dep, off in offsets.items():
        rset = set(R[dep])
        for l in set(L[dep]):
            if (l - off) in rset:
                shared.add((dep, l))
                shared.add((dep, l - off))

    # ---- step 4: per taxon ----------------------------------------------
    # key on the ALIGNED frame so an L frame and its offset R partner land
    # in the same bucket
    idx = defaultdict(lambda: defaultdict(Counter))
    for i, (dep, c, f) in parsed.items():
        if (dep, f) not in shared:
            continue
        key = (dep, f if c == "L" else f + offsets.get(dep, 0))
        for cat in taxa.get(i, ()):
            idx[key][cat][c] += 1

    dup_pairs, dup_crops, by_taxon = 0, 0, Counter()
    for key, per_cat in idx.items():
        for cat, k in per_cat.items():
            if k.get("L") and k.get("R"):
                n = min(k["L"], k["R"])
                dup_pairs += n
                dup_crops += 2 * n
                by_taxon[cat] += n

    print("\n" + "=" * 66)
    print("STEP 4  per taxon, within ALIGNED frame pairs")
    print("=" * 66)
    print(f"  probable duplicate pairs  {dup_pairs:>8,}")
    print(f"  crops involved            {dup_crops:>8,} "
          f"({100*dup_crops/tot:.1f}% of OzFish)")
    eff = tot - dup_pairs
    print(f"  effective crops           {eff:>8,}  (one per pair, not two)")
    print(f"  design effect from stereo   {tot/eff:>8.3f}" if eff else "")
    if by_taxon:
        print("\n  worst-affected taxa:")
        for cat, n in by_taxon.most_common(10):
            print(f"    {n:>6,}  {cats.get(cat, cat)}")

    print("\n  This is an UPPER bound. A school of twenty fish annotated in")
    print("  both views produces the same signature as one fish recorded")
    print("  twice, and this test cannot tell them apart. If the figure is")
    print("  small, the question is closed either way; if it is large, the")
    print("  strict version compares box coordinates for plausible stereo")
    print("  disparity.")


if __name__ == "__main__":
    main()

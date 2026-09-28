#!/usr/bin/env python3
"""Why does FathomNet's background still sit 2.6 points off its animals, and
does 2.6 points matter?

Those are two different questions and the second one is the one that decides
anything. The build report answers neither: it prints a sigma, and sigma
answers "is this difference real?", which at n=48,146 is never interesting.
What the set has to guarantee is that a model CANNOT SEPARATE background from
animal on scale, and separability is an effect size, not a p-value.

So:

  PART A  -- arithmetic on the collation alone, no build required. Tests one
             specific mechanism: the background draws one box per frame, so
             its mixture over frame sizes is FRAME-weighted, while the animal
             distribution it is matched against is BOX-weighted. Those two are
             equal only if annotations-per-frame is constant across frame
             sizes. This part computes both marginals from the same per-frame-
             size box distributions, so any difference it reports is that
             mechanism and nothing else.

  PART B  -- the effect size. The AUC of the best possible classifier that
             sees ONLY the crop's short side. 0.50 is "size carries no
             information"; 1.00 is "size alone separates the classes". This is
             the number to hold the set to, and it should have been the
             threshold from the start.

RUN B FIRST. It can close the question on its own: if size carries almost no
information, there is nothing to chase and no reason to rebuild anything.

    python scratch/background_drift_audit.py separability --config configs/shards_background.yaml

Run A only if B says the drift matters, because A's answer is about how to fix
it and is wasted effort until that is established.

    python scratch/background_drift_audit.py mixture --config configs/shards_background.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from collections import defaultdict

import yaml


# --------------------------------------------------------------- shared
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pct_under(vals, t):
    return 100.0 * sum(1 for v in vals if v < t) / len(vals) if vals else 0.0


def median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def collation_boxes(cfg):
    """Per source: a list of (frame_key, frame_w, frame_h, short_side).

    One entry per ANNOTATION, carrying the frame it sits on. That is enough to
    build both marginals, because the background's draw pool at frame size s
    is exactly the boxes whose frame is size s.

    Deliberately mirrors frame_index() in the builder rather than inventing a
    second reading of the data -- if the two disagree the audit is worthless.
    """
    out = defaultdict(list)
    for path in cfg["coco"]:
        print(f"  reading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        by_image = defaultdict(list)
        for an in doc.get("annotations", []):
            by_image[an["image_id"]].append(an)

        for im in doc.get("images", []):
            src = src_of.get(im.get("dataset_id"), "unknown")
            prov = im.get("crop_provenance")
            sm = im.get("source_meta") or {}

            if prov == "frame":
                fw, fh_ = im.get("width"), im.get("height")
                if not fw or not fh_:
                    continue
                key = os.path.splitext(im["file_name"])[0]
                for an in by_image.get(im["id"], []):
                    b = an.get("bbox") or []
                    if len(b) >= 4:
                        out[src].append((key, int(fw), int(fh_),
                                         float(min(b[2], b[3]))))

            elif prov == "cut_from_frame":
                fb = sm.get("frame_bbox_used") or sm.get("frame_bbox")
                fs = sm.get("frame_size")
                uu = sm.get("fathomnet_image_uuid")
                if not (fb and fs and uu and len(fb) >= 4):
                    continue
                out[src].append((str(uu), int(fs[0]), int(fs[1]),
                                 float(min(fb[2], fb[3]))))

            elif sm.get("ozfish_uid") is not None:
                # OzFish frame sizes are not in the collation. They are
                # uniform 1920x1080 in the frames the builder read, and the
                # builder reads them from the headers; this audit takes the
                # uniform value and SAYS SO rather than pretending to measure.
                fb = sm.get("frame_bbox")
                if not (fb and len(fb) >= 4):
                    continue
                key = (f"{sm.get('video')}_"
                       f"{str(sm.get('camera') or '').upper()}_"
                       f"{sm.get('frame')}")
                out[src].append((key, 1920, 1080, float(min(fb[2], fb[3]))))
        del doc, by_image
    return out


# ------------------------------------------------------------- PART A
def mixture(cfg, threshold=64):
    print(__doc__.split("Run A first")[0].rstrip())
    print("\n" + "=" * 74)
    print("PART A  frame-weighted against box-weighted, from the collation")
    print("=" * 74)
    rows = collation_boxes(cfg)

    rc = 0
    for src in sorted(rows):
        ann = rows[src]
        # boxes and distinct frames, grouped by frame size
        by_sz_short = defaultdict(list)
        frames_at = defaultdict(set)
        for key, fw, fh_, short in ann:
            by_sz_short[(fw, fh_)].append(short)
            frames_at[(fw, fh_)].add(key)

        n_boxes = sum(len(v) for v in by_sz_short.values())
        n_frames = sum(len(v) for v in frames_at.values())

        # The animal marginal weights a frame size by its BOX count.
        # The background marginal weights it by its FRAME count, because the
        # builder visits frames and takes one (or max_per_frame) from each.
        box_w = {s: len(v) / n_boxes for s, v in by_sz_short.items()}
        frm_w = {s: len(frames_at[s]) / n_frames for s in by_sz_short}

        p_under = {s: pct_under(v, threshold) for s, v in by_sz_short.items()}
        med_at = {s: median(v) for s, v in by_sz_short.items()}

        an_under = sum(box_w[s] * p_under[s] for s in by_sz_short)
        bg_under = sum(frm_w[s] * p_under[s] for s in by_sz_short)
        an_med = sum(box_w[s] * med_at[s] for s in by_sz_short)
        bg_med = sum(frm_w[s] * med_at[s] for s in by_sz_short)

        print(f"\n  {src}")
        print(f"    {n_boxes:,} boxes on {n_frames:,} frames, "
              f"{len(by_sz_short):,} distinct frame sizes, "
              f"{n_boxes / n_frames:.2f} boxes per frame")
        print(f"    <{threshold}px   box-weighted (the animals) "
              f"{an_under:5.1f}%")
        print(f"              frame-weighted (the background) "
              f"{bg_under:5.1f}%")
        print(f"              predicted drift from this mechanism alone "
              f"{bg_under - an_under:+.1f} pts")
        print(f"    median    box-weighted {an_med:6.1f}   "
              f"frame-weighted {bg_med:6.1f}")

        if len(by_sz_short) == 1:
            print("    -- one frame size, so the two weightings are "
                  "identical by construction and this mechanism cannot "
                  "apply here. A drift in the build for this source is "
                  "something else.")
            continue

        # Where the weighting bites: the frame sizes whose share of frames
        # differs most from their share of boxes, weighted by how unusual
        # their box sizes are.
        contrib = sorted(
            ((frm_w[s] - box_w[s]) * p_under[s], s) for s in by_sz_short)
        worst = [c for c in contrib if abs(c[0]) > 0.05][:4] \
            + [c for c in reversed(contrib) if abs(c[0]) > 0.05][:4]
        seen = set()
        shown = []
        for c, s in worst:
            if s not in seen:
                seen.add(s)
                shown.append((c, s))
        if shown:
            print("    biggest contributors (frame size: share of frames vs "
                  "share of boxes, <%dpx there):" % threshold)
            for c, s in sorted(shown, key=lambda t: -abs(t[0]))[:6]:
                print(f"      {s[0]}x{s[1]:<6} frames "
                      f"{100 * frm_w[s]:5.1f}%  boxes {100 * box_w[s]:5.1f}%"
                      f"  <{threshold}px {p_under[s]:5.1f}%"
                      f"  -> {c:+.2f} pts")
        if abs(bg_under - an_under) > 0.5:
            rc = 1
    print("\n  A predicted drift near the measured one means the mechanism is "
          "found and the fix is to weight the draw by boxes rather than "
          "frames. A predicted drift near zero means it is NOT this, and the "
          "placement-rejection loss is the remaining suspect.")
    return rc


# ------------------------------------------------------------- PART B
def shard_shorts(cfg):
    out = defaultdict(list)
    d = os.path.join(cfg["out_root"], cfg["name"])
    tars = sorted(f for f in os.listdir(d) if f.endswith(".tar"))
    if not tars:
        sys.exit(f"no shards in {d}")
    for t in tars:
        with tarfile.open(os.path.join(d, t)) as tf:
            for m in tf:
                if not m.name.endswith(".json"):
                    continue
                meta = json.load(tf.extractfile(m))
                src = meta.get("source")
                nw, nh = meta.get("native_width"), meta.get("native_height")
                if src and nw and nh:
                    out[src].append(float(min(nw, nh)))
        print(f"  {t}: read", flush=True)
    return out


def auc(pos, neg):
    """P(a random background is smaller than a random animal), ties at half.

    Mann-Whitney U over the merged ranking -- exact, not sampled. Reported
    folded to >= 0.5 so it reads as "how separable", with the direction named
    separately, because a background that is systematically LARGER is just as
    separable as one that is smaller and the number should not hide it.
    """
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum = 0.0
    i = 0
    r = 1
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg = (r + (r + (j - i) - 1)) / 2.0
        rank_sum += avg * sum(1 for k in range(i, j) if merged[k][1] == 1)
        r += j - i
        i = j
    n1, n2 = len(pos), len(neg)
    u = rank_sum - n1 * (n1 + 1) / 2.0
    return u / (n1 * n2)


def separability(cfg, threshold=64):
    print("=" * 74)
    print("PART B  how separable are the two classes on size alone?")
    print("=" * 74)
    print("\nreading the built shards")
    bg = shard_shorts(cfg)
    print("reading the collation")
    ann = collation_boxes(cfg)

    rc = 0
    print(f"\n  {'source':<12} {'AUC':>6}  {'direction':<22}"
          f" {'<%dpx drift' % threshold:>12}")
    print("  " + "-" * 60)
    for src in sorted(bg):
        a = [s for _k, _w, _h, s in ann.get(src, [])]
        b = bg[src]
        if not a or not b:
            continue
        raw = auc(a, b)
        folded = max(raw, 1.0 - raw)
        which = ("background runs smaller" if raw > 0.5
                 else "background runs larger" if raw < 0.5 else "-")
        drift = pct_under(b, threshold) - pct_under(a, threshold)
        print(f"  {src:<12} {folded:6.3f}  {which:<22} {drift:+11.1f}")
        if folded >= 0.55:
            rc = 1

    print("""
  READING THIS. AUC is the probability that a size-only classifier ranks a
  random animal above a random background, so 0.500 means size carries no
  information at all and the set has done its job.

  There is no clean published cutoff for "small enough", so treat it as a
  budget rather than a test: whatever the AUC is, a model trained on this set
  could in principle reach that much of its animal/background decision on
  scale alone, before looking at any content. 0.52 is a rounding error next to
  the differences the experiments are trying to measure. 0.60 is not, and 0.70
  would mean the set is broken in the way the 22-point first build was broken.

  A large <64px drift with an AUC near 0.5 means the two distributions differ
  in one tail and nowhere else, which is a much weaker problem than the drift
  figure on its own suggests -- and the reverse is also possible.""")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mixture", "separability"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--threshold", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mode == "mixture":
        sys.exit(mixture(cfg, args.threshold))
    sys.exit(separability(cfg, args.threshold))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Why did the OzFish frame gate fail, and is there a rule that fixes it?

THE EVIDENCE THIS STARTS FROM
    ozfish_frame_check.py cut 400 boxes out of their joined frames: 14 were
    pixel-identical to the distributed crop and 382 were not. Fourteen matters
    more than it looks. A pixel-exact match on a textured underwater patch
    cannot happen by accident -- a rescaled-frame control gives zero -- so for
    those fourteen the frame and the box agree perfectly. The frames therefore
    share the crops' pixel grid, and what is wrong is which frame is paired
    with which crop.

    The mixed difference magnitudes agree: 72 and 74 alongside 255. An
    unrelated region differs a lot consistently; a temporally NEARBY frame of
    the same scene differs a little. And a few percent matching exactly is
    what an index offset looks like when it happens to land on one of the
    duplicate frames video encoders emit.

TWO PHASES, CHEAP ONE FIRST
    Phase 1 costs nothing: for all 80,809 crops, does frame_metadata's
    file_name agree with the name rebuilt from the crop's own
    video/camera/container/frame? Those are two independent statements about
    which frame a crop came from. If they disagree systematically, the shape
    of the disagreement names the fault before a single pixel is decoded.

    Phase 2 decodes. For sampled failures it tries the metadata frame, the
    reconstructed frame, the same frame from the other camera, and every frame
    within +/- a window, and reports which candidate reproduces the crop
    exactly. If one rule explains nearly all of them, that rule is the fix.

    It also prints real filenames off the disk before doing any of this,
    because the last time a path was constructed from documentation here it
    cost four rounds of 404s. Look at what is actually there first.

    python scratch/ozfish_frame_diagnose.py                 # both phases
    python scratch/ozfish_frame_diagnose.py --sample 0      # phase 1 only
    python scratch/ozfish_frame_diagnose.py --window 12     # widen the search
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

from PIL import Image, ImageChops

D_FRAMES = "D:/marineai/dataset/raw/ozfish/frames"
D_META = "D:/marineai/dataset/raw/ozfish/frame_metadata.csv"
D_COCO = "D:/marineai/dataset/collated/seavision.json"
D_IMAGES = "D:/marineai/dataset/collated/images"

SUFFIX = re.compile(r"^(?P<base>.+?\.png)-\d+-\d+\.png$", re.I)
# {video}_{camera}.{container}.{frame}.png
FRAME_PAT = re.compile(
    r"^(?P<vid>[A-Za-z]+\d+)_(?P<cam>[LR])\.(?P<ext>avi|mp4|mpeg)\."
    r"(?P<frame>\d+)\.png$", re.I)


def walk_frames(root):
    """-> (by_name, by_key, samples)

    by_key is {(video, camera, frame_index): path}, which is what an offset
    search needs. Built by PARSING the filenames rather than trusting the
    convention, so files that do not parse are counted and reported.
    """
    by_name, by_key, samples, unparsed = {}, {}, [], 0
    for dirpath, _d, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".png"):
                continue
            p = os.path.join(dirpath, f)
            by_name.setdefault(f, p)
            m = SUFFIX.match(f)
            base = m.group("base") if m else f
            by_name.setdefault(base, p)
            if len(samples) < 6:
                samples.append(f)
            fm = FRAME_PAT.match(base)
            if fm:
                by_key[(fm["vid"], fm["cam"].upper(), int(fm["frame"]))] = p
            else:
                unparsed += 1
    return by_name, by_key, samples, unparsed


def load_meta(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        uid_c = next((cols[c] for c in ("uid", "id", "crop_uid") if c in cols),
                     None)
        fn_c = next((cols[c] for c in ("file_name", "filename", "frame",
                                       "frame_file", "image") if c in cols),
                    None)
        if not uid_c or not fn_c:
            sys.exit(f"cannot find uid/file_name in {rd.fieldnames}")
        out, dupes = {}, 0
        for row in rd:
            try:
                u = int(row[uid_c])
            except (TypeError, ValueError):
                continue
            name = os.path.basename(str(row[fn_c]).strip().replace("\\", "/"))
            if u in out and out[u] != name:
                dupes += 1
            out[u] = name
    return out, dupes


def ozfish_records(coco_path):
    with open(coco_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    ds = {d["id"] for d in doc.get("datasets", []) if d["name"] == "ozfish"}
    out = [(im["file_name"], im.get("source_meta") or {})
           for im in doc.get("images", []) if im.get("dataset_id") in ds]
    del doc
    return out


def rebuilt(sm):
    try:
        return (f"{sm['video']}_{sm['camera']}.{sm['container']}."
                f"{sm['frame']}.png")
    except KeyError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=D_FRAMES)
    ap.add_argument("--frame-metadata", default=D_META)
    ap.add_argument("--coco", default=D_COCO)
    ap.add_argument("--images", default=D_IMAGES)
    ap.add_argument("--sample", type=int, default=60,
                    help="crops to search in phase 2; 0 = phase 1 only")
    ap.add_argument("--window", type=int, default=6,
                    help="+/- frames to search around the stated index")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("PHASE 0  what is actually on disk")
    print("-" * 70)
    by_name, by_key, samples, unparsed = walk_frames(args.frames)
    for s in samples:
        print(f"  {s}")
    print(f"  {len(by_key):,} filenames parse as "
          f"video_camera.container.frame.png; {unparsed:,} do not")
    if not by_key:
        sys.exit("  no filename parsed. The offset search needs to know which "
                 "frame is which -- look at the names above and tell me the "
                 "convention before going further.")

    meta, dupes = load_meta(args.frame_metadata)
    print(f"  frame_metadata.csv: {len(meta):,} distinct uids"
          + (f", {dupes:,} uids appear twice with DIFFERENT file_names "
             f"(last wins -- this alone could be the fault)" if dupes else ""))

    recs = ozfish_records(args.coco)
    print(f"  {len(recs):,} OzFish crops in the collation")

    # ---- phase 1: two independent claims about the same frame ---------
    print("\nPHASE 1  does frame_metadata agree with the crop's own filename?")
    print("-" * 70)
    verdict = Counter()
    offsets = Counter()
    examples = []
    for fn, sm in recs:
        uid = sm.get("ozfish_uid")
        m_name = meta.get(uid)
        r_name = rebuilt(sm)
        if not m_name or not r_name:
            verdict["no comparison possible"] += 1
            continue
        if m_name == r_name:
            verdict["agree"] += 1
            continue
        mm, rm = FRAME_PAT.match(m_name), FRAME_PAT.match(r_name)
        if not mm or not rm:
            verdict["disagree, unparsable"] += 1
        elif mm["vid"] != rm["vid"]:
            verdict["disagree: different video"] += 1
        elif mm["cam"].upper() != rm["cam"].upper():
            verdict["disagree: different camera"] += 1
        else:
            d = int(mm["frame"]) - int(rm["frame"])
            verdict["disagree: different frame index"] += 1
            offsets[d] += 1
        if len(examples) < 6:
            examples.append(f"{fn}: metadata {m_name}  vs  rebuilt {r_name}")

    for k, v in verdict.most_common():
        print(f"  {k:<34} {v:>8,}  ({v / max(1, len(recs)):.1%})")
    if offsets:
        print(f"\n  frame-index differences (metadata minus rebuilt), "
              f"commonest first:")
        for d, n in offsets.most_common(8):
            print(f"    {d:+6d}  {n:,}")
        if len(offsets) == 1:
            print("  A single constant offset. That is the fault, and it is "
                  "a one-line fix.")
    if examples:
        print("\n  examples:")
        for e in examples:
            print(f"    {e}")

    on_disk = sum(1 for _fn, sm in recs
                  if (rebuilt(sm) or "") in by_name)
    print(f"\n  crops whose REBUILT name exists on disk: {on_disk:,} of "
          f"{len(recs):,} ({on_disk / max(1, len(recs)):.1%})")

    # Where a gap is matters more than how big it is. A gap spread evenly
    # costs a fraction of every deployment; a gap concentrated in a survey
    # costs that survey entirely, and OzFish's four surveys are genuinely
    # different studies (plan §3.3), so losing one is not losing 22% of a
    # homogeneous whole.
    # TWO match rules, because the container token is a live suspect. The
    # crop filenames carry .avi for surveys A and G and .mp4/.mpeg for B and
    # E; if the frame archive spells the container differently, an exact-name
    # join misses those frames entirely while they sit on disk. Matching on
    # (video, camera, frame) and ignoring the extension says whether that is
    # what happened -- and the gap between the two columns is the answer.
    need, have, have_loose = (defaultdict(set), defaultdict(set),
                              defaultdict(set))
    miss_dep = Counter()
    for _fn, sm in recs:
        r = rebuilt(sm)
        if not r:
            continue
        vid = str(sm.get("video") or "?")
        survey = vid[0] if vid else "?"
        need[survey].add(r)
        loose = (vid, str(sm.get("camera") or "").upper(),
                 int(sm.get("frame", -1)))
        if loose in by_key:
            have_loose[survey].add(r)
        if r in by_name:
            have[survey].add(r)
        else:
            miss_dep[vid] += 1
    print(f"\n  {'survey':<8} {'frames needed':>14} {'exact name':>11} "
          f"{'':>7} {'ignoring ext':>13} {'':>7}")
    for s in sorted(need):
        n, h, hl = len(need[s]), len(have[s]), len(have_loose[s])
        print(f"  {s:<8} {n:>14,} {h:>11,} {h / max(1, n):>6.1%} "
              f"{hl:>13,} {hl / max(1, n):>6.1%}")
    gained = (len(set().union(*have_loose.values())) if have_loose else 0) \
        - (len(set().union(*have.values())) if have else 0)
    if gained > 0:
        print(f"\n  ! ignoring the container token finds {gained:,} more "
              f"frames. The archive spells the container differently from "
              f"the crop filenames, and the fix is to key the lookup on "
              f"(video, camera, frame) -- NOT to write off those surveys.")
        exts = Counter()
        for _fn, sm in recs:
            loose = (str(sm.get("video") or "?"),
                     str(sm.get("camera") or "").upper(),
                     int(sm.get("frame", -1)))
            p_ = by_key.get(loose)
            if p_ and rebuilt(sm) not in by_name:
                m_ = FRAME_PAT.match(os.path.basename(p_).split(".png")[0]
                                     + ".png")
                exts[(sm.get("container"), m_["ext"] if m_ else "?")] += 1
        for (want_e, got_e), n_ in exts.most_common(6):
            print(f"      crop says .{want_e:<5} archive has .{got_e:<5} "
                  f"{n_:,} crops")
    all_need = set().union(*need.values()) if need else set()
    print(f"  {'TOTAL':<8} {len(all_need):>14,} "
          f"{len(set().union(*have.values())) if have else 0:>10,}")
    print(f"  {len(by_name) // 2:,} distinct frames are on disk in total, so "
          f"{max(0, len(by_name) // 2 - len(set().union(*have.values()) if have else set())):,} "
          f"of them are not wanted by any crop")
    dep_all = Counter()
    for _fn, sm in recs:
        dep_all[str(sm.get("video") or "?")] += 1
    whole = [d for d, n in miss_dep.items() if n == dep_all[d]]
    print(f"\n  deployments with NO frames at all: {len(whole):,} of "
          f"{len(dep_all):,}, holding "
          f"{sum(dep_all[d] for d in whole):,} crops")
    if whole:
        print(f"    e.g. {', '.join(sorted(whole)[:8])}")
    partial = [d for d, n in miss_dep.items() if 0 < n < dep_all[d]]
    print(f"  deployments PARTIALLY covered: {len(partial):,}, missing "
          f"{sum(miss_dep[d] for d in partial):,} crops between them")

    if args.sample <= 0:
        return 0

    # ---- phase 2: which candidate actually reproduces the crop? -------
    print(f"\nPHASE 2  searching for the frame that does reproduce the crop")
    print("-" * 70)
    rng = random.Random(args.seed)
    pool = [(fn, sm) for fn, sm in recs
            if sm.get("frame_bbox") and rebuilt(sm)]
    picks = rng.sample(pool, min(args.sample, len(pool)))
    print(f"  {len(picks)} crops, window +/-{args.window} frames, "
          f"both cameras")

    found = Counter()
    off_hist = Counter()
    by_dep = defaultdict(set)        # deployment -> winning tags seen
    for fn, sm in picks:
        b = sm["frame_bbox"]
        x0, y0, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        try:
            with Image.open(os.path.join(args.images, fn)) as c:
                c.load()
                dist = c.convert("RGB")
        except Exception:                                # noqa: BLE001
            found["crop unreadable"] += 1
            continue

        vid, cam, fr = sm["video"], str(sm["camera"]).upper(), int(sm["frame"])
        other = "R" if cam == "L" else "L"
        cands = []
        uid = sm.get("ozfish_uid")
        if uid in meta and meta[uid] in by_name:
            cands.append(("metadata frame", by_name[meta[uid]], 0))
        for d in sorted(range(-args.window, args.window + 1), key=abs):
            for c_cam, tag in ((cam, "same camera"), (other, "other camera")):
                p = by_key.get((vid, c_cam, fr + d))
                if p:
                    cands.append((f"{tag} {d:+d}", p, d))

        hit = None
        for tag, p, d in cands:
            try:
                with Image.open(p) as f_im:
                    f_im.load()
                    cut = f_im.convert("RGB").crop((x0, y0, x0 + w, y0 + h))
            except Exception:                            # noqa: BLE001
                continue
            if cut.size != dist.size:
                continue
            if ImageChops.difference(cut, dist).getbbox() is None:
                hit = (tag, d)
                break
        if hit:
            found["MATCHED"] += 1
            off_hist[hit[0]] += 1        # the tag verbatim; no re-derivation
            by_dep[sm["video"]].add(hit[0])
        else:
            found["no candidate matched"] += 1

    n = len(picks)
    print(f"\n  matched: {found['MATCHED']:,} of {n:,} "
          f"({found['MATCHED'] / max(1, n):.1%})")
    print(f"  unmatched: {found['no candidate matched']:,}")
    if off_hist:
        print(f"\n  {'winning candidate':<28} {'count':>7}")
        for tag, c in off_hist.most_common(12):
            print(f"  {tag:<28} {c:>7,}")
        # WP3 found the two stereo cameras offset by a constant PER
        # DEPLOYMENT. If the same is true here, the winning tag will be
        # consistent within a deployment and vary between them -- still a
        # rule, just one keyed on deployment rather than global.
        mixed = [d for d, tags in by_dep.items() if len(tags) > 1]
        top_tag, top_n = off_hist.most_common(1)[0]
        share = top_n / max(1, sum(off_hist.values()))
        print(f"\n  deployments sampled: {len(by_dep):,}; "
              f"{len(mixed):,} of them needed more than one candidate")
        print(f"  dominant candidate: {top_tag} ({share:.0%} of matches)")
        # Judge by how much the top candidate DOMINATES, not by how many
        # distinct tags appeared. An earlier version counted tags, and so
        # read a 59:1 split as systematic per-deployment variation when it
        # was one rule plus a single coincidence.
        if share < 0.9 and by_dep and not mixed:
            print("  No single rule dominates, yet each deployment is "
                  "internally consistent -- that is a PER-DEPLOYMENT offset, "
                  "the same phenomenon as the stereo offset WP3 measured.")

    print("\nREAD IT LIKE THIS")
    print("-" * 70)
    dom = (off_hist.most_common(1)[0][1] / max(1, sum(off_hist.values()))
           if off_hist else 0.0)
    if found["MATCHED"] and dom >= 0.9:
        tag, _c = off_hist.most_common(1)[0]
        print(f"  One candidate wins every time: {tag}. That is a rule, not a")
        print(f"  coincidence. Apply it in the square builder's frame lookup,")
        print(f"  re-run ozfish_frame_check.py, and expect it to pass.")
    elif found["MATCHED"] / max(1, n) > 0.8:
        print("  Most crops matched, but by more than one candidate. Look at")
        print("  whether the winning offset varies BY DEPLOYMENT -- the two")
        print("  stereo cameras are already known to sit at a constant offset")
        print("  per deployment (WP3), so a per-deployment offset here would")
        print("  be the same phenomenon and is still a rule, just a keyed one.")
    else:
        print("  Most crops did not match any nearby frame. Widen --window")
        print("  first; if that does not help, the published frames are not a")
        print("  frame-for-frame export of the footage the boxes were")
        print("  measured on, and OzFish square expansion should be parked")
        print("  the way SEAMAPD21 was, with the reason recorded.")
        print("  OzFish still contributes letterbox and distort from the")
        print("  `crops` set, which is already built and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

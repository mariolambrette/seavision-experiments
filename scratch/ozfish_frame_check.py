#!/usr/bin/env python3
"""Gate the OzFish frames before anything is built from them.

WHY THIS COMES FIRST
    Square expansion widens a crop's box into a square by taking in real
    pixels from around the animal. That is only meaningful if the box
    coordinates in the crop filename address the SAME pixel grid as the frame
    on disk. If the published frames are at a different resolution from the
    footage the boxes were measured against -- a different export, a rescale,
    a letterboxed encode -- then every square crop is a correctly-sized
    rectangle of the wrong part of the seabed. It would look entirely
    plausible: right dimensions, right file count, real underwater imagery,
    and the animal simply not in it.

    That is exactly the failure mode level 4 of the shard verification exists
    for, and it is cheaper to catch here, once, than to find it in an
    embedding. The Roboflow mirror was rejected for precisely this reason --
    no stated resolution. "The official frames should be right" is not
    evidence.

    So: cut each sampled box out of its frame and compare the result against
    the crop OzFish actually distributed. They should be pixel-identical,
    because both are PNG and one is meant to be a crop of the other.

WHAT ELSE IT CHECKS, BEFORE SPENDING AN HOUR ON THE PIXELS
    * coverage -- how many of the 80,809 crops resolve to a frame that exists
    * the join -- by uid through frame_metadata.csv, with a reconstructed
      `{video}_{camera}.{container}.{frame}.png` as a fallback, reported
      separately so it is visible which route worked
    * the filename artefact -- the crop archive appends `-1-1` to every name
      (WP3). Whether the frame archive does the same is checked, not assumed
    * frame sizes, and whether any box falls outside its own frame

NEGATIVE-ORIGIN BOXES ARE THEIR OWN BUCKET, NOT FAILURES
    WP3 recorded boxes whose origin is negative, running off the frame edge.
    PIL pads such a crop with black; whoever cut the distributed crop may have
    padded, clipped or something else. Those are reported separately, because
    counting them as mismatches would hide the rate that matters and counting
    them as matches would hide a real problem.

    python scratch/ozfish_frame_check.py --sample 400
    python scratch/ozfish_frame_check.py --sample 0        # inventory only

Exit code is non-zero if the geometry gate fails, so this can sit in front of
the square build rather than beside it.
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

# the crop archive's artefact: X.png -> X.png-1-1.png (WP3)
SUFFIX = re.compile(r"^(?P<base>.+?\.png)-\d+-\d+\.png$", re.I)
IMG_EXT = (".png", ".jpg", ".jpeg")
# {video}_{camera}.{container}.{frame}.png -- re.I because the archive spells
# the container .MP4 where the crop filenames say .mp4, which is what hid
# 17,680 crops and 1,054 deployments behind an exact-name join.
FRAME_PAT = re.compile(
    r"^(?P<vid>[A-Za-z]+\d+)_(?P<cam>[LR])\.(?P<ext>avi|mp4|mpeg)\."
    r"(?P<frame>\d+)\.png$", re.I)


def index_frames(root):
    """-> (by_name, by_key, n_files, n_artefact)

    Walks recursively, so it does not matter whether the archive extracted
    flat or nested by survey. Both the name as found and the name with the
    `-1-1` artefact stripped are indexed.

    by_key is {(video, CAMERA, frame): path}, parsed from the filename and
    deliberately CARRYING NO CONTAINER. That is the join that works: the
    container token is spelled inconsistently between the crop filenames and
    the frame archive, and it identifies nothing anyway -- a video, a camera
    and a frame index already name exactly one frame.
    """
    by_name, by_key, n, arte = {}, {}, 0, 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue
            p = os.path.join(dirpath, f)
            by_name.setdefault(f, p)
            m = SUFFIX.match(f)
            base = m.group("base") if m else f
            if m:
                arte += 1
                by_name.setdefault(base, p)
            fm = FRAME_PAT.match(base)
            if fm:
                by_key.setdefault(
                    (fm["vid"], fm["cam"].upper(), int(fm["frame"])), p)
            n += 1
    return by_name, by_key, n, arte


def load_frame_metadata(path):
    """-> {uid: file_name}. Columns are detected rather than assumed."""
    if not path or not os.path.exists(path):
        return {}, None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        uid_c = next((cols[c] for c in ("uid", "id", "crop_uid") if c in cols),
                     None)
        fn_c = next((cols[c] for c in ("file_name", "filename", "frame",
                                       "frame_file", "image") if c in cols),
                    None)
        if not uid_c or not fn_c:
            return {}, (f"could not find uid/file_name columns in "
                        f"{rd.fieldnames}")
        out = {}
        for row in rd:
            try:
                out[int(row[uid_c])] = os.path.basename(
                    str(row[fn_c]).strip().replace("\\", "/"))
            except (TypeError, ValueError):
                continue
    return out, None


def ozfish_records(coco_path):
    """-> list of (uid_file_name, source_meta) for OzFish images only."""
    with open(coco_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    ds = {d["id"] for d in doc.get("datasets", []) if d["name"] == "ozfish"}
    if not ds:
        sys.exit("no dataset named 'ozfish' in that COCO file")
    out = [(im["file_name"], im.get("source_meta") or {})
           for im in doc.get("images", []) if im.get("dataset_id") in ds]
    del doc
    return out


def frame_name_from_meta(sm):
    """`{video}_{camera}.{container}.{frame}.png` -- the crop filename minus
    the box. Only a fallback: frame_metadata's own file_name is authoritative
    and this is a guess at a convention."""
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
    ap.add_argument("--images", default=D_IMAGES,
                    help="the collated crop store -- the crops compared "
                         "against are the ones the shards actually hold")
    ap.add_argument("--sample", type=int, default=400,
                    help="crops to verify pixel-for-pixel; 0 = inventory only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fail-under", type=float, default=0.99,
                    help="minimum in-bounds pixel-identity rate to pass")
    args = ap.parse_args()

    if not os.path.isdir(args.frames):
        sys.exit(f"{args.frames} is not a directory -- has the extract "
                 f"finished?")

    print("indexing the extracted frames")
    by_name, by_key, n_files, n_arte = index_frames(args.frames)
    print(f"  {n_files:,} image files, {len(by_name):,} distinct names, "
          f"{len(by_key):,} (video, camera, frame) keys")
    if n_arte:
        print(f"  {n_arte:,} carry the '-1-1' archive artefact; both the raw "
              f"and stripped names are indexed, so the join works either way")

    meta, err = load_frame_metadata(args.frame_metadata)
    if err:
        print(f"  ! {err}")
    print(f"  frame_metadata.csv: {len(meta):,} uid -> file_name rows")

    print("\nreading the collation")
    recs = ozfish_records(args.coco)
    print(f"  {len(recs):,} OzFish crops")

    # ---- coverage -----------------------------------------------------
    # frame_metadata.csv is NOT a join route. Measured over all 80,809
    # crops, its uid pairs 67% of them with a frame from a different video
    # entirely -- its uid space is not the crops' uid space, whatever plan
    # §3.3 says. Keeping it as a fallback would silently pair a wrong frame
    # with any crop the real routes miss, which is worse than a gap, because
    # a gap is visible. It is read only to report the disagreement.
    resolved, how, missing, meta_disagree = {}, Counter(), [], 0
    for fn, sm in recs:
        name = frame_name_from_meta(sm)
        uid = sm.get("ozfish_uid")
        if uid in meta and name and meta[uid] != name:
            meta_disagree += 1
        p = by_name.get(name) if name else None
        route = "rebuilt name"
        if p is None:
            try:
                key = (str(sm["video"]), str(sm["camera"]).upper(),
                       int(sm["frame"]))
            except (KeyError, TypeError, ValueError):
                key = None
            p = by_key.get(key) if key else None
            route = "video/camera/frame"
        if p:
            resolved[fn] = (p, sm)
            how[route] += 1
        else:
            missing.append((fn, sm))

    print(f"\nCOVERAGE")
    print("-" * 70)
    print(f"  {len(resolved):,} of {len(recs):,} crops "
          f"({len(resolved) / max(1, len(recs)):.1%}) resolve to a frame on "
          f"disk")
    for route, n in how.most_common():
        print(f"    via {route}: {n:,}")
    print(f"  distinct frames used: "
          f"{len({p for p, _ in resolved.values()}):,}")
    if meta_disagree:
        print(f"  ({meta_disagree:,} crops where frame_metadata.csv names a "
              f"different frame -- ignored by design, see the code comment)")
    if missing:
        print(f"  ! {len(missing):,} crops have no frame. First few:")
        for fn, sm in missing[:6]:
            print(f"      {fn}  uid={sm.get('ozfish_uid')} "
                  f"guess={frame_name_from_meta(sm)}")
        if how["frame_metadata"] == 0 and how["reconstructed"] == 0:
            print("  Nothing joined at all. Either the extract is not "
                  "finished, --frames points at the wrong place, or the "
                  "frame filenames follow a convention neither route "
                  "guesses. Look at a few real filenames before assuming.")

    if args.sample <= 0 or not resolved:
        return 0 if not missing else 1

    # ---- the gate -----------------------------------------------------
    print(f"\nGEOMETRY GATE  ({min(args.sample, len(resolved)):,} sampled)")
    print("-" * 70)
    rng = random.Random(args.seed)
    keys = rng.sample(sorted(resolved), min(args.sample, len(resolved)))

    by_frame = defaultdict(list)          # decode each frame once
    for fn in keys:
        p, sm = resolved[fn]
        by_frame[p].append((fn, sm))

    stat = Counter()
    worst = []
    sizes = Counter()
    for p, items in by_frame.items():
        try:
            with Image.open(p) as fr:
                fr.load()
                frame = fr.convert("RGB")
        except Exception as exc:                     # noqa: BLE001
            stat["unreadable_frame"] += len(items)
            print(f"  ! unreadable frame {p}: {exc}")
            continue
        sizes[frame.size] += 1
        fw, fh = frame.size
        for fn, sm in items:
            b = sm.get("frame_bbox") or []
            if len(b) < 4:
                stat["no_bbox"] += 1
                continue
            x0, y0, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            crop_path = os.path.join(args.images, fn)
            try:
                with Image.open(crop_path) as c:
                    c.load()
                    dist = c.convert("RGB")
            except Exception as exc:                 # noqa: BLE001
                stat["unreadable_crop"] += 1
                print(f"  ! unreadable crop {crop_path}: {exc}")
                continue

            oob = x0 < 0 or y0 < 0 or x0 + w > fw or y0 + h > fh
            cut = frame.crop((x0, y0, x0 + w, y0 + h))

            if cut.size != dist.size:
                stat["size_mismatch"] += 1
                if len(worst) < 8:
                    worst.append(f"{fn}: frame cut {cut.size} vs distributed "
                                 f"{dist.size}")
                continue

            diff = ImageChops.difference(cut, dist)
            bbox = diff.getbbox()
            bucket = "oob" if oob else "in"
            if bbox is None:
                stat[f"{bucket}_identical"] += 1
            else:
                mx = max(diff.getextrema(), key=lambda t: t[1])[1]
                stat[f"{bucket}_differs"] += 1
                if mx <= 2:
                    stat[f"{bucket}_differs_trivially"] += 1
                if len(worst) < 8:
                    worst.append(f"{fn}: differs, max channel delta {mx}"
                                 + (" (box runs off the frame)" if oob else ""))

    n_in = stat["in_identical"] + stat["in_differs"]
    n_oob = stat["oob_identical"] + stat["oob_differs"]
    print(f"  frame sizes seen: "
          + ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.most_common(5)))
    print(f"\n  in-bounds boxes      {n_in:,}")
    print(f"    pixel-identical    {stat['in_identical']:,}"
          + (f"   ({stat['in_identical'] / n_in:.2%})" if n_in else ""))
    print(f"    differ             {stat['in_differs']:,}"
          f"   of which trivially (<=2/255): "
          f"{stat['in_differs_trivially']:,}")
    print(f"  boxes off the frame  {n_oob:,}")
    print(f"    pixel-identical    {stat['oob_identical']:,}")
    print(f"    differ             {stat['oob_differs']:,}"
          f"   -- expected: the overhang is padded, and by whose rule is "
          f"unknown")
    for k in ("size_mismatch", "unreadable_frame", "unreadable_crop",
              "no_bbox"):
        if stat[k]:
            print(f"  ! {k}: {stat[k]:,}")
    if worst:
        print("\n  examples:")
        for w in worst:
            print(f"    {w}")

    rate = stat["in_identical"] / n_in if n_in else 0.0
    print()
    if n_in == 0:
        print("VERDICT: no in-bounds box was checked. Nothing is proven.")
        return 1
    if rate >= args.fail_under:
        print(f"VERDICT: PASS. {rate:.2%} of in-bounds boxes cut out of the "
              f"frame are pixel-identical to the distributed crop, so the "
              f"frames share the crops' pixel grid and square expansion will "
              f"take in the right seabed.")
        if stat["oob_differs"]:
            print(f"         {stat['oob_differs']:,} off-frame boxes differ. "
                  f"Decide the padding rule explicitly in the square builder "
                  f"rather than inheriting PIL's black.")
        return 0
    print(f"VERDICT: FAIL. Only {rate:.2%} of in-bounds boxes match "
          f"(threshold {args.fail_under:.0%}).")
    print("  Do NOT build square crops from these frames. Either they are a")
    print("  different export from the one the boxes were measured on, or")
    print("  the join is pairing crops with the wrong frames. Check a single")
    print("  case by hand -- open the frame, draw the box, look at it --")
    print("  before writing any code to compensate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

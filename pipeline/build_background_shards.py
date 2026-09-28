#!/usr/bin/env python3
"""Build the `background` shard set (plan 6.1).

Background is an ordinary labelled class with its own prototype, so these are
training data, not a diagnostic sample. The design was settled and measured
during the WP7 review; this is the production build of it.

THE ONE RULE THAT DECIDES WHETHER ANY OF IT IS WORTH HAVING
    Match the background crops' size distribution to the animal crops'. Sample
    at arbitrary sizes and a model separates the classes by scale rather than
    content, and the whole rejection result is worthless -- 6.1 says so, and
    FishWIO's shipped background classes are a live example, at a median short
    side of 69 px against 88 for its fish.

    So each candidate's dimensions are drawn as a FRACTION of its frame from
    that source's own annotation boxes, then scaled to the frame it lands on.
    Relative rather than absolute, because frame sizes vary by a factor of
    four across the collation. Verified to reproduce the real distribution to
    within a percent at n = 400, and the build reports the match rather than
    asserting it.

WHAT COUNTS AS BACKGROUND, PER SOURCE -- AND IT IS NOT SYMMETRICAL
    A candidate must not come within `margin_px` of ANY annotation on its
    frame. Beyond that:

    PrePARED   frames flagged `has_unlabelled_animal` are excluded. There the
               flag means an animal nobody boxed, so a box-free region of that
               frame is not background.
    FathomNet  the flag is NOT an exclusion. There it means a box exists with
               an unidentified concept -- the box is already in the exclusion
               set, and dropping those frames would discard ~50,000 usable
               donors for nothing.
    OzFish     no flag is populated, and OzFish annotates roughly one frame
               per fish event (3.3), so other fish in the same frame may well
               be unboxed. **Its contamination rate is UNMEASURED** -- the WP7
               review covered PrePARED at 4.1% and FathomNet at 2.9% before
               these frames existed. Measure it the same way before using
               OzFish backgrounds in anything that matters.

    FishWIO's own background classes are excluded entirely. They are somebody
    else's sampling decision, they are measurably not size-matched, and mixing
    them in would confound our contamination rate with theirs.

WHY THE COUNTS ARE UNEQUAL
    See the config. Equal counts across sources would put a dozen crops on
    every PrePARED frame and 0.07 on each FathomNet one, so the same number of
    records would sit behind a 176-fold difference in distinct scenes. Every
    record carries `frame_key`, so the correlation that remains is measurable
    by the machinery 7.4 already has rather than hoped away.

ONLY THE FRAMES NEEDED ARE DECODED
    Unlike the square build this does not need every frame: it picks the
    frames first, then decodes those. A 108,000-crop set touches roughly a
    fifth of what the square sets did.

    python pipeline/build_background_shards.py --config configs/shards_background.yaml
    python pipeline/build_background_shards.py --config configs/shards_background.yaml --verify
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required")

SUFFIX = re.compile(r"^(?P<base>.+?\.png)-\d+-\d+\.png$", re.I)
OZ_FRAME = re.compile(
    r"^(?P<vid>[A-Za-z]+\d+)_(?P<cam>[LR])\.(?P<ext>avi|mp4|mpeg)\."
    r"(?P<frame>\d+)\.png$", re.I)


def safe(code):
    return "".join(c if c.isalnum() else "_" for c in str(code))


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=here, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=here,
                                       text=True)
        return {"commit": commit, "dirty": bool(diff.strip()),
                "diff": diff if diff.strip() else None}
    except Exception:                                    # noqa: BLE001
        return {"commit": None, "dirty": None, "diff": None}


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for k in ("coco", "image_dir", "out_root", "targets"):
        if k not in cfg:
            sys.exit(f"{path}: missing {k}")
    cfg.setdefault("name", "background")
    cfg.setdefault("margin_px", 8)
    cfg.setdefault("placement_tries", 60)
    cfg.setdefault("cap_long_side", 1024)
    cfg.setdefault("jpeg_quality", 95)
    cfg.setdefault("shard_bytes", 1 << 30)
    cfg.setdefault("seed", 0)
    cfg.setdefault("max_per_frame", {})
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


class ShardWriter:
    def __init__(self, out_dir, prefix, shard_bytes):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.limit = out_dir, prefix, shard_bytes
        self.idx, self.tar, self.path, self.written = -1, None, None, 0
        self.records, self.shards = 0, []
        self._rotate()

    def _rotate(self):
        self.close()
        self.idx += 1
        self.path = os.path.join(self.out_dir,
                                 f"{self.prefix}-{self.idx:06d}.tar")
        self.tar = tarfile.open(self.path + ".part", "w")
        self.written = 0

    def add(self, key, payload, meta):
        if self.written >= self.limit:
            self._rotate()
        for name, blob in ((f"{key}.jpg", payload),
                           (f"{key}.json",
                            json.dumps(meta, separators=(",", ":")).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            info.mtime = 0
            self.tar.addfile(info, io.BytesIO(blob))
            self.written += len(blob)
        self.records += 1

    def close(self):
        if self.tar is None:
            return
        self.tar.close()
        os.replace(self.path + ".part", self.path)
        self.shards.append(os.path.basename(self.path))
        self.tar = None


def index_ozfish(root):
    idx = {}
    if not root or not os.path.isdir(root):
        return idx
    for dirpath, _d, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".png"):
                continue
            m = SUFFIX.match(f)
            fm = OZ_FRAME.match(m.group("base") if m else f)
            if fm:
                idx.setdefault((fm["vid"], fm["cam"].upper(),
                                int(fm["frame"])), os.path.join(dirpath, f))
    return idx


def ozfish_frame_sizes(oz_idx):
    """-> {frame key: (w, h)} from the PNG headers. Image.open parses the
    header and stops, so this is a read of a few bytes per file, not a
    decode."""
    from PIL import Image
    out = {}
    for k, p in oz_idx.items():
        try:
            with Image.open(p) as im:
                out[k] = im.size
        except Exception:                                # noqa: BLE001
            continue
    return out


def frame_index(cfg, oz_idx, oz_sizes, stats):
    """-> {source: {"frames": [...], "rel": [(rw, rh), ...],
                    "ann_short": [int, ...]}}

    `rel` is every annotation box as a fraction of its own frame, pooled per
    source -- the distribution candidates are drawn from. `ann_short` is the
    same boxes' short sides in pixels, kept so the build can REPORT the size
    match against the real thing instead of claiming it.
    """
    # `rel` is the global fallback. `by_size` is the pool that actually
    # matters: annotation box sizes in ABSOLUTE pixels, keyed on the size of
    # the frame they were measured on.
    #
    # Relative matching is only equivalent to real matching when every frame
    # is the same size. FathomNet's are not -- they run from 1280x720 to
    # 8192x5464 -- and large absolute boxes come disproportionately from
    # large frames, so taking their RELATIVE size and applying it to a
    # typical smaller frame shrinks them. Measured on the first build: the
    # background ran 11% below the animals on median short side, 15 sigma,
    # while OzFish and PrePARED (uniform 1920x1080) matched to within half a
    # point. One source drifting and two not is the mechanism stating itself.
    idx = defaultdict(lambda: {"frames": [], "rel": [], "ann_short": [],
                               "by_size": defaultdict(list)})
    roots = cfg.get("frame_roots") or {}
    layouts = cfg.get("frame_layout") or {}

    for path in cfg["coco"]:
        print(f"  reading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        by_image = defaultdict(list)
        for an in doc.get("annotations", []):
            by_image[an["image_id"]].append(an)

        cut = {}                       # (src, frame key) -> record
        for im in doc.get("images", []):
            src = src_of.get(im.get("dataset_id"), "unknown")
            prov = im.get("crop_provenance")
            sm = im.get("source_meta") or {}
            uid = os.path.splitext(im["file_name"])[0]

            if prov == "frame":                                # PrePARED
                fw, fh = im.get("width"), im.get("height")
                if not fw or not fh:
                    continue
                boxes = []
                for an in by_image.get(im["id"], []):
                    b = an.get("bbox") or []
                    if len(b) < 4:
                        continue
                    boxes.append([float(b[0]), float(b[1]),
                                  float(b[2]), float(b[3])])
                    idx[src]["rel"].append((b[2] / fw, b[3] / fh))
                    idx[src]["ann_short"].append(int(min(b[2], b[3])))
                    idx[src]["by_size"][(int(fw), int(fh))].append(
                        (float(b[2]), float(b[3])))
                if im.get("has_unlabelled_animal"):
                    stats[f"{src}/frames excluded (unlabelled animal)"] += 1
                    continue
                idx[src]["frames"].append({
                    "key": uid, "path": os.path.join(cfg["image_dir"],
                                                     im["file_name"]),
                    "w": int(fw), "h": int(fh), "boxes": boxes})

            elif prov == "cut_from_frame":                     # FathomNet
                fb = sm.get("frame_bbox_used") or sm.get("frame_bbox")
                fs = sm.get("frame_size")
                uu = sm.get("fathomnet_image_uuid")
                root = roots.get(src)
                if not (fb and fs and uu and root and len(fb) >= 4):
                    continue
                fw, fh = int(fs[0]), int(fs[1])
                idx[src]["rel"].append((fb[2] / fw, fb[3] / fh))
                idx[src]["ann_short"].append(int(min(fb[2], fb[3])))
                idx[src]["by_size"][(fw, fh)].append((float(fb[2]),
                                                      float(fb[3])))
                k = (src, uu)
                if k not in cut:
                    rel = layouts.get(src, "{uuid}.jpg").format(
                        uuid=uu,
                        institution=safe(sm.get("owner_institution", "")),
                        source=src)
                    cut[k] = {"key": f"{src}-{uu}",
                              "path": os.path.join(root,
                                                   rel.replace("/", os.sep)),
                              "w": fw, "h": fh, "boxes": []}
                cut[k]["boxes"].append([float(fb[0]), float(fb[1]),
                                        float(fb[2]), float(fb[3])])

            elif sm.get("ozfish_uid") is not None:             # OzFish
                fb = sm.get("frame_bbox")
                fk = (str(sm.get("video")),
                      str(sm.get("camera") or "").upper(),
                      int(sm.get("frame", -1)))
                fpath = oz_idx.get(fk)
                if not (fb and fpath and len(fb) >= 4):
                    continue
                k = (src, fk)
                if k not in cut:
                    cut[k] = {"key": f"{src}-{fk[0]}_{fk[1]}_{fk[2]}",
                              "path": fpath, "w": None, "h": None,
                              "boxes": []}
                cut[k]["boxes"].append([float(fb[0]), float(fb[1]),
                                        float(fb[2]), float(fb[3])])
                idx[src]["ann_short"].append(int(min(fb[2], fb[3])))
                # OzFish frame sizes are not in the collation, so they are
                # read from the frame HEADERS (see oz_sizes) rather than
                # assumed. Without them this source contributes nothing to
                # the relative-size pool and is silently skipped -- which is
                # what the first version of this file did.
                fs = oz_sizes.get(fk)
                if fs:
                    cut[k]["w"], cut[k]["h"] = fs
                    idx[src]["rel"].append((fb[2] / fs[0], fb[3] / fs[1]))
                    idx[src]["by_size"][(int(fs[0]), int(fs[1]))].append(
                        (float(fb[2]), float(fb[3])))

        for (src, _k), ent in cut.items():
            idx[src]["frames"].append(ent)
        del doc, by_image, cut
    return idx


def intersects(box, boxes, margin):
    x0, y0, x1, y1 = box
    for bx, by, bw, bh in boxes:
        if (x0 < bx + bw + margin and x1 > bx - margin
                and y0 < by + bh + margin and y1 > by - margin):
            return True
    return False


def summarise(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return (s[n // 10], s[n // 2], s[min(n - 1, int(n * 0.9))],
            100.0 * sum(1 for v in s if v < 64) / n)


def build(cfg):
    from PIL import Image

    stats = Counter()
    rng = random.Random(cfg["seed"])
    oz_idx = index_ozfish((cfg.get("frame_roots") or {}).get("ozfish"))
    print(f"  {len(oz_idx):,} OzFish frames indexed")

    print("reading OzFish frame headers")
    oz_sizes = ozfish_frame_sizes(oz_idx)
    print(f"  {len(oz_sizes):,} sized")

    print("indexing frames and annotation boxes")
    idx = frame_index(cfg, oz_idx, oz_sizes, stats)
    for src, d in sorted(idx.items()):
        print(f"  {src:<12} {len(d['frames']):>8,} donor frames, "
              f"{len(d['rel']):>9,} boxes to match size against")

    writers, made_short = {}, defaultdict(list)
    out_dir = os.path.join(cfg["out_root"], cfg["name"])
    cap = int(cfg["cap_long_side"])
    q = int(cfg["jpeg_quality"])
    margin = int(cfg["margin_px"])
    tries = int(cfg["placement_tries"])

    for src, target in sorted(cfg["targets"].items()):
        d = idx.get(src)
        if not d or not d["frames"]:
            print(f"\n{src}: no donor frames -- skipped")
            continue
        if not d["rel"]:
            print(f"\n{src}: no box pool to match sizes against; skipping "
                  f"rather than inventing a size distribution")
            continue
        per_frame = int((cfg["max_per_frame"] or {}).get(src, 1))
        need_frames = min(len(d["frames"]),
                          -(-int(target) // max(1, per_frame)))
        frames = d["frames"][:]
        rng.shuffle(frames)
        frames = frames[:need_frames]
        print(f"\n{src}: {target:,} wanted, <= {per_frame} per frame, "
              f"visiting {len(frames):,} of {len(d['frames']):,} frames")

        made, t0 = 0, time.time()
        for i, fr in enumerate(frames, start=1):
            if made >= target:
                break
            try:
                with Image.open(fr["path"]) as f_im:
                    f_im.load()
                    frame = (f_im.convert("RGB") if f_im.mode != "RGB"
                             else f_im.copy())
            except Exception as exc:                     # noqa: BLE001
                stats[f"{src}/unreadable frame"] += 1
                if stats[f"{src}/unreadable frame"] <= 3:
                    print(f"    ! unreadable {fr['path']}: {exc}")
                continue
            fw, fh = frame.size
            if fr["w"] and (fr["w"], fr["h"]) != (fw, fh):
                stats[f"{src}/frame size disagrees with file"] += 1

            for j in range(per_frame):
                if made >= target:
                    break
                # DRAW THE SIZE ONCE, then try to place THAT size. Redrawing
                # inside the loop looks equivalent and is not: a large box is
                # harder to place without overlapping, so it gets rejected
                # more often and the retry substitutes a smaller one. That is
                # a rejection bias towards small boxes, and it destroys the
                # only property this set needs -- on the first version of this
                # file it pulled FathomNet's background 22 points below its
                # animals on '<64px', which the size-match report caught.
                # Draw from the boxes measured on frames THIS SIZE where
                # there are enough of them; fall back to the relative pool
                # otherwise, and count how often, because a high fallback
                # rate means the match is the weaker kind.
                pool = d["by_size"].get((fw, fh))
                if pool and len(pool) >= 50:
                    aw, ah = rng.choice(pool)
                    bw, bh = max(8, int(round(aw))), max(8, int(round(ah)))
                    stats[f"{src}/size drawn at frame size"] += 1
                else:
                    rw, rh = rng.choice(d["rel"])
                    bw = max(8, int(round(rw * fw)))
                    bh = max(8, int(round(rh * fh)))
                    stats[f"{src}/size drawn from the relative fallback"] += 1
                placed = None
                if bw < fw and bh < fh:
                    for _ in range(tries):
                        x0 = rng.randrange(0, fw - bw)
                        y0 = rng.randrange(0, fh - bh)
                        if not intersects((x0, y0, x0 + bw, y0 + bh),
                                          fr["boxes"], margin):
                            placed = (x0, y0, bw, bh)
                            break
                if placed is None:
                    # This box does not fit this frame. Move on rather than
                    # shrinking it -- a smaller substitute is exactly the bias.
                    stats[f"{src}/no room for the drawn size"] += 1
                    continue
                x0, y0, bw, bh = placed
                crop = frame.crop((x0, y0, x0 + bw, y0 + bh))
                native = crop.size
                if max(native) > cap:
                    s = cap / max(native)
                    crop = crop.resize((max(1, round(native[0] * s)),
                                        max(1, round(native[1] * s))),
                                       Image.LANCZOS)
                    stats[f"{src}/resized"] += 1
                buf = io.BytesIO()
                crop.save(buf, "JPEG", quality=q, optimize=True)

                if src not in writers:
                    writers[src] = ShardWriter(
                        out_dir, f"{cfg['name']}-{src}",
                        int(cfg["shard_bytes"]))
                key = f"{fr['key']}-bg{j:02d}"
                writers[src].add(key, buf.getvalue(), {
                    "source": src, "kind": "background", "geometry": "native",
                    "frame_key": fr["key"], "frame_size": [fw, fh],
                    "frame_bbox": [x0, y0, bw, bh],
                    "width": crop.size[0], "height": crop.size[1],
                    "native_width": native[0], "native_height": native[1],
                    "category_id": None, "category_name": None,
                })
                made_short[src].append(min(native))
                made += 1
            if i % 5000 == 0:
                el = time.time() - t0
                print(f"    {i:,}/{len(frames):,} frames, {made:,} crops, "
                      f"{i / el:.0f} frames/s", flush=True)
        print(f"  {src}: produced {made:,}")
        if made < target:
            print(f"  ! short of {target:,}. Either the frames are too "
                  f"crowded for the size distribution, or too few were "
                  f"reachable. The counters below say which.")

    per = {}
    for src, w in writers.items():
        w.close()
        per[src] = {"records": w.records, "shards": w.shards}

    print("\n" + "=" * 74)
    for s in sorted(per):
        print(f"  {s:<12} {per[s]['records']:>9,} records  "
              f"{len(per[s]['shards']):>3} shards")
    print()
    for k in sorted(stats):
        print(f"  {k:<48} {stats[k]:>10,}")

    # THE MATCH IS REPORTED, NOT ASSERTED -- WP7's done-when asks for exactly
    # this, because a size-matched claim is the difference between a rejection
    # result and an artefact.
    print(f"\nSIZE MATCH  background against the animal crops it must not be "
          f"separable from")
    print(f"  {'source':<12} {'':<12} {'p10':>6} {'median':>7} {'p90':>6} "
          f"{'<64px%':>8}")
    print("-" * 60)
    ok = True
    for src in sorted(made_short):
        a = summarise(idx[src]["ann_short"])
        b = summarise(made_short[src])
        if not a or not b:
            continue
        print(f"  {src:<12} {'animals':<12} {a[0]:>6,} {a[1]:>7,} "
              f"{a[2]:>6,} {a[3]:>7.1f}")
        print(f"  {'':<12} {'background':<12} {b[0]:>6,} {b[1]:>7,} "
              f"{b[2]:>6,} {b[3]:>7.1f}")
        # Judge the drift against SAMPLING ERROR, not against a fixed number
        # of points. A 5-point gap on 50,000 crops is a real bias; the same
        # gap on 200 is noise, and a check that cannot tell them apart will
        # either cry wolf or wave a real one through depending only on how
        # much was built.
        n = min(len(idx[src]["ann_short"]), len(made_short[src]))
        p = a[3] / 100.0
        se = 100.0 * ((p * (1 - p) / n) ** 0.5) if n else 0.0
        drift = abs(a[3] - b[3])
        sigmas = drift / se if se else 0.0
        print(f"  {'':<12} {'drift':<12} {drift:>6.1f} pts "
              f"= {sigmas:.1f} sigma at n={n:,}")
        if sigmas > 3:
            ok = False
            print(f"  {'':<12} ! that is beyond sampling error. A model "
                  f"could separate the classes on scale, which is the one "
                  f"thing this set must not allow.")
    if ok:
        print("\n  No source drifts beyond sampling error. The classes are "
              "not separable by scale.")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "build_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git": git_info(), "config": cfg["_config_path"],
            "per_source": per, "stats": dict(stats),
            "size_match": {s: {"animals": summarise(idx[s]["ann_short"]),
                               "background": summarise(made_short[s])}
                           for s in made_short},
            "caveat": "OzFish background contamination is UNMEASURED. The WP7 "
                      "review covered PrePARED (4.1%) and FathomNet (2.9%) "
                      "before OzFish frames existed. OzFish annotates about "
                      "one frame per fish event, so unboxed fish in a donor "
                      "frame are likely. Measure before use.",
        }, fh, indent=2)
    print(f"\nmanifest: {os.path.join(out_dir, 'build_manifest.json')}")
    print("\n! OzFish background contamination is UNMEASURED -- run "
          "scratch/annotation_audit.py over this set before using it.")
    print("Now run with --verify.")
    return 0


def verify(cfg, sample_n=400):
    """Structural, size match, and an overlap re-check against the COCO.

    The overlap check reads the frame's boxes from the collation rather than
    from anything the builder wrote. That is what makes it worth running: the
    realistic bug is not a broken rejection loop but a frame keyed to the
    wrong box list, and only an independent lookup catches that.
    """
    from PIL import Image

    out_dir = os.path.join(cfg["out_root"], cfg["name"])
    tars = sorted(f for f in os.listdir(out_dir) if f.endswith(".tar"))
    if not tars:
        sys.exit(f"no shards in {out_dir}")
    parts = [f for f in os.listdir(out_dir) if f.endswith(".part")]
    rc = 0
    if parts:
        print(f"! {len(parts)} unfinished .part file(s) -- build incomplete")
        rc = 1

    print("LEVEL 1  structural")
    seen, bad, decoded, metas = Counter(), 0, 0, {}
    rng = random.Random(cfg["seed"])
    reservoir, n_seen = [], 0
    for t in tars:
        per_key = defaultdict(set)
        with tarfile.open(os.path.join(out_dir, t)) as tf:
            for info in tf:
                key, _, ext = info.name.partition(".")
                per_key[key].add(ext)
                payload = tf.extractfile(info).read()
                if ext == "json":
                    m = json.loads(payload)
                    metas[key] = m
                    n_seen += 1
                    if len(reservoir) < sample_n:
                        reservoir.append((key, m))
                    else:
                        j = rng.randrange(n_seen)
                        if j < sample_n:
                            reservoir[j] = (key, m)
                else:
                    try:
                        with Image.open(io.BytesIO(payload)) as im:
                            im.verify()
                        decoded += 1
                    except Exception as exc:             # noqa: BLE001
                        bad += 1
                        print(f"    ! {t}:{info.name}: {exc}")
        for key, exts in per_key.items():
            seen[key] += 1
            if "jpg" not in exts or "json" not in exts:
                bad += 1
        print(f"  {t}: {len(per_key):,} records")
    dupes = {k: c for k, c in seen.items() if c > 1}
    print(f"  {len(seen):,} records, {decoded:,} decoded, {bad:,} problems"
          + (f", {len(dupes):,} DUPLICATE keys" if dupes else ""))
    rc |= 1 if (bad or dupes) else 0

    print("\nLEVEL 2  no candidate touches an annotation (boxes re-read from "
          "the collation)")
    want_frames = {m["frame_key"] for _k, m in reservoir}
    boxes_by_frame = defaultdict(list)
    for path in cfg["coco"]:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        by_image = defaultdict(list)
        for an in doc.get("annotations", []):
            by_image[an["image_id"]].append(an)
        for im in doc.get("images", []):
            src = src_of.get(im.get("dataset_id"), "unknown")
            sm = im.get("source_meta") or {}
            prov = im.get("crop_provenance")
            uid = os.path.splitext(im["file_name"])[0]
            if prov == "frame":
                fk = uid
                if fk not in want_frames:
                    continue
                for an in by_image.get(im["id"], []):
                    b = an.get("bbox") or []
                    if len(b) >= 4:
                        boxes_by_frame[fk].append([float(b[0]), float(b[1]),
                                                   float(b[2]), float(b[3])])
            elif prov == "cut_from_frame":
                fk = f"{src}-{sm.get('fathomnet_image_uuid')}"
                fb = sm.get("frame_bbox_used") or sm.get("frame_bbox")
                if fk in want_frames and fb and len(fb) >= 4:
                    boxes_by_frame[fk].append([float(x) for x in fb[:4]])
            elif sm.get("ozfish_uid") is not None:
                fk = (f"{src}-{sm.get('video')}_"
                      f"{str(sm.get('camera') or '').upper()}_"
                      f"{sm.get('frame')}")
                fb = sm.get("frame_bbox")
                if fk in want_frames and fb and len(fb) >= 4:
                    boxes_by_frame[fk].append([float(x) for x in fb[:4]])
        del doc, by_image

    checked = hits = nomatch = 0
    missing_eg = []
    for key, m in reservoir:
        fk = m["frame_key"]
        if fk not in boxes_by_frame:
            nomatch += 1
            if len(missing_eg) < 6:
                missing_eg.append(f"{m.get('source')}  {key}  frame={fk}")
            continue
        checked += 1
        x0, y0, bw, bh = m["frame_bbox"]
        for bx, by, bw2, bh2 in boxes_by_frame[fk]:
            ix = max(0.0, min(x0 + bw, bx + bw2) - max(x0, bx))
            iy = max(0.0, min(y0 + bh, by + bh2) - max(y0, by))
            if ix * iy > 0:
                hits += 1
                if hits <= 5:
                    print(f"    ! {key} overlaps an annotation on {fk}")
                break
    print(f"  {checked - hits:,}/{checked:,} clear of every annotation "
          f"on their frame")
    if nomatch:
        print(f"  ! {nomatch:,} of {len(reservoir):,} sampled records name "
              f"a frame the collation does not have. Those records were NOT "
              f"checked for overlap:")
        for e in missing_eg:
            print(f"      {e}")
        print("    A frame key that does not round-trip is either a builder "
              "bug or a verifier one, and which it is decides whether the "
              "set needs rebuilding. Compare the key above against how that "
              "source's frame_key is composed in frame_index().")
        rc = 1
    rc |= 1 if hits else 0

    print("\nLEVEL 3  the size match still holds in the built set")
    with open(os.path.join(out_dir, "build_manifest.json"),
              encoding="utf-8") as fh:
        man = json.load(fh)
    for src, d in sorted((man.get("size_match") or {}).items()):
        a, b = d.get("animals"), d.get("background")
        if a and b:
            print(f"  {src:<12} animals <64px {a[3]:.1f}%   background "
                  f"{b[3]:.1f}%   drift {abs(a[3] - b[3]):.1f} pts")

    print("\nLEVEL 4  semantic -- NOT CHECKED HERE. A candidate clear of every")
    print("  box can still hold an animal nobody boxed. That residual is what")
    print("  contamination means, it was 2.9% and 4.1% for FathomNet and")
    print("  PrePARED, and it is UNMEASURED for OzFish.")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sample", type=int, default=400)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.verify:
        sys.exit(verify(cfg, args.sample))
    d = os.path.join(cfg["out_root"], cfg["name"])
    if os.path.isdir(d) and any(f.endswith(".tar") for f in os.listdir(d)):
        sys.exit(f"{d} already holds shards. Move them aside deliberately.")
    sys.exit(build(cfg))


if __name__ == "__main__":
    main()

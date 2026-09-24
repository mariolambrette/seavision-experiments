#!/usr/bin/env python3
"""Build the `square` shard sets (plan 10.2).

Square expansion needs pixels from outside the box, so it is the one geometry
that cannot be applied at load time. Everything else about the design follows
from that one fact plus three decisions taken on measurement:

SIDE = max(w, h) * (1 + margin)
    Two margins, 0% and 10%, each CUT FOR REAL rather than one derived from
    the other. A smaller square is only a centre crop of a larger one while
    both stay centred on the box; shifting to fit breaks that, and shifting is
    what keeps edge animals in the set.

SHIFT, DO NOT CLIP
    The square's origin is clamped into the frame; its SIDE never changes.
    That has a property worth stating because it is why no animal is lost:
    provided the side is no larger than the frame's short side, a clamped
    square still contains the whole box. Unclamped it is centred on the box
    and the side is at least max(w, h); clamped to an edge, the box was
    already within half a side of that edge. Profiled at 16.9% of records
    shifting at 10% margin -- not a cost, since a real deployment produces
    off-centre animals, but recorded per record so it can be stratified.

PAD ONLY WHERE NO SQUARE EXISTS
    Two fallbacks, in order. If the box fits a square but the MARGIN tips it
    over the frame's short side, shrink the side and keep the whole animal at
    a smaller margin (`margin_reduced`). If max(w, h) itself exceeds the
    frame's short side -- an animal filling or exceeding the frame -- no
    square contains it at any margin, so pad rather than clip, because
    clipping loses the animal and padding only loses honesty, which the
    `pad_fraction` field recovers. Profiled at 1.35% of records and
    margin-invariant. Filtering those is an experiment-config decision (plan
    10.5), not a build decision.

FRAME SIZE COMES FROM THE PIXELS
    Not from `source_meta.frame_size`. The frame has to be decoded to cut
    from it, so its real dimensions are in hand, and using them makes the
    build immune to a wrong recorded size. Disagreements are counted and
    reported rather than silently absorbed.

KEYS MATCH THE `crops` SET
    `{uid}-a{ann_id}` where the collation image is a frame, `{uid}` where it
    is already a crop -- the same rule build_shards.py uses. So a crop and its
    two squares share a key and the geometry comparison is PAIRED, the same
    animal under three treatments, rather than three populations that happen
    to resemble each other. FishWIO has no frames and is absent from these
    sets, so any join against `crops` is an inner join on 1,370,528 of
    1,485,215 keys.

ONE FRAME PASS
    ~535,000 frames decoded once each, every box on the frame cut at both
    margins while it is in memory. Decoding twice for two margins would be
    hours wasted.

    python pipeline/build_square_shards.py --config configs/shards_square.yaml
    python pipeline/build_square_shards.py --config configs/shards_square.yaml --verify
"""

from __future__ import annotations

import argparse
import io
import json
import os
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

REQUIRED = ["coco", "image_dir", "out_root", "margins", "cap_long_side"]

SUFFIX = re.compile(r"^(?P<base>.+?\.png)-\d+-\d+\.png$", re.I)
OZ_FRAME = re.compile(
    r"^(?P<vid>[A-Za-z]+\d+)_(?P<cam>[LR])\.(?P<ext>avi|mp4|mpeg)\."
    r"(?P<frame>\d+)\.png$", re.I)


def safe(code):
    """The FathomNet converter's own institution-to-directory rule."""
    return "".join(c if c.isalnum() else "_" for c in str(code))


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=here, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD"],
                                       cwd=here, text=True)
        return {"commit": commit, "dirty": bool(diff.strip()),
                # A dirty build that records its diff is still reproducible.
                # Flagging without recording only tells you it is lost.
                "diff": diff if diff.strip() else None}
    except Exception:                                    # noqa: BLE001
        return {"commit": None, "dirty": None, "diff": None}


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        sys.exit(f"{path}: missing {missing}")
    cfg.setdefault("jpeg_quality", 95)
    cfg.setdefault("shard_bytes", 1 << 30)
    cfg.setdefault("pad_rgb", [128, 128, 128])
    cfg.setdefault("name", "square")
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


# --------------------------------------------------------------- geometry

def square_for(x0, y0, w, h, fw, fh, margin):
    """-> dict describing the square, or None if the box is unusable.

    The single place the rule lives. Everything else in this file is
    bookkeeping around it.
    """
    long_side = max(w, h)
    short_frame = min(fw, fh)
    if long_side < 2:
        return None

    side = int(round(long_side * (1.0 + margin)))
    tier, pad_fraction = "fits", 0.0

    if long_side > short_frame:
        tier = "padded"                      # no square contains the animal
    elif side > short_frame:
        side, tier = short_frame, "margin_reduced"

    ideal_x = (x0 + w / 2.0) - side / 2.0
    ideal_y = (y0 + h / 2.0) - side / 2.0
    if tier == "padded":
        x, y = ideal_x, ideal_y              # do not shift; pad symmetrically
    else:
        x = min(max(ideal_x, 0.0), max(0.0, fw - side))
        y = min(max(ideal_y, 0.0), max(0.0, fh - side))

    xi, yi = int(round(x)), int(round(y))
    if tier == "padded":
        vis_w = max(0, min(xi + side, fw) - max(xi, 0))
        vis_h = max(0, min(yi + side, fh) - max(yi, 0))
        pad_fraction = 1.0 - (vis_w * vis_h) / float(side * side)

    return {
        "side": side, "x": xi, "y": yi, "tier": tier,
        "shift": [round(xi - ideal_x, 1), round(yi - ideal_y, 1)],
        "pad_fraction": round(max(0.0, pad_fraction), 4),
        # where the animal sits inside the stored square, which is what
        # "back-calculate the shift" actually needs
        "box_in_square": [int(round(x0 - xi)), int(round(y0 - yi)),
                          int(round(w)), int(round(h))],
    }


def cut_square(frame, sq, pad_rgb):
    """PIL pads an out-of-bounds crop with black. Black is a decision nobody
    made, so the padded region is filled explicitly instead."""
    from PIL import Image
    side, x, y = sq["side"], sq["x"], sq["y"]
    box = (x, y, x + side, y + side)
    if sq["tier"] != "padded":
        return frame.crop(box)
    canvas = Image.new("RGB", (side, side), tuple(pad_rgb))
    fw, fh = frame.size
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(fw, x + side), min(fh, y + side)
    if sx1 > sx0 and sy1 > sy0:
        canvas.paste(frame.crop((sx0, sy0, sx1, sy1)), (sx0 - x, sy0 - y))
    return canvas


# ---------------------------------------------------------------- writer

class ShardWriter:
    """Byte-target rotation, .part until closed -- as build_shards.py, so the
    two sets are read by identical loader code."""

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


# ------------------------------------------------------------- frame jobs

def index_ozfish(root):
    """-> {(video, CAMERA, frame): path}, container token deliberately absent."""
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


def base_meta(im, src):
    return {
        "source": src, "dataset_id": im.get("dataset_id"),
        "groups": im.get("groups") or {},
        "group_sources": im.get("group_sources") or {},
        "has_unlabelled_animal": bool(im.get("has_unlabelled_animal")),
        "lat": im.get("lat"), "lon": im.get("lon"),
        "datetime": im.get("datetime"), "depth_m": im.get("depth_m"),
    }


def frame_jobs(cfg, oz_idx, stats):
    """-> {frame_path: [record dicts]}, one entry per frame to decode.

    Grouped by frame so each decode serves every box on it, at both margins.
    """
    jobs = defaultdict(list)
    roots = cfg.get("frame_roots") or {}
    layouts = cfg.get("frame_layout") or {}

    for path in cfg["coco"]:
        print(f"  reading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        cats = {c["id"]: c.get("name") for c in doc.get("categories", [])}
        by_image = defaultdict(list)
        for an in doc.get("annotations", []):
            by_image[an["image_id"]].append(an)

        for im in doc.get("images", []):
            src = src_of.get(im.get("dataset_id"), "unknown")
            prov = im.get("crop_provenance")
            sm = im.get("source_meta") or {}
            uid = os.path.splitext(im["file_name"])[0]
            anns = sorted(by_image.get(im["id"], []), key=lambda a: a["id"])

            if prov == "frame":                                  # PrePARED
                fpath = os.path.join(cfg["image_dir"], im["file_name"])
                for an in anns:
                    b = an.get("bbox") or []
                    if len(b) < 4 or b[2] < 2 or b[3] < 2:
                        continue
                    jobs[fpath].append({
                        "key": f"{uid}-a{an['id']}", "uid": uid,
                        "box": [float(b[0]), float(b[1]),
                                float(b[2]), float(b[3])],
                        "meta": dict(base_meta(im, src), **{
                            "annotation_id": an["id"],
                            "category_id": an.get("category_id"),
                            "category_name": cats.get(an.get("category_id")),
                            "declared_frame_size": [im.get("width"),
                                                    im.get("height")],
                        })})

            elif prov == "cut_from_frame":                       # FathomNet
                fb = sm.get("frame_bbox_used") or sm.get("frame_bbox")
                uu = sm.get("fathomnet_image_uuid")
                root = roots.get(src)
                if not (fb and uu and root and len(fb) >= 4):
                    stats[f"{src}/no frame route"] += 1
                    continue
                rel = layouts.get(src, "{uuid}.jpg").format(
                    uuid=uu, institution=safe(sm.get("owner_institution", "")),
                    source=src)
                fpath = os.path.join(root, rel.replace("/", os.sep))
                an = anns[0] if anns else None
                jobs[fpath].append({
                    "key": uid, "uid": uid,
                    "box": [float(fb[0]), float(fb[1]),
                            float(fb[2]), float(fb[3])],
                    "meta": dict(base_meta(im, src), **{
                        "annotation_id": an["id"] if an else None,
                        "category_id": an.get("category_id") if an else None,
                        "category_name": (cats.get(an.get("category_id"))
                                          if an else None),
                        "declared_frame_size": sm.get("frame_size"),
                    })})

            elif sm.get("ozfish_uid") is not None:               # OzFish
                fb = sm.get("frame_bbox")
                key = (str(sm.get("video")),
                       str(sm.get("camera") or "").upper(),
                       int(sm.get("frame", -1)))
                fpath = oz_idx.get(key)
                if not (fb and fpath and len(fb) >= 4):
                    stats[f"{src}/no frame route"] += 1
                    continue
                an = anns[0] if anns else None
                jobs[fpath].append({
                    "key": uid, "uid": uid,
                    "box": [float(fb[0]), float(fb[1]),
                            float(fb[2]), float(fb[3])],
                    "meta": dict(base_meta(im, src), **{
                        "annotation_id": an["id"] if an else None,
                        "category_id": an.get("category_id") if an else None,
                        "category_name": (cats.get(an.get("category_id"))
                                          if an else None),
                        "declared_frame_size": None,
                    })})
            else:
                stats[f"{src}/no frames exist"] += 1
        del doc, by_image
    return jobs


# ------------------------------------------------------------------ build

def build(cfg):
    from PIL import Image

    stats = Counter()
    oz_idx = index_ozfish((cfg.get("frame_roots") or {}).get("ozfish"))
    print(f"  {len(oz_idx):,} OzFish frames indexed")

    print("grouping records by frame")
    jobs = frame_jobs(cfg, oz_idx, stats)
    n_rec = sum(len(v) for v in jobs.values())
    print(f"  {len(jobs):,} frames, {n_rec:,} records, "
          f"{n_rec / max(1, len(jobs)):.1f} per frame")
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  excluded  {k:<32} {v:>10,}")

    margins = cfg["margins"]
    writers = {}                             # (margin_key, source) -> writer
    cap = int(cfg["cap_long_side"])
    q = int(cfg["jpeg_quality"])
    pad_rgb = cfg["pad_rgb"]

    t0, n = time.time(), 0
    for fpath in sorted(jobs):
        recs = jobs[fpath]
        try:
            with Image.open(fpath) as fr:
                fr.load()
                frame = fr.convert("RGB") if fr.mode != "RGB" else fr.copy()
        except Exception as exc:                         # noqa: BLE001
            stats["unreadable_frame"] += 1
            stats["records_lost_to_unreadable_frame"] += len(recs)
            if stats["unreadable_frame"] <= 5:
                print(f"    ! unreadable frame {fpath}: {exc}")
            continue
        fw, fh = frame.size

        for r in recs:
            declared = r["meta"].get("declared_frame_size")
            if declared and len(declared) >= 2 and declared[0] and \
                    (int(declared[0]), int(declared[1])) != (fw, fh):
                stats["frame_size_disagrees_with_file"] += 1
            x0, y0, w, h = r["box"]
            src = r["meta"]["source"]
            for mkey, margin in margins.items():
                sq = square_for(x0, y0, w, h, fw, fh, float(margin))
                if sq is None:
                    stats[f"{src}/degenerate box"] += 1
                    continue
                img = cut_square(frame, sq, pad_rgb)
                native = img.size[0]
                if native > cap:
                    img = img.resize((cap, cap), Image.LANCZOS)
                    stats[f"{src}/{mkey}/resized"] += 1
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=q, optimize=True)

                wk = (mkey, src)
                if wk not in writers:
                    out_dir = os.path.join(cfg["out_root"],
                                           f"{cfg['name']}-{mkey}")
                    writers[wk] = ShardWriter(
                        out_dir, f"{cfg['name']}-{mkey}-{src}",
                        int(cfg["shard_bytes"]))
                meta = dict(r["meta"])
                meta.update({
                    "uid": r["uid"], "geometry": "square",
                    "margin": float(margin), "margin_key": mkey,
                    "frame_size": [fw, fh],
                    "square_origin": [sq["x"], sq["y"]],
                    "square_side_native": sq["side"],
                    "box_in_square": sq["box_in_square"],
                    "shift": sq["shift"], "tier": sq["tier"],
                    "pad_fraction": sq["pad_fraction"],
                    "width": img.size[0], "height": img.size[1],
                    "native_width": native, "native_height": native,
                })
                writers[wk].add(r["key"], buf.getvalue(), meta)
                stats[f"{src}/{mkey}/records"] += 1
                stats[f"{src}/{mkey}/tier_{sq['tier']}"] += 1

        n += 1
        if n % 5000 == 0:
            el = time.time() - t0
            print(f"    {n:,}/{len(jobs):,} frames  {n / el:.0f}/s  "
                  f"~{(len(jobs) - n) / max(n / el, 1e-9) / 60:.0f} min left",
                  flush=True)

    per = {}
    for (mkey, src), w in writers.items():
        w.close()
        per[f"{mkey}/{src}"] = {"records": w.records, "shards": w.shards}

    print("\n" + "=" * 70)
    for k in sorted(per):
        print(f"  {k:<28} {per[k]['records']:>10,} records  "
              f"{len(per[k]['shards']):>3} shards")
    print()
    for k in sorted(stats):
        print(f"  {k:<44} {stats[k]:>12,}")

    for mkey in margins:
        out_dir = os.path.join(cfg["out_root"], f"{cfg['name']}-{mkey}")
        man = {
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git": git_info(), "config": cfg["_config_path"],
            "margin": float(margins[mkey]),
            "rule": "side = max(w,h) * (1 + margin); shift into frame; "
                    "pad only where no square fits",
            "per_source": {k.split("/", 1)[1]: v for k, v in per.items()
                           if k.startswith(mkey + "/")},
            "stats": {k: v for k, v in stats.items() if f"/{mkey}/" in k},
            "note": "FishWIO has no frames and is absent from this set; join "
                    "against the crops set on key as an INNER join.",
        }
        with open(os.path.join(out_dir, "build_manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(man, fh, indent=2)
        print(f"manifest: {os.path.join(out_dir, 'build_manifest.json')}")
    print("\nNow run with --verify.")
    return 0


# ----------------------------------------------------------------- verify

def verify(cfg, crops_dir=None, sample_n=300):
    """Levels 1-3. Level 4 is still eyes on a contact sheet.

    Level 3 here is stronger than the crops set's could be. There is no
    byte-copy to hash, but there IS an independent ground truth: the `crops`
    set holds the same animal under the same key, built by different code
    from different inputs. Extracting the recorded `box_in_square` region
    from a square must reproduce that crop. Re-deriving the geometry from the
    stored numbers would instead share any bug the builder has, and prove
    nothing.
    """
    from PIL import Image, ImageChops
    import random

    rc = 0
    for mkey in cfg["margins"]:
        out_dir = os.path.join(cfg["out_root"], f"{cfg['name']}-{mkey}")
        tars = sorted(f for f in os.listdir(out_dir) if f.endswith(".tar"))
        parts = [f for f in os.listdir(out_dir) if f.endswith(".part")]
        print(f"\n{'=' * 70}\n{mkey}  ({len(tars)} shards)")
        if parts:
            print(f"  ! {len(parts)} unfinished .part file(s): {parts[:4]}")
            rc = 1

        print("\nLEVEL 1  structural")
        seen, bad, decoded = Counter(), 0, 0
        rng = random.Random(cfg.get("seed", 0))
        reservoir, n_seen = [], 0
        for t in tars:
            per_key = defaultdict(set)
            blobs = {}
            with tarfile.open(os.path.join(out_dir, t)) as tf:
                for info in tf:
                    key, _, ext = info.name.partition(".")
                    per_key[key].add(ext)
                    payload = tf.extractfile(info).read()
                    if ext == "json":
                        blobs.setdefault(key, {})["meta"] = json.loads(payload)
                    else:
                        try:
                            with Image.open(io.BytesIO(payload)) as im:
                                im.verify()
                            decoded += 1
                        except Exception as exc:         # noqa: BLE001
                            bad += 1
                            print(f"    ! {t}:{info.name}: {exc}")
                        blobs.setdefault(key, {})["img"] = payload
            for key, exts in per_key.items():
                seen[key] += 1
                if "jpg" not in exts or "json" not in exts:
                    print(f"    ! {t}:{key} incomplete: {sorted(exts)}")
                    bad += 1
            for key, b in blobs.items():
                n_seen += 1
                if len(reservoir) < sample_n:
                    reservoir.append((key, b))
                else:
                    j = rng.randrange(n_seen)
                    if j < sample_n:
                        reservoir[j] = (key, b)
            print(f"  {t}: {len(per_key):,} records")
        dupes = {k: c for k, c in seen.items() if c > 1}
        print(f"  {len(seen):,} records, {decoded:,} decoded, {bad:,} problems"
              + (f", {len(dupes):,} DUPLICATE keys" if dupes else ""))
        rc |= 1 if (bad or dupes) else 0

        print("\nLEVEL 2  the stored numbers obey the rule")
        wrong = 0
        for key, b in reservoir:
            m = b.get("meta") or {}
            fw, fh = m.get("frame_size", [0, 0])
            bis = m.get("box_in_square") or [0, 0, 0, 0]
            sq = square_for(m["square_origin"][0] + bis[0],
                            m["square_origin"][1] + bis[1],
                            bis[2], bis[3], fw, fh, m.get("margin", 0.0))
            if not sq or sq["side"] != m.get("square_side_native") or \
                    [sq["x"], sq["y"]] != m.get("square_origin"):
                wrong += 1
                if wrong <= 5:
                    print(f"    ! {key}: stored side "
                          f"{m.get('square_side_native')} origin "
                          f"{m.get('square_origin')}, rule says "
                          f"{sq and sq['side']} / {sq and [sq['x'], sq['y']]}")
        print(f"  {len(reservoir) - wrong:,}/{len(reservoir):,} consistent")
        print("  (this shares the builder's rule, so it catches a slip in the")
        print("   bookkeeping, not a misreading of the geometry -- level 3 is")
        print("   the one that could)")
        rc |= 1 if wrong else 0

        if not crops_dir:
            print("\nLEVEL 3  SKIPPED -- pass --crops to cross-check against "
                  "the crops set")
            continue

        print(f"\nLEVEL 3  the animal is where the metadata says it is")
        crops = {}
        want = {k for k, _b in reservoir}
        for t in sorted(f for f in os.listdir(crops_dir)
                        if f.endswith(".tar")):
            with tarfile.open(os.path.join(crops_dir, t)) as tf:
                for info in tf:
                    key, _, ext = info.name.partition(".")
                    if key in want and ext != "json":
                        crops[key] = tf.extractfile(info).read()
            if len(crops) >= len(want):
                break
        checked = close = 0
        for key, b in reservoir:
            if key not in crops:
                continue
            m = b["meta"]
            bx, by, bw, bh = m["box_in_square"]
            scale = m["width"] / float(m["square_side_native"])
            try:
                with Image.open(io.BytesIO(b["img"])) as s:
                    sub = s.convert("RGB").crop(
                        (int(bx * scale), int(by * scale),
                         int((bx + bw) * scale), int((by + bh) * scale)))
                with Image.open(io.BytesIO(crops[key])) as c:
                    ref = c.convert("RGB")
            except Exception:                            # noqa: BLE001
                continue
            if sub.size[0] < 4 or sub.size[1] < 4:
                continue
            checked += 1
            ref = ref.resize(sub.size, Image.LANCZOS)
            diff = ImageChops.difference(sub, ref)
            mean = sum(c * i for i, c in enumerate(diff.convert("L")
                                                   .histogram())) / \
                max(1, sub.size[0] * sub.size[1])
            if mean < 12:            # JPEG re-encode + resample, not content
                close += 1
            elif checked - close <= 5:
                print(f"    ! {key}: mean abs difference {mean:.1f} against "
                      f"the crops set")
        print(f"  {close:,}/{checked:,} squares reproduce their crop at the "
              f"recorded offset")
        if checked and close / checked < 0.95:
            print("  FAIL: the animal is not where the metadata says. The")
            print("  square is being cut from the wrong place, or the offset")
            print("  is being recorded wrong. Either way do not build on it.")
            rc = 1

    print("\nLEVEL 4  semantic -- still needs eyes. Use "
          "scratch/annotation_audit.py against these sets.")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--crops", help="the crops shard dir, for level 3")
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.verify:
        sys.exit(verify(cfg, args.crops, args.sample))

    for mkey in cfg["margins"]:
        d = os.path.join(cfg["out_root"], f"{cfg['name']}-{mkey}")
        if os.path.isdir(d) and any(f.endswith(".tar") for f in os.listdir(d)):
            sys.exit(f"{d} already holds shards. Move them aside "
                     f"deliberately rather than interleaving two builds.")
    sys.exit(build(cfg))


if __name__ == "__main__":
    main()

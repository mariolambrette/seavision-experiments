#!/usr/bin/env python3
"""
Build the `crops` shard set (plan 10.2).

One tar per ~1 GB, WebDataset convention: for each record a key, an image file
and a .json of its metadata. PyTorch reads these natively and sequentially,
which is the entire point -- the collation is 1.47M small files and opening a
file costs the same whether it holds 7 kB or 7 MB.

WHAT THIS SET IS
    `crops`   every annotated crop at NATIVE resolution and NATIVE aspect
              ratio, long side capped. Serves BOTH letterbox and distort,
              because those are pure functions of the stored crop and are
              applied on the GPU at load time. Square expansion is not here:
              it needs pixels from outside the box and gets its own set.

THE ONE LOSSY DECISION, AND THE FLOOR THAT DEFUSES IT
    A crop whose long side is within --cap is copied into the tar BYTE FOR
    BYTE: not decoded, not re-encoded, so no JPEG generation loss. Only crops
    above the cap are resampled. Measured on the collation, a 1024 cap touches
    2.09% of crops.

    Capping the long side also shrinks the SHORT side, and the short side is
    where the species signal lives. So a short-side floor overrides the cap: a
    crop that would fall below `min_short_side` is left oversized instead.
    That makes the build lossless in the sense that matters, at the cost of a
    handful of large files (three, at cap 1024).

TWO PROVENANCES, TWO PATHS -- and this is easy to get wrong
    Where `crop_provenance` is `frame` the stored image is a FRAME, not a
    crop: yolo-bruv is 15,945 crops across 2,667 frames. Those must be cut
    here, so they are always decoded and re-encoded, and one frame yields many
    records. Everywhere else the stored image IS the crop, so it is copied.
    Treating all sources the same would either lose 13,000 yolo-bruv crops or
    shard whole frames as if they were animals.

    Frames carrying no annotation are skipped: a frame with no box is
    background material, and background is its own set with its own sampling
    design (plan 6.1).

    Images with no annotation that ARE crops are kept and flagged -- FishWIO's
    1,600 background crops and FathomNet's 49,967 unidentified-animal crops
    are both wanted downstream, and they are not the same thing.

    python pipeline/build_shards.py --config configs/shards_crops.yaml
    python pipeline/build_shards.py --config configs/shards_crops.yaml --verify
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required")

REQUIRED = ["coco", "image_dir", "out_dir", "cap_long_side", "min_short_side"]
IMAGE_EXT = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------

def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=here, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        cwd=here, text=True).strip() != ""
        return {"commit": commit, "dirty": dirty}
    except Exception:                                    # noqa: BLE001
        return {"commit": None, "dirty": None}


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        sys.exit(f"{path}: missing required keys {missing}")
    cfg.setdefault("shard_bytes", 1 << 30)
    cfg.setdefault("jpeg_quality", 95)
    cfg.setdefault("name", "crops")
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def target_size(w, h, cap, floor):
    """-> (new_w, new_h) or None when the crop should be left alone.

    None means BYTE COPY. The floor wins over the cap deliberately: better a
    handful of oversized files than a crop resampled below the size at which
    its species is still legible.
    """
    long_side, short_side = max(w, h), min(w, h)
    if long_side <= cap:
        return None
    scale = cap / long_side
    if short_side * scale < floor:
        return None                     # exempt: leaving it oversized is safer
    return max(1, round(w * scale)), max(1, round(h * scale))


# --------------------------------------------------------------------------

class ShardWriter:
    """Rotates tars at a byte target. Writes to .part and renames on close, so
    an interrupted build never leaves a truncated tar that looks finished."""

    def __init__(self, out_dir, prefix, shard_bytes):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix = out_dir, prefix
        self.limit = shard_bytes
        self.idx, self.tar, self.path, self.written = -1, None, None, 0
        self.records, self.shards = 0, []
        self._rotate()

    def _rotate(self):
        self.close()
        self.idx += 1
        self.path = os.path.join(self.out_dir, f"{self.prefix}-{self.idx:06d}.tar")
        self.tar = tarfile.open(self.path + ".part", "w")
        self.written = 0

    def add(self, key, ext, image_bytes, meta):
        if self.written >= self.limit:
            self._rotate()
        for name, payload in ((f"{key}{ext}", image_bytes),
                              (f"{key}.json",
                               json.dumps(meta, separators=(",", ":")).encode())):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0                       # reproducible byte-for-byte
            self.tar.addfile(info, io.BytesIO(payload))
            self.written += len(payload)
        self.records += 1

    def close(self):
        if self.tar is None:
            return
        self.tar.close()
        os.replace(self.path + ".part", self.path)
        self.shards.append(os.path.basename(self.path))
        self.tar = None


# --------------------------------------------------------------------------

def build_one(coco_path, cfg, stats):
    from PIL import Image

    print(f"\n  loading {os.path.basename(coco_path)} ...", flush=True)
    with open(coco_path, encoding="utf-8") as fh:
        doc = json.load(fh)

    src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
    by_image = defaultdict(list)
    for an in doc.get("annotations", []):
        by_image[an["image_id"]].append(an)

    # Deterministic order: a run must be reproducible without keeping a
    # manifest of 1.47M paths in step (plan 10.2).
    images = sorted(doc.get("images", []), key=lambda i: i["id"])
    cats = {c["id"]: c.get("name") for c in doc.get("categories", [])}

    cap = int(cfg["cap_long_side"])
    floor = int(cfg["min_short_side"])
    image_dir = cfg["image_dir"]

    writers, t0, n = {}, time.time(), 0
    for im in images:
        src = src_of.get(im.get("dataset_id"), "unknown")
        anns = sorted(by_image.get(im["id"], []), key=lambda a: a["id"])
        provenance = im.get("crop_provenance")
        path = os.path.join(image_dir, im["file_name"])
        uid = os.path.splitext(im["file_name"])[0]

        if src not in writers:
            writers[src] = ShardWriter(cfg["out_dir"],
                                       f"{cfg['name']}-{src}",
                                       int(cfg["shard_bytes"]))
        w = writers[src]

        def record(key, ext, payload, meta):
            w.add(key, ext, payload, meta)
            stats[f"{src}/records"] += 1

        base = {
            "uid": uid, "source": src, "dataset_id": im.get("dataset_id"),
            "crop_provenance": provenance,
            "groups": im.get("groups") or {},
            "group_sources": im.get("group_sources") or {},
            "has_unlabelled_animal": bool(im.get("has_unlabelled_animal")),
            "lat": im.get("lat"), "lon": im.get("lon"),
            "datetime": im.get("datetime"), "depth_m": im.get("depth_m"),
        }

        if provenance == "frame":
            if not anns:
                stats[f"{src}/frames_skipped_no_box"] += 1
                continue
            try:
                with Image.open(path) as frame:
                    frame.load()
                    if frame.mode != "RGB":
                        frame = frame.convert("RGB")
                    fw, fh = frame.size
                    for an in anns:
                        b = an.get("bbox") or []
                        if len(b) < 4:
                            stats[f"{src}/bad_bbox"] += 1
                            continue
                        x0, y0 = int(round(b[0])), int(round(b[1]))
                        x1, y1 = x0 + int(round(b[2])), y0 + int(round(b[3]))
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(fw, x1), min(fh, y1)
                        if x1 - x0 < 2 or y1 - y0 < 2:
                            stats[f"{src}/degenerate"] += 1
                            continue
                        crop = frame.crop((x0, y0, x1, y1))
                        cw, ch = crop.size
                        tgt = target_size(cw, ch, cap, floor)
                        if tgt:
                            crop = crop.resize(tgt, Image.LANCZOS)
                            stats[f"{src}/resized"] += 1
                        else:
                            stats[f"{src}/cut_native"] += 1
                        buf = io.BytesIO()
                        crop.save(buf, "JPEG", quality=int(cfg["jpeg_quality"]),
                                  optimize=True)
                        meta = dict(base)
                        meta.update({
                            "annotation_id": an["id"],
                            "category_id": an.get("category_id"),
                            "category_name": cats.get(an.get("category_id")),
                            "frame_bbox": [x0, y0, x1 - x0, y1 - y0],
                            "frame_size": [fw, fh],
                            "width": crop.size[0], "height": crop.size[1],
                            "native_width": cw, "native_height": ch,
                            "byte_copied": False,
                        })
                        record(f"{uid}-a{an['id']}", ".jpg", buf.getvalue(), meta)
            except Exception as exc:                     # noqa: BLE001
                stats[f"{src}/unreadable_frame"] += 1
                print(f"    ! unreadable frame {im['file_name']}: {exc}")
        else:
            an = anns[0] if anns else None
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
                with Image.open(io.BytesIO(raw)) as probe:
                    cw, ch = probe.size
                    fmt = (probe.format or "").upper()
            except Exception as exc:                     # noqa: BLE001
                stats[f"{src}/unreadable_crop"] += 1
                print(f"    ! unreadable crop {im['file_name']}: {exc}")
                continue

            tgt = target_size(cw, ch, cap, floor)
            ext = os.path.splitext(im["file_name"])[1].lower() or ".jpg"
            if tgt is None:
                payload, out_ext, copied = raw, ext, True
                stats[f"{src}/byte_copied"] += 1
            else:
                with Image.open(io.BytesIO(raw)) as img:
                    img.load()
                    if fmt == "PNG" and img.mode in ("RGBA", "P", "LA"):
                        img = img.convert("RGB")
                    elif img.mode != "RGB":
                        img = img.convert("RGB")
                    img = img.resize(tgt, Image.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=int(cfg["jpeg_quality"]),
                             optimize=True)
                payload, out_ext, copied = buf.getvalue(), ".jpg", False
                stats[f"{src}/resized"] += 1

            meta = dict(base)
            meta.update({
                "annotation_id": an["id"] if an else None,
                "category_id": an.get("category_id") if an else None,
                "category_name": cats.get(an.get("category_id")) if an else None,
                "width": tgt[0] if tgt else cw,
                "height": tgt[1] if tgt else ch,
                "native_width": cw, "native_height": ch,
                "byte_copied": copied,
                "source_format": fmt,
            })
            record(uid, out_ext, payload, meta)

        n += 1
        if n % 25000 == 0:
            el = time.time() - t0
            print(f"    {n:,}/{len(images):,} images  {n/el:.0f}/s", flush=True)

    out = {}
    for src, w in writers.items():
        w.close()
        out[src] = {"records": w.records, "shards": w.shards}
    del doc, by_image, images
    return out


# --------------------------------------------------------------------------

def expected_records(cfg):
    """The key set the COCO files imply, by the same rules the builder uses.

    This shares the KEY rule with the builder, which is fine -- the rule is two
    lines and is stated in the docstring. It deliberately does NOT re-derive
    the pixels: a check that recomputes a value the same way the builder did
    inherits any misreading and proves nothing. Whether the frame cuts are
    correct is settled by looking at them, not here (level 4).
    """
    want = {}
    for path in cfg["coco"]:
        print(f"  reading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        by_image = defaultdict(list)
        for an in doc.get("annotations", []):
            by_image[an["image_id"]].append(an)
        for im in doc.get("images", []):
            uid = os.path.splitext(im["file_name"])[0]
            src = src_of.get(im.get("dataset_id"), "unknown")
            anns = by_image.get(im["id"], [])
            if im.get("crop_provenance") == "frame":
                fw, fh = im.get("width"), im.get("height")
                for an in anns:                 # one record per BOX
                    b = an.get("bbox") or []
                    if len(b) < 4:
                        continue
                    # The builder clips the box to the frame BEFORE testing it,
                    # so a box hanging off the edge can survive here and die
                    # there. Clip the same way or every such box is reported
                    # missing. This is the key rule, not the pixel rule.
                    x0, y0 = max(0, int(round(b[0]))), max(0, int(round(b[1])))
                    x1 = int(round(b[0])) + int(round(b[2]))
                    y1 = int(round(b[1])) + int(round(b[3]))
                    if fw and fh:
                        x1, y1 = min(int(fw), x1), min(int(fh), y1)
                    if x1 - x0 < 2 or y1 - y0 < 2:
                        continue                # builder drops degenerate boxes
                    want[f"{uid}-a{an['id']}"] = {
                        "source": src, "category_id": an.get("category_id"),
                        "file_name": None,
                    }
            else:                               # the image IS the crop
                an = anns[0] if anns else None
                want[uid] = {
                    "source": src,
                    "category_id": an.get("category_id") if an else None,
                    "file_name": im["file_name"],
                }
        del doc, by_image
    return want


def verify(cfg, sample_n=500, skip_completeness=False):
    """Four levels, of which this does three. The fourth is looking at them.

    1  structural   every record has an image and a json; every image decodes
    2  completeness the key set matches the COCO exactly -- no missing, no
                    extras, no duplicates. A correct TOTAL can hide a wrong
                    CONTENT, and a duplicate key silently shadows a record.
    3  fidelity     sampled byte-copied records hash-match the file on disk;
                    sampled resized records obey the cap-and-floor rule;
                    category_id matches the collation.
    4  semantic     NOT HERE. If the frame bbox convention were misread, every
                    yolo-bruv crop would be a patch of seabed and levels 1-3
                    would all pass. Only an independent implementation or a
                    human looking at a contact sheet settles that.
    """
    from PIL import Image
    import random

    out_dir = cfg["out_dir"]
    tars = sorted(f for f in os.listdir(out_dir) if f.endswith(".tar"))
    if not tars:
        sys.exit(f"no shards in {out_dir}")
    parts = [f for f in os.listdir(out_dir) if f.endswith(".part")]
    if parts:
        print(f"  ! {len(parts)} unfinished .part file(s) -- the build did not "
              f"complete: {parts[:5]}")

    print("\nLEVEL 1  structural")
    print("-" * 74)
    seen = {}                       # key -> count, for duplicate detection
    got_meta = {}                   # key -> (source, category_id, byte_copied)
    bad = 0
    decoded = 0
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
                    m = json.loads(payload)
                    got_meta[key] = (m.get("source"), m.get("category_id"),
                                     bool(m.get("byte_copied")))
                    blobs.setdefault(key, {})["meta"] = m
                else:
                    try:
                        with Image.open(io.BytesIO(payload)) as im:
                            im.verify()
                        decoded += 1
                    except Exception as exc:             # noqa: BLE001
                        bad += 1
                        print(f"    ! {t}:{info.name} does not decode: {exc}")
                    blobs.setdefault(key, {})["img"] = payload
                    blobs[key]["ext"] = "." + ext
        for key, exts in per_key.items():
            seen[key] = seen.get(key, 0) + 1
            if not (exts & {"jpg", "jpeg", "png"}) or "json" not in exts:
                print(f"    ! {t}:{key} incomplete record: {sorted(exts)}")
                bad += 1
        # reservoir sample whole records for level 3
        for key, b in blobs.items():
            n_seen += 1
            if len(reservoir) < sample_n:
                reservoir.append((key, b))
            else:
                j = rng.randrange(n_seen)
                if j < sample_n:
                    reservoir[j] = (key, b)
        print(f"  {t}: {len(per_key):,} records")

    print(f"\n  {len(tars)} shards, {len(seen):,} records, "
          f"{decoded:,} images decoded, {bad:,} structural problems")

    dupes = {k: c for k, c in seen.items() if c > 1}
    if dupes:
        print(f"  ! {len(dupes):,} DUPLICATE keys -- later records shadow "
              f"earlier ones: {list(dupes)[:5]}")
        bad += len(dupes)

    # ---- level 2 --------------------------------------------------------
    missing = extra = 0
    if skip_completeness:
        print("\nLEVEL 2  completeness -- SKIPPED (--quick)")
    else:
        print("\nLEVEL 2  completeness")
        print("-" * 74)
        want = expected_records(cfg)
        wk, gk = set(want), set(seen)
        missing_set, extra_set = wk - gk, gk - wk
        missing, extra = len(missing_set), len(extra_set)
        print(f"  COCO implies {len(wk):,} records; shards hold {len(gk):,}")
        print(f"  missing from shards: {missing:,}")
        for k in sorted(missing_set)[:10]:
            print(f"      {k}  ({want[k]['source']})")
        print(f"  present but not in the COCO: {extra:,}")
        for k in sorted(extra_set)[:10]:
            print(f"      {k}")
        mism = [k for k in (wk & gk)
                if want[k]["category_id"] != got_meta.get(k, (None, None, None))[1]]
        print(f"  category_id disagreements: {len(mism):,}")
        for k in mism[:10]:
            print(f"      {k}: COCO {want[k]['category_id']} vs shard "
                  f"{got_meta[k][1]}")
        bad += missing + extra + len(mism)

    # ---- level 3 --------------------------------------------------------
    print(f"\nLEVEL 3  fidelity ({len(reservoir):,} sampled records)")
    print("-" * 74)
    cap, floor = int(cfg["cap_long_side"]), int(cfg["min_short_side"])
    hashed = hash_ok = size_ok = size_bad = 0
    for key, b in reservoir:
        m = b.get("meta") or {}
        if m.get("byte_copied"):
            src_path = os.path.join(cfg["image_dir"], key + b.get("ext", ""))
            if os.path.exists(src_path):
                hashed += 1
                with open(src_path, "rb") as fh:
                    same = hashlib.sha256(fh.read()).hexdigest() == \
                           hashlib.sha256(b["img"]).hexdigest()
                hash_ok += same
                if not same:
                    print(f"    ! {key}: byte-copied record differs from "
                          f"{src_path}")
        nw, nh = m.get("native_width"), m.get("native_height")
        w, h = m.get("width"), m.get("height")
        if None in (nw, nh, w, h):
            continue
        want_size = target_size(nw, nh, cap, floor)
        expect = want_size if want_size else (nw, nh)
        if (w, h) == tuple(expect):
            size_ok += 1
        else:
            size_bad += 1
            print(f"    ! {key}: stored {w}x{h}, cap rule says {expect} "
                  f"(native {nw}x{nh})")

    print(f"  byte-copied sampled: {hashed:,}   hash-identical to source: "
          f"{hash_ok:,}")
    print(f"  cap-and-floor rule:  {size_ok:,} correct, {size_bad:,} wrong")
    bad += (hashed - hash_ok) + size_bad

    print("\nLEVEL 4  semantic -- NOT CHECKED HERE")
    print("-" * 74)
    print("  yolo-bruv crops are CUT from frames at build time. If the bbox")
    print("  convention were misread, every one would be a patch of seabed and")
    print("  levels 1-3 would all pass: they decode, the count is right, the")
    print("  sizes obey the rule. Re-cutting them here would inherit the same")
    print("  misreading. Only looking at them settles it -- see the contact")
    print("  sheets from the annotation-quality check.")

    print()
    if bad:
        print(f"VERIFY FAILED: {bad:,} problem(s). Do not build on these shards.")
        return 1
    print("Levels 1-3 pass. Level 4 still needs eyes on a sample.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="read the shards back instead of building")
    ap.add_argument("--sample", type=int, default=500,
                    help="records to check byte-for-byte in level 3")
    ap.add_argument("--quick", action="store_true",
                    help="skip level 2, which re-reads the COCO files")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.verify:
        sys.exit(verify(cfg, sample_n=args.sample,
                        skip_completeness=args.quick))

    if os.path.isdir(cfg["out_dir"]) and any(
            f.endswith(".tar") for f in os.listdir(cfg["out_dir"])):
        sys.exit(f"{cfg['out_dir']} already holds shards. Building into it "
                 f"would interleave two builds. Move them aside deliberately.")

    stats = Counter()
    t0 = time.time()
    per_source = {}
    for path in cfg["coco"]:
        per_source.update(build_one(path, cfg, stats))

    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_info(),
        "config_path": cfg["_config_path"],
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "elapsed_s": round(time.time() - t0, 1),
        "per_source": per_source,
        "counts": dict(stats),
    }
    mpath = os.path.join(cfg["out_dir"], "build_manifest.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 70)
    for src in sorted(per_source):
        print(f"  {src:<12} {per_source[src]['records']:>10,} records  "
              f"{len(per_source[src]['shards']):>4} shards")
    print()
    for k in sorted(stats):
        print(f"  {k:<34} {stats[k]:>10,}")
    copied = sum(v for k, v in stats.items() if k.endswith("/byte_copied"))
    resized = sum(v for k, v in stats.items() if k.endswith("/resized"))
    if copied + resized:
        print(f"\n  byte-copied (lossless): {copied:,}  "
              f"({100*copied/(copied+resized):.2f}% of non-frame crops)")
    print(f"\nManifest: {mpath}")
    if manifest["git"]["dirty"]:
        print("  [WARNING: uncommitted changes -- this build is not reproducible]")
    print("Now run with --verify before using these shards.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Level 4: put eyes on the shards.

Levels 1-3 in build_shards.py prove the shards are structurally sound, count
out exactly right and obey the size rule. None of that can tell you whether a
crop contains the animal it claims to. If the bbox convention were misread,
every yolo-bruv record would be a patch of seabed and all three levels would
still pass. The only check that does not inherit the builder's assumptions is
a person looking at the pixels.

THE TASK IS BINARY AND BLIND
    Two questions get answered by one review pass:

      contamination  do background candidates contain an animal? (they must
                     not, or the background class teaches the model nothing)
      gross quality  do annotation crops contain an animal? (a crop that does
                     not means the box was wrong, the frame was wrong, or the
                     cut was wrong)

    Both reduce to "is there an animal in this tile", so annotation and
    background tiles are SHUFFLED TOGETHER and labelled with nothing but a
    number. A reviewer who can see which is which will find what they expect
    to find in each. Blinding is not pedantry here; it is the only reason the
    two rates are comparable.

    Note it is "an animal", not "a fish". FathomNet is mostly not fish and
    FishWIO has invertebrates. Asking for fish would score most of FathomNet
    as empty.

WHAT THIS CHECK CANNOT SEE
    A crop holding the WRONG animal passes the binary. That is deliberate:
    this pass is looking for gross geometry failures, which are the failure
    mode levels 1-3 are blind to and which would destroy the whole set at
    once. Species-level label quality is a different review with a different
    design (it needs the label shown, so it cannot be blind) and belongs after
    this one.

BACKGROUND SAMPLING -- the production rule, not a convenience
    A background candidate is a box cut from a frame that:
      * does not intersect any annotation on that frame (with a margin),
      * comes from a frame NOT flagged has_unlabelled_animal, and
      * has a RELATIVE size drawn from that source's real annotation boxes.

    The size matching matters. If backgrounds were uniformly large and
    annotations small, both the reviewer and later a model would be separating
    the classes on size rather than content, and the background class would be
    worthless. Relative rather than absolute, because frame sizes vary.

    Backgrounds come from any source whose FRAMES we still hold, which is two
    of them and by two different routes (see frame_index):

      PrePARED   the collation stores the frames themselves.
      FathomNet  the collation stores crops, but `source_meta` carries the
                 frame uuid, the box on the frame and the frame's size, so a
                 frame is reassembled by grouping its crops. The pixels are in
                 `frame_roots['fathomnet']`.

    FathomNet matters here and is not a nice-to-have. PrePARED is UK BRUV
    seabed at 18 sites; FathomNet is deep-sea ROV and is about 86% of the
    collation. A background class drawn from PrePARED alone would not be thin,
    it would be the wrong habitat -- it teaches what empty British seabed looks
    like and nothing about an empty water column. The contamination rate would
    likewise be measured on the easy source and silently assumed to hold for
    the hard one.

    OzFish joins when its frames land. FishWIO's 1,600 background crops already
    exist as crops and are not resampled here -- they are a different thing,
    chosen by whoever built that dataset, and mixing them in would confound the
    contamination rate with somebody else's sampling decision.

    Excluding frames flagged `has_unlabelled_animal` is the rule the production
    sampler will use on the PrePARED route, so the contamination rate measured
    there is the residual rate of the real sampler rather than of a strawman. A
    zero-overlap box in an unflagged frame can still hold an unlabelled animal
    -- that residual is exactly the number this is trying to bound.

USAGE
    python scratch/annotation_audit.py sample --config configs/shards_crops.yaml \
        --out "D:/marineai/_audit" --per-source 200 --background 200

    python scratch/annotation_audit.py sheets --out "D:/marineai/_audit"

    # review the sheets, write down the numbers with NO animal, then:
    python scratch/annotation_audit.py score --out "D:/marineai/_audit" \
        --no-animal "3,17,22-25,101" --unsure "44,90"

Nothing here writes into the collation, the shards or the NAS. Everything
lands under --out, which is disposable.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
import tarfile
from collections import Counter, defaultdict

try:
    import yaml
except ImportError:
    sys.exit("pyyaml is required")

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

MANIFEST = "manifest.csv"
TILES = "tiles"
SHEETS = "sheets"


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def source_of_shard(fn, set_name):
    """crops-yolo-bruv-000003.tar -> yolo-bruv.

    Source names contain hyphens, so strip the known prefix and the numeric
    suffix rather than splitting on hyphens and hoping.
    """
    stem = fn[:-4]
    if stem.startswith(set_name + "-"):
        stem = stem[len(set_name) + 1:]
    return stem.rsplit("-", 1)[0]


# ---------------------------------------------------------------- sampling

def sample_annotations(cfg, n_default, per_source, rng):
    """-> list of dicts with image bytes, sampled evenly per source.

    Two passes over the tars. The first reads only member HEADERS, which is
    cheap even over a few hundred GB because the tars are uncompressed and
    tarfile seeks past the payloads. Only the chosen members are then read
    back. Sampling by reading everything would take longer than the build did.
    """
    out_dir = cfg["out_dir"]
    set_name = cfg.get("name", "crops")
    tars = sorted(f for f in os.listdir(out_dir) if f.endswith(".tar"))
    if not tars:
        sys.exit(f"no shards in {out_dir} -- build them first")

    print("pass 1: listing shard members")
    by_source = defaultdict(list)            # source -> [(tar, key, ext)]
    for t in tars:
        src = source_of_shard(t, set_name)
        n = 0
        with tarfile.open(os.path.join(out_dir, t)) as tf:
            for info in tf:
                key, _, ext = info.name.partition(".")
                if ext == "json":
                    continue
                by_source[src].append((t, key, ext))
                n += 1
        print(f"  {t}: {n:,} records  ({src})")

    unknown = set(per_source) - set(by_source)
    if unknown:
        sys.exit(f"--per-source names source(s) not in the shards: "
                 f"{sorted(unknown)}. Known: {sorted(by_source)}")

    picks = defaultdict(list)                # tar -> [(key, ext, source)]
    for src, recs in sorted(by_source.items()):
        want = per_source.get(src, n_default)
        if want <= 0:
            print(f"  {src}: skipped")
            continue
        take = min(want, len(recs))
        if take < want:
            print(f"  ! {src} holds only {len(recs):,} records; taking all")
        for t, key, ext in rng.sample(recs, take):
            picks[t].append((key, ext, src))
        print(f"  {src}: sampling {take:,} of {len(recs):,}")

    print("pass 2: extracting the sampled records")
    rows = []
    for t in sorted(picks):
        wanted = {}
        for key, ext, src in picks[t]:
            wanted[f"{key}.{ext}"] = (key, src)
            wanted[f"{key}.json"] = (key, src)
        got_img, got_meta = {}, {}
        with tarfile.open(os.path.join(out_dir, t)) as tf:
            for info in tf:
                if info.name not in wanted:
                    continue
                key, src = wanted[info.name]
                payload = tf.extractfile(info).read()
                if info.name.endswith(".json"):
                    got_meta[key] = json.loads(payload)
                else:
                    got_img[key] = payload
        for key in got_img:
            m = got_meta.get(key, {})
            rows.append({
                "kind": "annotation",
                "source": m.get("source") or wanted[key + "." + "json"][1],
                "key": key,
                "category_id": m.get("category_id"),
                "category_name": m.get("category_name"),
                "detail": f"{m.get('width')}x{m.get('height')}",
                "bytes": got_img[key],
            })
        print(f"  {t}: {len(got_img):,} extracted")
    return rows


def safe(code):
    """Exactly the converter's own rule for turning an institution code into a
    directory name. Duplicated deliberately: importing the converter would drag
    its config machinery in, and a silent divergence here shows up as every
    frame missing, which the resolve count below makes impossible to miss."""
    return "".join(c if c.isalnum() else "_" for c in str(code))


def frame_index(cfg, rng=None):
    """-> source -> {"frames": [...], "rel": [(rw, rh), ...]}

    `rel` are annotation box sizes as a FRACTION of their frame, pooled per
    source. Background boxes are drawn from this so the two classes are
    indistinguishable on size alone.

    TWO KINDS OF DONOR, and they are not symmetrical
    ------------------------------------------------
    `crop_provenance == "frame"`  the stored image IS the frame (PrePARED).
        Boxes are the annotations on it. A frame flagged
        `has_unlabelled_animal` is DROPPED: the flag means an animal nobody
        boxed, so a box-free region of that frame is not background.

    `crop_provenance == "cut_from_frame"`  the stored image is a crop we cut
        ourselves from a frame still on disk (FathomNet). Frame identity, the
        box on the frame and the frame's size all live in `source_meta`, so
        the frame is reassembled by grouping crops on their frame id. Here
        `has_unlabelled_animal` means a box exists with an UNIDENTIFIED
        concept -- the box is in the exclusion set already, so dropping the
        frame would discard ~50,000 usable donors for nothing.

        The real hazard on this path is the one nothing flags: deep-sea frames
        are not exhaustively annotated, so unboxed animals will be present.
        That residual is what the contamination number exists to bound, and it
        should be expected to be worse than PrePARED's.

    Frames for the second kind are not in `image_dir`, so their location comes
    from config:

        frame_roots:
          fathomnet: "D:/marineai/dataset/raw/fathomnet/images"
        frame_layout:
          fathomnet: "{institution}/{uuid}.jpg"

    `{institution}` is passed through the converter's own `safe()`. How many
    frames actually resolved on disk is printed, because a wrong root yields
    zero candidates silently otherwise.
    """
    roots = cfg.get("frame_roots") or {}
    layouts = cfg.get("frame_layout") or {}
    idx = defaultdict(lambda: {"frames": [], "rel": []})

    for path in cfg["coco"]:
        print(f"  reading {os.path.basename(path)} ...", flush=True)
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        src_of = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        by_image = defaultdict(list)
        for an in doc.get("annotations", []):
            by_image[an["image_id"]].append(an)

        cut = defaultdict(lambda: {"path": None, "w": 0, "h": 0, "boxes": []})
        for im in doc.get("images", []):
            src = src_of.get(im.get("dataset_id"), "unknown")
            prov = im.get("crop_provenance")

            if prov == "frame":
                fw, fh_ = im.get("width"), im.get("height")
                if not fw or not fh_:
                    continue
                boxes = []
                for an in by_image.get(im["id"], []):
                    b = an.get("bbox") or []
                    if len(b) < 4:
                        continue
                    boxes.append([float(b[0]), float(b[1]),
                                  float(b[2]), float(b[3])])
                    idx[src]["rel"].append((b[2] / fw, b[3] / fh_))
                if im.get("has_unlabelled_animal"):
                    continue                 # an animal nobody boxed
                idx[src]["frames"].append(
                    {"tag": os.path.splitext(im["file_name"])[0],
                     "path": os.path.join(cfg["image_dir"], im["file_name"]),
                     "w": int(fw), "h": int(fh_), "boxes": boxes})

            elif prov == "cut_from_frame":
                sm = im.get("source_meta") or {}
                fb, fs = sm.get("frame_bbox"), sm.get("frame_size")
                uuid = sm.get("fathomnet_image_uuid") or sm.get("frame_id")
                if not (fb and fs and uuid) or len(fb) < 4 or len(fs) < 2:
                    continue
                fw, fh_ = int(fs[0]), int(fs[1])
                if not fw or not fh_:
                    continue
                idx[src]["rel"].append((fb[2] / fw, fb[3] / fh_))
                root = roots.get(src)
                if not root:
                    continue                 # no route to the pixels
                tmpl = layouts.get(src, "{uuid}.jpg")
                rel = tmpl.format(
                    uuid=uuid,
                    institution=safe(sm.get("owner_institution", "")),
                    source=src)
                ent = cut[(src, uuid)]
                ent["path"] = os.path.join(root, rel.replace("/", os.sep))
                ent["tag"] = f"{src}-{uuid}"
                ent["w"], ent["h"] = fw, fh_
                ent["boxes"].append([float(fb[0]), float(fb[1]),
                                     float(fb[2]), float(fb[3])])

        # Only the crops that share a frame reconstruct that frame's boxes, so
        # this grouping has to finish before any of them can be used.
        by_src = defaultdict(list)
        for (src, _uuid), ent in cut.items():
            by_src[src].append(ent)
        for src, ents in by_src.items():
            print(f"  {src}: {len(ents):,} distinct frames reassembled from "
                  f"their crops", flush=True)
            # Existence is checked on a sample rather than all 469k, because
            # os.path.exists on half a million files is minutes of stat calls
            # to answer a question a sample answers.
            probe = (rng or random).sample(ents, min(200, len(ents)))
            hit = sum(1 for e in probe if os.path.exists(e["path"]))
            print(f"    {hit}/{len(probe)} sampled frames found on disk")
            if not hit:
                print(f"    ! none of the sampled frames exist. Check "
                      f"frame_roots['{src}'] and frame_layout['{src}'].")
                print(f"      tried: {probe[0]['path']}")
                continue
            idx[src]["frames"].extend(ents)
        del doc, by_image, cut, by_src
    return idx


def intersects(box, boxes, margin):
    x0, y0, x1, y1 = box
    for bx, by, bw, bh in boxes:
        if (x0 < bx + bw + margin and x1 > bx - margin
                and y0 < by + bh + margin and y1 > by - margin):
            return True
    return False


def parse_counts(spec, default=0):
    """'200' -> a flat default; 'fathomnet=400,yolo-bruv=200' -> per source.

    Weighting belongs here rather than in a post-hoc reweighting, because the
    point of asking for more FathomNet is a tighter interval on FathomNet, and
    you cannot widen a sample after the fact.
    """
    if spec is None or spec == "":
        return default, {}
    spec = str(spec)
    if "=" not in spec:
        return int(spec), {}
    per = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            sys.exit(f"cannot read '{tok}': expected source=count")
        src, n = tok.rsplit("=", 1)
        per[src.strip()] = int(n)
    return 0, per


def sample_backgrounds(cfg, n_default, per_source, rng, margin=8, tries=60):
    print("indexing frames")
    idx = frame_index(cfg, rng)
    rows = []
    cap = int(cfg["cap_long_side"])
    floor = int(cfg["min_short_side"])
    quality = int(cfg.get("jpeg_quality", 95))

    unknown = set(per_source) - set(idx)
    if unknown:
        sys.exit(f"--background names source(s) with no frames in the "
                 f"collation: {sorted(unknown)}. Known: {sorted(idx)}")

    for src, d in sorted(idx.items()):
        n_per_source = per_source.get(src, n_default)
        if n_per_source <= 0:
            continue
        frames, rel = d["frames"], d["rel"]
        if not frames:
            print(f"  {src}: no eligible frames -- either this source stores "
                  f"no frames, every frame is flagged, or frame_roots/"
                  f"frame_layout do not reach the pixels (see above)")
            continue
        if not rel:
            print(f"  {src}: no annotation boxes to match sizes against; "
                  f"skipping rather than inventing a size distribution")
            continue
        print(f"  {src}: {len(frames):,} eligible frames, "
              f"{len(rel):,} boxes to match size against")
        made, attempts, missing = 0, 0, 0
        while made < n_per_source and attempts < n_per_source * 40:
            attempts += 1
            fr = rng.choice(frames)
            rw, rh = rng.choice(rel)
            bw = max(8, int(round(rw * fr["w"])))
            bh = max(8, int(round(rh * fr["h"])))
            if bw >= fr["w"] or bh >= fr["h"]:
                continue
            placed = None
            for _ in range(tries):
                x0 = rng.randrange(0, fr["w"] - bw)
                y0 = rng.randrange(0, fr["h"] - bh)
                box = (x0, y0, x0 + bw, y0 + bh)
                if not intersects(box, fr["boxes"], margin):
                    placed = box
                    break
            if placed is None:
                continue                      # crowded frame; try another
            try:
                with Image.open(fr["path"]) as frame:
                    frame.load()
                    if frame.mode != "RGB":
                        frame = frame.convert("RGB")
                    fw_real, fh_real = frame.size
                    # The COCO's frame_size must agree with the file, or the
                    # box was placed in a coordinate system the pixels do not
                    # use and the "background" is somewhere else entirely.
                    if (fw_real, fh_real) != (fr["w"], fr["h"]):
                        missing += 1
                        if missing <= 3:
                            print(f"    ! {fr['tag']}: COCO says "
                                  f"{fr['w']}x{fr['h']}, file is "
                                  f"{fw_real}x{fh_real} -- skipped")
                        continue
                    crop = frame.crop(placed)
            except Exception as exc:          # noqa: BLE001
                missing += 1
                if missing <= 3:
                    print(f"    ! unreadable frame {fr['path']}: {exc}")
                continue
            cw, ch = crop.size
            long_side, short_side = max(cw, ch), min(cw, ch)
            if long_side > cap:
                scale = cap / long_side
                if short_side * scale >= floor:
                    crop = crop.resize((max(1, round(cw * scale)),
                                        max(1, round(ch * scale))),
                                       Image.LANCZOS)
            buf = io.BytesIO()
            crop.save(buf, "JPEG", quality=quality, optimize=True)
            rows.append({
                "kind": "background",
                "source": src,
                "key": f"{fr['tag']}-bg{made:04d}",
                "category_id": "",
                "category_name": "",
                "detail": f"{placed[0]},{placed[1]},{bw},{bh}",
                "bytes": buf.getvalue(),
            })
            made += 1
        print(f"    produced {made:,} candidates in {attempts:,} attempts"
              + (f", {missing:,} frames skipped as missing, unreadable or the "
                 f"wrong size" if missing else ""))
        if made < n_per_source:
            print(f"    ! short of {n_per_source}: either the frames are too "
                  f"crowded for the size distribution being asked for, or too "
                  f"many are unreachable. The two numbers above say which.")
    return rows


def cmd_sample(args):
    cfg = load_config(args.config)
    rng = random.Random(args.seed)
    ann_n, ann_per = parse_counts(args.per_source, 200)
    bg_n, bg_per = parse_counts(args.background, 0)
    rows = sample_annotations(cfg, ann_n, ann_per, rng)
    if bg_n or bg_per:
        rows += sample_backgrounds(cfg, bg_n, bg_per, rng)

    rng.shuffle(rows)                          # the blinding
    tile_dir = os.path.join(args.out, TILES)
    os.makedirs(tile_dir, exist_ok=True)
    with open(os.path.join(args.out, MANIFEST), "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["index", "kind", "source", "key", "category_id",
                     "category_name", "detail", "tile"])
        for i, r in enumerate(rows, start=1):
            name = f"{i:05d}.jpg"
            with open(os.path.join(tile_dir, name), "wb") as out:
                out.write(r["bytes"])
            wr.writerow([i, r["kind"], r["source"], r["key"],
                         r["category_id"], r["category_name"], r["detail"],
                         name])

    counts = Counter((r["kind"], r["source"]) for r in rows)
    print(f"\n{len(rows):,} tiles -> {tile_dir}")
    for (kind, src), n in sorted(counts.items()):
        print(f"  {kind:<11} {src:<12} {n:,}")
    print(f"manifest: {os.path.join(args.out, MANIFEST)}")
    print("\nThe manifest says which is which. Do not read it before "
          "reviewing the sheets.")
    return 0


# ---------------------------------------------------------------- sheets

def cmd_sheets(args):
    tile_dir = os.path.join(args.out, TILES)
    sheet_dir = os.path.join(args.out, SHEETS)
    os.makedirs(sheet_dir, exist_ok=True)
    tiles = sorted(f for f in os.listdir(tile_dir) if f.endswith(".jpg"))
    if not tiles:
        sys.exit(f"no tiles in {tile_dir} -- run `sample` first")

    cell, cols, rows = args.tile, args.cols, args.rows
    label_h = max(18, cell // 10)
    head_h = label_h + 10
    per_sheet = cols * rows
    font = load_font(max(12, cell // 16))
    head_font = load_font(max(14, cell // 13))

    n_sheets = (len(tiles) + per_sheet - 1) // per_sheet
    pad = 6
    sheet_w = cols * (cell + pad) + pad
    sheet_h = head_h + rows * (cell + label_h + pad) + pad

    for s in range(n_sheets):
        chunk = tiles[s * per_sheet:(s + 1) * per_sheet]
        sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
        dr = ImageDraw.Draw(sheet)
        first = int(os.path.splitext(chunk[0])[0])
        last = int(os.path.splitext(chunk[-1])[0])
        dr.text((pad, 6), f"sheet {s + 1} of {n_sheets}    "
                          f"tiles {first}-{last}    "
                          f"write down every number with NO animal",
                fill=(0, 0, 0), font=head_font)
        for i, fn in enumerate(chunk):
            cx = pad + (i % cols) * (cell + pad)
            cy = head_h + (i // cols) * (cell + label_h + pad)
            dr.rectangle([cx, cy, cx + cell, cy + label_h],
                         fill=(255, 255, 255))
            dr.text((cx + 4, cy + 2), os.path.splitext(fn)[0].lstrip("0"),
                    fill=(0, 0, 0), font=font)
            try:
                with Image.open(os.path.join(tile_dir, fn)) as im:
                    im = im.convert("RGB")
                    # letterbox, never distort: a stretched crop is harder to
                    # judge and this sheet is the judgement
                    im.thumbnail((cell, cell), Image.LANCZOS)
                    box = Image.new("RGB", (cell, cell), (20, 20, 20))
                    box.paste(im, ((cell - im.width) // 2,
                                   (cell - im.height) // 2))
                    sheet.paste(box, (cx, cy + label_h))
            except Exception as exc:          # noqa: BLE001
                dr.text((cx + 4, cy + label_h + 4), f"unreadable: {exc}",
                        fill=(200, 0, 0), font=font)
            # FathomNet crops are often nearly black, and so is the letterbox
            # padding. Without an outline a reviewer cannot tell where the
            # image stops and the padding starts, which is the difference
            # between "empty crop" and "dark crop".
            dr.rectangle([cx, cy, cx + cell, cy + label_h + cell],
                         outline=(120, 120, 120), width=1)
        out = os.path.join(sheet_dir, f"sheet_{s + 1:03d}.jpg")
        sheet.save(out, "JPEG", quality=90, optimize=True)
        print(f"  {out}  ({len(chunk)} tiles)")
    print(f"\n{n_sheets} sheets, {per_sheet} tiles each -> {sheet_dir}")
    return 0


# ---------------------------------------------------------------- scoring

def parse_indices(text):
    """'3,17,22-25 101' -> {3,17,22,23,24,25,101}. Accepts commas, spaces,
    newlines and ranges, because a reviewer writing 400 numbers will use all
    four and should not have to care."""
    if not text:
        return set()
    if os.path.exists(text):
        with open(text, encoding="utf-8") as fh:
            text = fh.read()
    out = set()
    for tok in re.split(r"[,\s]+", text.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.update(range(min(a, b), max(a, b) + 1))
        elif tok.isdigit():
            out.add(int(tok))
        else:
            sys.exit(f"cannot read '{tok}' as an index or range")
    return out


def parse_sheet_scores(path, n_tiles):
    """Read a per-sheet score sheet and return (animal, no_animal, unsure).

        # 0 = no animal, 1 = animal, 2 = unclear
        1: 2,1,0,0,0,2,1,1,1,1,1,1,0,1,1,1,1,1,1,1
        2: ...

    This is a better record than a list of indices: you write one row per
    sheet in the order the tiles are printed, so you never have to read a
    number off a tile at all.

    It is also more dangerous, and that is why the checking below is heavy. A
    single dropped value in one row shifts every tile after it by one
    position, and the result is a complete, plausible-looking, entirely wrong
    set of rates. So every tile must be scored exactly once: the indices
    reconstructed here have to cover 1..n_tiles with nothing missing and
    nothing doubled, or this refuses to score at all.
    """
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.split("#")[0].strip()
            if not line:
                continue
            if ":" not in line:
                sys.exit(f"{path}:{ln}: expected 'sheet: v,v,v,...'")
            s, vals = line.split(":", 1)
            if not s.strip().isdigit():
                sys.exit(f"{path}:{ln}: '{s.strip()}' is not a sheet number")
            s = int(s)
            if s in rows:
                sys.exit(f"{path}:{ln}: sheet {s} scored twice")
            out = []
            for tok in re.split(r"[,\s]+", vals.strip()):
                if tok == "":
                    continue
                if tok not in ("0", "1", "2"):
                    sys.exit(f"{path}:{ln}: '{tok}' is not 0, 1 or 2")
                out.append(int(tok))
            if not out:
                sys.exit(f"{path}:{ln}: sheet {s} has no scores")
            rows[s] = out

    if not rows:
        sys.exit(f"{path}: no scores found")
    gap = sorted(set(range(1, max(rows) + 1)) - set(rows))
    if gap:
        sys.exit(f"{path}: sheets missing entirely: {gap[:10]}")

    per_sheet = len(rows[1])
    lengths = {len(v) for s, v in rows.items() if s != max(rows)}
    if lengths != {per_sheet}:
        odd = [s for s, v in rows.items()
               if s != max(rows) and len(v) != per_sheet]
        sys.exit(f"{path}: sheet 1 has {per_sheet} scores but sheet(s) "
                 f"{odd[:10]} differ. Every sheet but the last is full, so "
                 f"this is a dropped or doubled value, and it would shift "
                 f"every tile after it.")

    animal, no_animal, unsure = set(), set(), set()
    for s in sorted(rows):
        base = (s - 1) * per_sheet
        for p, v in enumerate(rows[s], start=1):
            i = base + p
            (animal if v == 1 else no_animal if v == 0 else unsure).add(i)

    scored = animal | no_animal | unsure
    if len(scored) != sum(len(v) for v in rows.values()):
        sys.exit(f"{path}: indices collided -- check the sheet numbering")
    missing = sorted(set(range(1, n_tiles + 1)) - scored)
    extra = sorted(scored - set(range(1, n_tiles + 1)))
    if missing or extra:
        sys.exit(
            f"{path}: reconstructed {len(scored):,} scored tiles against "
            f"{n_tiles:,} in the manifest.\n"
            f"  unscored tiles: {len(missing):,} {missing[:8]}\n"
            f"  scores with no tile: {len(extra):,} {extra[:8]}\n"
            f"  {per_sheet} tiles per sheet was taken from sheet 1. If the "
            f"sheets were built with a different --cols/--rows, rebuild them "
            f"or fix the rows; do not score a partial set.")
    print(f"{path}: {len(rows)} sheets x {per_sheet} tiles, all "
          f"{n_tiles:,} tiles scored exactly once")
    return animal, no_animal, unsure


def cmd_score(args):
    path = os.path.join(args.out, MANIFEST)
    if not os.path.exists(path):
        sys.exit(f"{path} not found -- run `sample` first")
    with open(path, encoding="utf-8") as fh:
        man = {int(r["index"]): r for r in csv.DictReader(fh)}

    if args.scores:
        # A bare filename means the one in the review directory. The scores
        # belong to that review, and typing the full path twice on one command
        # line is how you end up scoring one review's numbers against
        # another's manifest.
        sp = args.scores
        if not os.path.exists(sp) and not os.path.isabs(sp):
            alt = os.path.join(args.out, sp)
            if os.path.exists(alt):
                sp = alt
        if not os.path.exists(sp):
            sys.exit(f"no score file at '{args.scores}' or "
                     f"'{os.path.join(args.out, args.scores)}'")
        animal, no_animal, unsure = parse_sheet_scores(sp, len(man))
    else:
        no_animal = parse_indices(args.no_animal)
        unsure = parse_indices(args.unsure)
        animal = set(man) - no_animal - unsure
    stray = sorted((no_animal | unsure) - set(man))
    if stray:
        sys.exit(f"indices not in the manifest: {stray[:10]} -- a "
                 f"transcription slip here silently moves the rate, so fix it "
                 f"rather than let it through")
    overlap = no_animal & unsure
    if overlap:
        sys.exit(f"{len(overlap)} index(es) marked both no-animal and unsure: "
                 f"{sorted(overlap)[:10]}")

    tally = defaultdict(Counter)
    for i, r in man.items():
        k = (r["kind"], r["source"])
        tally[k]["n"] += 1
        if i in unsure:
            tally[k]["unsure"] += 1
        elif i in no_animal:
            tally[k]["no_animal"] += 1
        elif i in animal:
            tally[k]["animal"] += 1
        else:
            sys.exit(f"tile {i} carries no score at all -- refusing to "
                     f"guess it")

    print(f"\nreviewed {len(man):,} tiles; {len(animal):,} animal, "
          f"{len(no_animal):,} no animal, {len(unsure):,} unclear "
          f"({len(unsure) / max(1, len(man)):.1%})\n")
    print(f"{'kind':<11} {'source':<12} {'n':>6} {'animal':>7} "
          f"{'none':>6} {'unsure':>7}  {'rate':>8}")
    print("-" * 66)
    for (kind, src), c in sorted(tally.items()):
        denom = c["n"] - c["unsure"]
        if kind == "annotation":
            rate = c["no_animal"] / denom if denom else 0.0
            label = f"{rate:7.2%}"          # crops with no animal = failures
        else:
            rate = c["animal"] / denom if denom else 0.0
            label = f"{rate:7.2%}"          # backgrounds with an animal
        print(f"{kind:<11} {src:<12} {c['n']:>6,} {c['animal']:>7,} "
              f"{c['no_animal']:>6,} {c['unsure']:>7,}  {label}")
    print("\nrate = the failure rate for that row, unsure tiles excluded from")
    print("the denominator: for annotations, crops holding no animal; for")
    print("backgrounds, candidates that do hold one.")

    # ---- the interval, which is what actually decides anything -----------
    print(f"\n{'kind':<11} {'source':<12} {'floor':>8} {'ceiling':>9}   "
          f"unclear")
    print("-" * 66)
    for (kind, src), c in sorted(tally.items()):
        bad = c["no_animal"] if kind == "annotation" else c["animal"]
        lo = bad / c["n"] if c["n"] else 0.0
        hi = (bad + c["unsure"]) / c["n"] if c["n"] else 0.0
        print(f"{kind:<11} {src:<12} {lo:7.1%} {hi:8.1%}   "
              f"{c['unsure']:>4,}/{c['n']:,}")
    print("\nfloor   = every unclear tile is fine")
    print("ceiling = every unclear tile is a failure")
    print("The truth is between them. Where the two straddle the threshold")
    print("you set, this review cannot decide -- and no amount of staring at")
    print("the middle column changes that. Narrow the interval or change the")
    print("question; do not quietly adopt the floor because it is the nicer")
    print("number.")

    ann = [c for (k, _s), c in tally.items() if k == "annotation"]
    n_ann = sum(c["n"] - c["unsure"] for c in ann)
    bad_ann = sum(c["no_animal"] for c in ann)
    if n_ann and bad_ann / n_ann > 0.05:
        print(f"\n! {bad_ann}/{n_ann} annotation crops hold no animal. Above a "
              f"few percent this is a geometry fault, not label noise -- look "
              f"at whether the failures share a source before rebuilding "
              f"anything.")

    worst = sorted(((c["no_animal"] / max(1, c["n"] - c["unsure"]), s)
                    for (k, s), c in tally.items() if k == "annotation"),
                   reverse=True)
    if len(worst) > 1 and worst[0][0] > 3 * max(0.001, worst[1][0]):
        print(f"\n! {worst[0][1]} fails {worst[0][0]:.1%} against "
              f"{worst[1][0]:.1%} for the next worst. A per-source failure "
              f"rate is what a misread bbox convention looks like.")

    print("\nfailing annotation tiles, for a second look:")
    for i in sorted(no_animal):
        r = man[i]
        if r["kind"] == "annotation":
            print(f"  {i:>5}  {r['source']:<12} {r['category_name']} "
                  f"({r['detail']})  {r['key']}")
    print("\nbackground tiles holding an animal:")
    for i, r in sorted(man.items()):
        if r["kind"] == "background" and i not in no_animal and i not in unsure:
            print(f"  {i:>5}  {r['source']:<12} box {r['detail']}  {r['key']}")
    return 0


# ------------------------------------------------------------- breakdown

SIZE_BINS = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 10 ** 9)]


def tile_size(row):
    """-> (w, h) of the pixels the tile actually holds.

    Annotations record 'WxH'; backgrounds record the box as 'x,y,w,h'.
    """
    d = row.get("detail") or ""
    try:
        if "x" in d and "," not in d:
            w, h = d.split("x")
            return int(w), int(h)
        parts = [int(v) for v in d.split(",")]
        if len(parts) == 4:
            return parts[2], parts[3]
    except ValueError:
        pass
    return None


def bin_of(short_side):
    for lo, hi in SIZE_BINS:
        if lo <= short_side < hi:
            return f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
    return "?"


def cmd_breakdown(args):
    """Split the scores by crop size, to separate two things the headline
    rates confound.

    An 'unclear' can mean the crop carries no answer -- 17x15 px of dark
    seabed -- or it can mean the CONTACT SHEET threw the answer away, because
    a 1209x986 background box shown in a 300 px cell is a four-fold
    downsample and a small animal in it stops existing. The first is a fact
    about the dataset. The second is a fault in how I built the sheets, and
    is fixable by re-showing those tiles larger rather than by more review.

    The same split tells you whether background contamination scales with box
    area, which it should: a bigger box in a sparsely annotated frame has more
    chances to catch something nobody labelled.
    """
    path = os.path.join(args.out, MANIFEST)
    with open(path, encoding="utf-8") as fh:
        man = {int(r["index"]): r for r in csv.DictReader(fh)}
    sp = args.scores
    if not os.path.exists(sp) and not os.path.isabs(sp):
        sp = os.path.join(args.out, args.scores)
    animal, no_animal, unsure = parse_sheet_scores(sp, len(man))

    cell = args.tile
    grid = defaultdict(Counter)
    for i, r in man.items():
        wh = tile_size(r)
        if wh is None:
            continue
        key = (r["kind"], r["source"], bin_of(min(wh)))
        grid[key]["n"] += 1
        grid[key]["animal" if i in animal else
                  "no_animal" if i in no_animal else "unsure"] += 1
        if max(wh) > cell:
            grid[key]["downsampled"] += 1
            if i in unsure:
                grid[key]["unsure_downsampled"] += 1

    print(f"\nscores by crop short side (sheet cell {cell} px)")
    print(f"{'kind':<11} {'source':<11} {'short side':>11} {'n':>5} "
          f"{'animal':>7} {'none':>5} {'unclear':>8} {'uncl%':>7} "
          f"{'shrunk':>7}")
    print("-" * 82)
    order = {f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": k
             for k, (lo, hi) in enumerate(SIZE_BINS)}
    for kind, src, b in sorted(grid, key=lambda t: (t[0], t[1],
                                                    order.get(t[2], 99))):
        c = grid[(kind, src, b)]
        print(f"{kind:<11} {src:<11} {b:>11} {c['n']:>5,} {c['animal']:>7,} "
              f"{c['no_animal']:>5,} {c['unsure']:>8,} "
              f"{c['unsure'] / c['n']:>6.0%} {c['downsampled']:>7,}")

    shrunk = sum(c["downsampled"] for c in grid.values())
    shrunk_unclear = sum(c["unsure_downsampled"] for c in grid.values())
    all_unclear = sum(c["unsure"] for c in grid.values())
    print(f"\n{shrunk:,} tiles were shown smaller than their own pixels; "
          f"{shrunk_unclear:,} of those were called unclear.")
    if all_unclear and shrunk_unclear:
        print(f"That is {shrunk_unclear / all_unclear:.0%} of all unclear "
              f"tiles. Those are the ones worth showing again at full size -- "
              f"re-reviewing a 17 px crop would only produce the same answer "
              f"more slowly.")
    elif all_unclear:
        print("No unclear tile lost resolution to the sheet, so the sheets "
              "are not the problem: these crops genuinely do not carry the "
              "answer, and re-reviewing them will not change that.")
    return 0


# --------------------------------------------------------------- taxonomy

# WoRMS has no subphylum in our cached lineage (LINEAGE_RANKS stops at
# Phylum), so "vertebrate" is decided on CLASS. Listing the vertebrate
# classes explicitly, rather than treating all Chordata as vertebrates, is
# not pedantry: Bathochordaeus is a larvacean -- phylum Chordata, and about
# as far from a fish as an animal gets. Calling it a vertebrate would put one
# of the hardest tiles in the review on the easy side of the split and bias
# the very comparison being made.
VERTEBRATE_CLASSES = {
    "actinopteri", "teleostei", "actinopterygii", "elasmobranchii",
    "holocephali", "myxini", "petromyzonti", "coelacanthi", "dipneusti",
    "cladistia", "chondrostei", "chondrichthyes", "sarcopterygii",
    "mammalia", "aves", "reptilia", "amphibia",
}

VERT, INVERT, NONANIMAL, UNKNOWN = ("vertebrate", "invertebrate",
                                    "not an animal", "unknown")


def lineage_of(cache, aphia_id):
    """The cache holds two shapes: lineage entries under the bare AphiaID and
    slim records under 'rec:<id>'. Only the first carries a lineage, and
    different vintages of the file nest it differently, so accept both."""
    e = cache.get(str(aphia_id))
    if not isinstance(e, dict):
        return None
    if isinstance(e.get("lineage"), dict):
        return e["lineage"]
    if "phylum" in e or "kingdom" in e:
        return e
    return None


def taxon_bucket(cache, aphia_id):
    lin = lineage_of(cache, aphia_id)
    if not lin:
        return UNKNOWN
    kingdom = str(lin.get("kingdom") or "").strip().lower()
    phylum = str(lin.get("phylum") or "").strip().lower()
    cls = str(lin.get("class") or "").strip().lower()
    if kingdom and kingdom != "animalia":
        return NONANIMAL
    if cls in VERTEBRATE_CLASSES:
        return VERT
    if phylum == "chordata" and not cls:
        return UNKNOWN          # could be either; do not guess
    if kingdom == "animalia":
        return INVERT
    return UNKNOWN


def _rate_row(label, c, width=34):
    n = c["n"]
    called = n - c["unsure"]
    print(f"{label:<{width}} {n:>5,} {c['animal']:>7,} {c['no_animal']:>5,} "
          f"{c['unsure']:>8,} {c['unsure'] / n:>7.0%}"
          + (f" {c['no_animal'] / called:>7.1%}" if called else f" {'--':>7}"))


def cmd_taxon(args):
    """Is FathomNet harder because of the imagery, or because of the animals?

    Both stories predict FathomNet's high unclear rate. They differ on what
    happens when you hold the animal constant: if it is the imagery, FathomNet
    fish should still be harder than yolo-bruv fish; if it is the animals,
    the gap should mostly close.

    Read the result as a hypothesis worth testing, not a test. The hypothesis
    was formed after seeing these scores and is being checked on the same
    scores, so it cannot fail in the way a real test can.
    """
    with open(os.path.join(args.out, MANIFEST), encoding="utf-8") as fh:
        man = {int(r["index"]): r for r in csv.DictReader(fh)}
    sp = args.scores
    if not os.path.exists(sp) and not os.path.isabs(sp):
        sp = os.path.join(args.out, args.scores)
    animal, no_animal, unsure = parse_sheet_scores(sp, len(man))
    with open(args.lineage_cache, encoding="utf-8") as fh:
        cache = json.load(fh)

    buckets, by_size, examples = (defaultdict(Counter), defaultdict(Counter),
                                  defaultdict(list))
    for i, r in man.items():
        if r["kind"] != "annotation":
            continue
        b = taxon_bucket(cache, r["category_id"])
        score = ("animal" if i in animal else
                 "no_animal" if i in no_animal else "unsure")
        buckets[(r["source"], b)]["n"] += 1
        buckets[(r["source"], b)][score] += 1
        wh = tile_size(r)
        if wh:
            by_size[(r["source"], b, bin_of(min(wh)))]["n"] += 1
            by_size[(r["source"], b, bin_of(min(wh)))][score] += 1
        if b in (UNKNOWN, NONANIMAL):
            lab = f"{r['category_name']} ({r['category_id']})"
            if lab not in examples[b] and len(examples[b]) < 12:
                examples[b].append(lab)

    hdr = (f"{'n':>5} {'animal':>7} {'none':>5} {'unclear':>8} {'uncl%':>7} "
           f"{'fail%':>7}")
    print(f"\nannotation tiles by taxon group\n{'':<34}{hdr}")
    print("-" * 76)
    for src, b in sorted(buckets):
        _rate_row(f"{src}  {b}", buckets[(src, b)])

    print(f"\nVERTEBRATES ONLY -- the like-for-like comparison\n{'':<34}{hdr}")
    print("-" * 76)
    order = {f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+": k
             for k, (lo, hi) in enumerate(SIZE_BINS)}
    for src, b, sz in sorted(by_size, key=lambda t: (t[0], order.get(t[2], 9))):
        if b != VERT:
            continue
        _rate_row(f"{src}  vertebrate  {sz} px", by_size[(src, b, sz)])

    thin = [f"{s}/{b}" for (s, b), c in buckets.items()
            if b == VERT and c["n"] < 30]
    if thin:
        print(f"\n! fewer than 30 vertebrate tiles in: {', '.join(thin)}. "
              f"A rate on that many tiles moves by whole percentage points "
              f"per tile; treat the size split as indicative only.")
    for b in (NONANIMAL, UNKNOWN):
        if examples[b]:
            print(f"\nexamples classed '{b}': {', '.join(examples[b][:8])}")
    if buckets and all(b == UNKNOWN for _s, b in buckets):
        print("\n! every taxon came back unknown -- the lineage cache "
              "probably has no entries for these AphiaIDs, or is the wrong "
              "file. Nothing below this line means anything.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--per-source", default="200",
                   help="annotation crops: a flat number, or "
                        "src=n,src=n to weight")
    s.add_argument("--background", default="200",
                   help="background candidates: a flat number applied to "
                        "every frame-carrying source, or src=n,src=n to "
                        "weight. 0 to skip")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)

    s = sub.add_parser("sheets")
    s.add_argument("--out", required=True)
    s.add_argument("--tile", type=int, default=300)
    s.add_argument("--cols", type=int, default=5)
    s.add_argument("--rows", type=int, default=4)
    s.set_defaults(fn=cmd_sheets)

    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--scores", default="",
                   help="a per-sheet score file: '1: 0,1,2,1,...' per line, "
                        "0=no animal 1=animal 2=unclear. Preferred over "
                        "--no-animal: every tile is scored explicitly, so a "
                        "dropped value is caught instead of silently read as "
                        "'animal'")
    s.add_argument("--no-animal", default="",
                   help="indices, ranges, or a path to a file of them")
    s.add_argument("--unsure", default="")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("breakdown")
    s.add_argument("--out", required=True)
    s.add_argument("--scores", required=True)
    s.add_argument("--tile", type=int, default=300,
                   help="the --tile the sheets were built with")
    s.set_defaults(fn=cmd_breakdown)

    s = sub.add_parser("taxon")
    s.add_argument("--out", required=True)
    s.add_argument("--scores", required=True)
    s.add_argument("--lineage-cache", required=True,
                   help="sw/taxon_maps/worms_lineage_cache.json -- read only, "
                        "and no WoRMS calls are made")
    s.set_defaults(fn=cmd_taxon)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()

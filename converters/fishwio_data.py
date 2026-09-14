#!/usr/bin/env python3
"""
FishWIO -> SeaVision converter.

FishWIO (Zenodo 10.5281/zenodo.17297730) is a pre-cropped fish image library
from fixed GoPro cameras around Mayotte, western Indian Ocean.

Structure:
    data_paper_dataset/<label>/<crop>.jpeg
    metadata.html   per-video table (gt/HTML, not CSV)
    species.html    per-label taxonomy table

Source-specific facts this converter has to know:

  * TWO filename conventions, both ending in ImageMagick geometry:
        <video>_<frame>_<W>x<H>+<X>+<Y>.jpeg
        <video>_<frame>_<Genus_species>_<W>x<H>+<X>+<Y>.jpeg
    Four folders use the second exclusively. Parsing anchors on the geometry
    suffix and works backwards, so both are handled by one path.

  * Offsets may be NEGATIVE (a box overhanging the top or left frame edge).

  * macOS junk: '.DS_Store' and AppleDouble '._*' shadow files. The latter
    parse as plausible filenames and will silently inflate every count if not
    excluded.

  * Labels carry a life-stage or sex suffix with INCONSISTENT spelling
    (_juv and _juvenile both occur), alongside _adu, _male, _female. These are
    life-stage attributes of one species, not separate taxa: the suffix is
    stripped for taxonomy and preserved in source_meta.

  * Eight background classes, prefixed '_'. Ingested as images with ZERO
    annotations - the same treatment as PrePARED's empty frames - with the
    specific class kept in source_meta. Background has no AphiaID, and
    inventing a pseudo-taxon to hold it would corrupt the category space.

  * Original frames are NOT distributed, so crop_provenance is 'pre_cropped'.
    But the filename geometry means the crop's size and position in its source
    frame ARE known, so pixel_scale_known is true and the frame coordinates are
    preserved in source_meta.

  * track_id and individual_id are left NULL deliberately. Consecutive frames
    of one video with overlapping boxes are plainly the same individual, but
    that is an INFERENCE with tunable parameters. The ingredients (video, frame,
    box) are all in source_meta, so tracks can be derived at analysis time and
    re-derived with different thresholds without re-ingesting. Freezing one
    guess into the master file would be a claim we have not validated.
    `validate` reports track statistics so the scale of the effect is visible.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import statistics
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                       # noqa: E402

EXTRA_REQUIRED = ["image_root_rel", "metadata_html", "species_html",
                  "variant_suffixes", "background_prefix"]

GEOM = re.compile(r"^(?P<rest>.+)_(?P<w>\d+)x(?P<h>\d+)"
                  r"\+(?P<x>-?\d+)\+(?P<y>-?\d+)$")
FRAME = re.compile(r"^(?P<video>.+)_(?P<frame>\d+)$")


# --------------------------------------------------------------------------
# Source-specific parsing
# --------------------------------------------------------------------------

def is_junk(fn):
    """macOS litter. AppleDouble files parse as plausible names and will
    inflate every count if they are not excluded here."""
    return fn == ".DS_Store" or fn.startswith("._")


def parse_crop(fn, label, variants):
    stem, ext = os.path.splitext(fn)
    if ext.lower() not in (".jpg", ".jpeg"):
        return None
    m = GEOM.match(stem)
    if not m:
        return None
    rest = m["rest"]
    binom_label = strip_variant(label, variants)[0].replace(" ", "_")
    for cand in (label, binom_label):          # folder label, then bare binomial
        if rest.endswith("_" + cand):
            rest = rest[: -(len(cand) + 1)]
            break
    f = FRAME.match(rest)
    if not f:
        return None
    return {"video": f["video"], "frame": int(f["frame"]),
            "w": int(m["w"]), "h": int(m["h"]),
            "x": int(m["x"]), "y": int(m["y"])}


def strip_variant(label, suffixes):
    """'Chaetodon_trifascialis_juv' -> ('Chaetodon trifascialis', 'juv')."""
    for s in suffixes:
        if label.endswith("_" + s):
            return label[: -(len(s) + 1)].replace("_", " "), s
    return label.replace("_", " "), None


def coord_to_decimal(value):
    """'E044.9700' -> 44.97 ; 'S12.9100' -> -12.91."""
    v = C.norm(value)
    if not v:
        return None
    hemi, num = v[0].upper(), v[1:]
    try:
        d = float(num)
    except ValueError:
        return None
    return -d if hemi in ("S", "W") else d


def load_metadata(path):
    """video_name -> row dict, from the gt HTML table."""
    tables = pd.read_html(path)
    md = tables[0]
    need = {"video_name", "site_name", "shooting_date", "longitude",
            "latitude", "depth_shooting", "shooting_id"}
    missing = need - set(md.columns)
    if missing:
        sys.exit(f"{path}: metadata table is missing columns {sorted(missing)}")
    return {str(r["video_name"]): r for _, r in md.iterrows()}


def load_species(path):
    """'Genus species' -> row dict, from the gt HTML table."""
    md = pd.read_html(path)[0]
    need = {"Order", "Family", "Genus", "Species"}
    missing = need - set(md.columns)
    if missing:
        sys.exit(f"{path}: species table is missing columns {sorted(missing)}")
    return {C.norm(r["Species"]): r for _, r in md.iterrows()}


def walk(cfg):
    root = os.path.join(cfg["source_root"], cfg["image_root_rel"])
    variants = cfg["variant_suffixes"]
    for label in sorted(os.listdir(root)):
        d = os.path.join(root, label)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if is_junk(fn):
                continue
            if not fn.lower().endswith((".jpg", ".jpeg")):
                continue
            yield label, fn, os.path.join(d, fn), parse_crop(fn, label, variants)

# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def cmd_validate(args, cfg):
    from PIL import Image

    bg_prefix = cfg["background_prefix"]
    variants = cfg["variant_suffixes"]
    meta = load_metadata(os.path.join(cfg["source_root"], cfg["metadata_html"]))
    spec = load_species(os.path.join(cfg["source_root"], cfg["species_html"]))

    rows, bad = [], []
    for label, fn, path, p in walk(cfg):
        (rows if p else bad).append((label, fn, path, p))

    print(f"crops parsed {len(rows)}   unparsed {len(bad)}")
    for b in bad[:5]:
        print(f"   unparsed: {b[0]}/{b[1]}")

    labels = collections.Counter(r[0] for r in rows)
    bg = {l for l in labels if l.startswith(bg_prefix)}
    fish = {l for l in labels if l not in bg}
    print(f"labels {len(labels)}  fish {len(fish)}  background {len(bg)}")

    # -- taxonomy join ----------------------------------------------------
    sp_counts = collections.Counter()
    unmatched = []
    for l in fish:
        binom, life = strip_variant(l, variants)
        if binom in spec:
            sp_counts[binom] += labels[l]
        else:
            unmatched.append((l, binom))
    print(f"species matched to species.html {len(sp_counts)}  "
          f"labels unmatched {len(unmatched)}")
    for u in unmatched[:10]:
        print(f"   {u[0]!r} not in species.html -> taxonomy from WoRMS via {u[1]!r}")
    unused = set(spec) - set(sp_counts)
    if unused:
        print(f"species.html rows with no images: {sorted(unused)[:10]}")

    genera = collections.Counter(b.split(" ")[0] for b in sp_counts)
    multi = {g: n for g, n in genera.items() if n >= 2}
    print(f"genera {len(genera)}  with >=2 species {len(multi)}  "
          f"species in those genera {sum(multi.values())}")
    print(f"species with <{args.min_crops} crops: "
          f"{sorted([(k, v) for k, v in sp_counts.items() if v < args.min_crops])}")

    # -- video join -------------------------------------------------------
    vids = {r[3]["video"] for r in rows}
    print(f"videos in filenames {len(vids)}  in metadata {len(meta)}  "
          f"matched {len(vids & set(meta))}  unmatched {len(vids - set(meta))}")
    for v in sorted(vids - set(meta))[:5]:
        print(f"   no metadata row for video {v!r}")

    # -- sizes: fish vs background ---------------------------------------
    def quant(vals):
        vals = sorted(vals)
        q = lambda p: vals[int(p * len(vals))] if vals else 0
        return (f"n={len(vals)} min={vals[0]} q1={q(.25)} med={q(.5)} "
                f"q3={q(.75)} max={vals[-1]}  <64px={100*sum(1 for s in vals if s<64)/len(vals):.1f}%")

    fish_short = [min(r[3]["w"], r[3]["h"]) for r in rows if r[0] not in bg]
    bg_short = [min(r[3]["w"], r[3]["h"]) for r in rows if r[0] in bg]
    print(f"fish  short side: {quant(fish_short)}")
    if bg_short:
        print(f"bg    short side: {quant(bg_short)}")
        print("  NOTE: background crops must be size-matched to fish crops, or "
              "the class separates on scale rather than content (plan 6.1).")
    print(f"crops with short side < {args.min_px}px: "
          f"{sum(1 for s in fish_short if s < args.min_px)}")

    # -- filename geometry vs actual image dimensions (sample) ------------
    import random
    sample = random.Random(0).sample(rows, min(args.sample, len(rows)))
    mismatch = 0
    for label, fn, path, p in sample:
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:                                # noqa: BLE001
            mismatch += 1
            continue
        if (w, h) != (p["w"], p["h"]):
            mismatch += 1
    print(f"geometry check on {len(sample)} sampled crops: {mismatch} mismatched")

    # -- tracks -----------------------------------------------------------
    by = collections.defaultdict(list)
    for label, fn, path, p in rows:
        if label in bg:
            continue
        by[(label, p["video"])].append((p["frame"], p["x"], p["y"], p["w"], p["h"]))
    lens, cur = [], 0
    for v in by.values():
        v.sort()
        cur = 0
        for i, (fr, x, y, w, h) in enumerate(v):
            prev = v[i - 1] if i else None
            if prev and fr - prev[0] <= args.track_gap and \
               _iou((x, y, w, h), (prev[1], prev[2], prev[3], prev[4])) > args.track_iou:
                cur += 1
            else:
                if cur:
                    lens.append(cur)
                cur = 1
        if cur:
            lens.append(cur)
    if lens:
        m = statistics.mean(lens)
        print(f"tracks {len(lens)}  mean crops/track {m:.2f}  "
              f"median {statistics.median(lens)}  max {max(lens)}")
        print(f"  design effect at rho=0.5: {1 + (m - 1) * 0.5:.2f}  "
              f"-> {len(fish_short)} crops ~ "
              f"{len(fish_short) / (1 + (m - 1) * 0.5):.0f} effective")


# --------------------------------------------------------------------------
# taxon-map
# --------------------------------------------------------------------------

def cmd_taxon_map(args, cfg):
    out = cfg["taxon_map_csv"]
    if os.path.exists(out):
        sys.exit(f"{out} already exists. It holds manual decisions - refusing "
                 f"to overwrite. Move it aside deliberately to rebuild it.")

    variants = cfg["variant_suffixes"]
    spec = load_species(os.path.join(cfg["source_root"], cfg["species_html"]))
    bg_prefix = cfg["background_prefix"]

    counts = collections.Counter()
    for label, fn, path, p in walk(cfg):
        if p and not label.startswith(bg_prefix):
            counts[label] += 1

    seen, rows = set(), []
    for label in sorted(counts):
        binom, life = strip_variant(label, variants)
        s = spec.get(binom)
        family = C.norm(s["Family"]) if s is not None else ""
        genus = C.norm(s["Genus"]) if s is not None else binom.split(" ")[0]
        gbif = C.norm(s["GBIF taxon ID"]) if s is not None and \
            "GBIF taxon ID" in s else ""
        key = (family, genus, binom)
        if key in seen:
            continue
        seen.add(key)

        aphia = valid = rank = ""
        status = "UNRESOLVED"
        hit = C.worms_by_name(binom)
        time.sleep(0.3)
        if hit:
            aphia, valid, rank = hit
            status = "AUTO"
        rows.append({"family": family, "genus": genus, "species": binom,
                     "n_rows": sum(v for k, v in counts.items()
                                   if strip_variant(k, variants)[0] == binom),
                     "proposed_name": binom, "aphia_id": aphia,
                     "valid_name": valid, "rank": rank, "status": status,
                     "gbif_taxon_id": gbif})

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_todo = sum(r["status"] == "UNRESOLVED" for r in rows)
    print(f"Wrote {out}: {len(rows)} taxa "
          f"({len(rows) - n_todo} auto-resolved, {n_todo} need an AphiaID).")


# --------------------------------------------------------------------------
# coco
# --------------------------------------------------------------------------

def cmd_coco(args, cfg):
    from PIL import Image

    variants = cfg["variant_suffixes"]
    bg_prefix = cfg["background_prefix"]
    meta = load_metadata(os.path.join(cfg["source_root"], cfg["metadata_html"]))
    spec = load_species(os.path.join(cfg["source_root"], cfg["species_html"]))
    taxon_map = C.load_taxon_map(cfg["taxon_map_csv"])
    ds_id = cfg["dataset_meta"]["id"]

    coco, st = C.load_or_init_coco(cfg["output_json"])
    C.register_source(coco, cfg, st)
    C.ensure_categories(coco, taxon_map, cfg.get("lineage_cache"))

    os.makedirs(cfg["output_image_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output_json"]) or ".", exist_ok=True)

    review, n_bg, n_fish = [], 0, 0
    for label, fn, path, p in walk(cfg):
        if p is None:
            review.append((f"{label}/{fn}", path, "filename matched neither convention"))
            continue

        uid = C.uid_for(path, cfg["source_root"], cfg["uid_prefix"])
        if uid in st["seen_uid"]:
            continue          # already ingested; not a review item

        is_bg = label.startswith(bg_prefix)
        aid = None
        binom, life = strip_variant(label, variants)
        if not is_bg:
            s = spec.get(binom)
            family = C.norm(s["Family"]) if s is not None else ""
            genus = C.norm(s["Genus"]) if s is not None else binom.split(" ")[0]
            aid = taxon_map.get((family, genus, binom))
            if not aid:
                review.append((f"{label}/{fn}", path,
                               f"unresolved taxon (no AphiaID): {binom}"))
                continue

        try:
            if args.trust_geometry:
                width, height = p["w"], p["h"]
            else:
                with Image.open(path) as im:
                    width, height = im.size
        except Exception as exc:                         # noqa: BLE001
            review.append((f"{label}/{fn}", path, f"unreadable image: {exc}"))
            continue

        md = meta.get(p["video"])
        if md is None:
            review.append((f"{label}/{fn}", path,
                           f"no metadata row for video {p['video']}"))

        groups, gsrc = {}, {}
        if md is not None:
            site, shoot = C.norm(md["site_name"]), C.norm(md["shooting_id"])
            if site:
                groups["site"], gsrc["site"] = site, "site_name"
            if shoot:
                groups["deployment"], gsrc["deployment"] = shoot, "shooting_id"

        date = C.norm(md["shooting_date"]) if md is not None else ""
        iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 else None
        depth = float(md["depth_shooting"]) if md is not None and \
            C.norm(md["depth_shooting"]) else None

        dest = os.path.join(cfg["output_image_dir"], f"{uid}.jpg")
        if not os.path.exists(dest):
            C.copy_file(path, dest)

        image_id = st["next_img"]
        st["next_img"] += 1
        st["seen_uid"].add(uid)

        source_meta = {
            "label": label,
            "video_name": p["video"],
            "frame": p["frame"],
            "frame_bbox": [p["x"], p["y"], p["w"], p["h"]],
            "licence_note": "Zenodo record states CC BY 4.0; the preprint "
                            "states CC BY-NC 4.0. Zenodo governs, as the "
                            "channel the data was obtained through.",
        }
        if life:
            source_meta["lifestage"] = life
        if is_bg:
            source_meta["background_class"] = label.lstrip(bg_prefix)
            n_bg += 1
        else:
            n_fish += 1
            if md is not None and "GBIF taxon ID" in spec.get(binom, {}):
                source_meta["gbif_taxon_id"] = C.norm(spec[binom]["GBIF taxon ID"])

        coco["images"].append({
            "id": image_id,
            "file_name": f"{uid}.jpg",
            "width": width, "height": height,
            "dataset_id": ds_id,
            "crop_provenance": cfg["crop_provenance"],
            "pixel_scale_known": cfg["pixel_scale_known"],
            "gear": cfg["gear"],
            "groups": groups, "group_sources": gsrc,
            "has_unlabelled_animal": False,
            "lat": coord_to_decimal(md["latitude"]) if md is not None else None,
            "lon": coord_to_decimal(md["longitude"]) if md is not None else None,
            "datetime": iso,
            "depth_m": depth,
            "source_meta": source_meta,
        })

        if not is_bg:
            coco["annotations"].append({
                "id": st["next_ann"], "image_id": image_id, "category_id": aid,
                "bbox": [0.0, 0.0, float(width), float(height)],
                "area": float(width * height), "iscrowd": 0,
                "individual_id": None, "track_id": None,
            })
            st["next_ann"] += 1

    import json
    with open(cfg["output_json"], "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)
    C.write_review(cfg["review_csv"], review)
    C.write_manifest(cfg, coco, len(review),
                     extra={"fishwio_fish": n_fish, "fishwio_background": n_bg})

    print(f"Wrote {cfg['output_json']}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories.")
    print(f"  this source: {n_fish} fish crops, {n_bg} background crops")
    print(f"Review items: {len(review)}"
          + (f" -> {cfg['review_csv']}" if review else ""))


def main():
    ap = argparse.ArgumentParser(description="FishWIO -> SeaVision converter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="survey the archive (offline)")
    v.add_argument("--min-crops", type=int, default=100)
    v.add_argument("--min-px", type=int, default=16)
    v.add_argument("--sample", type=int, default=300)
    v.add_argument("--track-gap", type=int, default=5)
    v.add_argument("--track-iou", type=float, default=0.2)
    v.set_defaults(func=cmd_validate)

    t = sub.add_parser("taxon-map", help="write the taxon map (needs WoRMS)")
    t.set_defaults(func=cmd_taxon_map)

    c = sub.add_parser("coco", help="build/merge the COCO JSON")
    c.add_argument("--trust-geometry", action="store_true",
                   help="take width/height from the filename instead of opening "
                        "each image - only after validate reports 0 mismatches")
    c.set_defaults(func=cmd_coco)

    for p in (v, t, c):
        p.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = C.load_config(args.config, EXTRA_REQUIRED)
    args.func(args, cfg)


if __name__ == "__main__":
    main()

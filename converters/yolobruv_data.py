#!/usr/bin/env python3
"""
YOLO-BRUVS (PrePARED) -> SeaVision converter.

EventMeasure ``.txt`` point exports plus their frames, from a UK BRUV survey.
Shared machinery lives in common.py; this file holds only what is specific to
this source.

    validate    Parse the exports and print a tally. Offline.
    taxon-map   List every distinct (Family, Genus, Species) triple and
                auto-resolve unambiguous binomials against WoRMS.
    coco        Build/merge the COCO JSON using the reviewed taxon map.

Source-specific decisions, fixed during design review:
  * An image marked ``Styelidae / Botryllus / spp`` is EMPTY: zero annotations
    regardless of any box drawn on it. `spp` is used for nothing else here.
  * Bounding box = [ImageCol, ImageRow, RectWidth, RectHeight], absolute pixels,
    top-left origin. No normalisation.
  * A zero-area box is degenerate: dropped and logged. If that leaves an image
    with no valid box and it was NOT an empty sentinel, the image is excluded -
    an unboxed animal must not be taught as background.
  * Resolution never auto-downgrades rank: each distinct label triple maps to
    exactly one human-approved taxon, so a typo surfaces in the taxon map rather
    than silently borrowing a coarser rank.
  * Image UID hashes the path RELATIVE to source_root. A content hash would have
    merged genuinely duplicated images and dropped the second copy's
    annotations, breaking the deliberate per-export scoping.
  * `opcode` carries site and deployment, and lives in source_meta - the master
    file holds no source-specific conventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                       # noqa: E402

EXTRA_REQUIRED = ["empty_genus", "export_pairs"]
_CLEAN_TOKEN = re.compile(r"^[A-Za-z]+$")


def expand_exports(cfg):
    root = cfg["source_root"]
    return [(os.path.join(root, txt), os.path.join(root, img))
            for txt, img in cfg["export_pairs"]]


# --------------------------------------------------------------------------
# Parsing / classification  (pure, offline-testable)
# --------------------------------------------------------------------------

def read_export(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ("ImageCol", "ImageRow", "RectWidth", "RectHeight"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def triple_of(row):
    return (C.norm(row["Family"]), C.norm(row["Genus"]), C.norm(row["Species"]))


def is_empty_row(row, empty_genus):
    return C.norm(row["Genus"]) == empty_genus and C.norm(row["Species"]) == "spp"


def is_valid_box(row):
    w, h = row["RectWidth"], row["RectHeight"]
    return pd.notna(w) and pd.notna(h) and w > 0 and h > 0


def classify_export(df, empty_genus):
    empty_files, annot_rows, degenerate_rows = set(), [], []
    for _, row in df.iterrows():
        if is_empty_row(row, empty_genus):
            empty_files.add(row["Filename"])
        elif is_valid_box(row):
            annot_rows.append(row)
        else:
            degenerate_rows.append(row)
    return empty_files, annot_rows, degenerate_rows


def groups_from_opcode(opcode):
    """'MET01_03_BRUV_3' -> site MET01, deployment MET01_03."""
    if not opcode:
        return {}, {}
    parts = opcode.split("_")
    if len(parts) < 2:
        return {}, {}
    return ({"site": parts[0], "deployment": f"{parts[0]}_{parts[1]}"},
            {"site": "opcode", "deployment": "opcode"})


def propose_name(family, genus, species):
    if _CLEAN_TOKEN.match(genus) and _CLEAN_TOKEN.match(species) and species != "spp":
        return f"{genus} {species}", True
    if _CLEAN_TOKEN.match(genus) and species == "spp":
        return genus, True
    return "", False


def _resolve(export_path, override_dir):
    if override_dir:
        return os.path.join(override_dir, os.path.basename(export_path))
    return export_path


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_validate(args, cfg):
    total_imgs = total_empty = total_annot = total_degen = 0
    triples = Counter()
    for export_path, _pic in expand_exports(cfg):
        path = _resolve(export_path, args.dir)
        if not os.path.exists(path):
            print(f"  [missing] {path}")
            continue
        df = read_export(path)
        empty_files, annot_rows, degen = classify_export(df, cfg["empty_genus"])
        imgs = df["Filename"].nunique()
        total_imgs += imgs
        total_empty += len(empty_files)
        total_annot += len(annot_rows)
        total_degen += len(degen)
        for r in annot_rows:
            triples[triple_of(r)] += 1
        print(f"  {os.path.basename(path):<28} images={imgs:>5}  "
              f"empty={len(empty_files):>4}  annots={len(annot_rows):>5}  "
              f"degenerate={len(degen):>4}")
    print("-" * 72)
    print(f"  TOTAL  images={total_imgs}  empty={total_empty}  "
          f"annotations={total_annot}  degenerate={total_degen}")
    print(f"  distinct (Family,Genus,Species) annotation triples: {len(triples)}")
    print("\n  Triple inventory (count | Family | Genus | Species):")
    for t, n in triples.most_common():
        print(f"    {n:>6} | {t[0] or 'NA':<16} | {t[1] or 'NA':<20} | {t[2] or 'NA'}")


def cmd_taxon_map(args, cfg):
    out = cfg["taxon_map_csv"]
    if os.path.exists(out):
        sys.exit(f"{out} already exists. It holds manual decisions - refusing "
                 f"to overwrite. Move it aside deliberately to rebuild it.")

    triples = Counter()
    for export_path, _pic in expand_exports(cfg):
        path = _resolve(export_path, args.dir)
        if not os.path.exists(path):
            continue
        df = read_export(path)
        _, annot_rows, _ = classify_export(df, cfg["empty_genus"])
        for r in annot_rows:
            triples[triple_of(r)] += 1

    rows = []
    for (family, genus, species), n in triples.most_common():
        name, auto = propose_name(family, genus, species)
        aphia = rank = valid = ""
        status = "UNRESOLVED"
        if auto and name:
            hit = C.worms_by_name(name)
            time.sleep(0.3)
            if hit:
                aphia, valid, rank = hit
                status = "AUTO"
        rows.append({"family": family, "genus": genus, "species": species,
                     "n_rows": n, "proposed_name": name, "aphia_id": aphia,
                     "valid_name": valid, "rank": rank, "status": status})

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_todo = sum(r["status"] == "UNRESOLVED" for r in rows)
    print(f"Wrote {out}: {len(rows)} triples "
          f"({len(rows) - n_todo} auto-resolved, {n_todo} need an AphiaID).")


def cmd_coco(args, cfg):
    from PIL import Image

    empty_genus = cfg["empty_genus"]
    ds_id = cfg["dataset_meta"]["id"]
    taxon_map = C.load_taxon_map(cfg["taxon_map_csv"])

    coco, st = C.load_or_init_coco(cfg["output_json"])
    C.register_source(coco, cfg, st)
    C.ensure_categories(coco, taxon_map, cfg.get("lineage_cache"))

    review = []
    os.makedirs(cfg["output_image_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output_json"]) or ".", exist_ok=True)
    content_hashes = {}

    for export_path, pic_dir in expand_exports(cfg):
        if not os.path.exists(export_path):
            print(f"  [missing export] {export_path}")
            continue
        df = read_export(export_path)

        for filename, group in df.groupby("Filename"):
            source_path = os.path.join(pic_dir, filename)
            uid = C.uid_for(source_path, cfg["source_root"], cfg["uid_prefix"])

            if uid in st["seen_uid"]:
                continue          # already ingested; not a review item
            if not os.path.exists(source_path):
                review.append((filename, source_path,
                               "image file not found on disk"))
                continue

            try:
                with Image.open(source_path) as im:
                    width, height = im.size
                    im.load()
                content_sha = C.file_sha(source_path)
            except Exception as exc:                     # noqa: BLE001
                review.append((filename, source_path, f"unreadable image: {exc}"))
                continue

            if content_sha in content_hashes:
                review.append((filename, source_path,
                               f"identical image bytes to {content_hashes[content_sha]}"))
            else:
                content_hashes[content_sha] = uid

            opcodes = {C.norm(v) for v in group["OpCode"] if C.norm(v)}
            if len(opcodes) > 1:
                review.append((filename, source_path,
                               f"multiple OpCodes: {sorted(opcodes)}"))
            opcode = next(iter(opcodes), None)

            image_id = st["next_img"]
            st["next_img"] += 1
            st["seen_uid"].add(uid)

            n_valid = 0
            is_sentinel_empty = False
            has_unlabelled_animal = False
            pending = []
            for _, row in group.iterrows():
                if is_empty_row(row, empty_genus):
                    is_sentinel_empty = True
                    continue
                if not is_valid_box(row):
                    has_unlabelled_animal = True
                    review.append((filename, source_path,
                                   f"degenerate box dropped: {triple_of(row)}"))
                    continue
                aid = taxon_map.get(triple_of(row))
                if not aid:
                    has_unlabelled_animal = True
                    review.append((filename, source_path,
                                   f"unresolved taxon (no AphiaID): {triple_of(row)}"))
                    continue
                pending.append((row, aid))
                n_valid += 1

            if n_valid == 0 and not is_sentinel_empty:
                review.append((filename, source_path,
                               "no valid box and not an empty sentinel -> excluded"))
                st["next_img"] -= 1
                st["seen_uid"].discard(uid)
                continue

            dest = os.path.join(cfg["output_image_dir"], f"{uid}.jpg")
            if not os.path.exists(dest):
                C.copy_file(source_path, dest)

            groups, gsrc = groups_from_opcode(opcode)
            coco["images"].append({
                "id": image_id,
                "file_name": f"{uid}.jpg",
                "width": width, "height": height,
                "dataset_id": ds_id,
                "crop_provenance": cfg["crop_provenance"],
                "pixel_scale_known": cfg["pixel_scale_known"],
                "gear": cfg["gear"],
                "groups": groups, "group_sources": gsrc,
                "has_unlabelled_animal": has_unlabelled_animal,
                "lat": None, "lon": None,
                "datetime": None, "depth_m": None,
                "source_meta": {"opcode": opcode},
            })

            for row, aid in pending:
                x, y = float(row["ImageCol"]), float(row["ImageRow"])
                w, h = float(row["RectWidth"]), float(row["RectHeight"])
                coco["annotations"].append({
                    "id": st["next_ann"], "image_id": image_id,
                    "category_id": aid, "bbox": [x, y, w, h],
                    "area": w * h, "iscrowd": 0,
                    "individual_id": None, "track_id": None,
                })
                st["next_ann"] += 1

    with open(cfg["output_json"], "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)
    C.write_review(cfg["review_csv"], review)
    C.write_manifest(cfg, coco, len(review))

    print(f"Wrote {cfg['output_json']}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories.")
    print(f"Review items: {len(review)}"
          + (f" -> {cfg['review_csv']}" if review else ""))


def main():
    ap = argparse.ArgumentParser(description="YOLO-BRUVS -> SeaVision converter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_text, fn in [
        ("validate", "parse exports and print a tally (offline)", cmd_validate),
        ("taxon-map", "write the taxon map for review (needs WoRMS)", cmd_taxon_map),
        ("coco", "build/merge the COCO JSON", cmd_coco),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True)
        p.add_argument("--dir", help="folder holding the export .txt files")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    cfg = C.load_config(args.config, EXTRA_REQUIRED)
    args.func(args, cfg)


if __name__ == "__main__":
    main()

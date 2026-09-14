#!/usr/bin/env python3
"""
FathomNet -> SeaVision converter.

FathomNet is unlike the other three sources: it publishes full frames with
bounding boxes, not pre-cut crops. This converter cuts the crops.

Source-specific facts this converter has to know:

  * The concept map (fathomnet_concept_final.csv) is the AUTHORITY on
    taxonomy, not the aphia_id carried in the manifest. A correction to the
    CSV therefore never requires re-enumerating half a million images.

  * Three actions. 'class' emits a crop and an annotation. 'unlabelled'
    emits a crop with NO annotation and sets has_unlabelled_animal - the
    concept 'marine organism' means an annotator saw an animal and did not
    identify it, which is a useful crop and a useless class. 'drop' emits
    nothing.

  * gear is null for every image. imagingType is unusable across FathomNet
    and there is no per-image evidence of platform; inferring one from the
    institution would be a guess recorded as a fact.

  * groups is empty for every image. Deployment identity lives in the
    image-set-upload darwinCore record, reachable only per image UUID, so
    that pass is deferred. collectionCode is NOT in source_meta because it
    was not collected; the FathomNet image UUID is, which is what a later
    pass needs to fill groups in without re-downloading anything.

  * Boxes can extend beyond the frame. They are clamped, and clamping is
    recorded per crop so the affected crops can be excluded from any size
    analysis rather than silently skewing it.

  * taxon_map_csv holds this source's manual decisions, like the others, but
    it is keyed on the FathomNet concept string rather than a family/genus/
    species triple, so it is read by load_concept_map here and NOT by
    C.load_taxon_map.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                       # noqa: E402

EXTRA_REQUIRED = ["manifest_dir", "image_root"]


def safe(code):
    return "".join(c if c.isalnum() else "_" for c in code)


def load_concept_map(path):
    """concept -> (action, aphia_id)."""
    m = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            aid = (row.get("aphia_id") or "").strip()
            m[row["concept"]] = (row["action"].strip(),
                                 int(aid) if aid else None)
    return m


def iter_manifest(cfg, limit=None):
    n = 0
    for fn in sorted(os.listdir(cfg["manifest_dir"])):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(cfg["manifest_dir"], fn), encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
                n += 1
                if limit and n >= limit:
                    return


def frame_path(cfg, rec):
    return os.path.join(cfg["image_root"], safe(rec["institution"]),
                        rec["uuid"] + ".jpg")


def uid_for_box(box_uuid, prefix):
    """FathomNet box UUIDs are globally unique and stable, so the UID comes
    from the box, not from a file path - the crop does not exist on disk
    until this converter makes it."""
    return f"{prefix}-{hashlib.sha256(box_uuid.encode()).hexdigest()[:10]}"


def clamp_box(b, W, H):
    """-> (x0, y0, x1, y1, was_clamped) or None if degenerate."""
    rx0, ry0 = int(b["x"]), int(b["y"])
    rx1, ry1 = rx0 + int(b["w"]), ry0 + int(b["h"])
    x0, y0 = max(0, rx0), max(0, ry0)
    x1, y1 = min(W, rx1), min(H, ry1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1, (x0, y0, x1, y1) != (rx0, ry0, rx1, ry1)


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def cmd_validate(args, cfg):
    cmap = load_concept_map(cfg["taxon_map_csv"])
    by_action = collections.Counter()
    by_inst = collections.Counter()
    unknown = collections.Counter()
    shorts, clamped, missing, degenerate = [], 0, 0, 0
    n_rec = 0

    for rec in iter_manifest(cfg, args.limit):
        n_rec += 1
        by_inst[rec["institution"]] += 1
        if args.check_files and not os.path.exists(frame_path(cfg, rec)):
            missing += 1
        W, H = rec.get("width") or 0, rec.get("height") or 0
        for b in rec["boxes"]:
            hit = cmap.get(b["concept"])
            if hit is None:
                unknown[b["concept"]] += 1
                continue
            by_action[hit[0]] += 1
            if hit[0] == "drop":
                continue
            if W and H:
                c = clamp_box(b, W, H)
                if c is None:
                    degenerate += 1
                    continue
                x0, y0, x1, y1, was = c
                clamped += bool(was)
                shorts.append(min(x1 - x0, y1 - y0))

    print(f"frames {n_rec:,}")
    if args.check_files:
        print(f"frames missing on disk: {missing:,}")
    print(f"boxes by action: {dict(by_action)}")
    print(f"concepts absent from the map: {len(unknown)}"
          f"  boxes {sum(unknown.values()):,}")
    for c, n in unknown.most_common(15):
        print(f"   {n:>7,}  {c}")

    print(f"\nclamped to frame edge: {clamped:,}   degenerate (<2px): {degenerate:,}")
    if shorts:
        shorts.sort()
        def pc(p): return shorts[int(p * (len(shorts) - 1))]
        print(f"crop short side  median {pc(.5)}px  p10 {pc(.1)}  p90 {pc(.9)}  "
              f"<64px {100*sum(1 for s in shorts if s < 64)/len(shorts):.1f}%")

    print("\nper institution (frames):")
    for k, v in by_inst.most_common():
        print(f"  {k:<52} {v:>8,}")


# --------------------------------------------------------------------------
# coco
# --------------------------------------------------------------------------

def cmd_coco(args, cfg):
    from PIL import Image

    cmap = load_concept_map(cfg["taxon_map_csv"])
    ds_id = cfg["dataset_meta"]["id"]

    coco, st = C.load_or_init_coco(cfg["output_json"])
    C.register_source(coco, cfg, st)
    C.ensure_categories(coco, {k: v[1] for k, v in cmap.items()
                               if v[0] == "class"}, cfg.get("lineage_cache"))

    os.makedirs(cfg["output_image_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output_json"]) or ".", exist_ok=True)

    review, n_crop, n_frame, n_clamp = [], 0, 0, 0
    for rec in iter_manifest(cfg, args.limit):
        src = frame_path(cfg, rec)
        keep = [b for b in rec["boxes"]
                if cmap.get(b["concept"], ("drop", None))[0] != "drop"]
        if not keep:
            continue
        # every crop from this frame already exists - skip before touching the
        # file, so a re-run or crash-resume decodes no images at all
        if all(uid_for_box(b["uuid"], cfg["uid_prefix"]) in st["seen_uid"]
               for b in keep):
            continue
        if not os.path.exists(src):
            review.append((rec["uuid"], src, "frame not downloaded"))
            continue
        try:
            im = Image.open(src)
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
        except Exception as exc:                          # noqa: BLE001
            review.append((rec["uuid"], src, f"unreadable frame: {exc}"))
            continue
        W, H = im.size
        n_frame += 1

        for b in keep:
            action, aid = cmap[b["concept"]]
            if action == "class" and not aid:
                review.append((b["uuid"], src,
                               f"class with no AphiaID: {b['concept']}"))
                continue
            c = clamp_box(b, W, H)
            if c is None:
                review.append((b["uuid"], src,
                               f"degenerate box after clamping: {b['concept']}"))
                continue
            x0, y0, x1, y1, was_clamped = c
            n_clamp += bool(was_clamped)

            uid = uid_for_box(b["uuid"], cfg["uid_prefix"])
            if uid in st["seen_uid"]:
                continue
            dest = os.path.join(cfg["output_image_dir"], f"{uid}.jpg")
            if not os.path.exists(dest):
                im.crop((x0, y0, x1, y1)).save(
                    dest, "JPEG", quality=cfg.get("crop_quality", 95),
                    optimize=True)

            cw, ch = x1 - x0, y1 - y0
            image_id = st["next_img"]
            st["next_img"] += 1
            st["seen_uid"].add(uid)
            n_crop += 1

            coco["images"].append({
                "id": image_id, "file_name": f"{uid}.jpg",
                "width": cw, "height": ch, "dataset_id": ds_id,
                "crop_provenance": cfg["crop_provenance"],
                "pixel_scale_known": cfg["pixel_scale_known"],
                "gear": cfg["gear"],
                "groups": {}, "group_sources": {},
                "has_unlabelled_animal": action == "unlabelled",
                "lat": rec.get("lat"), "lon": rec.get("lon"),
                "datetime": rec.get("timestamp"),
                "depth_m": rec.get("depth_m"),
                "source_meta": {
                    "fathomnet_image_uuid": rec["uuid"],
                    "fathomnet_box_uuid": b["uuid"],
                    "owner_institution": rec["institution"],
                    "concept_raw": b["concept"],
                    "alt_concept": b.get("alt_concept"),
                    "review_state": b.get("review_state"),
                    "reviewer": b.get("reviewer"),
                    "observer": b.get("observer"),
                    "group_of": b.get("group_of"),
                    "occluded": b.get("occluded"),
                    "truncated": b.get("truncated"),
                    "frame_bbox": [int(b["x"]), int(b["y"]),
                                   int(b["w"]), int(b["h"])],
                    "frame_bbox_used": [x0, y0, cw, ch],
                    "frame_clamped": bool(was_clamped),
                    "frame_size": [W, H],
                    "frame_url": rec.get("url"),
                    "frame_sha256": rec.get("sha256"),
                    "imaging_type": rec.get("imaging_type"),
                    "contributors_email": rec.get("contributors_email"),
                    "altitude": rec.get("altitude"),
                    "temperature_c": rec.get("temperature_c"),
                    "salinity": rec.get("salinity"),
                    "oxygen_ml_l": rec.get("oxygen_ml_l"),
                    "grouping_note": "deployment identity not collected; it "
                                     "lives on the image-set-upload "
                                     "darwinCore record, reachable by "
                                     "fathomnet_image_uuid",
                },
            })

            if action == "class":
                coco["annotations"].append({
                    "id": st["next_ann"], "image_id": image_id,
                    "category_id": aid,
                    "bbox": [0.0, 0.0, float(cw), float(ch)],
                    "area": float(cw * ch), "iscrowd": 0,
                    "individual_id": None, "track_id": None,
                })
                st["next_ann"] += 1

        im.close()
        if n_frame % 5000 == 0:
            print(f"  {n_frame:,} frames, {n_crop:,} crops", flush=True)

    with open(cfg["output_json"], "w", encoding="utf-8") as fh:
        json.dump(coco, fh, separators=(",", ":"))
    C.write_review(cfg["review_csv"], review)
    C.write_manifest(cfg, coco, len(review),
                     extra={"frames_this_run": n_frame,
                            "crops_this_run": n_crop,
                            "clamped_this_run": n_clamp})

    print(f"\nWrote {cfg['output_json']}")
    print(f"  frames used {n_frame:,}   crops {n_crop:,}   clamped {n_clamp:,}")
    print(f"  images {len(coco['images']):,}  "
          f"annotations {len(coco['annotations']):,}  "
          f"categories {len(coco['categories']):,}  review {len(review):,}")


def main():
    ap = argparse.ArgumentParser(description="FathomNet -> SeaVision converter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="survey the manifest (offline)")
    v.add_argument("--check-files", action="store_true",
                   help="also confirm every frame exists on disk")
    v.add_argument("--limit", type=int, default=None)
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("coco", help="cut crops and build the COCO JSON")
    c.add_argument("--limit", type=int, default=None)
    c.set_defaults(func=cmd_coco)

    for p in (v, c):
        p.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = C.load_config(args.config, EXTRA_REQUIRED)
    args.func(args, cfg)


if __name__ == "__main__":
    main()

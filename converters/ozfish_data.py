#!/usr/bin/env python3
"""
OzFish -> SeaVision converter.

OzFish (AIMS/UWA/Curtin, DOI 10.25845/5e28f062c5097) is pre-cut fish crops from
stereo BRUVS across northern Australia, distributed via Pawsey.

    crop_metadata.csv   uid, file_name, family, genus, species
    crops/<file_name>   the crops themselves

Source-specific facts this converter has to know:

  * The CSV columns carry no grouping variable, but the FILENAME does:
        A000001_L.avi.5107.806.371.922.448.png
        survey+deployment, camera, container, frame, x0, y0, x1, y1
    Three containers occur: .avi, .mp4 and .mpeg.

  * One video IS one deployment, not a GoPro chapter. Verified from the
    measurement files: elapsed Time maxes near 60 min for surveys A and G
    (frames/minute giving exactly 25.0 and 30.0 fps), and surveys B and E carry
    annotations at 68 and 76 minutes, which a ~35-minute chapter could not
    contain. So groups.deployment is real.

  * FOUR surveys with different gear and protocols (A, B, E, G). Survey E has
    675 deployments but only EIGHT species and double everyone else's crop size
    - a targeted survey, not a community one. It is ingested anyway; excluding
    it is an experiment-time decision recorded in a config, not an ingest-time
    one, because that choice is reversible and a missing ingest is not.

  * _L and _R are the two cameras of one stereo rig, so THE SAME FISH IS
    ANNOTATED TWICE, at a constant frame offset per deployment. A random split
    would put the two views on opposite sides. Splitting by deployment closes
    this; the camera is recorded in source_meta so the pairing stays visible.

  * Species values of the form 'spN' are a generic unidentified marker, not a
    species: they appear under many unrelated families. Such rows resolve to
    GENUS where a genus is given and to FAMILY where it is not. Rows with
    neither are excluded and logged. Two triples may map to one AphiaID; the
    raw triple is preserved in source_meta.

  * uid is shared with frame_metadata.csv - the crops are cut from the
    published frames. Ingesting both products would enter every fish twice.

  * Original frames are not downloaded (WP7 defers a designed subset), so
    crop_provenance is 'pre_cropped'. The filename gives the box in frame
    coordinates, so pixel_scale_known is true and the geometry is preserved.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import statistics
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C                                       # noqa: E402

EXTRA_REQUIRED = ["image_root_rel", "crop_metadata_csv"]

PAT = re.compile(
    r"^(?P<vid>[A-Za-z]+\d+)_(?P<cam>[LR])\.(?P<ext>avi|mp4|mpeg)\.(?P<frame>\d+)"
    r"\.(?P<x0>-?\d+)\.(?P<y0>-?\d+)\.(?P<x1>-?\d+)\.(?P<y1>-?\d+)\.png$",
    re.IGNORECASE)

BINOMIAL = re.compile(r"^[a-z][a-z\-]+$")
UNIDENT_SPECIES = {"sp", "spp", "sp.", "spp."}
SUFFIX = re.compile(r"^(?P<base>.+?\.png)-\d+-\d+\.png$", re.I)
_DISK_INDEX = None


def disk_index(root):
    """Map crop_metadata file_name -> the file as it actually sits on disk.

    The web-portal zip appends a global row index to every crop:
        X.png  ->  X.png-1-1.png
    Verified 1:1 across all 80,823 crops (index 1..80,823, no collisions).
    The raw archive is left untouched; the mapping lives here so that UIDs
    are computed from the clean metadata name, not the portal's artefact.
    """
    global _DISK_INDEX
    if _DISK_INDEX is None:
        _DISK_INDEX = {}
        for fn in os.listdir(root):
            m = SUFFIX.match(fn)
            _DISK_INDEX[m["base"] if m else fn] = fn
    return _DISK_INDEX


def src_for(root, file_name):
    fn = str(file_name)
    return os.path.join(root, disk_index(root).get(fn, fn))


def parse_name(fn):
    m = PAT.match(str(fn))
    if not m:
        return None
    x0, y0, x1, y1 = (int(m["x0"]), int(m["y0"]), int(m["x1"]), int(m["y1"]))
    return {"vid": m["vid"], "cam": m["cam"].upper(), "ext": m["ext"].lower(),
            "frame": int(m["frame"]), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "w": x1 - x0, "h": y1 - y0,
            "survey": re.match(r"^([A-Za-z]+)", m["vid"]).group(1)}


def rank_intent(family, genus, species):
    """
    -> (name_to_resolve, rank_intent) or (None, None) if unidentifiable.

    'spN' is a generic unidentified marker in this source, not a species, so it
    never yields a binomial.
    """
    family, genus, species = C.norm(family), C.norm(genus), C.norm(species)
    if (genus and species and BINOMIAL.match(species)
            and species.lower() not in UNIDENT_SPECIES):
        return f"{genus} {species}", "species"
    if genus:
        return genus, "genus"
    if family:
        return family, "family"
    return None, None


def load_rows(cfg):
    """Yield (row, parsed) for every crop metadata row."""
    path = os.path.join(cfg["source_root"], cfg["crop_metadata_csv"])
    d = pd.read_csv(path)
    for r in d.itertuples(index=False):
        yield r, parse_name(r.file_name)


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def cmd_validate(args, cfg):
    from PIL import Image

    root = os.path.join(cfg["source_root"], cfg["image_root_rel"])
    rows, bad, missing = [], [], 0
    for r, p in load_rows(cfg):
        if p is None:
            bad.append(r.file_name)
            continue
        rows.append((r, p))
        if args.check_files and not os.path.exists(src_for(root, r.file_name)):
            missing += 1

    print(f"rows {len(rows) + len(bad)}   unparsed {len(bad)}")
    for b in bad[:10]:
        print("   ", b)
    if args.check_files:
        print(f"files missing on disk: {missing}")

    print()
    print("PER SURVEY")
    by = collections.defaultdict(list)
    for r, p in rows:
        by[p["survey"]].append((r, p))
    for sv in sorted(by):
        sub = by[sv]
        deps = {p["vid"] for _, p in sub}
        spp = {(C.norm(r.genus), C.norm(r.species)) for r, _ in sub}
        short = sorted(min(p["w"], p["h"]) for _, p in sub)
        print(f"  {sv}: crops {len(sub):>6}  deployments {len(deps):>4}  "
              f"taxa {len(spp):>4}  median short side {short[len(short)//2]:>4}px  "
              f"<64px {100*sum(1 for s in short if s < 64)/len(short):>5.1f}%")

    print()
    print("TAXONOMY")
    intents = collections.Counter()
    triples = {}
    unident = 0
    for r, _ in rows:
        name, ri = rank_intent(r.family, r.genus, r.species)
        intents[ri] += 1
        if ri is None:
            unident += 1
            continue
        triples[(C.norm(r.family), C.norm(r.genus), C.norm(r.species))] = (name, ri)
    print("  rows by intended rank:", dict(intents))
    print("  distinct triples:", len(triples))
    print("  distinct names to resolve:", len({n for n, _ in triples.values()}))
    print("  unidentifiable rows (excluded):", unident)
    byrank = collections.Counter(ri for _, ri in triples.values())
    print("  triples by rank:", dict(byrank))

    print()
    print("GROUPING VIABILITY")
    g = collections.defaultdict(lambda: [0, set(), set()])
    for r, p in rows:
        k = (C.norm(r.genus), C.norm(r.species))
        g[k][0] += 1
        g[k][1].add(p["vid"])
        g[k][2].add(p["survey"])
    print("  taxa:", len(g))
    for n, v in ((100, 3), (100, 5), (50, 3)):
        print(f"  >={n} crops and >={v} deployments:",
              sum(1 for c, d, _ in g.values() if c >= n and len(d) >= v))

    print()
    print("GEOMETRY CHECK")
    import random
    sample = random.Random(0).sample(rows, min(args.sample, len(rows)))
    mm = unreadable = 0
    for r, p in sample:
        try:
            with Image.open(src_for(root, r.file_name)) as im:
                if im.size != (p["w"], p["h"]):
                    mm += 1
        except Exception:                                # noqa: BLE001
            unreadable += 1
    print(f"  {mm} mismatched, {unreadable} unreadable, of {len(sample)} sampled")

    print()
    print("TRACKS (same deployment+camera, frame gap <= gap, boxes overlap)")
    seq = collections.defaultdict(list)
    for r, p in rows:
        seq[(p["vid"], p["cam"], C.norm(r.genus), C.norm(r.species))].append(
            (p["frame"], p["x0"], p["y0"], p["w"], p["h"]))
    lens = []
    for v in seq.values():
        v.sort()
        cur = 0
        for i, (fr, x, y, w, h) in enumerate(v):
            pv = v[i - 1] if i else None
            ok = False
            if pv and fr - pv[0] <= args.track_gap:
                ix = max(0, min(x + w, pv[1] + pv[3]) - max(x, pv[1]))
                iy = max(0, min(y + h, pv[2] + pv[4]) - max(y, pv[2]))
                inter = ix * iy
                ok = inter and inter / (w * h + pv[3] * pv[4] - inter) > args.track_iou
            if ok:
                cur += 1
            else:
                if cur:
                    lens.append(cur)
                cur = 1
        if cur:
            lens.append(cur)
    m = statistics.mean(lens)
    print(f"  tracks {len(lens)}  mean {m:.2f}  median {statistics.median(lens)}  "
          f"max {max(lens)}   design effect at rho=0.5: {1 + (m - 1) * 0.5:.2f}")


# --------------------------------------------------------------------------
# taxon-map
# --------------------------------------------------------------------------

def cmd_taxon_map(args, cfg):
    out = cfg["taxon_map_csv"]
    if os.path.exists(out):
        sys.exit(f"{out} already exists. It holds manual decisions - refusing "
                 f"to overwrite. Move it aside deliberately to rebuild it.")

    counts = collections.Counter()
    for r, p in load_rows(cfg):
        if p is None:
            continue
        counts[(C.norm(r.family), C.norm(r.genus), C.norm(r.species))] += 1

    # one WoRMS lookup per distinct NAME, not per triple
    resolved = {}
    rows = []
    for (family, genus, species), n in sorted(counts.items(), key=lambda x: -x[1]):
        name, ri = rank_intent(family, genus, species)
        if name is None:
            rows.append({"family": family, "genus": genus, "species": species,
                         "n_rows": n, "proposed_name": "", "rank_intent": "",
                         "aphia_id": "", "valid_name": "", "rank": "",
                         "status": "UNIDENTIFIABLE"})
            continue
        if name not in resolved:
            hit = C.worms_by_name(name)
            time.sleep(0.3)
            resolved[name] = hit
            print(f"  {'AUTO ' if hit else 'MANUAL'} {ri:<8} {name}")
        hit = resolved[name]
        rows.append({
            "family": family, "genus": genus, "species": species, "n_rows": n,
            "proposed_name": name, "rank_intent": ri,
            "aphia_id": hit[0] if hit else "", "valid_name": hit[1] if hit else "",
            "rank": hit[2] if hit else "", "status": "AUTO" if hit else "UNRESOLVED",
        })

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_todo = sum(r["status"] == "UNRESOLVED" for r in rows)
    n_none = sum(r["status"] == "UNIDENTIFIABLE" for r in rows)
    print(f"\nWrote {out}: {len(rows)} triples, {len(resolved)} distinct names "
          f"({n_todo} need an AphiaID, {n_none} unidentifiable).")


# --------------------------------------------------------------------------
# coco
# --------------------------------------------------------------------------

def cmd_coco(args, cfg):
    from PIL import Image

    root = os.path.join(cfg["source_root"], cfg["image_root_rel"])
    ds_id = cfg["dataset_meta"]["id"]
    taxon_map = C.load_taxon_map(cfg["taxon_map_csv"])

    coco, st = C.load_or_init_coco(cfg["output_json"])
    C.register_source(coco, cfg, st)
    C.ensure_categories(coco, taxon_map, cfg.get("lineage_cache"))

    os.makedirs(cfg["output_image_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output_json"]) or ".", exist_ok=True)

    review, n = [], 0
    for r, p in load_rows(cfg):
        fn = str(r.file_name)
        src = src_for(root, fn)
        clean = os.path.join(root, fn)      # UID comes from the metadata name
        if p is None:
            review.append((fn, src, "filename did not parse"))
            continue

        triple = (C.norm(r.family), C.norm(r.genus), C.norm(r.species))
        aid = taxon_map.get(triple)
        if not aid:
            review.append((fn, src, f"unresolved taxon (no AphiaID): {triple}"))
            continue

        uid = C.uid_for(src, cfg["source_root"], cfg["uid_prefix"])
        if uid in st["seen_uid"]:
            continue          # already ingested; not a review item

        try:
            if args.trust_geometry:
                width, height = p["w"], p["h"]
            else:
                with Image.open(src) as im:
                    width, height = im.size
        except Exception as exc:                         # noqa: BLE001
            review.append((fn, src, f"unreadable image: {exc}"))
            continue

        ext = os.path.splitext(fn)[1].lower() or ".png"
        dest = os.path.join(cfg["output_image_dir"], f"{uid}{ext}")
        if not os.path.exists(dest):
            C.copy_file(src, dest)

        image_id = st["next_img"]
        st["next_img"] += 1
        st["seen_uid"].add(uid)
        n += 1

        coco["images"].append({
            "id": image_id, "file_name": f"{uid}{ext}",
            "width": width, "height": height, "dataset_id": ds_id,
            "crop_provenance": cfg["crop_provenance"],
            "pixel_scale_known": cfg["pixel_scale_known"],
            "gear": cfg["gear"],
            "groups": {"survey": p["survey"], "deployment": p["vid"]},
            "group_sources": {"survey": "file_name", "deployment": "file_name"},
            "has_unlabelled_animal": False,
            "lat": None, "lon": None, "datetime": None, "depth_m": None,
            "source_meta": {
                "ozfish_uid": int(r.uid),
                "video": p["vid"], "camera": p["cam"], "container": p["ext"],
                "frame": p["frame"],
                "frame_bbox": [p["x0"], p["y0"], p["w"], p["h"]],
                "raw_taxon": {"family": triple[0], "genus": triple[1],
                              "species": triple[2]},
                "stereo_note": "_L and _R are the two cameras of one rig; the "
                               "same individual is annotated twice",
            },
        })

        coco["annotations"].append({
            "id": st["next_ann"], "image_id": image_id, "category_id": aid,
            "bbox": [0.0, 0.0, float(width), float(height)],
            "area": float(width * height), "iscrowd": 0,
            "individual_id": None, "track_id": None,
        })
        st["next_ann"] += 1

    with open(cfg["output_json"], "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)
    C.write_review(cfg["review_csv"], review)
    C.write_manifest(cfg, coco, len(review), extra={"ozfish_crops": n})

    print(f"Wrote {cfg['output_json']}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories.  this source: {n}")
    print(f"Review items: {len(review)}"
          + (f" -> {cfg['review_csv']}" if review else ""))


def main():
    ap = argparse.ArgumentParser(description="OzFish -> SeaVision converter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="survey the archive (offline)")
    v.add_argument("--sample", type=int, default=300)
    v.add_argument("--track-gap", type=int, default=5)
    v.add_argument("--track-iou", type=float, default=0.2)
    v.add_argument("--check-files", action="store_true",
                   help="also confirm every crop exists on disk")
    v.set_defaults(func=cmd_validate)

    t = sub.add_parser("taxon-map", help="write the taxon map (needs WoRMS)")
    t.set_defaults(func=cmd_taxon_map)

    c = sub.add_parser("coco", help="build/merge the COCO JSON")
    c.add_argument("--trust-geometry", action="store_true")
    c.set_defaults(func=cmd_coco)

    for p in (v, t, c):
        p.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = C.load_config(args.config, EXTRA_REQUIRED)
    args.func(args, cfg)


if __name__ == "__main__":
    main()

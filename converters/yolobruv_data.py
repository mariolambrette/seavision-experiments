#!/usr/bin/env python3
"""
SeaVision annotation converter
==============================

Converts EventMeasure ``.txt`` point exports into a single, merge-aware,
COCO-style JSON dataset for the unified SeaVision collection.

Three subcommands, each driven by a YAML config (``--config``):

    validate    Parse the exports and print a classification tally.
                Needs only the .txt files (no images, no network). Use this
                first to sanity-check the config and the data.

    taxon-map   Scan the exports, list every distinct (Family, Genus, Species)
                triple, auto-resolve the unambiguous binomials against WoRMS,
                and write the taxon map CSV. YOU then fill in the AphiaID for
                every row left as UNRESOLVED. Needs the .txt files and network.

    coco        Using the *reviewed* taxon map, enumerate images, copy each to
                a shared store under a hashed UID, read its real width/height,
                attach annotations, and emit / merge the COCO JSON. Needs the
                .txt files and the image directories; needs the network only
                for taxa not already in the lineage cache.

Design decisions (yolo-bruv), fixed during design review:
  * An image marked ``Styelidae / Botryllus / spp`` is EMPTY: it gets zero
    annotations regardless of any box drawn on it. This is the empty sentinel;
    `spp` is used for nothing else in this dataset.
  * Bounding box = [ImageCol, ImageRow, RectWidth, RectHeight], absolute pixels,
    top-left origin (COCO convention). No normalisation.
  * A row whose box has zero width or height is degenerate: it is dropped and
    written to the review log. If that leaves an image with zero valid boxes and
    it was NOT an empty sentinel, the image is excluded and logged (an unboxed
    animal must not be taught as background).
  * category_id == WoRMS AphiaID. Each category stores its rank and a full
    lineage snapshot so the file is self-contained offline. Class indices for
    a given YOLO/COCO subset are materialised at usage time, not stored here.
  * Resolution never auto-downgrades rank: each distinct label triple maps to
    exactly one human-approved taxon. A typo is its own triple and surfaces in
    the taxon map rather than silently borrowing a coarser rank.

Changed in WP0b (September 2026):
  * All paths come from a YAML config. Nothing machine-specific in the code.
  * Image UID = prefix + sha256(path RELATIVE to source_root)[:10]. The old
    absolute-path hash made UIDs change whenever the data moved, so a re-run
    merged a duplicate copy of the whole dataset instead of being idempotent.
    Relative paths are portable, and unlike a content hash they preserve
    per-export scoping: identical filenames in different exports stay distinct
    images, as intended.
  * Images carry ``has_unlabelled_animal``. Previously an unresolved taxon was
    logged while the image was kept, so an image could enter the dataset with a
    real animal in it and no box on it. Harmless for classifying crops; not
    harmless once a detector is evaluated against it.
  * Every build writes ``build_manifest.json`` beside the output, recording the
    git commit, the config and the resulting counts.

Changed in WP1 (September 2026):
  * Schema v1.1 fields: crop_provenance, pixel_scale_known, gear, groups,
    group_sources, datetime, depth_m, source_meta on images; individual_id and
    track_id on annotations; licences registered from the config.
  * ``opcode`` demoted from a top-level image field into ``source_meta``. The
    master file carries no source-specific conventions; the validator enforces
    it by rejecting unknown keys.
  * Grouping is a dict of levels (site, deployment, ...) rather than one field,
    because yolo-bruv genuinely has two nested levels and the split ladder
    needs both. A source populates only the levels it really has; an absent
    level means that split is not available for that source, which is the
    honest failure rather than a remembered caveat.
  * Config loading is strict: duplicate YAML keys raise instead of silently
    keeping the last one, and dataset_meta / licence references are checked
    before any work starts.
  * WoRMS lineages are cached to a committed JSON file, and every request
    retries with backoff and tolerates a non-JSON body. Lineages are stable
    reference data; re-fetching them on every build made the build fail
    whenever the service was slow.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants that are genuinely fixed (everything else lives in the config)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.1"
WORMS = "https://www.marinespecies.org/rest"
WORMS_UA = "SeaVision-collation/1.1 (University of Exeter)"
LINEAGE_RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

_CLEAN_TOKEN = re.compile(r"^[A-Za-z]+$")   # a single, plain scientific token


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    "name", "uid_prefix", "empty_genus", "source_root", "output_json",
    "output_image_dir", "taxon_map_csv", "review_csv", "dataset_meta",
    "export_pairs", "crop_provenance", "pixel_scale_known", "gear",
]
REQUIRED_DATASET_META = ["id", "name", "license_id"]


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys instead of silently
    keeping the last one - which is how a config can lose half its content."""


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def load_config(path):
    """Read the YAML config, check it, and expand export pairs to full paths."""
    with open(path, encoding="utf-8") as fh:
        try:
            cfg = yaml.load(fh, Loader=_StrictLoader)
        except yaml.constructor.ConstructorError as exc:
            sys.exit(f"config {path}: {exc}")

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"config {path} is missing required keys: {missing}")

    missing = [k for k in REQUIRED_DATASET_META if k not in cfg["dataset_meta"]]
    if missing:
        sys.exit(f"config {path}: dataset_meta is missing {missing}")

    lic_ids = {l["id"] for l in cfg.get("licenses", [])}
    if cfg["dataset_meta"]["license_id"] not in lic_ids | {None}:
        sys.exit(f"config {path}: dataset_meta.license_id "
                 f"{cfg['dataset_meta']['license_id']} is not defined in licenses")

    root = cfg["source_root"]
    cfg["exports"] = [
        (os.path.join(root, txt), os.path.join(root, img))
        for txt, img in cfg["export_pairs"]
    ]
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def git_info():
    """Commit hash and dirty flag for the repo this script lives in."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=here, text=True).strip() != ""
        return {"commit": commit, "dirty": dirty}
    except Exception:                                   # noqa: BLE001
        return {"commit": None, "dirty": None}


# ---------------------------------------------------------------------------
# Parsing / classification  (pure, offline-testable)
# ---------------------------------------------------------------------------

def read_export(path):
    """Read one EventMeasure export as a DataFrame with clean column names."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ("ImageCol", "ImageRow", "RectWidth", "RectHeight"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def norm(value):
    """Normalise a taxon cell: strip, treat blank/NA as empty string."""
    if value is None:
        return ""
    v = str(value).strip()
    return "" if v.upper() in ("", "NA", "NAN", "NULL") else v


def triple_of(row):
    return (norm(row["Family"]), norm(row["Genus"]), norm(row["Species"]))


def is_empty_row(row, empty_genus):
    """True if this row is the empty-frame sentinel."""
    return norm(row["Genus"]) == empty_genus and norm(row["Species"]) == "spp"


def is_valid_box(row):
    w, h = row["RectWidth"], row["RectHeight"]
    return pd.notna(w) and pd.notna(h) and w > 0 and h > 0


def classify_export(df, empty_genus):
    """
    Split one export's rows into: empty-image filenames, valid-box annotation
    rows, and degenerate (zero-box, non-empty) rows. Grouping is by Filename
    within this single export only.
    """
    empty_files, annot_rows, degenerate_rows = set(), [], []
    for _, row in df.iterrows():
        if is_empty_row(row, empty_genus):
            empty_files.add(row["Filename"])
        elif is_valid_box(row):
            annot_rows.append(row)
        else:
            degenerate_rows.append(row)
    return empty_files, annot_rows, degenerate_rows


def cmd_validate(args, cfg):
    total_imgs = total_empty = total_annot = total_degen = 0
    triples = Counter()
    for export_path, _pic in cfg["exports"]:
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


# ---------------------------------------------------------------------------
# WoRMS resolution
# ---------------------------------------------------------------------------

def _requests():
    import requests  # imported lazily so `validate` needs no network stack
    return requests


def _worms_json(url, params=None, tries=4, timeout=60):
    """
    GET and parse JSON, with retries and exponential backoff.

    Returns the parsed object, or None for 'no such record' and for persistent
    failure. A non-JSON body is treated as a failure and retried: an
    intercepting proxy or an error page should not surface as a
    JSONDecodeError three hundred lines into a traceback.
    """
    requests = _requests()
    delay = 2
    for attempt in range(1, tries + 1):
        try:
            resp = requests.get(
                url, params=params, timeout=timeout,
                headers={"User-Agent": WORMS_UA, "Accept": "application/json"})
            if resp.status_code in (204, 404):
                return None
            if resp.status_code != 200:
                print(f"    ! WoRMS HTTP {resp.status_code} ({attempt}/{tries}) {url}")
            else:
                try:
                    return resp.json()
                except ValueError:
                    print(f"    ! WoRMS non-JSON ({attempt}/{tries}): "
                          f"{resp.headers.get('content-type')} {resp.text[:120]!r}")
        except Exception as exc:                    # noqa: BLE001
            print(f"    ! WoRMS {type(exc).__name__} ({attempt}/{tries}) {url}")
        if attempt < tries:
            time.sleep(delay)
            delay *= 2
    return None


def worms_by_name(name):
    """Return (aphia_id, valid_name, rank) for an exact accepted match, or None."""
    records = _worms_json(f"{WORMS}/AphiaRecordsByName/{name}",
                          params={"like": "false", "marine_only": "true"})
    if not records:
        return None
    accepted = [r for r in records if r.get("status") == "accepted"]
    pool = accepted or records
    if len(pool) != 1:                              # 0 or ambiguous -> manual
        return None
    r = pool[0]
    return r["AphiaID"], r.get("valid_name") or r.get("scientificname"), r.get("rank")


def load_lineage_cache(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_lineage_cache(path, cache):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)


def worms_lineage(aphia_id, cache=None):
    """
    (rank, valid_name, {rank_lower: name}) for an AphiaID.

    Cached locally. Lineages are stable reference data, so re-fetching them on
    every build only creates a network dependency the build does not need. The
    cache is committed, which makes rebuilds offline and versions the taxonomy
    snapshot alongside the taxon map. Entries carry the date they were fetched;
    delete the file to force a refresh.
    """
    key = str(aphia_id)
    if cache is not None and key in cache:
        e = cache[key]
        return e["rank"], e["valid_name"], e["lineage"]

    rec = _worms_json(f"{WORMS}/AphiaRecordByAphiaID/{aphia_id}")
    cls = _worms_json(f"{WORMS}/AphiaClassificationByAphiaID/{aphia_id}")
    if rec is None or cls is None:
        raise RuntimeError(
            f"WoRMS lookup failed for AphiaID {aphia_id} - either the service is "
            f"unreachable, or that ID does not exist (check the taxon map). "
            f"Re-run when it responds; cached taxa are skipped.")

    lineage = {}
    node = cls
    while node:
        if node.get("rank") in LINEAGE_RANKS:
            lineage[node["rank"].lower()] = node["scientificname"]
        node = node.get("child")

    rank = rec.get("rank")
    valid = rec.get("valid_name") or rec.get("scientificname")
    if cache is not None:
        cache[key] = {"rank": rank, "valid_name": valid, "lineage": lineage,
                      "fetched": datetime.date.today().isoformat()}
    return rank, valid, lineage


def propose_name(family, genus, species):
    """Best-effort query name + whether it is safe to auto-resolve."""
    if _CLEAN_TOKEN.match(genus) and _CLEAN_TOKEN.match(species) and species != "spp":
        return f"{genus} {species}", True          # clean binomial -> species
    if _CLEAN_TOKEN.match(genus) and species == "spp":
        return genus, True                          # genus-only
    return "", False                                # hybrid / NA / odd -> manual


def cmd_taxon_map(args, cfg):
    triples = Counter()
    for export_path, _pic in cfg["exports"]:
        path = _resolve(export_path, args.dir)
        if not os.path.exists(path):
            continue
        df = read_export(path)
        _, annot_rows, _ = classify_export(df, cfg["empty_genus"])
        for r in annot_rows:
            triples[triple_of(r)] += 1

    out = cfg["taxon_map_csv"]
    if os.path.exists(out):
        sys.exit(f"{out} already exists. It holds manual decisions - refusing to "
                 f"overwrite. Move it aside deliberately if you mean to rebuild it.")

    rows = []
    for (family, genus, species), n in triples.most_common():
        name, auto = propose_name(family, genus, species)
        aphia = rank = valid = ""
        status = "UNRESOLVED"
        if auto and name:
            hit = worms_by_name(name)
            time.sleep(0.3)                          # be polite to the API
            if hit:
                aphia, valid, rank = hit
                status = "AUTO"
        rows.append({
            "family": family, "genus": genus, "species": species, "n_rows": n,
            "proposed_name": name, "aphia_id": aphia, "valid_name": valid,
            "rank": rank, "status": status,
        })

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_auto = sum(r["status"] == "AUTO" for r in rows)
    n_todo = sum(r["status"] == "UNRESOLVED" for r in rows)
    print(f"Wrote {out}: {len(rows)} triples "
          f"({n_auto} auto-resolved, {n_todo} need an AphiaID from you).")
    print("Fill the aphia_id column for every UNRESOLVED row, correct any wrong "
          "AUTO rows, then run the `coco` command.")


# ---------------------------------------------------------------------------
# COCO build
# ---------------------------------------------------------------------------

def load_taxon_map(path):
    """triple -> aphia_id (int) for every row that has one."""
    mapping = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (norm(row["family"]), norm(row["genus"]), norm(row["species"]))
            aid = row.get("aphia_id", "").strip()
            mapping[key] = int(aid) if aid else None
    return mapping


def groups_from_opcode(opcode):
    """
    'MET01_03_BRUV_3' -> site MET01, deployment MET01_03.
    Returns ({level: value}, {level: raw field it came from}).
    """
    if not opcode:
        return {}, {}
    parts = opcode.split("_")
    if len(parts) < 2:
        return {}, {}
    return ({"site": parts[0], "deployment": f"{parts[0]}_{parts[1]}"},
            {"site": "opcode", "deployment": "opcode"})


def uid_for(source_path, source_root, uid_prefix):
    """
    Stable image UID from the path RELATIVE to source_root, so moving the
    dataset does not change any UID. Per-export scoping is preserved: the same
    filename under two different export directories gives two different UIDs.
    """
    rel = os.path.relpath(source_path, source_root).replace("\\", "/")
    digest = hashlib.sha256(rel.encode("utf-8")).hexdigest()[:10]
    return f"{uid_prefix}-{digest}"


def cmd_coco(args, cfg):
    from PIL import Image

    source_root = cfg["source_root"]
    uid_prefix = cfg["uid_prefix"]
    empty_genus = cfg["empty_genus"]
    dataset_meta = cfg["dataset_meta"]
    output_json = cfg["output_json"]
    output_image_dir = cfg["output_image_dir"]

    taxon_map = load_taxon_map(cfg["taxon_map_csv"])

    # -- resume / merge into an existing master file --------------------------
    if os.path.exists(output_json):
        with open(output_json, encoding="utf-8") as fh:
            coco = json.load(fh)
        next_img = max((im["id"] for im in coco["images"]), default=0) + 1
        next_ann = max((a["id"] for a in coco["annotations"]), default=0) + 1
        cat_ids = {c["id"] for c in coco["categories"]}
        seen_uid = {os.path.splitext(im["file_name"])[0] for im in coco["images"]}
        ds_ids = {d["id"] for d in coco["datasets"]}
    else:
        coco = {
            "info": {"schema_version": SCHEMA_VERSION,
                     "description": "SeaVision unified detection dataset"},
            "licenses": [],
            "datasets": [],
            "categories": [],
            "images": [],
            "annotations": [],
        }
        next_img = next_ann = 1
        cat_ids, seen_uid, ds_ids = set(), set(), set()

    # -- licences and dataset -------------------------------------------------
    for lic in cfg.get("licenses", []):
        if lic["id"] not in {l["id"] for l in coco["licenses"]}:
            coco["licenses"].append(dict(lic))

    if dataset_meta["id"] not in ds_ids:
        coco["datasets"].append(dict(dataset_meta))

    # -- categories: one per unique AphiaID in the (reviewed) map -------------
    # The cache is saved in a finally block so a network failure part way
    # through leaves the fetched taxa cached, and a re-run resumes.
    cache_path = cfg.get("lineage_cache")
    lineage_cache = load_lineage_cache(cache_path)
    try:
        for aid in sorted({a for a in taxon_map.values() if a}):
            if aid in cat_ids:
                continue
            was_cached = str(aid) in lineage_cache
            rank, valid, lineage = worms_lineage(aid, lineage_cache)
            if not was_cached:
                print(f"  fetched lineage for {aid} ({valid})")
                time.sleep(0.3)
            coco["categories"].append({
                "id": aid,
                "name": valid,
                "rank": rank,
                "supercategory": lineage.get("family", ""),
                "aphia_id": aid,
                "lineage": lineage,
            })
            cat_ids.add(aid)
    finally:
        save_lineage_cache(cache_path, lineage_cache)

    review = []
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cfg["review_csv"]) or ".", exist_ok=True)
    content_hashes = {}   # sha(image bytes) -> uid, to flag true duplicates

    for export_path, pic_dir in cfg["exports"]:
        if not os.path.exists(export_path):
            print(f"  [missing export] {export_path}")
            continue
        df = read_export(export_path)

        for filename, group in df.groupby("Filename"):
            source_path = os.path.join(pic_dir, filename)
            uid = uid_for(source_path, source_root, uid_prefix)

            if uid in seen_uid:
                review.append((filename, source_path, "duplicate UID (already ingested)"))
                continue

            if not os.path.exists(source_path):
                review.append((filename, source_path, "image file not found on disk"))
                continue

            try:
                with Image.open(source_path) as im:
                    width, height = im.size
                    im.load()
                content_sha = _file_sha(source_path)
            except Exception as exc:                # noqa: BLE001
                review.append((filename, source_path, f"unreadable image: {exc}"))
                continue

            if content_sha in content_hashes:
                review.append((filename, source_path,
                               f"identical image bytes to {content_hashes[content_sha]}"))
            else:
                content_hashes[content_sha] = uid

            opcodes = {norm(v) for v in group["OpCode"] if norm(v)}
            if len(opcodes) > 1:
                review.append((filename, source_path, f"multiple OpCodes: {sorted(opcodes)}"))
            opcode = next(iter(opcodes), None)

            image_id = next_img
            next_img += 1
            seen_uid.add(uid)

            # classify this image's rows
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

            # decision A: an image with no valid box that is not a sentinel
            # empty is ambiguous -> exclude and log, never treat as background.
            if n_valid == 0 and not is_sentinel_empty:
                review.append((filename, source_path,
                               "no valid box and not an empty sentinel -> excluded"))
                next_img -= 1
                seen_uid.discard(uid)
                continue

            # copy image into the shared store and register it
            dest = os.path.join(output_image_dir, f"{uid}.jpg")
            if not os.path.exists(dest):
                _copy(source_path, dest)

            groups, group_sources = groups_from_opcode(opcode)
            coco["images"].append({
                "id": image_id,
                "file_name": f"{uid}.jpg",
                "width": width,
                "height": height,
                "dataset_id": dataset_meta["id"],
                "crop_provenance": cfg["crop_provenance"],
                "pixel_scale_known": cfg["pixel_scale_known"],
                "gear": cfg["gear"],
                "groups": groups,
                "group_sources": group_sources,
                "has_unlabelled_animal": has_unlabelled_animal,
                "lat": None,
                "lon": None,
                "datetime": None,
                "depth_m": None,
                "source_meta": {"opcode": opcode},
            })

            for row, aid in pending:
                x, y = float(row["ImageCol"]), float(row["ImageRow"])
                w, h = float(row["RectWidth"]), float(row["RectHeight"])
                coco["annotations"].append({
                    "id": next_ann,
                    "image_id": image_id,
                    "category_id": aid,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "individual_id": None,
                    "track_id": None,
                })
                next_ann += 1

    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)

    if review:
        with open(cfg["review_csv"], "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filename", "source_path", "reason"])
            writer.writerows(review)

    # -- build manifest -------------------------------------------------------
    manifest = {
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "config_path": cfg["_config_path"],
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "counts": {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": len(coco["categories"]),
            "review_items": len(review),
            "images_with_unlabelled_animal":
                sum(1 for im in coco["images"] if im.get("has_unlabelled_animal")),
        },
    }
    manifest_path = os.path.join(os.path.dirname(output_json) or ".",
                                 "build_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote {output_json}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations, {len(coco['categories'])} categories.")
    print(f"Review items logged: {len(review)}"
          + (f" -> {cfg['review_csv']}" if review else ""))
    print(f"Manifest: {manifest_path}"
          + ("  [WARNING: uncommitted changes]" if manifest["git"]["dirty"] else ""))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _resolve(export_path, override_dir):
    """For `validate`/`taxon-map`: optionally look for the basename in --dir."""
    if override_dir:
        return os.path.join(override_dir, os.path.basename(export_path))
    return export_path


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy(src, dst):
    import shutil
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="EventMeasure -> SeaVision COCO converter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_text, fn in [
        ("validate",  "parse exports and print a tally (offline)", cmd_validate),
        ("taxon-map", "write the taxon map for review (needs WoRMS)", cmd_taxon_map),
        ("coco",      "build/merge the COCO JSON (needs images; WoRMS if uncached)", cmd_coco),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="path to the YAML config")
        p.add_argument("--dir", help="folder holding the export .txt files (basename match)")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()

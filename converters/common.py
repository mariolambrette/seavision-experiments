#!/usr/bin/env python3
"""
Shared machinery for SeaVision source converters.

Every converter imports from here. Source-specific parsing lives in the
converter; nothing source-specific belongs in this file.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import yaml

SCHEMA_VERSION = "1.1"
WORMS = "https://www.marinespecies.org/rest"
WORMS_UA = "SeaVision-collation/1.1 (University of Exeter)"
LINEAGE_RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

BASE_REQUIRED_KEYS = [
    "name", "uid_prefix", "source_root", "output_json", "output_image_dir",
    "taxon_map_csv", "review_csv", "dataset_meta", "crop_provenance",
    "pixel_scale_known", "gear",
]
REQUIRED_DATASET_META = ["id", "name", "license_id"]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys instead of silently
    keeping the last one - which is how a config loses half its content."""


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


def load_config(path, extra_required=()):
    with open(path, encoding="utf-8") as fh:
        try:
            cfg = yaml.load(fh, Loader=_StrictLoader)
        except yaml.constructor.ConstructorError as exc:
            sys.exit(f"config {path}: {exc}")

    missing = [k for k in list(BASE_REQUIRED_KEYS) + list(extra_required)
               if k not in cfg]
    if missing:
        sys.exit(f"config {path} is missing required keys: {missing}")

    missing = [k for k in REQUIRED_DATASET_META if k not in cfg["dataset_meta"]]
    if missing:
        sys.exit(f"config {path}: dataset_meta is missing {missing}")

    lic_ids = {l["id"] for l in cfg.get("licenses", [])}
    if cfg["dataset_meta"]["license_id"] not in lic_ids | {None}:
        sys.exit(f"config {path}: dataset_meta.license_id "
                 f"{cfg['dataset_meta']['license_id']} is not in licenses")

    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=here, text=True).strip() != ""
        return {"commit": commit, "dirty": dirty}
    except Exception:                                   # noqa: BLE001
        return {"commit": None, "dirty": None}


# --------------------------------------------------------------------------
# WoRMS
# --------------------------------------------------------------------------

def _requests():
    import requests
    return requests


def worms_json(url, params=None, tries=4, timeout=60):
    """GET and parse JSON with retries. None for 'no record' or persistent
    failure. A non-JSON body is retried, not raised - an intercepting proxy or
    an error page should not surface as a JSONDecodeError."""
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
        except Exception:                               # noqa: BLE001
            print(f"    ! WoRMS request failed ({attempt}/{tries}) {url}")
        if attempt < tries:
            time.sleep(delay)
            delay *= 2
    return None


def worms_by_name(name):
    records = worms_json(f"{WORMS}/AphiaRecordsByName/{name}",
                         params={"like": "false", "marine_only": "true"})
    if not records:
        return None
    accepted = [r for r in records if r.get("status") == "accepted"]
    pool = accepted or records
    if len(pool) != 1:
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
    """Cached. Lineages are stable reference data; making the build depend on a
    live service was the original fault."""
    key = str(aphia_id)
    if cache is not None and key in cache:
        e = cache[key]
        return e["rank"], e["valid_name"], e["lineage"]

    rec = worms_json(f"{WORMS}/AphiaRecordByAphiaID/{aphia_id}")
    cls = worms_json(f"{WORMS}/AphiaClassificationByAphiaID/{aphia_id}")
    if rec is None or cls is None:
        raise RuntimeError(
            f"WoRMS lookup failed for AphiaID {aphia_id} - the service is "
            f"unreachable, or that ID does not exist. Cached taxa are skipped "
            f"on re-run.")

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


def ensure_categories(coco, taxon_map, cache_path):
    """Add a category for every AphiaID in the taxon map that is not already
    present. The cache is saved even if a fetch fails part way, so a re-run
    resumes rather than restarting."""
    cat_ids = {c["id"] for c in coco["categories"]}
    cache = load_lineage_cache(cache_path)
    try:
        for aid in sorted({a for a in taxon_map.values() if a}):
            if aid in cat_ids:
                continue
            was_cached = str(aid) in cache
            rank, valid, lineage = worms_lineage(aid, cache)
            if not was_cached:
                print(f"  fetched lineage for {aid} ({valid})")
                time.sleep(0.3)
            coco["categories"].append({
                "id": aid, "name": valid, "rank": rank,
                "supercategory": lineage.get("family", ""),
                "aphia_id": aid, "lineage": lineage,
            })
            cat_ids.add(aid)
    finally:
        save_lineage_cache(cache_path, cache)


# --------------------------------------------------------------------------
# Shared dataset helpers
# --------------------------------------------------------------------------

def norm(value):
    if value is None:
        return ""
    v = str(value).strip()
    return "" if v.upper() in ("", "NA", "NAN", "NULL", "NONE") else v


def uid_for(source_path, source_root, uid_prefix):
    """UID from the path RELATIVE to source_root, so moving the dataset does not
    change it, and identical filenames in different folders stay distinct."""
    rel = os.path.relpath(source_path, source_root).replace("\\", "/")
    return f"{uid_prefix}-{hashlib.sha256(rel.encode('utf-8')).hexdigest()[:10]}"


def load_taxon_map(path):
    mapping = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (norm(row["family"]), norm(row["genus"]), norm(row["species"]))
            aid = row.get("aphia_id", "").strip()
            mapping[key] = int(aid) if aid else None
    return mapping


def load_or_init_coco(output_json):
    if os.path.exists(output_json):
        with open(output_json, encoding="utf-8") as fh:
            coco = json.load(fh)
        state = {
            "next_img": max((i["id"] for i in coco["images"]), default=0) + 1,
            "next_ann": max((a["id"] for a in coco["annotations"]), default=0) + 1,
            "seen_uid": {os.path.splitext(i["file_name"])[0] for i in coco["images"]},
            "ds_ids": {d["id"] for d in coco["datasets"]},
        }
    else:
        coco = {
            "info": {"schema_version": SCHEMA_VERSION,
                     "description": "SeaVision unified detection dataset"},
            "licenses": [], "datasets": [], "categories": [],
            "images": [], "annotations": [],
        }
        state = {"next_img": 1, "next_ann": 1, "seen_uid": set(), "ds_ids": set()}
    return coco, state


def register_source(coco, cfg, state):
    for lic in cfg.get("licenses", []):
        if lic["id"] not in {l["id"] for l in coco["licenses"]}:
            coco["licenses"].append(dict(lic))
    if cfg["dataset_meta"]["id"] not in state["ds_ids"]:
        coco["datasets"].append(dict(cfg["dataset_meta"]))


def write_review(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "source_path", "reason"])
        w.writerows(rows)


def write_manifest(cfg, coco, n_review, extra=None):
    manifest = {
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "config_path": cfg["_config_path"],
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "counts": {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "categories": len(coco["categories"]),
            "review_items": n_review,
        },
    }
    if extra:
        manifest["counts"].update(extra)
    path = os.path.join(os.path.dirname(cfg["output_json"]) or ".",
                        "build_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest: {path}"
          + ("  [WARNING: uncommitted changes]" if manifest["git"]["dirty"] else ""))


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src, dst):
    import shutil
    shutil.copy2(src, dst)

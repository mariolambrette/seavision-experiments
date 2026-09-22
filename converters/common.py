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
from collections import defaultdict

import yaml

SCHEMA_VERSION = "1.2"
WORMS = "https://www.marinespecies.org/rest"
WORMS_UA = "SeaVision-collation/1.2 (University of Exeter)"
LINEAGE_RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

BASE_REQUIRED_KEYS = [
    "name", "uid_prefix", "source_root", "output_json", "output_image_dir",
    "taxon_map_csv", "review_csv", "dataset_meta", "crop_provenance",
    "pixel_scale_known", "gear",
]
REQUIRED_DATASET_META = ["id", "name", "license_id"]

# Statuses a machine must not resolve on its own. Kept deliberately identical
# to the set used by the WP6a migration script -- if the two ever disagree,
# the converter and the migration will resolve the same taxon differently and
# a rebuild will silently diverge from the migrated collation.
NEEDS_REVIEW = {"alternate representation", "nomen dubium",
                "taxon inquirendum", "uncertain", "interim unpublished"}

# Dispositions returned by resolve_accepted().
ACCEPTED = "accepted"               # status is accepted; use its own id
REMAPPED = "remapped"               # followed valid_AphiaID to an accepted id
KEPT = "kept_no_replacement"        # unaccepted, but WoRMS offers nothing else
REVIEW = "review"                   # a person decides; do NOT build a category

USABLE = (ACCEPTED, REMAPPED, KEPT)


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

class WormsUnavailable(RuntimeError):
    """WoRMS could not be reached, as distinct from WoRMS having no record.

    These two must never collapse into one return value. If they do, an
    outage mid-build looks exactly like 'this taxon does not exist', the whole
    taxon map goes to review, and the build produces a collation with no
    categories and a review CSV blaming WoRMS for not having common species.
    A build should stop instead, and be re-run when the service is back.
    """


def _requests():
    import requests
    return requests


def worms_json(url, params=None, tries=4, timeout=60):
    """GET and parse JSON with retries.

    Returns None ONLY when WoRMS answers definitively that there is no record
    (204/404). Raises WormsUnavailable when the service could not be reached
    or kept answering unusably. A non-JSON body is retried rather than raised
    as a JSONDecodeError - an intercepting proxy or an error page should not
    surface as a parse error.
    """
    requests = _requests()
    delay, last = 2, "no attempt made"
    for attempt in range(1, tries + 1):
        try:
            resp = requests.get(
                url, params=params, timeout=timeout,
                headers={"User-Agent": WORMS_UA, "Accept": "application/json"})
            if resp.status_code in (204, 404):
                return None                     # definitive: no such record
            if resp.status_code != 200:
                last = f"HTTP {resp.status_code}"
                print(f"    ! WoRMS {last} ({attempt}/{tries}) {url}")
            else:
                try:
                    return resp.json()
                except ValueError:
                    last = f"non-JSON {resp.headers.get('content-type')}"
                    print(f"    ! WoRMS {last} ({attempt}/{tries}): "
                          f"{resp.text[:120]!r}")
        except Exception as exc:                        # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            print(f"    ! WoRMS request failed ({attempt}/{tries}) {url}")
        if attempt < tries:
            time.sleep(delay)
            delay *= 2
    raise WormsUnavailable(
        f"WoRMS unreachable after {tries} attempts ({last}): {url}. This is "
        f"NOT the same as the taxon not existing - re-run when the service is "
        f"back rather than accepting a build made without it.")


def _cached_record(aphia_id, cache):
    """AphiaRecordByAphiaID, cached under a 'rec:' key so it shares the lineage
    cache file without colliding with lineage entries. Only the fields the
    resolver needs are stored, so the cache does not balloon."""
    key = f"rec:{aphia_id}"
    if cache is not None and key in cache:
        return cache[key]
    rec = worms_json(f"{WORMS}/AphiaRecordByAphiaID/{aphia_id}")
    if rec is None:
        return None
    slim = {"AphiaID": rec.get("AphiaID"),
            "status": rec.get("status"),
            "valid_AphiaID": rec.get("valid_AphiaID"),
            "valid_name": rec.get("valid_name"),
            "scientificname": rec.get("scientificname"),
            "rank": rec.get("rank"),
            "fetched": datetime.date.today().isoformat()}
    if cache is not None:
        cache[key] = slim
    return slim


def resolve_accepted(aphia_id, cache=None, depth=5):
    """Resolve an AphiaID to the id a category should actually be built under.

    Returns (disposition, resolved_id, origin_status, note).

        ACCEPTED  status is accepted; resolved_id == aphia_id
        REMAPPED  followed valid_AphiaID to an accepted record
        KEPT      unaccepted, but WoRMS offers no replacement, so the original
                  id is the best available identity. resolved_id == aphia_id
        REVIEW    a person must decide; resolved_id is None

    Two things here are deliberate and were got wrong before:

    *   KEPT exists so that an unaccepted taxon with no valid_AphiaID (e.g.
        Thecosomata) keeps its crops instead of being dropped to review. A fix
        for a data-integrity fault must not quietly lose data.
    *   NEEDS_REVIEW stops on the ORIGIN's status, not the final record's. An
        alternate representation is a real taxonomic decision, not a redirect,
        and following it would make the machine choose.
    """
    seen, cur, origin_status = [], int(aphia_id), None
    for _ in range(depth):
        rec = _cached_record(cur, cache)
        if rec is None:
            return REVIEW, None, origin_status, f"no WoRMS record for AphiaID {cur}"

        status = str(rec.get("status") or "").strip().lower()
        valid = rec.get("valid_AphiaID")
        has_choice = bool(valid) and int(valid) != cur

        if origin_status is None:
            origin_status = status
            # A doubtful status only needs a person when WoRMS actually offers
            # an alternative. 'taxon inquirendum' with nowhere to go presents
            # no decision to make, and sending it to review does not buy a
            # judgement -- it just drops the category and its crops.
            if status in NEEDS_REVIEW and has_choice:
                return (REVIEW, None, status,
                        f"status '{status}' with a replacement offered "
                        f"({valid}) needs a human decision")

        if status == "accepted":
            acc = int(rec["AphiaID"])
            disp = ACCEPTED if acc == int(aphia_id) else REMAPPED
            return disp, acc, origin_status, ""

        if not has_choice:
            # Unaccepted with nowhere to go. Keep the original identity.
            return (KEPT, int(aphia_id), origin_status,
                    f"status '{status}' with no replacement offered")

        seen.append(cur)
        if int(valid) in seen:
            return REVIEW, None, origin_status, "circular valid_AphiaID"
        cur = int(valid)

    return (REVIEW, None, origin_status,
            f"valid_AphiaID chain deeper than {depth}")


def worms_by_name(name):
    """Exact-name lookup, resolved to the id a category would be built under.

    marine_only is false: every taxon here is marine by construction, and
    like=false plus the single-match requirement already sends a homonym to
    review. Setting it true only adds a way to fail. It also matches the WP6a
    migration script, which must resolve names identically.
    """
    records = worms_json(f"{WORMS}/AphiaRecordsByName/{name}",
                         params={"like": "false", "marine_only": "false"})
    if not records:
        return None
    exact = [r for r in records
             if str(r.get("scientificname", "")).strip() == str(name).strip()]
    if not exact:
        return None
    accepted = [r for r in exact if r.get("status") == "accepted"]
    pool = accepted or exact
    if len(pool) != 1:
        return None                            # ambiguous: a person decides

    disp, rid, _status, _note = resolve_accepted(pool[0]["AphiaID"])
    if disp not in USABLE:
        return None
    rec = _cached_record(rid, None)
    if rec is None:
        return None
    return rid, rec.get("valid_name") or rec.get("scientificname"), rec.get("rank")


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


def _key_label(key):
    """Taxon-map keys are (family, genus, species) triples in three of the
    converters and plain concept strings in the FathomNet one. Joining a bare
    string character by character is not a helpful review row."""
    if isinstance(key, (tuple, list)):
        return "|".join(str(k) for k in key)
    return str(key)


def resolve_taxon_map(taxon_map, cache_path, verbose=True):
    """Rewrite every AphiaID in the taxon map to the id a category should be
    built under. MUST run before ensure_categories.

    Skipping this step is what let one species enter the collation twice: the
    category id came from the taxon map (unaccepted) while the category name
    came from valid_name (accepted), so two ids carried one name and the
    annotations split between them.

    Returns (resolved, changes, review, merges). Two map rows collapsing onto
    one id is EXPECTED - that is the duplication being removed at source - so
    it is reported, not treated as an error.
    """
    cache = load_lineage_cache(cache_path)
    resolved, changes, review = {}, [], []
    try:
        for key, aid in taxon_map.items():
            if not aid:
                resolved[key] = aid
                continue
            disp, rid, status, note = resolve_accepted(aid, cache)
            if disp not in USABLE:
                resolved[key] = None
                review.append((_key_label(key), f"aphia_id={aid}",
                               f"unresolved taxonomy ({status}): {note}"))
                continue
            resolved[key] = rid
            if disp == REMAPPED:
                changes.append((key, int(aid), rid, status))
            elif disp == KEPT:
                changes.append((key, int(aid), rid, f"{status} (kept)"))
    finally:
        save_lineage_cache(cache_path, cache)

    by_id = defaultdict(list)
    for key, aid in resolved.items():
        if aid:
            by_id[aid].append(key)
    merges = {a: ks for a, ks in by_id.items() if len(ks) > 1}

    if verbose:
        n_remap = sum(1 for c in changes if not str(c[3]).endswith("(kept)"))
        print(f"  taxon map: {n_remap} id(s) remapped to accepted, "
              f"{len(changes) - n_remap} kept without replacement, "
              f"{len(merges)} id(s) now reached by >1 map row, "
              f"{len(review)} sent to review")
        for key, old, new, status in changes:
            arrow = "==" if old == new else "->"
            print(f"    {_key_label(key)}: {old} ({status}) {arrow} {new}")
        for aid, keys in sorted(merges.items()):
            print(f"    merge onto {aid}: " +
                  "; ".join(_key_label(k) for k in keys))
    return resolved, changes, review, merges


def ensure_categories(coco, taxon_map, cache_path, strict=True):
    """Add a category for every AphiaID in the taxon map that is not already
    present. The taxon map MUST already have been through resolve_taxon_map.
    """
    cat_ids = {c["id"] for c in coco["categories"]}
    cache = load_lineage_cache(cache_path)
    try:
        for aid in sorted({a for a in taxon_map.values() if a}):
            if aid in cat_ids:
                continue
            # Belt and braces: refuse an id that resolve_taxon_map would have
            # changed, even if the caller forgot to call it. The absence of
            # this check is what produced the WP6a duplicate-species fault.
            disp, rid, status, note = resolve_accepted(aid, cache)
            if disp not in USABLE or rid != aid:
                sys.exit(f"ensure_categories: AphiaID {aid} resolves to "
                         f"{disp}/{rid} (status '{status}'; {note}). Run "
                         f"resolve_taxon_map on the taxon map first.")
            was_cached = str(aid) in cache
            rank, valid, lineage = worms_lineage(aid, cache)
            if not was_cached:
                print(f"  fetched lineage for {aid} ({valid})")
                time.sleep(0.3)
            coco["categories"].append({
                "id": aid, "name": valid, "rank": rank,
                "supercategory": lineage.get("family", ""),
                "aphia_id": aid, "lineage": lineage,
                "worms_status": status,
            })
            cat_ids.add(aid)
    finally:
        save_lineage_cache(cache_path, cache)

    check_unique_names(coco, strict=strict)


def check_unique_names(coco, strict=True):
    """One name, one category.

    Two ids sharing a name IS the WP6a fault. It should stop a build rather
    than be found months later by an audit.
    """
    names = defaultdict(list)
    for c in coco["categories"]:
        names[str(c.get("name", "")).strip().lower()].append(c["id"])
    dupes = {n: sorted(ids) for n, ids in names.items() if len(ids) > 1}
    if not dupes:
        return {}
    msg = "; ".join(f"{n} -> {ids}" for n, ids in sorted(dupes.items())[:10])
    more = "" if len(dupes) <= 10 else f" (+{len(dupes) - 10} more)"
    if strict:
        sys.exit(f"duplicate category name(s): {len(dupes)} found: {msg}{more}")
    print(f"  ! WARNING: {len(dupes)} duplicate category name(s): {msg}{more}")
    return dupes


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
    """Taxon map, refusing duplicate keys.

    Two rows with the same (family, genus, species) and different aphia_ids
    used to overwrite silently, so which id won depended on row order.
    """
    mapping, seen = {}, {}
    with open(path, newline="", encoding="utf-8") as fh:
        for n, row in enumerate(csv.DictReader(fh), start=2):
            key = (norm(row["family"]), norm(row["genus"]), norm(row["species"]))
            aid = row.get("aphia_id", "").strip()
            val = int(aid) if aid else None
            if key in mapping and mapping[key] != val:
                sys.exit(f"{path} line {n}: duplicate taxon key {key} with a "
                         f"different aphia_id ({mapping[key]} vs {val}); "
                         f"first seen on line {seen[key]}")
            mapping[key], seen[key] = val, n
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
    ds_id = cfg["dataset_meta"]["id"]
    if ds_id not in state["ds_ids"]:
        coco["datasets"].append(dict(cfg["dataset_meta"]))
        state["ds_ids"].add(ds_id)      # was missing: a second call in one run
                                        # appended the same dataset twice


def write_review(path, rows):
    """Always written, even when empty.

    Previously a run with nothing to review wrote no file, so the PREVIOUS
    run's CSV stayed on disk and read as current.
    """
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
    src = cfg.get("name") or cfg["dataset_meta"]["name"]
    path = os.path.join(os.path.dirname(cfg["output_json"]) or ".",
                        f"build_manifest_{src}.json")
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

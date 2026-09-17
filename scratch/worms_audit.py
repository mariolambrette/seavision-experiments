#!/usr/bin/env python
"""
Audit every category_id in the collation against WoRMS.

Read-only. Touches no dataset file. Produces a remap CSV for review.

The fault it exists to find: the converter resolved source labels to the
WoRMS *accepted name* but kept the *original* AphiaID as category_id, and
fetched the lineage for that original ID. So a category can carry an accepted
name, an unaccepted identity, and the classification of the taxon it was
supposed to have been merged into. Where the accepted name happened to collide
with a category already present under its accepted ID, that shows up as a
duplicate name. Where it did not collide, nothing shows up at all -- which is
why this audit asks WoRMS directly rather than looking for symptoms.

One call to AphiaRecordByAphiaID gives status, valid_AphiaID, valid_name and
genus, so one lookup per category settles it.

Responses are cached to JSON and the cache is written in a `finally` block, so
an interrupted run resumes rather than restarting -- the same pattern as the
lineage cache (WP1).

Usage (fast path, reads the WP6 category table):

    python worms_audit.py ^
        --categories-csv "N:/marineai/dataset/collated/logs/wp6_categories.csv" ^
        --out "D:/marineai/classification-experiments/sw/taxon_maps/worms_remap.csv" ^
        --cache "D:/marineai/classification-experiments/sw/taxon_maps/worms_record_cache.json"

Or straight from the COCO files (authoritative, slower):

    python worms_audit.py --coco "...seavision.json" --coco "...seavision_fathomnet.json" ...
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

try:
    import requests
except ImportError:
    sys.exit("requests is required: conda install requests")

REST = "https://www.marinespecies.org/rest/AphiaRecordByAphiaID/{}"

# statuses where valid_AphiaID pointing elsewhere is NOT automatically a
# remap -- these need a human decision, so they are reported, never actioned
NEEDS_REVIEW = {"alternate representation", "nomen dubium", "taxon inquirendum",
                "uncertain", "interim unpublished"}


# --------------------------------------------------------------------------

def load_cache(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            print(f"  ! cache unreadable ({e}); starting empty")
    return {}


def save_cache(cache, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def fetch_record(session, aphia_id, cache, delay):
    """WoRMS record for one AphiaID, cached. None means 'no record'."""
    key = str(aphia_id)
    if key in cache:
        return cache[key]

    url = REST.format(aphia_id)
    last = None
    for attempt in range(1, 5):
        try:
            r = session.get(url, timeout=30)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 204:                 # no content = no such record
            cache[key] = None
            return None
        if r.status_code == 200:
            try:
                rec = r.json()
            except ValueError:
                last = "non-JSON body"
                time.sleep(2 ** attempt)
                continue
            keep = {k: rec.get(k) for k in
                    ("AphiaID", "scientificname", "rank", "status",
                     "unacceptreason", "valid_AphiaID", "valid_name",
                     "genus", "family", "order", "class", "phylum", "kingdom")}
            cache[key] = keep
            time.sleep(delay)
            return keep
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}"
            time.sleep(2 ** attempt)
            continue
        last = f"HTTP {r.status_code}"
        break

    raise RuntimeError(f"AphiaID {aphia_id}: {last}")


def resolve_chain(session, aphia_id, cache, delay, depth=5):
    """Follow valid_AphiaID until it settles.

    Returns (origin_rec, final_rec, hops, note). The ORIGIN record is returned
    alongside the final one because the decision of whether a remap is safe
    depends on the origin's status, not the destination's -- an 'alternate
    representation' points at an accepted record, and reading only the
    destination would classify it as a routine rename.
    """
    seen = []
    origin = None
    cur = aphia_id
    for _ in range(depth):
        rec = fetch_record(session, cur, cache, delay)
        if rec is None:
            return origin, None, len(seen), "no WoRMS record"
        if origin is None:
            origin = rec
        seen.append(cur)
        status = (rec.get("status") or "").strip().lower()
        valid = rec.get("valid_AphiaID")
        if status == "accepted" or not valid or int(valid) == int(cur):
            return origin, rec, len(seen) - 1, ""
        if int(valid) in seen:
            return origin, rec, len(seen) - 1, "circular valid_AphiaID"
        cur = int(valid)
    return origin, rec, len(seen) - 1, "chain longer than depth limit"


# --------------------------------------------------------------------------

def categories_from_csv(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cid = int(row["category_id"])
            out[cid] = {
                "name": row.get("name", ""),
                "rank": row.get("rank", ""),
                "genus": row.get("genus", ""),
                "n_crops": int(row.get("n_crops") or 0),
                "sources": row.get("sources", ""),
            }
    return out


def categories_from_coco(paths):
    out = {}
    for p in paths:
        print(f"  loading {os.path.basename(p)} ...", flush=True)
        with open(p, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        counts = Counter(a.get("category_id") for a in doc.get("annotations", []))
        src = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        for c in doc.get("categories", []):
            cid = c["id"]
            e = out.setdefault(cid, {"name": c.get("name", ""),
                                     "rank": c.get("rank", ""),
                                     "genus": "", "n_crops": 0, "sources": ""})
            lin = c.get("lineage") or {}
            if not e["genus"] and isinstance(lin, dict):
                for k, v in lin.items():
                    if str(k).strip().lower() == "genus" and v:
                        e["genus"] = str(v)
            e["n_crops"] += counts.get(cid, 0)
        del doc
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="WoRMS category audit (read-only)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--categories-csv", help="wp6_categories.csv (fast)")
    g.add_argument("--coco", action="append", help="COCO file; repeat")
    ap.add_argument("--out", required=True, help="remap CSV to write")
    ap.add_argument("--cache", help="WoRMS record cache JSON")
    ap.add_argument("--delay", type=float, default=0.12,
                    help="seconds between live lookups (default 0.12)")
    args = ap.parse_args()

    print("collecting categories")
    cats = (categories_from_csv(args.categories_csv) if args.categories_csv
            else categories_from_coco(args.coco))
    print(f"  {len(cats):,} categories")

    present = set(cats)                       # ids already in the collation
    cache = load_cache(args.cache)
    print(f"  {len(cache):,} records cached")

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    rows, errors = [], []
    tally = Counter()

    try:
        for i, cid in enumerate(sorted(cats), 1):
            if i % 250 == 0:
                print(f"  {i:,}/{len(cats):,} ...", flush=True)
                save_cache(cache, args.cache)

            info = cats[cid]
            try:
                origin, rec, hops, note = resolve_chain(
                    session, cid, cache, args.delay)
            except Exception as e:
                errors.append((cid, info["name"], str(e)))
                tally["lookup_failed"] += 1
                continue

            # status of the ORIGINAL id -- this is what decides the action
            origin_status = ((origin or {}).get("status") or "no record").strip()
            origin_name = (origin or {}).get("scientificname") or ""
            origin_reason = (origin or {}).get("unacceptreason") or ""

            if rec is None:
                acc_id, acc_name, acc_rank, acc_genus = "", "", "", ""
                action = "REVIEW"
                tally["no_record"] += 1
            else:
                acc_id = rec.get("AphiaID") or ""
                acc_name = rec.get("scientificname") or ""
                acc_rank = rec.get("rank") or ""
                acc_genus = rec.get("genus") or ""

                if hops == 0 and origin_status.lower() == "accepted":
                    action = "ok"
                    tally["ok"] += 1
                elif origin_status.lower() in NEEDS_REVIEW or note:
                    action = "REVIEW"
                    tally["review"] += 1
                elif (origin_status.lower() == "unaccepted"
                      and acc_id and int(acc_id) != int(cid)):
                    if int(acc_id) in present:
                        action = "merge"
                        tally["merge"] += 1
                    else:
                        action = "rename"
                        tally["rename"] += 1
                else:
                    action = "REVIEW"
                    tally["review"] += 1

            rows.append({
                "category_id": cid,
                "current_name": info["name"],
                "current_rank": info["rank"],
                "current_genus": info["genus"],
                "n_crops": info["n_crops"],
                "sources": info["sources"],
                "worms_name_for_this_id": origin_name,
                "worms_status": origin_status,
                "unaccept_reason": origin_reason,
                "hops": hops if rec is not None else "",
                "accepted_id": acc_id,
                "accepted_name": acc_name,
                "accepted_rank": acc_rank,
                "accepted_genus": acc_genus,
                "target_already_present": (
                    int(acc_id) in present if str(acc_id).isdigit() else ""),
                "action": action,
                "note": note,
            })
    finally:
        save_cache(cache, args.cache)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["action"] == "ok", -r["n_crops"])):
            w.writerow(r)

    # ---- summary ---------------------------------------------------------
    todo = [r for r in rows if r["action"] in ("merge", "rename", "REVIEW")]
    crops_affected = sum(r["n_crops"] for r in todo)

    print()
    print("=" * 68)
    print("WoRMS AUDIT")
    print("=" * 68)
    print(f"  categories checked   {len(rows):,}")
    print(f"  already accepted     {tally['ok']:,}")
    print(f"  need MERGE           {tally['merge']:,}   "
          f"(accepted id already a category)")
    print(f"  need RENAME          {tally['rename']:,}   "
          f"(accepted id not yet present)")
    print(f"  need REVIEW          {tally['review'] + tally['no_record']:,}")
    print(f"  lookup failed        {tally['lookup_failed']:,}")
    print()
    print(f"  crops in affected categories: {crops_affected:,}")

    if todo:
        print()
        print("  largest affected categories:")
        for r in sorted(todo, key=lambda r: -r["n_crops"])[:15]:
            print(f"    {r['action']:<7}{r['n_crops']:>8,}  "
                  f"{r['category_id']:<9}{r['current_name'][:32]:<34}"
                  f"-> {r['accepted_id']} {r['accepted_name'][:28]}")

    if errors:
        print()
        print(f"  !! {len(errors)} lookups failed -- re-run to retry "
              f"(the cache keeps what succeeded):")
        for cid, nm, e in errors[:10]:
            print(f"     {cid} {nm}: {e}")

    print()
    print(f"  written: {args.out}")
    print("  nothing has been modified. Review the CSV before migrating.")


if __name__ == "__main__":
    main()

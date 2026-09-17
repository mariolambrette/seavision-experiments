#!/usr/bin/env python
"""
Migrate the collation onto accepted WoRMS AphiaIDs.

Metadata only. Touches `categories` and `annotations[].category_id` and
nothing else -- image records, image UIDs, file names and every crop file on
disk are left exactly as they are.

Driven by the `action` column of the remap CSV produced by worms_audit.py:

    remap | merge | rename   ->  point this category at its accepted AphiaID
    skip                     ->  leave it alone (a recorded decision)
    ok                       ->  nothing to do
    REVIEW                   ->  refuses to run; decide it first

merge and rename mean the same thing. Which one actually happens is decided
per file at migration time, because a target can be present in one COCO file
and absent from the other:

    target present in this file  ->  MERGE  (annotations repointed, old
                                             category dropped, class count -1)
    target absent from this file ->  RENAME (category's identity, name, rank
                                             and lineage replaced; count same)

Lineage is rebuilt from the accepted record, and **every rank in it is
resolved to its own accepted record** -- WoRMS can hold an accepted species
inside an unaccepted genus (Turrum gymnostethus is accepted; the genus Turrum
resolves to Carangoides), and taking the record at face value would write a
genus the authority rejects. Substitution is conservative: a name is replaced
only on an unambiguous single match at the expected rank in the same kingdom.
Anything ambiguous is left alone and reported.

Dry run by default. Nothing is written without --apply.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict

try:
    import requests
except ImportError:
    sys.exit("requests is required: conda install requests")

REC_BY_ID = "https://www.marinespecies.org/rest/AphiaRecordByAphiaID/{}"
# like=false forces an EXACT name match. The default is fuzzy, which returns
# near-spellings (a query for "Naso" also returns "Nasonia") and would let a
# single fuzzy hit be mistaken for an unambiguous answer.
REC_BY_NAME = ("https://www.marinespecies.org/rest/AphiaRecordsByName/{}"
               "?like=false&marine_only=false")

DO = {"remap", "merge", "rename"}
LINEAGE_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]


# ---------------------------------------------------------------- WoRMS ---

def cache_load(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def cache_save(cache, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _get(session, url, delay):
    for attempt in range(1, 5):
        try:
            r = session.get(url, timeout=30)
        except Exception:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 204:
            return None
        if r.status_code == 200:
            try:
                out = r.json()
            except ValueError:
                time.sleep(2 ** attempt)
                continue
            time.sleep(delay)
            return out
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"{url}: HTTP {r.status_code}")
    raise RuntimeError(f"{url}: gave up after 4 attempts")


def record_by_id(session, aid, cache, delay):
    key = str(aid)
    if key in cache:
        return cache[key]
    rec = _get(session, REC_BY_ID.format(aid), delay)
    if rec is not None:
        rec = {k: rec.get(k) for k in
               ("AphiaID", "scientificname", "rank", "status", "unacceptreason",
                "valid_AphiaID", "valid_name", "genus", "family", "order",
                "class", "phylum", "kingdom")}
    cache[key] = rec
    return rec


def accepted_name_for(session, name, rank, kingdom, cache, delay, notes):
    """Resolve one lineage rank name to its accepted spelling.

    Conservative: substitutes only when exactly one WoRMS record matches the
    name at the expected rank within the expected kingdom. Ambiguity, absence
    or homonymy leaves the name untouched and records why.
    """
    key = f"nm2:{rank}:{name}"          # nm2: entries below predate like=false
    if key in cache:
        hit = cache[key]
    else:
        try:
            recs = _get(session, REC_BY_NAME.format(name), delay) or []
        except Exception as e:
            notes.append(f"{name} ({rank}): lookup failed, left unchanged ({e})")
            cache[key] = None
            return name
        cand = [r for r in recs
                if str(r.get("scientificname", "")).strip() == str(name).strip()
                and str(r.get("rank", "")).strip().lower() == rank.lower()
                and (not kingdom or str(r.get("kingdom", "")).strip().lower()
                     == str(kingdom).strip().lower())]
        if len(cand) != 1:
            notes.append(f"{name} ({rank}): {len(cand)} candidate records, "
                         f"left unchanged")
            cache[key] = None
            return name
        r = cand[0]
        hit = {"status": r.get("status"), "AphiaID": r.get("AphiaID"),
               "valid_name": r.get("valid_name"),
               "valid_AphiaID": r.get("valid_AphiaID")}
        cache[key] = hit

    if hit is None:
        return name
    if str(hit.get("status", "")).strip().lower() == "accepted":
        return name
    new = hit.get("valid_name")
    if new and new != name:
        notes.append(f"{name} ({rank}) -> {new}  "
                     f"[{hit.get('AphiaID')} {hit.get('status')} "
                     f"-> {hit.get('valid_AphiaID')}]")
        return new
    return name


def build_category(session, aid, cache, delay, subs):
    """The category record for an accepted AphiaID, lineage ranks resolved."""
    rec = record_by_id(session, aid, cache, delay)
    if rec is None:
        raise RuntimeError(f"AphiaID {aid}: no WoRMS record")

    kingdom = rec.get("kingdom")
    lineage = {}
    for rk in LINEAGE_RANKS:
        val = rec.get(rk)
        if val:
            lineage[rk] = accepted_name_for(session, val, rk, kingdom,
                                            cache, delay, subs)
    own = str(rec.get("rank") or "").strip().lower()
    if own:
        lineage[own] = rec.get("scientificname")

    cat = {
        "id": int(aid),
        "name": rec.get("scientificname"),
        "rank": rec.get("rank"),
        "supercategory": lineage.get("family") or rec.get("scientificname"),
        "aphia_id": int(aid),
        "lineage": lineage,
    }
    return cat


# --------------------------------------------------------------- remap ----

def load_remap(path):
    """Rows to act on, rows still undecided, and rows that point at themselves.

    A row whose accepted_id equals its category_id is a no-op whatever its
    action says -- WoRMS marks the taxon unaccepted or doubtful but offers no
    replacement, so there is nowhere to remap it to. These are reported and
    then ignored, rather than being mistaken for a one-step cycle.
    """
    rows, undecided, selfpoint = [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            act = (r.get("action") or "").strip().lower()
            if act == "review":
                undecided.append(r)
                continue
            if act not in DO:
                continue
            acc = (r.get("accepted_id") or "").strip()
            if not acc.isdigit():
                raise SystemExit(
                    f"row {r.get('category_id')} is '{act}' but has no "
                    f"accepted_id")
            if int(acc) == int(r["category_id"]):
                selfpoint.append(r)
                continue
            rows.append((int(r["category_id"]), int(acc), r))
    return rows, undecided, selfpoint


def resolve_chains(pairs):
    """Collapse A->B->C to A->C. Raises on a cycle."""
    direct = {a: b for a, b, _ in pairs if a != b}
    out = {}
    for a in direct:
        seen, cur = [a], direct[a]
        while cur in direct and cur != direct[cur]:
            if cur in seen:
                raise SystemExit(f"cycle in remap: {seen + [cur]}")
            seen.append(cur)
            cur = direct[cur]
        out[a] = cur
    return out


# ------------------------------------------------------------- migrate ----

def migrate_file(path, remap, newcats, apply_, backup_dir):
    print(f"\n--- {os.path.basename(path)} ---")
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    n_img = len(doc.get("images", []))
    n_ann = len(doc.get("annotations", []))
    ann_ids = {a["id"] for a in doc.get("annotations", [])}
    cats = {c["id"]: c for c in doc.get("categories", [])}
    n_cat = len(cats)

    here = {old: new for old, new in remap.items() if old in cats}
    if not here:
        print("  no affected categories; unchanged")
        return None

    merges = {o: n for o, n in here.items() if n in cats and n != o}
    renames = {o: n for o, n in here.items() if n not in cats}
    selfsame = {o: n for o, n in here.items() if n == o}

    print(f"  merges  {len(merges)}   renames {len(renames)}"
          + (f"   already-correct {len(selfsame)}" if selfsame else ""))
    for o, n in sorted(merges.items()):
        print(f"    MERGE  {o} {cats[o].get('name')!r} -> {n} "
              f"{cats[n].get('name')!r}")
    for o, n in sorted(renames.items()):
        print(f"    RENAME {o} {cats[o].get('name')!r} -> {n} "
              f"{newcats[n]['name']!r}  genus "
              f"{(cats[o].get('lineage') or {}).get('genus')!r} -> "
              f"{newcats[n]['lineage'].get('genus')!r}")

    # -- annotations -------------------------------------------------------
    moved = 0
    for a in doc.get("annotations", []):
        cid = a.get("category_id")
        if cid in here and here[cid] != cid:
            a["category_id"] = here[cid]
            moved += 1
    print(f"  annotations repointed: {moved:,}")

    # -- categories --------------------------------------------------------
    out = []
    for cid, c in cats.items():
        if cid in merges:
            continue                         # absorbed into the target
        if cid in renames:
            out.append(newcats[renames[cid]])
            continue
        out.append(c)
    doc["categories"] = sorted(out, key=lambda c: c["id"])

    # -- invariants --------------------------------------------------------
    ok = True
    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}"
              + (f"   {detail}" if detail and not cond else ""))
        ok = ok and cond

    new_ids = {c["id"] for c in doc["categories"]}
    used = {a["category_id"] for a in doc["annotations"]}
    dup_ids = len(doc["categories"]) != len(new_ids)
    dup_names = [n for n, k in Counter(c["name"] for c in doc["categories"]
                                       ).items() if k > 1]

    check("image count unchanged", len(doc["images"]) == n_img)
    check("annotation count unchanged", len(doc["annotations"]) == n_ann)
    check("annotation ids unchanged",
          {a["id"] for a in doc["annotations"]} == ann_ids)
    check("category count == before - merges",
          len(doc["categories"]) == n_cat - len(merges),
          f"{len(doc['categories'])} vs {n_cat - len(merges)}")
    check("no duplicate category ids", not dup_ids)
    check("every annotation category exists", used <= new_ids,
          f"{len(used - new_ids)} dangling")
    check("no duplicate category names", not dup_names,
          f"{dup_names[:5]}")

    if not ok:
        print("  !! invariants failed -- NOT writing this file")
        return False

    if not apply_:
        print("  dry run: not written")
        return True

    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        dst = os.path.join(backup_dir, os.path.basename(path) + ".pre_migrate")
        if not os.path.exists(dst):
            print(f"  backing up -> {dst}")
            shutil.copy2(path, dst)
        else:
            print(f"  backup already exists, keeping it: {dst}")

    info = doc.setdefault("info", {})
    info["description"] = (str(info.get("description", "")).split(" | ")[0]
                           + " | categories migrated to accepted WoRMS AphiaIDs")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    os.replace(tmp, path)
    print("  written")
    return True


def main():
    ap = argparse.ArgumentParser(description="Migrate to accepted AphiaIDs")
    ap.add_argument("--coco", action="append", required=True)
    ap.add_argument("--remap", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--backup-dir")
    ap.add_argument("--delay", type=float, default=0.12)
    ap.add_argument("--apply", action="store_true",
                    help="actually write; omit for a dry run")
    args = ap.parse_args()

    pairs, undecided, selfpoint = load_remap(args.remap)
    if selfpoint:
        print(f"{len(selfpoint)} rows point at themselves (no accepted "
              f"replacement exists) -- left unchanged:")
        for r in selfpoint:
            print(f"  {r['category_id']:<9}{r['current_name']:<34}"
                  f"{r.get('worms_status','')}")
        print()
    if undecided:
        print(f"{len(undecided)} rows still say REVIEW. Decide them first:")
        for r in undecided[:20]:
            print(f"  {r['category_id']:<9}{r['current_name']:<34}"
                  f"{r['worms_status']}")
        sys.exit(1)

    remap = resolve_chains(pairs)
    print(f"{len(remap)} categories to remap")

    cache = cache_load(args.cache)
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    subs, newcats = [], {}
    try:
        for tgt in sorted(set(remap.values())):
            newcats[tgt] = build_category(session, tgt, cache, args.delay, subs)
    finally:
        cache_save(cache, args.cache)

    if subs:
        print(f"\nlineage rank substitutions ({len(subs)}):")
        for s in sorted(set(subs)):
            print(f"  {s}")
    else:
        print("\nno lineage rank substitutions needed")

    results = [migrate_file(p, remap, newcats, args.apply, args.backup_dir)
               for p in args.coco]

    print()
    if any(r is False for r in results):
        sys.exit("one or more files failed their invariants; nothing written "
                 "for those. Restore from backup if a partial write occurred.")
    if not args.apply:
        print("DRY RUN complete. Re-run with --apply to write.")
    else:
        print("Migration applied. Re-validate, then re-run the WP6 summary.")


if __name__ == "__main__":
    main()

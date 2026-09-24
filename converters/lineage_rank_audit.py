#!/usr/bin/env python3
"""The WP6a loose end: audit every lineage in the collation, not just the 37
that were rebuilt.

WHY THIS EXISTS
    WP6a established that every rank in a lineage must resolve to its own
    accepted record, not just the species. It applied that rule to the 37
    categories the migration rebuilt. The other ~2,800 still carry lineages
    built by the old logic, which took each WoRMS record at face value.

    That matters because WoRMS can hold an accepted species inside an
    UNACCEPTED genus. *Turrum gymnostethus* is accepted; the genus *Turrum*
    resolves to *Carangoides*. The old logic writes "genus: Turrum" -- a genus
    the authority rejects. Among the 37 rebuilt, three substitutions were
    needed (Turrum and Ferdauia -> Carangoides, Moolgarda -> Crenimugil), and
    without them the carangids fragmented into six genera of one to three
    species each.

    So anything that groups by genus is exposed: WP16 builds the taxonomy tree
    from lineages; plan §6.4's congener test groups ARE genera; and §5.4's
    "borrow statistics from a taxonomic neighbour" needs congeners correctly
    grouped. A synonym genus splits one group in two or merges two into one,
    and nothing downstream notices.

    On scale, one caution worth keeping in view while reading the output:
    3 of 37 is 8%, but those 37 were SELECTED for being unaccepted species, so
    they are enriched for taxonomic instability. The rate among the
    already-accepted majority is probably lower. 8% is an upper bound on what
    to expect, not an estimate of it.

TWO FAULTS, AND THEY ARE DIFFERENT
    A  unaccepted intermediate rank -- the lineage names a taxon WoRMS no
       longer accepts. Needs the network.
    B  gapped lineage -- a rank is simply missing. The WP7 review found fish
       carrying phylum Chordata with no class at all (Turrum spp., Azurina
       lepidolepis, Ferdauia orthogrammus, Stegastes lacrymatus, and
       `Actinopteri` itself), which makes them unplaceable in a rank-based
       tree. Needs no network, so --offline answers it in seconds.

    Two gap rules, both chosen because they need no guess about where a
    non-core rank sits in the ladder:
      G1  a core rank is absent while a FINER core rank is present
      G2  the category's own rank is a core rank and is absent from its own
          lineage (this is what catches a class-rank category with no class)

EVERYTHING IS WEIGHTED BY CROPS
    Three affected categories is noise; three affected categories holding
    200,000 crops is not. Every table is sorted by crops, not by count.

    python scratch/lineage_rank_audit.py --offline        # fault B only, instant
    python scratch/lineage_rank_audit.py                  # both; takes a while
    python scratch/lineage_rank_audit.py --limit 300      # a pilot, to time it

Interrupting is safe: the name cache is written on the way out and the run
resumes from it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # the repo root, for common
import common as C                                  # noqa: E402

CORE = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
CORE_INDEX = {r: i for i, r in enumerate(CORE)}

# Fault A audits the ranks ABOVE species by default. The species entry in a
# lineage is the category itself, and WP6a already proved all 2,847 category
# AphiaIDs accepted, by ID, with a test that converges on every row of
# worms_remap.csv. Re-checking it here by NAME would be a third of the
# network calls to re-answer a settled question -- and worse than useless,
# because name lookup returns nothing for a homonym, so every species binomial
# shared with an insect or a plant would land in the unresolved list and have
# to be dismissed by hand. Gaps (fault B) are still assessed at every rank.
DEFAULT_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]

# Which ranks to resolve FIRST. Genus decides congener grouping and the shape
# of WP16's tree; kingdom is three names nobody will ever have got wrong.
RANK_PRIORITY = {"genus": 0, "family": 1, "order": 2, "class": 3,
                 "phylum": 4, "kingdom": 5, "species": 6}

DEFAULT_COCO = [
    "D:/marineai/dataset/collated/seavision.json",
    "D:/marineai/dataset/collated/seavision_fathomnet.json",
]
DEFAULT_CACHE = "taxon_maps/lineage_name_cache.json"


# ------------------------------------------------------------------ loading

def load_categories(paths):
    """-> (categories, crops_per_category)

    Crops are counted from the annotations rather than assumed, because a
    category's importance here is how much of the collation it carries.
    """
    cats, crops = {}, Counter()
    for p in paths:
        print(f"  reading {os.path.basename(p)} ...", flush=True)
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        for c in doc.get("categories", []):
            cats.setdefault(c["id"], c)
        for an in doc.get("annotations", []):
            crops[an.get("category_id")] += 1
        del doc
    return cats, crops


# --------------------------------------------------------------- fault B

def gaps_for(cat):
    """-> list of missing core ranks, by rules G1 and G2."""
    lin = cat.get("lineage") or {}
    present = {r for r in CORE if (lin.get(r) or "").strip()}
    missing = []

    # G1: absent while something finer is present
    if present:
        finest = max(CORE_INDEX[r] for r in present)
        for r in CORE:
            if CORE_INDEX[r] < finest and r not in present:
                missing.append(r)

    # G2: the category's own rank, if it is a core rank
    own = str(cat.get("rank") or "").strip().lower()
    if own in CORE_INDEX and own not in present and own not in missing:
        missing.append(own)

    return sorted(missing, key=lambda r: CORE_INDEX[r])


# --------------------------------------------------------------- fault A

def load_cache(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_cache(path, cache):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def resolve_name(name, cache, sleep):
    """-> ('accepted'|'substitute'|'unresolved', accepted_name, aphia, rank)

    Delegates to C.worms_by_name rather than reimplementing exact-match and
    homonym handling. That matters: an audit that resolves names by different
    rules from the build would report differences that are its own.
    """
    if name in cache:
        return tuple(cache[name])
    hit = C.worms_by_name(name)
    if sleep:
        time.sleep(sleep)
    if hit is None:
        out = ("unresolved", "", "", "")
    else:
        rid, valid, rank = hit
        same = str(valid).strip().lower() == str(name).strip().lower()
        out = ("accepted" if same else "substitute", valid, rid, rank or "")
    cache[name] = list(out)
    return out


# ------------------------------------------------------------------ report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", nargs="*", default=DEFAULT_COCO)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=".",
                    help="directory for the three CSVs")
    ap.add_argument("--offline", action="store_true",
                    help="fault B only, plus whatever names are already cached")
    ap.add_argument("--limit", type=int,
                    help="resolve only this many uncached names (a pilot)")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent WoRMS lookups. 6 turns a six-hour run "
                         "into about one; be sparing, it is a free public "
                         "service")
    ap.add_argument("--ranks", nargs="*", default=DEFAULT_RANKS,
                    choices=CORE,
                    help="lineage ranks to audit for fault A")
    args = ap.parse_args()

    print("loading the collation")
    cats, crops = load_categories(args.coco)
    total_crops = sum(crops.values())
    print(f"  {len(cats):,} categories, {total_crops:,} annotations")

    # ---- fault B ------------------------------------------------------
    print("\nFAULT B  gapped lineages (no network needed)")
    print("-" * 74)
    gapped, gap_rank_crops, gap_rank_n = {}, Counter(), Counter()
    for cid, c in cats.items():
        g = gaps_for(c)
        if g:
            gapped[cid] = g
            for r in g:
                gap_rank_n[r] += 1
                gap_rank_crops[r] += crops.get(cid, 0)
    gcrops = sum(crops.get(cid, 0) for cid in gapped)
    print(f"  {len(gapped):,} of {len(cats):,} categories "
          f"({len(gapped) / max(1, len(cats)):.1%}) have a gapped lineage, "
          f"carrying {gcrops:,} crops ({gcrops / max(1, total_crops):.1%})")
    if gap_rank_n:
        print(f"\n  {'missing rank':<12} {'categories':>11} {'crops':>12}")
        for r in CORE:
            if gap_rank_n[r]:
                print(f"  {r:<12} {gap_rank_n[r]:>11,} {gap_rank_crops[r]:>12,}")
    worst = sorted(gapped, key=lambda c: -crops.get(c, 0))[:12]
    if worst:
        print("\n  worst by crops:")
        for cid in worst:
            c = cats[cid]
            print(f"    {crops.get(cid, 0):>9,}  {c.get('name', '?')} "
                  f"({c.get('rank', '?')}, id {cid}) missing "
                  f"{', '.join(gapped[cid])}")

    with open(os.path.join(args.out, "lineage_gaps.csv"), "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["category_id", "name", "rank", "crops", "missing_ranks"])
        for cid in sorted(gapped, key=lambda c: -crops.get(c, 0)):
            c = cats[cid]
            wr.writerow([cid, c.get("name"), c.get("rank"),
                         crops.get(cid, 0), " ".join(gapped[cid])])

    # ---- fault A ------------------------------------------------------
    # One entry per distinct (rank, name): genera and families are shared
    # heavily, so this is a few thousand lookups rather than ~17,000.
    audit_ranks = set(args.ranks)
    usage = defaultdict(lambda: {"cats": set(), "crops": 0})
    for cid, c in cats.items():
        for r, name in (c.get("lineage") or {}).items():
            if r in audit_ranks and (name or "").strip():
                u = usage[(r, name.strip())]
                u["cats"].add(cid)
                u["crops"] += crops.get(cid, 0)
    print(f"\nFAULT A  unaccepted intermediate ranks")
    print("-" * 74)
    print(f"  auditing ranks: {', '.join(args.ranks)}")
    if "species" not in audit_ranks:
        print("  (species excluded -- WP6a settled it by ID; see the note at "
              "the top)")
    print(f"  {len(usage):,} distinct (rank, name) pairs across "
          f"{len(cats):,} categories")

    cache = load_cache(args.cache)
    todo = [k for k in usage if k[1] not in cache]
    print(f"  {len(cache):,} already cached, {len(todo):,} to look up")

    # ORDER MATTERS, and the first version of this got it wrong. Sorting
    # alphabetically put every `class` name before every `genus` name, so an
    # interrupted run had audited the rank that barely matters and none of the
    # rank that does. Genus substitutions are what move plan §6.4's congener
    # groups and WP16's tree, so genus goes first, and within a rank the names
    # carrying the most crops go first. Now a partial run is a partial ANSWER
    # rather than an alphabetical accident.
    todo.sort(key=lambda k: (RANK_PRIORITY.get(k[0], 99), -usage[k]["crops"]))

    if args.offline:
        print("  --offline: resolving nothing; cached answers only")
        todo = []
    elif args.limit:
        todo = todo[:args.limit]
        print(f"  --limit: resolving the {len(todo):,} that carry the most "
              f"crops, genus first")

    t0, done = time.time(), 0
    lock = threading.Lock()

    def one(k):
        resolve_name(k[1], cache, args.sleep)

    try:
        if args.workers > 1:
            print(f"  {args.workers} workers")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for _ in pool.map(one, todo):
                    with lock:
                        done += 1
                        d = done
                    if d % 100 == 0:
                        el = time.time() - t0
                        rate = d / max(el, 1e-9)
                        print(f"    {d:,}/{len(todo):,}  {rate:.1f}/s  "
                              f"~{(len(todo) - d) / max(rate, 1e-9) / 60:.0f} "
                              f"min left", flush=True)
                        save_cache(args.cache, cache)
        else:
            for k in todo:
                one(k)
                done += 1
                if done % 100 == 0:
                    el = time.time() - t0
                    rate = done / max(el, 1e-9)
                    print(f"    {done:,}/{len(todo):,}  {rate:.1f}/s  "
                          f"~{(len(todo) - done) / max(rate, 1e-9) / 60:.0f} "
                          f"min left", flush=True)
                    save_cache(args.cache, cache)
    except C.WormsUnavailable as exc:
        print(f"\n  ! WoRMS unreachable: {exc}")
        print("  stopping cleanly -- the cache is saved and the run resumes.")
    except KeyboardInterrupt:
        print("\n  interrupted -- the cache is saved and the run resumes.")
    finally:
        save_cache(args.cache, cache)
        print(f"  cache: {args.cache} ({len(cache):,} names)")

    subs, unres, checked = [], [], 0
    for (r, name), u in usage.items():
        if name not in cache:
            continue
        checked += 1
        status, valid, aid, rank = cache[name]
        rec = (r, name, valid, aid, rank, len(u["cats"]), u["crops"])
        if status == "substitute":
            subs.append(rec)
        elif status == "unresolved":
            unres.append(rec)

    subs.sort(key=lambda t: -t[6])
    unres.sort(key=lambda t: -t[6])
    sub_crops = sum(t[6] for t in subs)
    sub_cats = len({c for (r, n), u in usage.items()
                    for c in u["cats"]
                    if n in cache and cache[n][0] == "substitute"})

    print(f"\n  checked {checked:,} of {len(usage):,} pairs")
    print(f"  {len(subs):,} name(s) are unaccepted and need substituting, "
          f"touching {sub_cats:,} categories and {sub_crops:,} crops "
          f"({sub_crops / max(1, total_crops):.1%} of the collation)")
    print(f"  {len(unres):,} name(s) could not be resolved unambiguously "
          f"(homonym, or no exact record) -- these need a person, not a rerun")

    if subs:
        print(f"\n  {'rank':<9} {'current':<28} {'accepted':<28} "
              f"{'cats':>5} {'crops':>10}")
        for r, name, valid, aid, rank, ncat, ncrop in subs[:25]:
            print(f"  {r:<9} {name:<28.28} {valid:<28.28} {ncat:>5,} "
                  f"{ncrop:>10,}")
        gsubs = [t for t in subs if t[0] == "genus"]
        if gsubs:
            print(f"\n  ! {len(gsubs)} of these are GENERA, carrying "
                  f"{sum(t[6] for t in gsubs):,} crops. Genus substitutions "
                  f"are the ones that move plan §6.4's congener test groups "
                  f"and WP16's tree, so these are the rows that matter.")

    if unres:
        print(f"\n  unresolved, worst by crops:")
        for r, name, _v, _a, _rk, ncat, ncrop in unres[:12]:
            print(f"    {ncrop:>9,}  {r} '{name}' ({ncat} categories)")

    for fn, rows in (("lineage_rank_substitutions.csv", subs),
                     ("lineage_rank_unresolved.csv", unres)):
        with open(os.path.join(args.out, fn), "w", newline="",
                  encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["lineage_rank", "current_name", "accepted_name",
                         "accepted_aphia", "accepted_rank", "n_categories",
                         "n_crops"])
            wr.writerows(rows)
        print(f"  wrote {os.path.join(args.out, fn)} ({len(rows):,} rows)")

    if todo and checked < len(usage):
        print(f"\n  NOT FINISHED: {len(usage) - checked:,} pairs still "
              f"unchecked. Re-run to continue; nothing above is the final "
              f"answer yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

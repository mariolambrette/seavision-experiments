#!/usr/bin/env python3
"""WP8: profile, and optionally freeze, the FathomNet slice of the dev set.

Decided 1 October 2026: the FathomNet slice is drawn ONLY from non-SEFSC
dive groups (MBARI, NOAA Ocean Exploration, Schmidt Ocean Institute), so
the SEFSC block -- 95% of FathomNet's teleosts and the only Gulf of Mexico
material -- stays whole. Its uses: WP17's deep-sea rejection thresholds, and
WP8's check on what the teleost-first rule costs. Groups are INFERRED
(WP5b); the slice is reported separately and never pooled.

Same machinery as wp8_profile.py, applied to FathomNet:
  - units are collapsed for near-duplicate frames (one unit per frame
    image; SEFSC per video-second, used only for gate protection);
  - a species is usable if its dev units split by group into >= ref_k
    reference and >= query_min query;
  - groups are added greedily, each chosen to bring the most species
    towards usable (objective "all species");
  - a group is never taken if it would push any species below a protected
    cell of FathomNet's OWN gate on collapsed units (default 50/2, 100/3),
    computed over ALL FathomNet groups including SEFSC.

Read-only. Default: a grid of sizes. With --fix N: freeze N groups to
fn_dev_groups.csv and fn_dev_selection.json.

    python scratch\\wp8_fathomnet_slice.py ^
        --coco-fn   D:\\marineai\\dataset\\collated\\seavision_fathomnet.json ^
        --fn-groups D:\\marineai\\scratch\\fathomnet_inferred_deployment.csv.gz ^
        --out       D:\\marineai\\scratch\\wp8_fn_profile_v1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

SEFSC = "NOAA NMFS SEFSC"
SEFSC_SECOND = re.compile(
    r"/([^/]+)/[^/]+\.(?:mp4|avi)\.(\d\d)\.(\d\d)\.(\d\d)\.\d+\.jpg$", re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def usable(by_group, ref_k, qmin):
    if not by_group:
        return False
    tot = sum(by_group.values())
    return tot >= ref_k + qmin and tot - max(by_group.values()) >= qmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-fn", required=True)
    ap.add_argument("--fn-groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--institutions", nargs="*",
                    default=["mbari", "ocean exploration", "schmidt"],
                    help="case-insensitive substrings of owner_institution")
    ap.add_argument("--levels", nargs="*", default=["dive", "deployment"],
                    help="derived_level values eligible for the slice")
    ap.add_argument("--ref-k", type=int, default=20)
    ap.add_argument("--query-min", type=int, default=10)
    ap.add_argument("--protect", default="50/2,100/3")
    ap.add_argument("--sizes", default="10,20,40,80,160")
    ap.add_argument("--fix", type=int, default=None, metavar="N")
    ap.add_argument("--bar-species", type=int, default=None, metavar="K",
                    help="find the smallest slice with >= K usable species "
                         "and freeze it (decided 1 October 2026: K = 40)")
    ap.add_argument("--search-max", type=int, default=400)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    os.makedirs(args.out)
    report = {"args": vars(args)}

    fn_sha = sha256(args.fn_groups)
    print(f"fn-groups sha256 {fn_sha}")

    # ---- groups, per crop (the CSV is one row per crop record)
    g_of, lvl_of, inst_of_g = {}, {}, {}
    crops_per_g = Counter()
    with gzip.open(args.fn_groups, "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = r["inferred_deployment"]
            if not g:
                continue
            g_of[r["uid"]] = g
            lvl_of[g] = r["derived_level"]
            inst_of_g[g] = r["owner_institution"]
            crops_per_g[g] += 1

    def eligible_inst(inst):
        return inst != SEFSC and any(p.lower() in inst.lower()
                                     for p in args.institutions)

    print(f"\n{'=' * 78}\nA. GROUPS BY INSTITUTION AND LEVEL  "
          f"(* = eligible for the slice)")
    tab = Counter((inst_of_g[g], lvl_of[g]) for g in inst_of_g)
    for (inst, lv), n in sorted(tab.items()):
        mark = "*" if eligible_inst(inst) and lv in args.levels else " "
        print(f"  {mark} {inst[:40]:<41}{lv:<22}{n:>6,} groups"
              f"{sum(crops_per_g[g] for g in inst_of_g if inst_of_g[g] == inst and lvl_of[g] == lv):>11,} crops")
    cands = sorted(g for g in inst_of_g
                   if eligible_inst(inst_of_g[g]) and lvl_of[g] in args.levels)
    if not cands:
        sys.exit("no eligible groups: check --institutions and --levels "
                 "against table A")

    # ---- annotations -> collapsed units per (species, group)
    print(f"\n  loading {os.path.basename(args.coco_fn)} ...", flush=True)
    with open(args.coco_fn, encoding="utf-8") as fh:
        doc = json.load(fh)
    cats = {c["id"]: c for c in doc["categories"]}
    lin = {cid: (c.get("lineage") or {}) for cid, c in cats.items()}
    is_sp = {cid for cid, c in cats.items()
             if str(c.get("rank", "")).lower() == "species"}
    tele = {cid for cid in cats if lin[cid].get("class") == "Teleostei"}
    img = {i["id"]: i for i in doc["images"]}
    units = defaultdict(lambda: defaultdict(set))     # sp -> g -> {unit}
    for an in doc["annotations"]:
        sp = an.get("category_id")
        if sp not in is_sp:
            continue
        im = img[an["image_id"]]
        g = g_of.get(os.path.splitext(im["file_name"])[0])
        if not g:
            continue
        sm = im.get("source_meta") or {}
        m = SEFSC_SECOND.search(sm.get("frame_url") or "") \
            if sm.get("owner_institution") == SEFSC else None
        units[sp][g].add((m[1], m[2], m[3], m[4]) if m
                         else sm.get("fathomnet_image_uuid"))
    del doc, img
    U = {sp: Counter({g: len(s) for g, s in d.items()})
         for sp, d in units.items()}
    by_group = defaultdict(Counter)                   # g -> sp -> units
    for sp, d in U.items():
        for g, n in d.items():
            by_group[g][sp] = n
    cset = set(cands)

    prot = []
    for c in args.protect.split(","):
        mc, mg = (int(x) for x in c.split("/"))
        q = {s for s, d in U.items() if sum(d.values()) >= mc and len(d) >= mg}
        prot.append((mc, mg, q))
        print(f"  FathomNet gate on collapsed units, >= {mc}/{mg}: "
              f"{len(q)} species (protected)")
    need = args.ref_k + args.query_min

    def levels(us):
        gen = Counter(lin[s].get("genus") for s in us if lin[s].get("genus"))
        fam, odr = defaultdict(set), defaultdict(set)
        for s in us:
            L = lin[s]
            if L.get("family") and L.get("genus"):
                fam[L["family"]].add(L["genus"])
            if L.get("order") and L.get("family"):
                odr[L["order"]].add(L["family"])
        return {"species": len(us),
                "teleost_species": sum(1 for s in us if s in tele),
                "genera_2sp": sum(1 for v in gen.values() if v >= 2),
                "families_2gen": sum(1 for v in fam.values() if len(v) >= 2),
                "orders_2fam": sum(1 for v in odr.values() if len(v) >= 2),
                "classes": dict(Counter(lin[s].get("class") or "?"
                                        for s in us).most_common())}

    def run(n, track=None):
        main_u = {s: Counter(d) for s, d in U.items()}
        n_us = 0
        dev = defaultdict(Counter)
        chosen, avail = [], list(cands)
        while len(chosen) < n:
            best, best_gain = None, 0
            for g in avail:
                bad = False
                for sp, u in by_group[g].items():
                    for mc, mg, q in prot:
                        if sp in q:
                            d = main_u[sp]
                            if sum(d.values()) - u < mc or len(d) - 1 < mg:
                                bad = True
                                break
                    if bad:
                        break
                if bad:
                    continue
                gain = sum(max(0, min(u, need - sum(dev[sp].values())))
                           for sp, u in by_group[g].items())
                if gain > best_gain:          # sorted, so ties go first
                    best, best_gain = g, gain
            if best is None:
                break
            chosen.append(best)
            avail.remove(best)
            for sp, u in by_group[best].items():
                was = usable(dev[sp], args.ref_k, args.query_min)
                dev[sp][best] = u
                del main_u[sp][best]
                n_us += usable(dev[sp], args.ref_k, args.query_min) and not was
            if track is not None:
                track.append(n_us)
        us = {s for s, d in dev.items() if usable(d, args.ref_k, args.query_min)}
        lost = [len(q - {s for s, d in main_u.items()
                         if sum(d.values()) >= mc and len(d) >= mg})
                for mc, mg, q in prot]
        return {"groups": len(chosen),
                "crops": sum(crops_per_g[g] for g in chosen),
                "units": sum(sum(by_group[g].values()) for g in chosen),
                "institutions": dict(Counter(inst_of_g[g] for g in chosen)),
                "levels": levels(us), "lost_protected": lost,
                "usable_species": sorted(us), "chosen": chosen}

    pool = levels({s for s, d in U.items() if usable(
        Counter({g: n for g, n in d.items() if g in cset}),
        args.ref_k, args.query_min)})
    print(f"  eligible: {len(cands):,} groups, "
          f"{sum(crops_per_g[g] for g in cands):,} crops; if EVERY eligible "
          f"group were reserved: {pool['species']} usable species "
          f"({pool['teleost_species']} teleost), {pool['genera_2sp']} genera "
          f">=2sp, {pool['families_2gen']} families >=2gen")
    report["pool_ceiling"] = pool

    if args.bar_species is not None:
        # Greedy is prefix-consistent: the first n groups of a longer run
        # are the n-group slice. One pass gives the usable count at every n,
        # and crops only grow with n, so the first n meeting the bar is the
        # smallest slice by crops as well as by groups.
        track = []
        run(args.search_max, track)
        hit = next((i + 1 for i, v in enumerate(track)
                    if v >= args.bar_species), None)
        print(f"\n{'=' * 78}\nB'. SMALLEST SLICE WITH >= {args.bar_species} "
              f"USABLE SPECIES")
        for n in sorted({x for x in (hit - 2, hit - 1, hit, hit + 1, hit + 5)
                         if hit and 1 <= x <= len(track)} or {len(track)}):
            print(f"  {n:>4} groups -> {track[n - 1]} usable species"
                  f"{'   <- smallest meeting the bar' if n == hit else ''}")
        if not hit:
            sys.exit(f"  ! no slice up to {len(track)} groups meets the bar")
        report["bar_track"] = track
        args.fix = hit

    if args.fix is None:
        print(f"\n{'=' * 78}\nB. CANDIDATE SLICES  (greedy, objective 'all "
              f"species', protected {args.protect}; usable = >= {args.ref_k} "
              f"ref + >= {args.query_min} query units, split by group)")
        print(f"  {'groups':>6}{'crops':>9}{'units':>8}{'sp':>5}{'tele':>5}"
              f"{'gen':>5}{'fam':>5}{'ord':>5}   lost   classes of usable species")
        res = []
        for n in [int(x) for x in args.sizes.split(",")]:
            r = run(n)
            res.append({k: v for k, v in r.items() if k != "chosen"})
            L = r["levels"]
            cls = ", ".join(f"{k} {v}" for k, v in list(L["classes"].items())[:5])
            print(f"  {r['groups']:>6}{r['crops']:>9,}{r['units']:>8,}"
                  f"{L['species']:>5}{L['teleost_species']:>5}"
                  f"{L['genera_2sp']:>5}{L['families_2gen']:>5}"
                  f"{L['orders_2fam']:>5}   {','.join(map(str, r['lost_protected'])):<6} {cls}")
        report["grid"] = res
    else:
        r = run(args.fix)
        L = r["levels"]
        print(f"\n{'=' * 78}\nFIXED FATHOMNET SLICE  {r['groups']} groups, "
              f"{r['crops']:,} crops, {r['units']:,} units\n  usable species "
              f"{L['species']} (teleost {L['teleost_species']}); genera >=2sp "
              f"{L['genera_2sp']}; families >=2gen {L['families_2gen']}; "
              f"orders >=2fam {L['orders_2fam']}; lost protected "
              f"{r['lost_protected']}\n  by institution {r['institutions']}")
        with open(os.path.join(args.out, "fn_dev_groups.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["order", "source", "deployment", "institution",
                        "derived_level", "crops", "units"])
            for i, g in enumerate(r["chosen"]):
                w.writerow([i + 1, "fathomnet", g, inst_of_g[g], lvl_of[g],
                            crops_per_g[g], sum(by_group[g].values())])
        with open(os.path.join(args.out, "fn_dev_selection.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"decided": "2026-10-01",
                       "basis": "non-SEFSC dive groups only; inferred groups; "
                                "reported separately, never pooled",
                       "args": vars(args), "fn_groups_sha256": fn_sha,
                       "coco_fn_sha256": sha256(args.coco_fn),
                       "result": r}, fh, indent=1)
        print("  wrote fn_dev_groups.csv and fn_dev_selection.json")
        report["fixed"] = {k: v for k, v in r.items() if k != "chosen"}

    with open(os.path.join(args.out, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("\nNothing was reserved; nothing in the collation was changed.")


if __name__ == "__main__":
    main()

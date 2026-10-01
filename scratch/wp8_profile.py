#!/usr/bin/env python3
"""WP8 development-split profiler. READ-ONLY: chooses nothing, writes CSVs.

Answers "what could a standing dev set look like, and what would it cost the
main set?" before any pool is chosen -- the same measure-then-decide
discipline that set the shard cap.

FIVE SECTIONS
  0. Self-check: recompute the WP6 gate from the COCO files and compare with
     the recorded 454/78, 304/57/35, 243/48. If this does not reproduce, this
     script's gate logic is not WP6's and every later section is suspect.
  1. Pool inventory: per OzFish deployment (by survey) and FishWIO shooting --
     crops, individuals, species.
  2. Candidate dev pools, chosen greedily for usable congener material, at a
     grid of sizes, each protected so that no species currently clearing the
     reference gate cell (>=50 crops in >=2 groups) is pushed below it.
  3. The WP6 gate re-run on what each pool leaves behind.
  4. The FathomNet slice: inferred groups, with counts collapsed for
     near-duplicate frames, beside the raw crop counts.

UNITS, stated because they decide what "usable" means
  OzFish: individuals = per deployment x species, max(left-camera crops,
          right-camera crops). WP3 measured 99.3% of deployment x taxon
          combinations at exactly equal counts in both cameras, so the larger
          of the two counts individuals with stereo pairs merged.
  FishWIO: crops. It has tracks (mean 2.63 crops per track, WP2) that are not
          derived here, so its units OVERSTATE individuals; flagged in output.
  FathomNet: SEFSC = distinct (video, whole second); elsewhere distinct
          frames. Raw crops are shown alongside so the inflation is visible.

A SPECIES IS USABLE IN A DEV POOL when its dev units can be split by
deployment into >= --ref-k reference units and >= --query-min query units:
operationalised as total >= ref_k + query_min and total - largest single
deployment >= query_min. A GENUS IS USABLE when it holds >= 2 usable species.

    python scratch/wp8_profile.py ^
      --coco-main D:\\marineai\\dataset\\collated\\seavision.json ^
      --coco-fn   D:\\marineai\\dataset\\collated\\seavision_fathomnet.json ^
      --fn-groups D:\\marineai\\scratch\\fathomnet_inferred_deployment.csv.gz ^
      --out       D:\\marineai\\scratch\\wp8_profile
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

GATE_CELLS = [(10, 2), (50, 2), (100, 3)]
WP6_EXPECTED = {(10, 2): (454, 78), (50, 2): (304, 57), (100, 3): (243, 48)}
PROTECT = (50, 2)
OZ_SIZES = [50, 100, 150, 200]
FW_SIZES = [0, 3, 5]
SEFSC_SECOND = re.compile(
    r"/([^/]+)/[^/]+\.(?:mp4|avi)\.(\d\d)\.(\d\d)\.(\d\d)\.\d+\.jpg$", re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    print(f"  loading {os.path.basename(path)} ...", flush=True)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ gate

def gate(species_groups, cells=GATE_CELLS, genus_of=None):
    """species_groups: {species_id: Counter(group -> crops)}.
    -> {cell: (n_species, genera_2plus, genera_3plus, set(species))}"""
    out = {}
    for mc, mg in cells:
        q = {s for s, g in species_groups.items()
             if sum(g.values()) >= mc and len(g) >= mg}
        per_genus = Counter(genus_of[s] for s in q if genus_of.get(s))
        out[(mc, mg)] = (len(q), sum(1 for v in per_genus.values() if v >= 2),
                         sum(1 for v in per_genus.values() if v >= 3), q)
    return out


def usable(units_by_dep, ref_k, qmin):
    tot = sum(units_by_dep.values())
    if not units_by_dep:
        return False
    return tot >= ref_k + qmin and tot - max(units_by_dep.values()) >= qmin


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-main", required=True)
    ap.add_argument("--coco-fn", required=True)
    ap.add_argument("--fn-groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-k", type=int, default=20)
    ap.add_argument("--query-min", type=int, default=10)
    ap.add_argument("--exclude-surveys", nargs="*", default=["E"])
    ap.add_argument("--random-draws", type=int, default=20)
    ap.add_argument("--bar-oz-genera", type=int, default=10)
    ap.add_argument("--bar-families", type=int, default=8)
    ap.add_argument("--search-oz-min", type=int, default=50)
    ap.add_argument("--search-oz-max", type=int, default=100)
    ap.add_argument("--search-oz-step", type=int, default=5)
    ap.add_argument("--search-fw-max", type=int, default=5)
    ap.add_argument("--skip-grid", action="store_true",
                    help="skip the section 2-3 grid and random baseline")
    ap.add_argument("--fix", default=None, metavar="OZ,FW",
                    help="freeze one size: write the chosen deployments to "
                         "dev_selection.json and dev_deployments.csv, and "
                         "skip the section 5 search")
    ap.add_argument("--protect", default="50/2,100/3",
                    help="gate cells no reserved deployment may break")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    os.makedirs(args.out)
    report = {"args": vars(args)}

    fn_sha = sha256(args.fn_groups)
    print(f"fn-groups sha256 {fn_sha}")
    report["fn_groups_sha256"] = fn_sha

    # ---- main collation: per annotation record
    doc = load(args.coco_main)
    cats = {c["id"]: c for c in doc["categories"]}
    src_of = {d["id"]: d["name"] for d in doc["datasets"]}
    img = {i["id"]: i for i in doc["images"]}
    recs = []                      # (source, dep, survey, species, camera)
    for an in doc["annotations"]:
        im = img[an["image_id"]]
        c = cats.get(an.get("category_id"))
        if not c:
            continue
        g = im.get("groups") or {}
        sm = im.get("source_meta") or {}
        recs.append((src_of.get(im["dataset_id"], "?"), g.get("deployment"),
                     g.get("survey"), an["category_id"],
                     str(sm.get("camera") or "").upper()))
    del doc, img

    genus_of = {cid: (c.get("lineage") or {}).get("genus")
                for cid, c in cats.items()}
    is_species = {cid for cid, c in cats.items()
                  if str(c.get("rank", "")).lower() == "species"}
    name_of = {cid: c.get("name") for cid, c in cats.items()}

    # species x group crop counts, grouped crops only (WP6 rule), no FathomNet
    sg = defaultdict(Counter)
    for src, dep, _sv, sp, _cam in recs:
        if dep and sp in is_species:
            sg[sp][(src, dep)] += 1

    # ---- 0. self-check
    print(f"\n{'=' * 78}\n0. SELF-CHECK: WP6 gate recomputed")
    base = gate(sg, genus_of=genus_of)
    ok = True
    for cell, (ns, g2, g3, _q) in base.items():
        exp = WP6_EXPECTED[cell]
        flag = "ok" if (ns, g2) == exp else "MISMATCH"
        ok &= flag == "ok"
        print(f"  >={cell[0]} crops, >={cell[1]} groups: {ns} species, {g2} "
              f"genera >=2, {g3} >=3   (WP6: {exp[0]}, {exp[1]})  {flag}")
    if not ok:
        print("  ! does not reproduce WP6. Later sections use this script's "
              "rule, not WP6's -- resolve before choosing anything.")
    report["self_check_ok"] = ok

    # ---- 1. pool inventory
    print(f"\n{'=' * 78}\n1. POOL INVENTORY")
    dep_info = defaultdict(lambda: {"crops": 0, "species": set(),
                                    "L": Counter(), "R": Counter(),
                                    "src": None, "survey": None})
    for src, dep, sv, sp, cam in recs:
        if src not in ("ozfish", "fishwio") or not dep:
            continue
        d = dep_info[(src, dep)]
        d["crops"] += 1
        d["src"], d["survey"] = src, sv
        if sp in is_species:
            d["species"].add(sp)
            if src == "ozfish":
                d["L" if cam == "L" else "R"][sp] += 1
            else:
                d["L"][sp] += 1                     # FishWIO: crops
    units = {}            # (src, dep) -> Counter(species -> units)
    for k, d in dep_info.items():
        u = Counter()
        for sp in set(d["L"]) | set(d["R"]):
            u[sp] = max(d["L"][sp], d["R"][sp]) if d["src"] == "ozfish" \
                else d["L"][sp]
        units[k] = u
    rows = []
    for (src, dep), d in sorted(dep_info.items()):
        rows.append([src, d["survey"] or "", dep, d["crops"],
                     sum(units[(src, dep)].values()), len(d["species"])])
    with open(os.path.join(args.out, "pool_inventory.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "survey", "deployment", "crops",
                    "species_units", "species"])
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[(r[0], r[1])].append(r)
    print(f"  {'source':<8}{'survey':<7}{'groups':>7}{'crops':>9}"
          f"{'units':>9}{'crops/grp':>10}{'units/grp':>10}")
    for (src, sv), rs in sorted(by.items()):
        n, c, u = len(rs), sum(r[3] for r in rs), sum(r[4] for r in rs)
        print(f"  {src:<8}{sv or '-':<7}{n:>7,}{c:>9,}{u:>9,}"
              f"{c / n:>10.1f}{u / n:>10.1f}")
    print("  (FishWIO units are crops: tracks are not collapsed, so they "
          "overstate individuals)")

    # ---- 2/3. greedy pools
    print(f"\n{'=' * 78}\n2-3. CANDIDATE DEV SETS (greedy; "
          f"OzFish chosen first; protected cells {args.protect})")
    excl = set(args.exclude_surveys or [])
    cands = sorted(k for k in units
                   if k[0] == "fishwio"
                   or (k[0] == "ozfish" and dep_info[k]["survey"] not in excl))
    sp_cnt = Counter()
    for k in cands:
        for sp in units[k]:
            sp_cnt[sp] += 1
    family_of = {cid: (c.get("lineage") or {}).get("family")
                 for cid, c in cats.items()}
    order_of = {cid: (c.get("lineage") or {}).get("order")
                for cid, c in cats.items()}
    genus_pot = Counter(genus_of[s] for s in sp_cnt if genus_of.get(s))
    # Two objectives. "all": make as many species usable as possible, which
    # serves discrimination at every level -- within genus, between genera
    # of one family, between families of one order. "congener": only species
    # whose genus has a second candidate species (the earlier objective).
    targets = {
        "all": {s for s in sp_cnt if s in is_species},
        "congener": {s for s in sp_cnt if s in is_species and
                     genus_of.get(s) and genus_pot[genus_of[s]] >= 2},
    }
    prot_cells = [tuple(int(x) for x in c.split("/"))
                  for c in args.protect.split(",")]
    prot_sets = [(mc, mg, base[(mc, mg)][3] if (mc, mg) in base else
                  gate(sg, [(mc, mg)], genus_of)[(mc, mg)][3])
                 for mc, mg in prot_cells]
    need_total = args.ref_k + args.query_min

    def levels(us):
        """Discrimination material at each taxonomic level, from a set of
        usable species: genera with >=2 usable species (within genus);
        families with >=2 genera each holding a usable species (between
        genera); orders with >=2 families likewise (between families)."""
        g = Counter(genus_of[s] for s in us if genus_of.get(s))
        fam_gen = defaultdict(set)
        ord_fam = defaultdict(set)
        for s in us:
            if family_of.get(s) and genus_of.get(s):
                fam_gen[family_of[s]].add(genus_of[s])
            if order_of.get(s) and family_of.get(s):
                ord_fam[order_of[s]].add(family_of[s])
        return {"species": len(us),
                "genera_2sp": sum(1 for v in g.values() if v >= 2),
                "families_2gen": sum(1 for v in fam_gen.values()
                                     if len(v) >= 2),
                "orders_2fam": sum(1 for v in ord_fam.values()
                                   if len(v) >= 2),
                "genera_list": sorted(k for k, v in g.items() if v >= 2)}

    def run(n_oz, n_fw, mode):
        target = targets[mode]
        caps = {"ozfish": n_oz, "fishwio": n_fw}
        main = {s: Counter(g) for s, g in sg.items()}
        dev = defaultdict(Counter)          # species -> Counter(dep -> units)
        chosen, used = [], Counter()
        # STAGED: OzFish first, scored on OzFish units alone, so the OzFish
        # portion stands on its own (the geometry check uses it alone);
        # FishWIO is then added, scored on the combined pool.
        for stage in ("ozfish", "fishwio"):
            avail = [k for k in cands if k[0] == stage and caps[stage] > 0]
            while used[stage] < caps[stage]:
                best, best_gain = None, 0.0
                for k in avail:
                    bad = False
                    for sp in units[k]:
                        for mc, mg, pset in prot_sets:
                            if sp in pset:
                                g = main[sp]
                                c = g.get(k, 0)
                                if c and (sum(g.values()) - c < mc or
                                          len(g) - 1 < mg):
                                    bad = True
                                    break
                        if bad:
                            break
                    if bad:
                        continue
                    gain = 0.0
                    for sp, u in units[k].items():
                        if sp not in target:
                            continue
                        have = sum(v for kk, v in dev[sp].items()
                                   if stage == "fishwio" or kk[0] == stage)
                        gain += max(0, min(u, need_total - have))
                    if gain > best_gain:          # avail is sorted, so ties
                        best, best_gain = k, gain  # go first: deterministic
                if not best:
                    break
                chosen.append(best)
                used[stage] += 1
                avail.remove(best)
                for sp, u in units[best].items():
                    dev[sp][best] = u
                    if sp in main and best in main[sp]:
                        del main[sp][best]
        us = {s for s, d in dev.items() if s in is_species and
              usable(d, args.ref_k, args.query_min)}
        oz_us = {s for s in us if usable(
            Counter({k: v for k, v in dev[s].items() if k[0] == "ozfish"}),
            args.ref_k, args.query_min)}
        rem = gate({s: g for s, g in main.items() if g}, genus_of=genus_of)
        return {
            "mode": mode, "oz": used["ozfish"], "fw": used["fishwio"],
            "crops": sum(dep_info[k]["crops"] for k in chosen),
            "units": sum(sum(units[k].values()) for k in chosen),
            "all": levels(us), "oz_alone": levels(oz_us),
            "gate_remainder": {f"{c[0]}/{c[1]}": rem[c][:3] for c in rem},
            "lost_50_2": len(base[(50, 2)][3] - rem[(50, 2)][3]),
            "lost_100_3": len(base[(100, 3)][3] - rem[(100, 3)][3]),
            "chosen": [f"{k[0]}:{k[1]}" for k in chosen],
        }

    if not args.skip_grid:
        print(f"  usable = >= {args.ref_k} reference and >= {args.query_min} "
              f"query individuals, split by deployment")
        print(f"  columns: species usable | genera with >=2 usable species | "
              f"families with >=2 usable genera | orders with >=2 usable "
              f"families;  'oz' = the OzFish portion alone")
        hdr = (f"  {'objective':<10}{'oz':>4}{'fw':>3}{'crops':>8}{'units':>8}"
               f"{'sp':>6}{'gen':>5}{'fam':>5}{'ord':>5}{'oz:sp':>7}{'gen':>5}"
               f"{'fam':>5}   remainder 10/2|50/2|100/3   lost 50/2,100/3")
        results = []
        for mode in ("all", "congener"):
            print(hdr)
            for n_oz in OZ_SIZES:
                for n_fw in FW_SIZES:
                    r = run(n_oz, n_fw, mode)
                    results.append(r)
                    a, o, gr = r["all"], r["oz_alone"], r["gate_remainder"]
                    print(f"  {mode:<10}{r['oz']:>4}{r['fw']:>3}"
                          f"{r['crops']:>8,}{r['units']:>8,}"
                          f"{a['species']:>6}{a['genera_2sp']:>5}"
                          f"{a['families_2gen']:>5}{a['orders_2fam']:>5}"
                          f"{o['species']:>7}{o['genera_2sp']:>5}"
                          f"{o['families_2gen']:>5}   "
                          f"{gr['10/2'][0]}/{gr['10/2'][1]}|"
                          f"{gr['50/2'][0]}/{gr['50/2'][1]}|"
                          f"{gr['100/3'][0]}/{gr['100/3'][1]}   "
                          f"{r['lost_50_2']},{r['lost_100_3']}")
            print()
        print(f"  baseline gate: 10/2 {base[(10, 2)][0]}/{base[(10, 2)][1]} | "
              f"50/2 {base[(50, 2)][0]}/{base[(50, 2)][1]} | "
              f"100/3 {base[(100, 3)][0]}/{base[(100, 3)][1]}")
        report["pools"] = results

        # ---- the simple alternative: whole deployments at random.
        import random
        print(f"\n  RANDOM BASELINE (OzFish only, {args.random_draws} draws per "
              f"size, no protection applied; median [min-max])")
        print(f"  {'oz':>4}{'species':>14}{'genera':>12}{'families':>12}"
              f"{'lost 50/2':>14}{'lost 100/3':>14}")
        oz_cands = [k for k in cands if k[0] == "ozfish"]
        rb = []
        for n_oz in OZ_SIZES:
            ss, gs, fs, l1, l2 = [], [], [], [], []
            for seed in range(args.random_draws):
                pick = random.Random(seed).sample(oz_cands, min(n_oz,
                                                                 len(oz_cands)))
                dev = defaultdict(Counter)
                main = {s_: Counter(g) for s_, g in sg.items()}
                for k in pick:
                    for sp, u in units[k].items():
                        dev[sp][k] = u
                        if sp in main:
                            main[sp].pop(k, None)
                us = {s_ for s_, d in dev.items() if s_ in is_species and
                      usable(d, args.ref_k, args.query_min)}
                lv = levels(us)
                rem = gate({s_: g for s_, g in main.items() if g},
                           genus_of=genus_of)
                ss.append(lv["species"])
                gs.append(lv["genera_2sp"])
                fs.append(lv["families_2gen"])
                l1.append(len(base[(50, 2)][3] - rem[(50, 2)][3]))
                l2.append(len(base[(100, 3)][3] - rem[(100, 3)][3]))
            def med(v):
                v = sorted(v)
                return f"{v[len(v) // 2]} [{v[0]}-{v[-1]}]"
            print(f"  {n_oz:>4}{med(ss):>14}{med(gs):>12}{med(fs):>12}"
                  f"{med(l1):>14}{med(l2):>14}")
            rb.append({"oz": n_oz, "species": ss, "genera": gs, "families": fs,
                       "lost_50_2": l1, "lost_100_3": l2})
        report["random_baseline"] = rb

    # ---- fixed size: freeze the selection (decided 1 October 2026: 90+3)
    if args.fix:
        n_oz, n_fw = (int(x) for x in args.fix.split(","))
        r = run(n_oz, n_fw, "all")
        a, o = r["all"], r["oz_alone"]
        ok = (o["genera_2sp"] >= args.bar_oz_genera and
              a["families_2gen"] >= args.bar_families)
        print(f"\n{'=' * 78}\nFIXED SELECTION {n_oz} OzFish + {n_fw} FishWIO"
              f"\n  crops {r['crops']:,}  units {r['units']:,}  species "
              f"{a['species']}  genera>=2sp {a['genera_2sp']} (OzFish alone "
              f"{o['genera_2sp']})  families>=2gen {a['families_2gen']}  "
              f"orders>=2fam {a['orders_2fam']}\n  bar met: {ok}   lost "
              f"50/2 {r['lost_50_2']}, 100/3 {r['lost_100_3']}")
        if r["oz"] != n_oz or r["fw"] != n_fw:
            print(f"  ! greedy stopped early: got {r['oz']}+{r['fw']}")
        rows = []
        for i, ck in enumerate(r["chosen"]):
            src, dep = ck.split(":", 1)
            k = next(kk for kk in cands if kk[0] == src and str(kk[1]) == dep)
            rows.append({"order": i + 1, "source": src, "deployment": dep,
                         "survey": dep_info[k].get("survey") or "",
                         "crops": dep_info[k]["crops"],
                         "units": sum(units[k].values())})
        with open(os.path.join(args.out, "dev_deployments.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        sel = {"decided": "2026-10-01", "size": {"ozfish": n_oz,
                                                  "fishwio": n_fw},
               "objective": "all", "bar": {"oz_genera_2sp": args.bar_oz_genera,
                                           "families_2gen": args.bar_families},
               "bar_note": "families bar relies on FishWIO crop counts, which "
                           "overstate individuals; accepted as approximate",
               "ref_k": args.ref_k, "query_min": args.query_min,
               "protect": args.protect,
               "exclude_surveys": args.exclude_surveys,
               "fn_groups_sha256": fn_sha,
               "coco_main_sha256": sha256(args.coco_main),
               "result": r}
        with open(os.path.join(args.out, "dev_selection.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(sel, fh, indent=1)
        print(f"  wrote dev_deployments.csv ({len(rows)} rows) and "
              f"dev_selection.json")
        report["fixed"] = {k: v for k, v in r.items() if k != "chosen"}

    # ---- 5. smallest dev set meeting the sufficiency bar
    # Bar (decided 1 October 2026): OzFish portion alone >= bar_oz_genera
    # genera with >=2 usable species; whole set >= bar_families families
    # with >=2 usable genera. Between-family material is reported, not
    # required. "Smallest" = fewest crops reserved.
    if not args.fix:
        print(f"\n{'=' * 78}\n5. SMALLEST DEV SET MEETING THE BAR  (objective "
              f"'all'; OzFish alone >= {args.bar_oz_genera} genera with >=2 usable "
              f"species; whole set >= {args.bar_families} families with >=2 "
              f"usable genera)")
        meets, seen = [], set()
        for n_oz in range(args.search_oz_min, args.search_oz_max + 1,
                          args.search_oz_step):
            for n_fw in range(0, args.search_fw_max + 1):
                r = run(n_oz, n_fw, "all")
                ok = (r["oz_alone"]["genera_2sp"] >= args.bar_oz_genera and
                      r["all"]["families_2gen"] >= args.bar_families)
                key = tuple(sorted(r["chosen"]))
                if ok and key not in seen:  # capped sizes repeat a set
                    seen.add(key)
                    meets.append(r)
        meets.sort(key=lambda r: (r["crops"], r["oz"], r["fw"]))
        print(f"  {'oz':>4}{'fw':>4}{'crops':>8}{'units':>8}{'sp':>6}{'gen':>5}"
              f"{'fam':>5}{'ord':>5}{'oz:gen':>8}   lost 50/2,100/3")
        for r in meets[:8]:
            a, o = r["all"], r["oz_alone"]
            print(f"  {r['oz']:>4}{r['fw']:>4}{r['crops']:>8,}{r['units']:>8,}"
                  f"{a['species']:>6}{a['genera_2sp']:>5}{a['families_2gen']:>5}"
                  f"{a['orders_2fam']:>5}{o['genera_2sp']:>8}   "
                  f"{r['lost_50_2']},{r['lost_100_3']}")
        if meets:
            best = meets[0]
            with open(os.path.join(args.out, "smallest_dev_set.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"bar": {"oz_genera_2sp": args.bar_oz_genera,
                                   "families_2gen": args.bar_families},
                           "ref_k": args.ref_k, "query_min": args.query_min,
                           "protect": args.protect,
                           "exclude_surveys": args.exclude_surveys,
                           "fn_groups_sha256": fn_sha,
                           "result": best}, fh, indent=1)
            print(f"\n  smallest: {best['oz']} OzFish deployments + {best['fw']} "
                  f"FishWIO shootings, {best['crops']:,} crops "
                  f"-> smallest_dev_set.json (a candidate, not a reservation)")
        else:
            print("  ! no size in the search range meets the bar")
        report["smallest_search"] = [{k: v for k, v in r.items() if k != "chosen"}
                                     for r in meets[:20]]

    # ---- 4. FathomNet slice
    print(f"\n{'=' * 78}\n4. FATHOMNET (inferred groups; units collapsed for "
          f"near-duplicate frames)")
    fgroup, flevel = {}, {}
    with gzip.open(args.fn_groups, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["inferred_deployment"]:
                fgroup[r["uid"]] = r["inferred_deployment"]
                flevel[r["uid"]] = r["derived_level"]
    fdoc = load(args.coco_fn)
    fcats = {c["id"]: c for c in fdoc["categories"]}
    for cid, c in fcats.items():
        genus_of.setdefault(cid, (c.get("lineage") or {}).get("genus"))
        name_of.setdefault(cid, c.get("name"))
    fsp = {cid for cid, c in fcats.items()
           if str(c.get("rank", "")).lower() == "species"}
    tele = {cid for cid, c in fcats.items()
            if (c.get("lineage") or {}).get("class") == "Teleostei"}
    fimg = {i["id"]: i for i in fdoc["images"]}
    crops_g = defaultdict(Counter)
    units_g = defaultdict(lambda: defaultdict(set))
    for an in fdoc["annotations"]:
        im = fimg[an["image_id"]]
        uid = os.path.splitext(im["file_name"])[0]
        g = fgroup.get(uid)
        sp = an.get("category_id")
        if not g or sp not in fsp:
            continue
        sm = im.get("source_meta") or {}
        url = sm.get("frame_url") or ""
        m = SEFSC_SECOND.search(url) if sm.get("owner_institution") == \
            "NOAA NMFS SEFSC" else None
        unit = (m[1], m[2], m[3], m[4]) if m else sm.get(
            "fathomnet_image_uuid")
        crops_g[sp][g] += 1
        units_g[sp][g].add(unit)
    del fdoc, fimg
    ug = {s: Counter({g: len(u) for g, u in d.items()})
          for s, d in units_g.items()}
    print(f"  {'':<30}{'by crops':>20}{'by collapsed units':>22}")
    for subset, label in ((None, "all species"), (tele, "teleost species")):
        for cell in GATE_CELLS:
            def filt(d):
                return {s: g for s, g in d.items()
                        if subset is None or s in subset}
            a = gate(filt(crops_g), [cell], genus_of)[cell]
            b = gate(filt(ug), [cell], genus_of)[cell]
            print(f"  {label:<18} >={cell[0]:>3}/{cell[1]}   "
                  f"{a[0]:>6} sp {a[1]:>4} gen   {b[0]:>8} sp {b[1]:>4} gen")
    shared = gate(ug, [PROTECT], genus_of)[PROTECT][3] & base[PROTECT][3]
    print(f"  species clearing >=50/2 on collapsed units in BOTH FathomNet "
          f"and the recorded-group sources: {len(shared)}")
    print("  (FathomNet rows rest on inferred groups; never pooled with the "
          "rows above)")

    with open(os.path.join(args.out, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=lambda o: sorted(o)
                  if isinstance(o, set) else str(o))
    print(f"\nwrote {args.out}\\pool_inventory.csv and report.json "
          f"(pool membership per grid cell is in report.json)")
    print("Nothing was reserved; nothing in the collation was changed.")


if __name__ == "__main__":
    main()

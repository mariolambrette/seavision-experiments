#!/usr/bin/env python3
"""WP10: make every draw once, from the committed config, and save it.

Every backbone, readout and geometry is then scored on these same draws, so
all comparisons are paired (decision 9). Nothing here reads an embedding.

One draw, for one cell (pair, direction, level, arm, matching, variant) and
one taxon:

  1. size matching (6): match the shape of the two sources' short-side
     distributions, randomly per draw (unmatched cells skip this);
  2. k_t = the largest k at which the taxon is testable (4): query source
     passes the WP8 usable rule, reference source holds >= k units in >= 2
     deployments;
  3. within-source baseline: the query source's deployments are shuffled and
     taken as reference deployments until they hold >= k_t units; the rest
     supply the QUERIES. If the remaining deployments hold < q_min units the
     shuffle is retried (up to 50 times), then k_t steps down a level;
  4. baseline references = the reference deployments' units in random order;
     cross-source references = the reference source's units in random order.
     A random order samples in proportion to species abundance and to
     deployment holdings (1, 4). Taking the first k of each list makes k
     nested within the draw (3).
  5. the same queries serve both arms at every k (4).

Output (--out): units.csv.gz (the representative units, decision 13), and one
<cell>.npz per cell holding, per draw and taxon, k_t and the three index
lists (queries, baseline references, cross-source references), plus
manifest.json with the config's sha256, counts and checks.

    python -m evaluate.wp10_draws --config configs\\evaluate\\wp10.yaml ^
        --out D:\\marineai\\classification-experiments\\results\\wp10\\draws
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

from evaluate import wp10_data as D

RETRIES = 50


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def draw_taxon(U, iq0, ir0, cell, cfg, d, taxon):
    """-> (k_t, queries, base_refs, cross_refs) or None."""
    ks, q, mg = cfg["k"], cfg["q_min"], cfg["ref_min_groups"]
    seed = cfg["seed"]
    if cell["matching"] == "matched":
        rng_m = np.random.default_rng(D.seed_for(seed, d, cell["id"], taxon,
                                                 "match"))
        iq, ir = D.match_shape(U, iq0, ir0,
                               cfg["size_matching"]["short_side_bins"], rng_m)
    else:
        iq, ir = iq0, ir0
    gq, gr = D.group_counts(U, iq), D.group_counts(U, ir)
    ks_ok = [k for k in ks if D.ok_query(gq, k, q) and D.ok_ref(gr, k, mg)]
    if not ks_ok:
        return None
    rng = np.random.default_rng(D.seed_for(seed, d, cell["id"], taxon,
                                           "split"))
    deps = np.array(sorted(gq))
    for kt in reversed(ks_ok):
        for _ in range(RETRIES):
            order = rng.permutation(deps)
            ref_deps, n = [], 0
            for g in order:
                if n >= kt:
                    break
                ref_deps.append(g)
                n += gq[int(g)]
            in_ref = np.isin(U.grp[iq], ref_deps)
            queries = iq[~in_ref]
            if n >= kt and len(queries) >= q:
                base = rng.permutation(iq[in_ref])[:kt]
                cross = rng.permutation(ir)[:kt]
                return kt, np.sort(queries), base, cross
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")
    cfg = D.load_config(args.config)
    t0 = time.time()
    print("reading locked-test records ...", flush=True)
    recs = D.read_records(cfg)
    units = D.representatives(recs)
    U = D.Units(units)
    print(f"  {len(recs):,} records -> {len(U):,} units "
          f"{dict(Counter(U.src.tolist()))}")
    os.makedirs(args.out)
    with gzip.open(os.path.join(args.out, "units.csv.gz"), "wt",
                   newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "key", "source", "group", "survey", "short_side",
                    "species", "genus", "family", "order"])
        for i in range(len(U)):
            tx = U.taxa[i]
            w.writerow([i, U.keys[i], U.src[i], U.grp_names[U.grp[i]],
                        U.survey[i], U.short[i], tx.get("species", ""),
                        tx.get("genus", ""), tx.get("family", ""),
                        tx.get("order", "")])

    variants = sorted({c["variant"] for c in D.cells(cfg)})
    pools = {v: D.make_pools(U, D.variant_mask(U, v)) for v in variants}
    nd = cfg["n_draws"]
    manifest = {"config_sha256": cfg["_sha256"], "seed": cfg["seed"],
                "n_draws": nd, "k": cfg["k"], "q_min": cfg["q_min"],
                "units": len(U), "cells": {}}
    problems = []
    for cell in D.cells(cfg):
        taxa = D.candidate_taxa(cell, pools)
        kt = np.zeros((nd, len(taxa)), np.int64)
        lists = {"q": [], "b": [], "c": []}
        offs = {k: np.zeros((nd, len(taxa) + 1), np.int64) for k in lists}
        for d in range(nd):
            pos = {k: 0 for k in lists}
            for j, t in enumerate(taxa):
                iq0, ir0 = D.cell_pools(U, cell, pools, t)
                r = draw_taxon(U, iq0, ir0, cell, cfg, d, t)
                if r is not None:
                    kt[d, j] = r[0]
                    for k, arr in zip("qbc", r[1:]):
                        lists[k].append(arr)
                        pos[k] += len(arr)
                        # invariants, checked per taxon per draw
                    qd = set(U.grp[r[1]].tolist())
                    bd = set(U.grp[r[2]].tolist())
                    if qd & bd:
                        problems.append(f"{cell['id']} d{d} {t}: baseline "
                                        f"references share a deployment "
                                        f"with queries")
                    if len(r[2]) != r[0] or len(r[3]) != r[0]:
                        problems.append(f"{cell['id']} d{d} {t}: reference "
                                        f"count != k_t")
                for k in lists:
                    offs[k][d, j + 1] = pos[k]
        cat = {k: (np.concatenate(v) if v else np.zeros(0, np.int64))
               for k, v in lists.items()}
        # offsets are per draw; make them absolute into the flat arrays
        base = {k: 0 for k in lists}
        for d in range(nd):
            for k in lists:
                offs[k][d] += base[k]
                base[k] = offs[k][d, -1]
        fname = safe(cell["id"]) + ".npz"
        np.savez_compressed(os.path.join(args.out, fname),
                            taxa=np.array(taxa, dtype=object), kt=kt,
                            q=cat["q"], q_off=offs["q"], b=cat["b"],
                            b_off=offs["b"], c=cat["c"], c_off=offs["c"])
        testable = {int(k): int((kt[0] >= k).sum()) for k in cfg["k"]}
        manifest["cells"][cell["id"]] = {**cell, "file": fname,
                                         "taxa": len(taxa),
                                         "testable_draw0": testable}
        print(f"  {cell['id']:<70} taxa {len(taxa):>3}  testable@k "
              f"{testable}", flush=True)
    manifest["problems"] = problems
    manifest["passed"] = not problems
    manifest["seconds"] = round(time.time() - t0)
    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    for p in problems[:20]:
        print("  !", p)
    print(f"{'PASSED' if not problems else 'FAILED'}: {len(manifest['cells'])}"
          f" cells x {nd} draws in {manifest['seconds']} s -> {args.out}")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()

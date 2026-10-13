#!/usr/bin/env python3
"""WP10: score one backbone on the saved draws. One backbone per command.

For every readout of the backbone (configs/extract/wp9_readouts.yaml) and
every geometry it holds that WP10 uses, and every cell, draw, k and taxon set
(per_k, fixed_k50, fixed_k100), nearest prototype on L2-normalised raw
embeddings (7), both arms on the same queries:

  cross     prototypes from the first k cross-source references
  baseline  prototypes from the first k within-source references

and macro recall over the candidate taxa (9). A cell-k-set with fewer than
min_candidates taxa is not scored; its count is still written (14). A cell is
skipped for a geometry whose shard set does not hold one of its sources
(FishWIO has no square sets) -- recorded, not silently dropped.

Also the source-level leakage margin (10): for each pair and level, on the
matched whole-taxon units of draw 0 (both directions' queries), mean cosine of
same-taxon-different-source pairs minus same-source-different-taxon pairs,
via evaluate/leakage.py with source in place of deployment.

Refuses to run if the draws were made from a different config, or (unless
--allow-dirty, for tests) from an uncommitted working tree.

    python -m evaluate.wp10_score --config configs\\evaluate\\wp10.yaml ^
        --draws D:\\marineai\\classification-experiments\\results\\wp10\\draws ^
        --backbone dinov3_vitl16
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

from evaluate import wp10_data as D
from evaluate.leakage import leakage, sample_pairs
from evaluate.wp9_store import Store, l2

TAXON_SETS = ("per_k", "fixed_k50", "fixed_k100")


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            text=True).strip())
    except Exception:                              # noqa: BLE001
        return "unknown", True
    return commit, dirty


def load_units(draws_dir):
    keys, src, taxa = [], [], []
    with gzip.open(os.path.join(draws_dir, "units.csv.gz"), "rt",
                   newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            keys.append(r["key"])
            src.append(r["source"])
            taxa.append({lv: r[lv] for lv in ("species", "genus", "family",
                                              "order") if r[lv]})
    return keys, np.array(src), taxa


def candidates(kt_row, k, tset):
    if tset == "per_k":
        return np.flatnonzero(kt_row >= k)
    need = 50 if tset == "fixed_k50" else 100
    if k > need:
        return np.zeros(0, np.int64)
    return np.flatnonzero(kt_row >= need)


def proto_matrix(X, row_of, refs, taxa_idx, k):
    """[len(taxa_idx), d] prototypes: L2-normalised mean of the first k
    references of each taxon (features already L2-normalised)."""
    return l2(np.stack([X[row_of[refs[j][:k]]].mean(0) for j in taxa_idx]))


def recall_from_scores(S, qlab, cand_cols):
    """S [nq, n_cols] scores against every taxon testable at this k; qlab the
    column of each query's true taxon. Restrict to the candidate columns and
    the queries belonging to them; -> (macro recall, per-candidate recall)."""
    cand_cols = np.asarray(cand_cols)
    keep = np.isin(qlab, cand_cols)
    pred = cand_cols[S[keep][:, cand_cols].argmax(1)]
    true = qlab[keep]
    per = [float((pred[true == c] == c).mean()) for c in cand_cols]
    return float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="tests only: run from an uncommitted tree")
    args = ap.parse_args()
    cfg = D.load_config(args.config)
    with open(os.path.join(args.draws, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    if man["config_sha256"] != cfg["_sha256"]:
        sys.exit("draws were made from a different config; remake them")
    if not man.get("passed"):
        sys.exit("the draws manifest did not pass its checks")
    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        sys.exit("uncommitted changes in the working tree; commit first")
    with open(cfg["readouts"], encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)["backbones"][args.backbone]
    out = args.out or os.path.join(cfg["out_dir"], "scores", args.backbone)
    os.makedirs(out, exist_ok=True)

    keys, usrc, utaxa = load_units(args.draws)
    geoms = [g for g in spec["geometries"] if g in cfg["geometries"]]
    k_list, min_c = cfg["k"], cfg["min_candidates"]
    t0 = time.time()
    for g in geoms:
        done = os.path.join(out, f"{g}.done.json")
        if os.path.exists(done):
            print(f"{g}: already scored, skipping")
            continue
        store = Store(cfg["embeddings"], args.backbone, spec["size"], g)
        present = np.array([store.has(k) for k in keys])
        src_present = {s: bool(present[usrc == s].all())
                       for s in set(usrc.tolist())}
        partial = {s: (present[usrc == s].any() and
                       not present[usrc == s].all())
                   for s in set(usrc.tolist())}
        if any(partial.values()):
            sys.exit(f"{g}: some but not all units of "
                     f"{[s for s, v in partial.items() if v]} are in the "
                     f"store; WP9 output is incomplete")
        avail = np.flatnonzero(present)
        row_of = np.full(len(keys), -1, np.int64)
        row_of[avail] = np.arange(len(avail))
        rows, taxon_rows, leak_rows, skipped = [], [], [], []
        for ro in spec["readouts"]:
            name = f"{ro['token']}@{ro['layer']}"
            X = l2(store.matrix([keys[i] for i in avail], ro["token"],
                                ro["layer"]))
            for cid, info in man["cells"].items():
                if not (src_present[info["query_source"]]
                        and src_present[info["reference_source"]]):
                    skipped.append(cid)
                    continue
                z = np.load(os.path.join(args.draws, info["file"]),
                            allow_pickle=True)
                taxa, kt = z["taxa"], z["kt"]
                for d in range(kt.shape[0]):
                    qs = [z["q"][z["q_off"][d, j]:z["q_off"][d, j + 1]]
                          for j in range(len(taxa))]
                    bs = [z["b"][z["b_off"][d, j]:z["b_off"][d, j + 1]]
                          for j in range(len(taxa))]
                    cs = [z["c"][z["c_off"][d, j]:z["c_off"][d, j + 1]]
                          for j in range(len(taxa))]
                    live = np.flatnonzero(kt[d] > 0)
                    if len(live) == 0:
                        Q, qtax = None, None
                    else:
                        Q = X[row_of[np.concatenate([qs[j] for j in live])]]
                        qtax = np.concatenate([np.full(len(qs[j]), j)
                                               for j in live])
                    for k in k_list:
                        cols = np.flatnonzero(kt[d] >= k)    # taxa usable at k
                        if len(cols) and Q is not None:
                            keep_q = np.isin(qtax, cols)
                            Qk, labk = Q[keep_q], qtax[keep_q]
                            col_of = {j: c for c, j in enumerate(cols)}
                            lab_col = np.array([col_of[j] for j in labk])
                            Sc = Qk @ proto_matrix(X, row_of, cs, cols, k).T
                            Sb = Qk @ proto_matrix(X, row_of, bs, cols, k).T
                        for tset in TAXON_SETS:
                            cand = candidates(kt[d], k, tset)
                            base_row = [args.backbone, name, g, cid,
                                        info["pair"], info["query_source"],
                                        info["reference_source"],
                                        info["level"], info["arm"],
                                        info["matching"], info["variant"],
                                        d, k, tset, len(cand)]
                            if len(cand) < min_c:
                                rows.append(base_row + [""] * 6)
                                continue
                            ccols = [col_of[j] for j in cand]
                            cross, per_c = recall_from_scores(Sc, lab_col, ccols)
                            base, per_b = recall_from_scores(Sb, lab_col, ccols)
                            nq = int(sum(len(qs[j]) for j in cand))
                            rows.append(base_row + [
                                nq, round(cross, 6), round(base, 6),
                                round(cross - base, 6),
                                round(cross / base, 6) if base > 0 else "",
                                round(1 / len(cand), 6)])
                            if tset == "per_k":
                                for c, j in enumerate(cand):
                                    taxon_rows.append(
                                        [args.backbone, name, g, cid, d, k,
                                         taxa[j], len(qs[j]),
                                         round(per_c[c], 6),
                                         round(per_b[c], 6)])
            # leakage margin (10): main variant, whole taxon, matched, draw 0
            for pair in sorted({i["pair"] for i in man["cells"].values()}):
                for lv in cfg["levels"]:
                    ids = [cid for cid, i in man["cells"].items()
                           if i["pair"] == pair and i["level"] == lv
                           and i["arm"] == "whole_taxon"
                           and i["matching"] == "matched"
                           and i["variant"] == "main"]
                    if not ids or any(cid in skipped for cid in ids):
                        continue
                    idx = []
                    for cid in ids:
                        z = np.load(os.path.join(args.draws,
                                                 man["cells"][cid]["file"]),
                                    allow_pickle=True)
                        idx.append(z["q"][z["q_off"][0, 0]:z["q_off"][0, -1]])
                    idx = np.unique(np.concatenate(idx))
                    if len(idx) < 2:
                        continue
                    lab = np.array([utaxa[i].get(lv, "") for i in idx])
                    src = usrc[idx]
                    a, b = sample_pairs(lab, src,
                                        n_pairs=cfg["leakage"]["pairs_per_type"],
                                        seed=cfg["seed"])
                    res = leakage(X[row_of[idx]], a, b)
                    leak_rows.append([args.backbone, name, g, pair, lv,
                                      round(res["same_species_diff_deployment"], 6),
                                      round(res["same_deployment_diff_species"], 6),
                                      round(res["margin"], 6),
                                      *res["n_pairs"]])
            print(f"  {g} {name}: {len(rows):,} rows "
                  f"({time.time() - t0:.0f} s)", flush=True)
        with gzip.open(os.path.join(out, f"{g}.results.csv.gz"), "wt",
                       newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["backbone", "readout", "geometry", "cell", "pair",
                        "query_source", "reference_source", "level", "arm",
                        "matching", "variant", "draw", "k", "taxon_set",
                        "n_candidates", "n_queries", "cross", "baseline",
                        "difference", "ratio", "chance"])
            w.writerows(rows)
        with gzip.open(os.path.join(out, f"{g}.taxa.csv.gz"), "wt",
                       newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["backbone", "readout", "geometry", "cell", "draw",
                        "k", "taxon", "n_queries", "cross", "baseline"])
            w.writerows(taxon_rows)
        with open(os.path.join(out, f"{g}.leakage.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["backbone", "readout", "geometry", "pair", "level",
                        "same_taxon_diff_source", "same_source_diff_taxon",
                        "margin", "n_pairs_same_taxon",
                        "n_pairs_same_source"])
            w.writerows(leak_rows)
        with open(done, "w", encoding="utf-8") as fh:
            json.dump({"backbone": args.backbone, "geometry": g,
                       "readouts": [f"{r['token']}@{r['layer']}"
                                    for r in spec["readouts"]],
                       "config_sha256": cfg["_sha256"], "git_commit": commit,
                       "dirty": dirty, "rows": len(rows),
                       "skipped_cells": sorted(set(skipped)),
                       "seconds": round(time.time() - t0)}, fh, indent=1)
        print(f"{g}: done, {len(rows):,} rows, "
              f"{len(set(skipped))} cells skipped (source not in this set)")


if __name__ == "__main__":
    main()

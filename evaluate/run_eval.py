#!/usr/bin/env python3
"""Score every readout of every extraction run on the shared draws.

Two subcommands:

  draws   make the reference/query draws for every pool, once, and save them
          (with their seed, settings and the labels file's sha256):

            python -m evaluate.run_eval draws configs\\evaluate\\wp8_sweep.yaml

  score   score extraction runs. Each run directory
          (<emb>/<backbone>/<size>px/<geometry>) gets results.csv and
          leakage.csv beside its manifest; a run that already has them is
          skipped, so the command can be re-run as extraction finishes:

            python -m evaluate.run_eval score configs\\evaluate\\wp8_sweep.yaml

results.csv is long format, one row per
  readout (token, layer) x pool x rule x k x draw x level x mode
with macro recall, number of classes and number of queries.

Pools per geometry: square sets hold no FishWIO (no frames), so
ozfish_fishwio is scored on letterbox, distort and native only; ozfish and
fathomnet are scored on every geometry.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

from . import classify, draws as drawmod, leakage as leak, metrics, pools
from .readouts import Run, l2

FIXED_GEOMS = {"letterbox", "distort", "native"}     # crops-set geometries


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def draws_path(cfg, pool):
    return os.path.join(cfg["draws_dir"], f"draws_{pool}.json")


def cmd_draws(cfg):
    labels = load_labels(cfg["labels"])
    os.makedirs(cfg["draws_dir"], exist_ok=True)
    for pool, sources in pools.POOLS.items():
        p = draws_path(cfg, pool)
        if os.path.exists(p):
            print(f"  {pool}: exists, kept ({p})")
            continue
        sp = pools.evaluable_species(labels, sources, cfg["k_max"],
                                     cfg["q_min"])
        dr = drawmod.make_draws(labels, sp, cfg["n_draws"], cfg["k_max"],
                                cfg["q_min"], cfg["seed"])
        drawmod.check_draws(dr)
        meta = {"pool": pool, "sources": sorted(sources), "seed": cfg["seed"],
                "n_draws": cfg["n_draws"], "k_max": cfg["k_max"],
                "q_min": cfg["q_min"], "labels_sha256": sha256(cfg["labels"]),
                "n_species": len(sp),
                "made": datetime.datetime.now().isoformat(timespec="seconds")}
        drawmod.save(dr, p, meta)
        nq = [sum(len(s["query"]) for s in d.values()) for d in dr["draws"]]
        print(f"  {pool}: {len(sp)} evaluable species, {dr['redraws']} "
              f"redraws, failed {dr['failed'] or 'none'}, queries per draw "
              f"{min(nq):,}-{max(nq):,}")


# readouts are grouped so at most two token arrays are in memory at once
TOKEN_PLAN = [(["patch_mean"], ()),
              (["cls", "cls+patch_mean"], ("patch_mean",)),
              (["patch_mean_normed", "cls+patch_mean_normed"], ("cls",))]


def score_run(run_dir, cfg, labels_by_key, lineage, dev):
    run = Run(run_dir)
    geom = run.manifest["geometry"]
    b = run.manifest["backbone"]
    base = {"backbone": b["name"], "size": run.manifest["size"],
            "geometry": geom, "interpolated": b["interpolated"],
            "input_mode": b.get("input_mode", "fixed")}
    rows, lrows = [], []
    pool_names = [p for p in pools.POOLS
                  if p != "ozfish_fishwio" or geom in FIXED_GEOMS]
    # per pool: draws, taxonomy, the run rows needed, leakage pairs
    P = {}
    for pool in pool_names:
        dr = drawmod.load(draws_path(cfg, pool))
        species = sorted(dr["draws"][0]) if dr["draws"] else []
        species = [s for s in species if s not in dr["failed"]]
        tax = metrics.Taxonomy(species, lineage)
        keys = sorted({k for d in dr["draws"] for s in d.values()
                       for k in s["ref"] + s["query"]})
        missing = [k for k in keys if k not in run.row]
        if missing:
            raise RuntimeError(f"{run_dir}: {len(missing)} draw keys absent "
                               f"from the run (pool {pool}), e.g. {missing[:3]}")
        pos = {k: i for i, k in enumerate(keys)}
        rows_idx = [run.row[k] for k in keys]
        sp_arr = np.array([labels_by_key[k]["species"] for k in keys],
                          dtype=object)
        dep_arr = np.array([labels_by_key[k]["source"] + "|" +
                            labels_by_key[k]["deployment"] for k in keys],
                           dtype=object)
        ssdd, sdds = leak.sample_pairs(sp_arr, dep_arr, cfg["leak_pairs"],
                                       cfg["seed"])
        P[pool] = dict(dr=dr, tax=tax, keys=keys, pos=pos, rows=rows_idx,
                       ssdd=ssdd, sdds=sdds)
    readouts = run.readouts()
    plan = []
    for toks, keep in TOKEN_PLAN:
        plan.append(([r for r in readouts if r[0] in toks], keep))
    plan.append(([r for r in readouts if r[1] == "pooled"], ()))
    for group, keep in plan:
        run.release(keep=keep)
        for token, layer in group:
            for pool, p in P.items():
                X = l2(run.matrix(token, layer, p["rows"]))
                lk = leak.leakage(X, p["ssdd"], p["sdds"])
                lrows.append({**base, "pool": pool, "token": token,
                              "layer": layer, **{k: v for k, v in lk.items()
                                                 if k != "n_pairs"},
                              "n_pairs": "/".join(map(str, lk["n_pairs"]))})
                tax = p["tax"]
                for d, draw in enumerate(p["dr"]["draws"]):
                    q_keys = [k for s in tax.species for k in draw[s]["query"]]
                    q_sp = [labels_by_key[k]["species"] for k in q_keys]
                    Q = X[[p["pos"][k] for k in q_keys]]
                    for k in cfg["k"]:
                        r_keys, r_y = [], []
                        for i, s in enumerate(tax.species):
                            ref = draw[s]["ref"][:k]
                            r_keys += ref
                            r_y += [i] * len(ref)
                        R = X[[p["pos"][x] for x in r_keys]]
                        rules = [("prototype", classify.prototype_scores)]
                        if k in cfg["probe_k"]:
                            rules.append(("linear_probe",
                                          classify.probe_scores))
                        for rule, fn in rules:
                            S = fn(R, np.array(r_y), Q, len(tax.species), dev)
                            for m in metrics.score_all(S, q_sp, tax):
                                rows.append({**base, "pool": pool,
                                             "token": token, "layer": layer,
                                             "rule": rule, "k": k, "draw": d,
                                             **m})
    run.release()
    return rows, lrows


def write_csv(path, rows):
    if not rows:
        return
    tmp = path + ".part"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)        # never a half-written results file


def cmd_score(cfg, args):
    dev = torch.device(args.device or cfg.get("device", "cuda:0")
                       if torch.cuda.is_available() else "cpu")
    labels = load_labels(cfg["labels"])
    labels_by_key = {r["key"]: r for r in labels}
    lineage = {}
    for r in labels:
        if r["rank"] == "species" and r["species"]:
            lineage[r["species"]] = {"genus": r["genus"],
                                     "family": r["family"],
                                     "order": r["order"]}
    pattern = os.path.join(cfg["emb"], args.backbone or "*", "*", "*",
                           "manifest.json")
    runs = sorted(os.path.dirname(p) for p in glob.glob(pattern))
    print(f"{len(runs)} extraction runs found under {cfg['emb']}")
    for rd in runs:
        out = os.path.join(rd, "results.csv")
        if os.path.exists(out):
            print(f"  skip {rd} (scored)")
            continue
        t0 = time.time()
        print(f"  scoring {rd} ...", flush=True)
        rows, lrows = score_run(rd, cfg, labels_by_key, lineage, dev)
        write_csv(os.path.join(rd, "leakage.csv"), lrows)
        write_csv(out, rows)
        print(f"    {len(rows):,} result rows, {time.time() - t0:,.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["draws", "score"])
    ap.add_argument("config")
    ap.add_argument("--backbone", default=None, help="score only this one")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.cmd == "draws":
        cmd_draws(cfg)
    else:
        cmd_score(cfg, args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Summarise the readout sweep: the decision per backbone, the stability
checks, and the three panels.

Reads every results.csv / leakage.csv under the extraction root, and writes
to --out:
  results_all.csv.gz       every result row, all runs
  summary.csv              mean, min, max over draws, per configuration
  decision.csv             per backbone: the readout(s) chosen, and why
  stability.csv            per backbone: the three stability checks
  leakage_chosen.csv       the leakage diagnostic for the chosen readouts
  fathomnet_check.csv      does the reef-chosen readout hold on FathomNet?
  panels.png               A: readout spread; B: which axis matters;
                           C: recall by taxonomic level

The decision rule (fixed in configs/evaluate/wp8_summary.yaml BEFORE the
results are looked at; schedule, WP8):

  A readout is (size, token, layer). For each backbone, at each level
  (species, genus, family; unconstrained), its score per draw is the mean
  over the backbone's crops geometries -- paired, since every geometry uses
  the same draws -- in the decision pool, decision rule and decision k.
  The WINNER is the readout with the highest mean over draws, averaged over
  the three levels.
  "Within noise" of the best at a level = NOT significantly worse than it on
  a paired t-test over the draws (two-sided, alpha from the config, default
  0.05). Paired because every readout is scored on the same draws, so the
  shared draw-to-draw difficulty cancels; overlapping ranges (the first
  rule, replaced 7 October 2026 before the results were read) were far too
  permissive.
    case 1  the winner is within noise of the best at every level: it is
            the only readout kept.
    case 2  it is not: the winner is kept, plus the best readout at each
            level where the winner is significantly worse (WP9 keeps all).

Stability checks, each pass/fail: is the chosen readout within noise
(paired test, as above) of the best (i) at every other k, (ii) under the linear probe, (iii) on every
geometry separately (OzFish pool, the only one present in every geometry)?
Each also reports the Spearman correlation of readout means between the
decision condition and the alternative.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

# A readout configuration within a backbone is identified by "size|token|layer".
LEVELS = ["species", "genus", "family"]


def load_all(emb):
    parts, leaks = [], []
    for p in sorted(glob.glob(os.path.join(emb, "*", "*", "*", "results.csv"))):
        parts.append(pd.read_csv(p, dtype={"layer": str}))
        lp = os.path.join(os.path.dirname(p), "leakage.csv")
        if os.path.exists(lp):
            leaks.append(pd.read_csv(lp, dtype={"layer": str}))
    if not parts:
        sys.exit(f"no results.csv under {emb}")
    return pd.concat(parts, ignore_index=True), \
        (pd.concat(leaks, ignore_index=True) if leaks else pd.DataFrame())


def per_draw(df, cfg, pool, rule, k, mode="unconstrained", geoms=None):
    """-> DataFrame indexed by KEY + level, columns = draw; values = recall
    averaged over the selected geometries (paired: same draws)."""
    d = df[(df.pool == pool) & (df.rule == rule) & (df.k == k) &
           (df["mode"] == mode) & df.level.isin(LEVELS)]
    if geoms is not None:
        d = d[d.geometry.isin(geoms)]
    d = d.assign(rid=d["size"].astype(str) + "|" + d.token + "|" +
                 d.layer.astype(str))
    g = d.groupby(["rid", "level", "draw"]).macro_recall.mean()
    return g.unstack("draw")


def parse(rid):
    size, token, layer = rid.split("|")
    return int(size), token, layer


def rid_of(r):
    return f"{r['size']}|{r['token']}|{r['layer']}"


def stats(w):
    return pd.DataFrame({"mean": w.mean(axis=1), "std": w.std(axis=1),
                         "min": w.min(axis=1), "max": w.max(axis=1)})


T_CRIT = {}


def t_crit(n, alpha):
    """Two-sided critical t for n paired draws (n - 1 degrees of freedom)."""
    key = (n, alpha)
    if key not in T_CRIT:
        from scipy.stats import t
        T_CRIT[key] = float(t.ppf(1 - alpha / 2, n - 1))
    return T_CRIT[key]


def worse(w, r, best, alpha):
    """Is readout r significantly worse than `best`? Paired t-test over the
    draws both were scored on. w: per-draw recall, readouts x draws."""
    d = (w.loc[best] - w.loc[r]).dropna()
    n = len(d)
    if n < 2 or r == best:
        return False
    m, sd = float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        return m > 0
    return m / (sd / np.sqrt(n)) > t_crit(n, alpha)


def tie_set(w, alpha):
    """w: per-draw recall for one backbone and level (readouts x draws).
    -> (readouts not significantly worse than the best, the best)."""
    best = w.mean(axis=1).idxmax()
    return {r for r in w.index if not worse(w, r, best, alpha)}, best


def decide(df, cfg):
    crops = set(cfg["crops_geometries"])
    alpha = cfg.get("alpha", 0.05)
    out = []
    for bb in sorted(df.backbone.unique()):
        d = df[df.backbone == bb]
        geoms = sorted(set(d.geometry) & crops)
        W = per_draw(d, cfg, cfg["decision_pool"], cfg["decision_rule"],
                     cfg["decision_k"], geoms=geoms)
        st = stats(W)
        ties, bests = {}, {}
        for lv in LEVELS:
            ties[lv], bests[lv] = tie_set(W.xs(lv, level="level"), alpha)
        lv_mean = st["mean"].unstack("level")[LEVELS].mean(axis=1)
        winner = lv_mean.idxmax()
        short = [lv for lv in LEVELS if winner not in ties[lv]]
        case = 1 if not short else 2
        keep = [winner] + sorted({bests[lv] for lv in short} - {winner})
        for r in keep:
            size, token, layer = parse(r)
            row = {"backbone": bb, "case": case, "size": size, "token": token,
                   "layer": layer, "geometries_averaged": "+".join(geoms),
                   "role": "winner" if r == winner else "best at " + "+".join(
                       lv for lv in short if bests[lv] == r),
                   "best_at": "+".join(lv for lv in LEVELS if bests[lv] == r)
                   or "none (within noise at every level)"}
            for lv in LEVELS:
                s = st.xs(lv, level="level")
                row[f"{lv}_mean"] = round(float(s.loc[r, "mean"]), 4)
                row[f"{lv}_std"] = round(float(s.loc[r, "std"]), 4)
                row[f"{lv}_min"] = round(float(s.loc[r, "min"]), 4)
                row[f"{lv}_max"] = round(float(s.loc[r, "max"]), 4)
                row[f"{lv}_best_mean"] = round(float(s.loc[bests[lv], "mean"]),
                                               4)
                row[f"{lv}_within_noise"] = r in ties[lv]
            out.append(row)
    return pd.DataFrame(out)


def spearman(a, b):
    j = a.index.intersection(b.index)
    if len(j) < 3:
        return float("nan")
    return float(a[j].rank().corr(b[j].rank()))


def stability(df, dec, cfg):
    crops = set(cfg["crops_geometries"])
    rows = []
    for _, r in dec.iterrows():
        bb = r.backbone
        d = df[df.backbone == bb]
        geoms = sorted(set(d.geometry) & crops)
        chosen = rid_of(r)
        base = stats(per_draw(d, cfg, cfg["decision_pool"],
                              cfg["decision_rule"], cfg["decision_k"],
                              geoms=geoms))
        conds = [("k", f"k={k}", dict(pool=cfg["decision_pool"],
                                      rule=cfg["decision_rule"], k=k,
                                      geoms=geoms))
                 for k in cfg["all_k"] if k != cfg["decision_k"]]
        conds.append(("rule", "linear_probe",
                      dict(pool=cfg["decision_pool"], rule="linear_probe",
                           k=cfg["decision_k"], geoms=geoms)))
        for g in sorted(set(d.geometry)):
            conds.append(("geometry", g, dict(pool=cfg["geometry_pool"],
                                              rule=cfg["decision_rule"],
                                              k=cfg["decision_k"], geoms=[g])))
        for check, cond, kw in conds:
            W = per_draw(d, cfg, kw["pool"], kw["rule"], kw["k"],
                         geoms=kw["geoms"])
            st = stats(W)
            if st.empty:
                continue
            for lv in LEVELS:
                s = st.xs(lv, level="level")
                b0 = base.xs(lv, level="level")["mean"]
                if chosen not in s.index:
                    continue
                ties, best = tie_set(W.xs(lv, level="level"),
                                     cfg.get("alpha", 0.05))
                rows.append({"backbone": bb, "chosen": chosen, "check": check, "condition": cond,
                             "level": lv, "pass": chosen in ties,
                             "chosen_mean": round(float(s.loc[chosen, "mean"]),
                                                  4),
                             "chosen_std": round(float(s.loc[chosen, "std"]),
                                                 4),
                             "best_mean": round(float(s.loc[best, "mean"]), 4),
                             "best_std": round(float(s.loc[best, "std"]), 4),
                             "best": best,
                             "spearman_vs_decision": round(
                                 spearman(b0, s["mean"]), 3)})
    return pd.DataFrame(rows)


def fathomnet_check(df, dec, cfg):
    crops = set(cfg["crops_geometries"])
    rows = []
    for _, r in dec.iterrows():
        d = df[df.backbone == r.backbone]
        geoms = sorted(set(d.geometry) & crops)
        W = per_draw(d, cfg, "fathomnet", cfg["decision_rule"],
                     cfg["decision_k"], geoms=geoms)
        st = stats(W)
        if st.empty:
            continue
        chosen = rid_of(r)
        for lv in LEVELS:
            s = st.xs(lv, level="level")
            if chosen not in s.index:
                continue
            ties, best = tie_set(W.xs(lv, level="level"),
                                 cfg.get("alpha", 0.05))
            rows.append({"backbone": r.backbone, "level": lv,
                         "chosen_within_noise_of_fathomnet_best":
                         chosen in ties,
                         "chosen_mean": round(float(s.loc[chosen, "mean"]), 4),
                         "chosen_std": round(float(s.loc[chosen, "std"]), 4),
                         "fathomnet_best": best,
                         "fathomnet_best_mean": round(float(s.loc[best, "mean"]),
                                                      4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ panels
def panels(df, dec, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#ececE7", "#fcfcfb"
    S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#d8d7d2"})
    crops = set(cfg["crops_geometries"])
    bbs = sorted(df.backbone.unique())
    fig = plt.figure(figsize=(12.4, 4.2 + 1.6 * len(bbs)))
    gs = GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.45, wspace=0.25,
                  left=0.16, right=0.93, top=0.93, bottom=0.07)
    # A: every readout, display metric, with the chosen one marked
    axA = fig.add_subplot(gs[0, :])
    lvA, modeA = cfg["panel_a_level"], cfg["panel_a_mode"]
    tokcol = {}
    pal = [S1, S2, S3, S4, "#8a8985", "#b03a2e", "#6b4fbb"]
    bests = {}
    for i, bb in enumerate(bbs):
        d = df[df.backbone == bb]
        geoms = sorted(set(d.geometry) & crops)
        st = stats(per_draw(d, cfg, cfg["decision_pool"], cfg["decision_rule"],
                            cfg["decision_k"], mode=modeA, geoms=geoms))
        s = st.xs(lvA, level="level").reset_index()
        s[["size", "token", "layer"]] = pd.DataFrame(
            [parse(x) for x in s.rid], index=s.index)
        y = len(bbs) - 1 - i
        axA.plot([s["mean"].min(), s["mean"].max()], [y, y], color="#dcdbd6",
                 lw=6, solid_capstyle="round", zorder=1)
        rng = np.random.default_rng(i)
        for _, row in s.iterrows():
            c = tokcol.setdefault(row.token, pal[len(tokcol) % len(pal)])
            axA.scatter(row["mean"], y + rng.uniform(-.17, .17), s=18, color=c,
                        edgecolor=SURF, lw=.6, zorder=3)
        for _, r in dec[dec.backbone == bb].iterrows():
            m = s[s.rid == rid_of(r)]["mean"]
            if len(m):
                axA.scatter(m, [y], marker="D", s=60, facecolor="none",
                            edgecolor=INK, lw=1.6, zorder=4)
        bests[bb] = s["mean"].max()
        axA.text(s["mean"].max() + .005, y, f"spread {s['mean'].max() - s['mean'].min():.3f}",
                 va="center", fontsize=8, color=INK2)
    axA.set_yticks(range(len(bbs)))
    axA.set_yticklabels(bbs[::-1], color=INK)
    axA.set_xlabel(f"macro recall, {lvA} ({modeA}), k = {cfg['decision_k']}, "
                   f"{cfg['decision_pool']}; mean over draws and crops geometries",
                   fontsize=8)
    axA.grid(axis="x", color=GRID)
    axA.set_title("A.  Readout spread within each backbone (◇ = chosen); "
                  f"spread between the bests {max(bests.values()) - min(bests.values()):.3f}",
                  loc="left", fontsize=11, weight="bold", color=INK)
    for t, c in tokcol.items():
        axA.scatter([], [], color=c, s=18, label=t)
    axA.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=6,
               bbox_to_anchor=(0, -0.2), handletextpad=0.2)
    # B: range attributable to each axis
    axB = fig.add_subplot(gs[1, 0])
    axes = [("token", S1), ("layer", S2), ("size", S4)]
    for i, bb in enumerate(bbs):
        d = df[df.backbone == bb]
        geoms = sorted(set(d.geometry) & crops)
        st = stats(per_draw(d, cfg, cfg["decision_pool"], cfg["decision_rule"],
                            cfg["decision_k"], mode=modeA, geoms=geoms))
        s = st.xs(lvA, level="level").reset_index()
        s[["size", "token", "layer"]] = pd.DataFrame(
            [parse(x) for x in s.rid], index=s.index)
        s = s[s.layer != "pooled"]
        y0 = len(bbs) - 1 - i
        for j, (ax, c) in enumerate(axes):
            m = s.groupby(ax)["mean"].mean()
            r = float(m.max() - m.min()) if len(m) > 1 else 0.0
            yy = y0 + (1 - j) * 0.26
            axB.barh(yy, r, height=.22, color=c, edgecolor=SURF)
            axB.text(r + .001, yy, f"{r:.3f}", va="center", fontsize=7,
                     color=INK2)
    axB.set_yticks(range(len(bbs)))
    axB.set_yticklabels(bbs[::-1], color=INK)
    axB.set_xlabel("range in mean recall attributable to that axis alone")
    axB.grid(axis="x", color=GRID)
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(color=c, label=a) for a, c in axes],
               frameon=False, fontsize=8, loc="lower right")
    axB.set_title("B.  Which choice is doing the work", loc="left",
                  fontsize=11, weight="bold", color=INK)
    # C: recall by level, unconstrained; chosen readout(s) highlighted
    nrow = int(np.ceil(len(bbs) / 3))
    sub = gs[1, 1].subgridspec(nrow + 1, 3, hspace=.6, wspace=.3,
                               height_ratios=[0.12] + [1] * nrow)
    head = fig.add_subplot(sub[0, :])
    head.axis("off")
    head.set_title("C.  Recall by level (unconstrained)", loc="left",
                   fontsize=11, weight="bold", color=INK)
    head.text(0, 0.1, "dark = one readout chosen; colours = one kept per level",
              fontsize=7.5, color=INK2, transform=head.transAxes)
    for i, bb in enumerate(bbs):
        ax = fig.add_subplot(sub[1 + i // 3, i % 3])
        d = df[df.backbone == bb]
        geoms = sorted(set(d.geometry) & crops)
        st = stats(per_draw(d, cfg, cfg["decision_pool"], cfg["decision_rule"],
                            cfg["decision_k"], geoms=geoms))["mean"]
        w = st.unstack("level")[LEVELS]
        for _, row in w.iterrows():
            ax.plot(range(3), row.values, color="#cfcec9", lw=.6, zorder=1)
        for j, (_, r) in enumerate(dec[dec.backbone == bb].iterrows()):
            key = rid_of(r)
            if key in w.index:
                ax.plot(range(3), w.loc[key].values, lw=2, zorder=3,
                        color=INK if r.case == 1 else [S1, S2, S3][j % 3])
        ax.set_xticks(range(3))
        ax.set_xticklabels(LEVELS if i // 3 == (len(bbs) - 1) // 3 else [],
                           fontsize=7)
        ax.set_title(bb, fontsize=8, loc="left", color=INK)
        ax.grid(axis="y", color=GRID)
        ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=160, facecolor=SURF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    os.makedirs(args.out, exist_ok=True)
    df, lk = load_all(cfg["emb"])
    df.to_csv(os.path.join(args.out, "results_all.csv.gz"), index=False)
    g = df.groupby(["backbone", "size", "geometry", "interpolated", "pool",
                    "token", "layer", "rule", "k", "level", "mode"])
    summ = g.macro_recall.agg(["mean", "min", "max", "std", "count"])
    summ.reset_index().to_csv(os.path.join(args.out, "summary.csv"),
                              index=False)
    dec = decide(df, cfg)
    dec.to_csv(os.path.join(args.out, "decision.csv"), index=False)
    stab = stability(df, dec, cfg)
    stab.to_csv(os.path.join(args.out, "stability.csv"), index=False)
    fathomnet_check(df, dec, cfg).to_csv(
        os.path.join(args.out, "fathomnet_check.csv"), index=False)
    if not lk.empty:
        sel = lk.merge(dec[["backbone", "size", "token", "layer"]],
                       on=["backbone", "size", "token", "layer"])
        sel.to_csv(os.path.join(args.out, "leakage_chosen.csv"), index=False)
    panels(df, dec, cfg, os.path.join(args.out, "panels.png"))
    # console summary
    print("\nDECISION")
    for _, r in dec.iterrows():
        print(f"  {r.backbone:<18} case {r.case}  {r.role[:7]:<7}  {r['size']}  {r.token:<22} "
              f"layer {r.layer:<6} "
              + "  ".join(f"{lv} {r[f'{lv}_mean']:.3f}"
                          f"{'' if r[f'{lv}_within_noise'] else '*'}"
                          for lv in LEVELS))
    print("  (* = significantly worse than the best at that level, paired "
          "t-test; case 2 also keeps that level's best)")
    print("\nSTABILITY (pass = chosen readout not significantly worse than "
          "the best, paired t-test)")
    if not stab.empty:
        t = stab.groupby(["backbone", "check"])["pass"].agg(["sum", "count"])
        for (bb, ch), v in t.iterrows():
            print(f"  {bb:<18} {ch:<9} {int(v['sum'])}/{int(v['count'])} pass")
    print(f"\nwrote decision.csv, stability.csv, fathomnet_check.csv, "
          f"leakage_chosen.csv, summary.csv, panels.png to {args.out}")


if __name__ == "__main__":
    main()

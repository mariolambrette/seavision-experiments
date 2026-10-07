#!/usr/bin/env python3
"""Check the WP9 readout spec against the WP8 summary, before the sweep
embeddings are deleted.

Fails loudly unless:
  1. every readout in the spec was actually scored in the sweep, for that
     backbone, at that size, on every geometry the spec lists;
  2. every readout WP8's decision kept (decision.csv) is in the spec.
Readouts the spec adds beyond decision.csv (the gate re-plans) are listed so
they are visible, not silently accepted.

    python -m evaluate.check_readout_spec configs\\extract\\wp9_readouts.yaml ^
        D:\\marineai\\scratch\\wp8_sweep\\summary
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("summary_dir")
    a = ap.parse_args()
    with open(a.spec, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)["backbones"]
    summ = pd.read_csv(os.path.join(a.summary_dir, "summary.csv"),
                       dtype={"layer": str})
    dec = pd.read_csv(os.path.join(a.summary_dir, "decision.csv"),
                      dtype={"layer": str})
    errors, extras = [], []
    for bb, s in spec.items():
        d = summ[(summ.backbone == bb) & (summ["size"] == s["size"])]
        if d.empty:
            errors.append(f"{bb}: nothing scored at size {s['size']}")
            continue
        for r in s["readouts"]:
            tok, lay = r["token"], str(r["layer"])
            got = set(d[(d.token == tok) & (d.layer == lay)].geometry)
            missing = set(s["geometries"]) - got
            if missing:
                errors.append(f"{bb} {tok}@{lay}: not scored on "
                              f"{sorted(missing)}")
        kept = {(r["token"], str(r["layer"])) for r in s["readouts"]}
        db = dec[dec.backbone == bb]
        for _, row in db.iterrows():
            if int(row["size"]) != s["size"]:
                errors.append(f"{bb}: decision size {row['size']} but spec "
                              f"size {s['size']}")
            if (row.token, row.layer) not in kept:
                errors.append(f"{bb}: decision keeps {row.token}@{row.layer}"
                              " but the spec does not")
        decided = {(t, str(l)) for t, l in zip(db.token, db.layer)}
        for t, l in sorted(kept - decided):
            extras.append(f"{bb}: {t}@{l} (beyond decision.csv)")
    for b in sorted(set(dec.backbone) - set(spec)):
        extras.append(f"{b}: in decision.csv, not in the spec (dropped)")
    for e in extras:
        print("  note   ", e)
    for e in errors:
        print("  ERROR  ", e)
    if errors:
        sys.exit(f"{len(errors)} error(s): spec and sweep disagree")
    n = sum(len(s["readouts"]) for s in spec.values())
    print(f"OK: {len(spec)} backbones, {n} readouts, all scored in the sweep "
          "and every kept decision covered")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end synthetic test of WP10 (draws + scoring) with known answers.

Builds a fake collation, a fake locked-test split and fake WP9 embeddings in
a temporary folder, then runs evaluate.wp10_draws and evaluate.wp10_score on
them three times, varying only how the embeddings are generated:

  noshift   v = taxonomy signal + noise       -> cross ~ baseline, margin > 0
  shift     v = per-source rotation of taxonomy + source offset
                                            -> cross << baseline, margin < 0
  noise     v = noise only                    -> recall ~ chance

Plus invariants: draws pass their own checks; FishWIO cells are skipped for
the square geometry (no frames); every scored candidate set has >= 3 taxa;
the '+' readout is built; unmatched testable counts agree with
scratch/wp10_counts.py on the same data.

    python tests/test_wp10.py
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 64


def build(tmp, mode):
    rng = np.random.default_rng(0)
    cats, cid = [], 1
    lineage = {}
    for o in range(2):
        for f in range(3):
            for g in range(3):
                for s in range(3):
                    cats.append({"id": cid, "name": f"G{o}{f}{g} s{s}",
                                 "rank": "Species",
                                 "lineage": {"order": f"O{o}",
                                             "family": f"F{o}{f}",
                                             "genus": f"G{o}{f}{g}"}})
                    lineage[cid] = (o, f, g, s)
                    cid += 1
    # one genus-rank category (counts at genus and above, never species)
    cats.append({"id": cid, "name": "G000", "rank": "Genus",
                 "lineage": {"order": "O0", "family": "F00", "genus": "G000"}})
    lineage[cid] = (0, 0, 0, None)
    gcid = cid
    ds = [{"id": 1, "name": "ozfish"}, {"id": 2, "name": "fishwio"},
          {"id": 3, "name": "yolo-bruv"}]
    imgs, anns, split = [], [], []
    iid = aid = 1
    sizes = {"ozfish": 120, "fishwio": 90, "yolo-bruv": 50, "fathomnet": 100}

    def species_for(src):
        # overlapping but not identical species sets per source
        lo, hi = {"ozfish": (1, 40), "fishwio": (10, 54), "yolo-bruv": (1, 6),
                  "fathomnet": (20, 54)}[src]
        return list(range(lo, hi + 1)) + [gcid]

    unit_meta = []                       # key, src, cid
    for d in range(60):                  # OzFish deployments
        sv = "ABEG"[d % 4]
        dep = f"{sv}{d:05d}"
        for _ in range(60):
            c = int(rng.choice(species_for("ozfish")))
            s = max(10, rng.normal(sizes["ozfish"], 30))
            for cam in "LR":
                uid = f"oz{iid}"
                imgs.append({"id": iid, "dataset_id": 1,
                             "file_name": uid + ".png",
                             "crop_provenance": "pre_cropped",
                             "groups": {"deployment": dep, "survey": sv},
                             "source_meta": {"camera": cam}})
                anns.append({"id": aid, "image_id": iid, "category_id": c,
                             "bbox": [0, 0, s, s * 1.3]})
                split.append([uid, "ozfish", "lock", dep, 0])
                unit_meta.append((uid, "ozfish", c))
                iid += 1
                aid += 1
    for d in range(30):                  # FishWIO shootings
        dep = f"fw{d}"
        for _ in range(80):
            c = int(rng.choice(species_for("fishwio")))
            s = max(10, rng.normal(sizes["fishwio"], 25))
            uid = f"fw{iid}"
            imgs.append({"id": iid, "dataset_id": 2, "file_name": uid + ".jpg",
                         "crop_provenance": "pre_cropped",
                         "groups": {"deployment": dep}, "source_meta": {}})
            anns.append({"id": aid, "image_id": iid, "category_id": c,
                         "bbox": [0, 0, s, s]})
            split.append([uid, "fishwio", "lock", dep, 0])
            unit_meta.append((uid, "fishwio", c))
            iid += 1
            aid += 1
    for d in range(10):                  # yolo-bruv frames, several boxes each
        dep = f"yb{d}"
        uid = f"yb{iid}"
        imgs.append({"id": iid, "dataset_id": 3, "file_name": uid + ".jpg",
                     "crop_provenance": "frame", "width": 1920,
                     "height": 1080, "groups": {"deployment": dep},
                     "source_meta": {}})
        for _ in range(20):
            c = int(rng.choice(species_for("yolo-bruv")))
            anns.append({"id": aid, "image_id": iid, "category_id": c,
                         "bbox": [0, 0, 50, 40]})
            key = f"{uid}-a{aid}"
            split.append([key, "yolo-bruv", "lock", dep, 0])
            unit_meta.append((key, "yolo-bruv", c))
            aid += 1
        iid += 1
    json.dump({"categories": cats, "datasets": ds, "images": imgs,
               "annotations": anns}, open(os.path.join(tmp, "m.json"), "w"))
    imgs, anns = [], []
    for g in range(40):                  # FathomNet inferred groups
        inst = ["MBARI", "NOAA NMFS SEFSC"][g % 2]
        for _ in range(70):
            c = int(rng.choice(species_for("fathomnet")))
            s = max(10, rng.normal(sizes["fathomnet"], 40))
            uid = f"fn{iid}"
            imgs.append({"id": iid, "file_name": uid + ".jpg",
                         "crop_provenance": "cut_from_frame",
                         "source_meta": {"owner_institution": inst,
                                         "fathomnet_image_uuid": uid,
                                         "frame_url": ""}})
            anns.append({"id": aid, "image_id": iid, "category_id": c,
                         "bbox": [0, 0, s, s]})
            split.append([uid, "fathomnet", "lock", f"g{g}", 0])
            unit_meta.append((uid, "fathomnet", c))
            iid += 1
            aid += 1
    json.dump({"categories": cats, "images": imgs, "annotations": anns},
              open(os.path.join(tmp, "f.json"), "w"))
    os.makedirs(os.path.join(tmp, "lock"), exist_ok=True)
    with gzip.open(os.path.join(tmp, "lock", "split_crops.csv.gz"), "wt",
                   newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "source", "split", "group", "heldout_genus"])
        w.writerows(split)

    # embeddings: hierarchical taxonomy signal (+ source shift) + noise
    basis = rng.normal(size=(200, DIM))
    src_vec = {s: rng.normal(size=DIM) for s in sizes}
    # per-source rotation: a source-specific distortion of the taxonomy
    # signal. A constant offset alone would NOT break transfer (it shifts all
    # of one source's prototypes equally, so rankings survive).
    rot = {s: np.linalg.qr(rng.normal(size=(DIM, DIM)))[0] for s in sizes}

    def emb(src, c):
        o, f, g, s = lineage[c]
        noise = rng.normal(size=DIM)
        if mode == "noise":
            return noise
        t = 3 * basis[o] + 2 * basis[10 + 3 * o + f] + \
            1.5 * basis[30 + 9 * o + 3 * f + g]
        if s is not None:
            t = t + 1.0 * basis[100 + 27 * o + 9 * f + 3 * g + s]
        if mode == "shift":
            return rot[src] @ t + 4 * src_vec[src] + 0.6 * noise
        return t + 5.0 * noise

    vecs = {k: emb(s, c) for k, s, c in unit_meta}
    for geom, sset in (("letterbox", "crops"), ("square_m00", "square-m00")):
        for src in sizes:
            ks = [k for k, s, _ in unit_meta if s == src]
            if sset != "crops" and src == "fishwio":
                continue                  # FishWIO has no frames
            d = os.path.join(tmp, "emb", "fake", "224px", geom, sset,
                             f"{sset}-{src}-000000")
            os.makedirs(d)
            a = np.stack([vecs[k] for k in ks]).astype(np.float16)
            np.save(os.path.join(d, "cls@3.npy"), a)
            np.save(os.path.join(d, "patch_mean@3.npy"),
                    (a + rng.normal(size=a.shape) * 0.1).astype(np.float16))
            open(os.path.join(d, "keys.txt"), "w").write("\n".join(ks))
            json.dump({"ok": True}, open(os.path.join(d, "done.json"), "w"))
    yaml.safe_dump({"backbones": {"fake": {
        "size": 224, "geometries": ["letterbox", "square_m00"],
        "readouts": [{"token": "cls", "layer": 3},
                     {"token": "cls+patch_mean", "layer": 3}]}}},
        open(os.path.join(tmp, "readouts.yaml"), "w"))
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "evaluate",
                                           "wp10.yaml")))
    cfg.update({"split_dir": os.path.join(tmp, "lock"),
                "coco": [os.path.join(tmp, "m.json"),
                         os.path.join(tmp, "f.json")],
                "embeddings": os.path.join(tmp, "emb"),
                "readouts": os.path.join(tmp, "readouts.yaml"),
                "out_dir": os.path.join(tmp, "out"),
                "k": [1, 5, 20], "n_draws": 3})
    yaml.safe_dump(cfg, open(os.path.join(tmp, "wp10.yaml"), "w"))


def run(*cmd):
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, text=True,
                       capture_output=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout


def results(tmp):
    rows = []
    for f in os.listdir(os.path.join(tmp, "out", "scores", "fake")):
        if f.endswith(".results.csv.gz"):
            with gzip.open(os.path.join(tmp, "out", "scores", "fake", f),
                           "rt") as fh:
                rows += list(csv.DictReader(fh))
    return rows


def leak(tmp):
    rows = []
    for f in os.listdir(os.path.join(tmp, "out", "scores", "fake")):
        if f.endswith(".leakage.csv"):
            rows += list(csv.DictReader(open(os.path.join(
                tmp, "out", "scores", "fake", f))))
    return rows


def mean(xs):
    xs = [float(x) for x in xs if x != ""]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}")

    try:
        sys.path.insert(0, ROOT)
        from extract.preprocess import SHARD_SET as ORIG
        from evaluate.wp9_store import SHARD_SET as COPY
        check("wp9_store.SHARD_SET == extract.preprocess.SHARD_SET",
              ORIG == COPY)
    except ImportError:
        print("  (torch not installed: SHARD_SET copy not compared here)")

    summary = {}
    for mode in ("noshift", "shift", "noise"):
        tmp = tempfile.mkdtemp(prefix=f"wp10_{mode}_")
        try:
            build(tmp, mode)
            cfgp = os.path.join(tmp, "wp10.yaml")
            run("-m", "evaluate.wp10_draws", "--config", cfgp,
                "--out", os.path.join(tmp, "draws"))
            man = json.load(open(os.path.join(tmp, "draws", "manifest.json")))
            run("-m", "evaluate.wp10_score", "--config", cfgp,
                "--draws", os.path.join(tmp, "draws"), "--backbone", "fake",
                "--allow-dirty")
            R = results(tmp)
            scored = [r for r in R if r["cross"] != ""]
            sel = [r for r in scored if r["pair"] == "ozfishxfishwio"
                   and r["arm"] == "whole_taxon" and r["matching"] == "matched"
                   and r["variant"] == "main" and r["taxon_set"] == "per_k"
                   and r["k"] == "20" and r["geometry"] == "letterbox"]
            summary[mode] = {
                "cross": mean(r["cross"] for r in sel),
                "baseline": mean(r["baseline"] for r in sel),
                "chance": mean(r["chance"] for r in sel),
                "margin": mean(r["margin"] for r in leak(tmp)
                               if r["pair"] == "ozfishxfishwio"),
                "rows": len(R), "scored": len(scored)}
            print(f"{mode}: {summary[mode]}")
            if mode == "noshift":
                check("draws pass their own checks", man["passed"])
                check("every scored cell has >= 3 candidates",
                      all(int(r["n_candidates"]) >= 3 for r in scored))
                done = json.load(open(os.path.join(
                    tmp, "out", "scores", "fake", "square_m00.done.json")))
                check("FishWIO cells skipped for square (no frames)",
                      all("fishwio" in c for c in done["skipped_cells"])
                      and len(done["skipped_cells"]) > 0)
                check("'+' readout scored",
                      any(r["readout"] == "cls+patch_mean@3" for r in scored))
                check("sensitivity variant present",
                      any(r["variant"] == "exclude_ozfish_survey_E"
                          for r in scored))
                # count script agreement on unmatched cells
                cnt = os.path.join(tmp, "counts")
                run("scratch/wp10_counts.py", "--config", cfgp, "--out", cnt)
                want = {}
                for r in csv.DictReader(open(os.path.join(cnt,
                                                          "wp10_counts.csv"))):
                    if r["matching"] == "unmatched" and \
                            r["taxon_set"] == "per_k":
                        want[(r["query_source"], r["reference_source"],
                              r["level"], r["arm"], int(r["k"]))] = int(r["taxa"])
                mism = 0
                for cid, info in man["cells"].items():
                    if info["matching"] != "unmatched" or \
                            info["variant"] != "main":
                        continue
                    for k, n in info["testable_draw0"].items():
                        key = (info["query_source"], info["reference_source"],
                               info["level"], info["arm"], int(k))
                        if key in want and want[key] != n:
                            mism += 1
                check("draws' unmatched testable counts == count script",
                      mism == 0, f"({mism} mismatches)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    s = summary
    check("noshift: baseline well above chance",
          s["noshift"]["baseline"] > s["noshift"]["chance"] + 0.3)
    check("noshift: cross close to baseline (within 0.1)",
          abs(s["noshift"]["cross"] - s["noshift"]["baseline"]) < 0.1)
    check("shift: cross well below baseline",
          s["shift"]["cross"] < s["shift"]["baseline"] - 0.1)
    check("noise: recall near chance",
          abs(s["noise"]["baseline"] - s["noise"]["chance"]) < 0.1
          and abs(s["noise"]["cross"] - s["noise"]["chance"]) < 0.1)
    check("leakage margin: positive without shift, negative with",
          s["noshift"]["margin"] > 0 > s["shift"]["margin"])
    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()

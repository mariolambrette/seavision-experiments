"""WP10 data rules, in one place: records, units, pools, size matching,
eligibility. Every rule here is a decision in the project record "WP10 config
decisions" (numbers in brackets) and a setting in configs/evaluate/wp10.yaml.

Shared by wp10_draws.py (which makes the draws) and the tests. The read-only
count script scratch/wp10_counts.py predates this module and implements the
same rules; tests/test_wp10.py checks the two agree.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import yaml

SEFSC = "NOAA NMFS SEFSC"
SEFSC_SECOND = re.compile(
    r"/([^/]+)/[^/]+\.(?:mp4|avi)\.(\d\d)\.(\d\d)\.(\d\d)\.\d+\.jpg$", re.I)
RANK_ORDER = {"order": 0, "family": 1, "genus": 2, "species": 3}


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def seed_for(*parts):
    """A stable 64-bit seed from any labels (seed, draw, cell, taxon ...)."""
    return int(hashlib.sha256(":".join(map(str, parts)).encode())
               .hexdigest()[:16], 16)


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_sha256"] = sha256(path)
    return cfg


def taxa_of(cat):
    """level -> name, from the lineage (never the name), only at or above the
    category's own rank (12: a crop counts at its level and every coarser)."""
    lin = cat.get("lineage") or {}
    rank = str(cat.get("rank", "")).lower()
    out = {lv: lin[lv] for lv in ("order", "family", "genus") if lin.get(lv)}
    if rank == "species":
        out["species"] = cat.get("name")
    if rank in RANK_ORDER:
        out = {lv: v for lv, v in out.items()
               if RANK_ORDER[lv] <= RANK_ORDER[rank]}
    return out


def read_records(cfg):
    """Locked-test annotation records (9b), one dict each."""
    keep = {}
    with gzip.open(os.path.join(cfg["split_dir"], "split_crops.csv.gz"),
                   "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in cfg["use_splits"]:
                keep[r["key"]] = (r["source"], r["group"])
    recs = []
    for path in cfg["coco"]:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        cats = {c["id"]: c for c in doc["categories"]}
        by_image = defaultdict(list)
        for an in doc["annotations"]:
            by_image[an["image_id"]].append(an)
        for im in doc["images"]:
            uid = os.path.splitext(im["file_name"])[0]
            sm = im.get("source_meta") or {}
            g = im.get("groups") or {}
            frame = im.get("crop_provenance") == "frame"
            for an in by_image.get(im["id"], []):
                c = cats.get(an.get("category_id"))
                if not c:
                    continue
                key = f"{uid}-a{an['id']}" if frame else uid
                if key in keep:
                    src, grp = keep[key]
                    if src == "fathomnet":
                        m = SEFSC_SECOND.search(sm.get("frame_url") or "") \
                            if sm.get("owner_institution") == SEFSC else None
                        unit = "|".join(m.groups()) if m \
                            else sm.get("fathomnet_image_uuid")
                    else:
                        unit = None
                    recs.append({
                        "key": key, "src": src, "grp": grp,
                        "cid": an["category_id"], "taxa": taxa_of(c),
                        "short": min(an["bbox"][2], an["bbox"][3]),
                        "camera": "L" if str(sm.get("camera") or "").upper()
                        == "L" else "R",
                        "survey": g.get("survey") or "", "unit": unit})
                if not frame:
                    break                       # the image IS one crop
        del doc, by_image
    return recs


def representatives(recs):
    """(13) One representative crop per unit. OzFish: one camera per
    deployment x category (more crops; left if tied). FathomNet: first crop
    by key per collapsed unit x category. FishWIO, yolo-bruv: every crop."""
    cam = defaultdict(Counter)
    for r in recs:
        if r["src"] == "ozfish":
            cam[(r["grp"], r["cid"])][r["camera"]] += 1
    chosen = {k: ("L" if c["L"] >= c["R"] else "R") for k, c in cam.items()}
    seen, out = set(), []
    for r in sorted(recs, key=lambda r: r["key"]):
        if r["src"] == "ozfish" and r["camera"] != chosen[(r["grp"], r["cid"])]:
            continue
        if r["src"] == "fathomnet":
            u = (r["grp"], r["unit"], r["cid"])
            if u in seen:
                continue
            seen.add(u)
        out.append(r)
    return out


class Units:
    """The representative units as arrays, so pools are index arrays and a
    draw's random choices come from one seeded numpy generator, not per-unit
    hashes (which would take hours over FathomNet's ~100,000 units)."""
    def __init__(self, units):
        self.keys = [u["key"] for u in units]
        self.src = np.array([u["src"] for u in units])
        self.survey = np.array([u["survey"] for u in units])
        self.short = np.array([u["short"] for u in units], np.float64)
        grps = sorted({u["grp"] for u in units})
        self.grp_names = grps
        code = {g: i for i, g in enumerate(grps)}
        self.grp = np.array([code[u["grp"]] for u in units], np.int64)
        self.taxa = [u["taxa"] for u in units]

    def __len__(self):
        return len(self.keys)


def variant_mask(U, variant):
    """Sensitivity variants (14). 'main' keeps everything."""
    if variant == "main":
        return np.ones(len(U), bool)
    if variant == "exclude_ozfish_survey_E":
        return ~((U.src == "ozfish") & (U.survey == "E"))
    raise ValueError(f"unknown variant {variant}")


def make_pools(U, mask):
    """-> pool[(src, level, taxon)] = index array of every unit labelled at
    that level or finer (12); sp_pool the same restricted to species-labelled
    units, for the shared-children control (1a)."""
    pool, sp_pool = defaultdict(list), defaultdict(list)
    for i in np.flatnonzero(mask):
        tx = U.taxa[i]
        for lv, t in tx.items():
            pool[(U.src[i], lv, t)].append(i)
            if "species" in tx:
                sp_pool[(U.src[i], lv, t)].append(i)
    return ({k: np.array(v, np.int64) for k, v in pool.items()},
            {k: np.array(v, np.int64) for k, v in sp_pool.items()})


def match_shape(U, ia, ib, bins, rng):
    """(6, corrected) Match the SHAPE of the short-side distribution, not the
    counts. Common shape p_i = min(share_A_i, share_B_i), renormalised; each
    source keeps as many units as that shape allows (its binding bin), chosen
    at random by `rng`. The larger source keeps its larger total."""
    if len(ia) == 0 or len(ib) == 0:
        return ia[:0], ib[:0]
    edges = np.asarray(bins[1:-1], np.float64)
    ba = np.searchsorted(edges, U.short[ia], side="right")
    bb = np.searchsorted(edges, U.short[ib], side="right")
    nb = len(bins) - 1
    ca, cb = np.bincount(ba, minlength=nb), np.bincount(bb, minlength=nb)
    m = np.minimum(ca / len(ia), cb / len(ib))
    if m.sum() == 0:
        return ia[:0], ib[:0]
    p = m / m.sum()
    live = np.flatnonzero(p > 0)
    out = []
    for idx, b, c in ((ia, ba, ca), (ib, bb, cb)):
        n = np.min(c[live] / p[live])
        kept = [rng.permutation(idx[b == i])[:min(int(c[i]),
                                                   int(n * p[i] + 1e-9))]
                for i in live]
        out.append(np.sort(np.concatenate(kept)))
    return out[0], out[1]


def group_counts(U, idx):
    return Counter(U.grp[idx].tolist())


def ok_query(groups, k, q):
    """(4) WP8 usable rule for the query source."""
    tot = sum(groups.values())
    return bool(groups) and tot >= k + q and tot - max(groups.values()) >= q


def ok_ref(groups, k, min_groups):
    """(4) Reference source: >= k units in >= min_groups deployments."""
    return sum(groups.values()) >= k and \
        sum(1 for v in groups.values() if v > 0) >= min_groups


def cells(cfg):
    """Every scored condition: pair, direction, level, arm, matching,
    variant. A cell id is a stable string used to salt its draws."""
    out = []
    levels_for = {tuple(k.split(",")): v
                  for k, v in (cfg.get("pair_levels") or {}).items()}
    shared = {tuple(p) for p in cfg["arms"]["shared_children"]["pairs"]}
    sens = cfg.get("sensitivity", {}).get("exclude_ozfish_survey_E", {})
    sens_pairs = {tuple(p) for p in sens.get("pairs", [])}
    for a, b in (tuple(p) for p in cfg["pairs"]):
        levels = levels_for.get((a, b), cfg["levels"])
        for qsrc, rsrc in ((a, b), (b, a)):
            for lv in levels:
                arms = ["whole_taxon"]
                if (a, b) in shared and lv != "species":
                    arms.append("shared_children")
                for arm in arms:
                    for matching in ("matched", "unmatched"):
                        variants = ["main"]
                        if (a, b) in sens_pairs and arm == "whole_taxon" \
                                and matching in sens.get("matching", []):
                            variants.append("exclude_ozfish_survey_E")
                        for v in variants:
                            out.append({
                                "id": f"{qsrc}<-{rsrc}|{lv}|{arm}|"
                                      f"{matching}|{v}",
                                "pair": f"{a}x{b}", "query_source": qsrc,
                                "reference_source": rsrc, "level": lv,
                                "arm": arm, "matching": matching,
                                "variant": v})
    return out


def cell_pools(U, cell, pools_by_variant, taxon):
    """Query and reference index arrays for one taxon in one cell, before
    matching. The shared-children control keeps only species both hold."""
    pool, sp_pool = pools_by_variant[cell["variant"]]
    q, r, lv = cell["query_source"], cell["reference_source"], cell["level"]
    empty = np.zeros(0, np.int64)
    if cell["arm"] == "whole_taxon":
        return pool.get((q, lv, taxon), empty), pool.get((r, lv, taxon), empty)
    iq = sp_pool.get((q, lv, taxon), empty)
    ir = sp_pool.get((r, lv, taxon), empty)
    both = {U.taxa[i]["species"] for i in iq} & \
        {U.taxa[i]["species"] for i in ir}
    return (np.array([i for i in iq if U.taxa[i]["species"] in both], np.int64),
            np.array([i for i in ir if U.taxa[i]["species"] in both], np.int64))


def candidate_taxa(cell, pools_by_variant):
    pool, _ = pools_by_variant[cell["variant"]]
    lv = cell["level"]
    tq = {t for (s, l, t) in pool if s == cell["query_source"] and l == lv}
    tr = {t for (s, l, t) in pool if s == cell["reference_source"] and l == lv}
    return sorted(tq & tr)

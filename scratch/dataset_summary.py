#!/usr/bin/env python
"""
WP6 -- dataset summary for the SeaVision collation.

Reads one or more COCO files written by the converters and reports what the
collation actually contains. The question it exists to answer is whether the
collation supports species-level work: are there genera holding several
well-populated species, and do those species appear across more than one group?

Design notes, because they affect how the numbers should be read:

*   COCO files are loaded ONE AT A TIME and discarded once their aggregates
    are accumulated, so peak memory is one file rather than all of them.
    seavision_fathomnet.json is ~1.2 GB on disk and several times that parsed.

*   Crop size depends on crop_provenance. Where provenance is `frame` the
    image is a full frame and the crop is the bounding box, so size comes from
    bbox. Where provenance is `pre_cropped` or `cut_from_frame` the image IS
    the crop, so size comes from the image dimensions. Using one rule for both
    would report yolo-bruv crops as 1080 px.

*   Group values are namespaced by source. Two sources can both call something
    "deployment 1" and they are not the same deployment.

*   Genus is read from the category's lineage by case-insensitive key match,
    and there is NO fallback to splitting the binomial. WoRMS can accept a
    species whose name implies a genus it is not classified in, so the first
    token of a name is not a genus. A species-rank category with no lineage
    genus is reported as the data fault it is.

*   The gate counts only crops that carry the grouping level it is testing at.
    Counting collation-wide crops would let a species clear the crop bar on
    FathomNet -- which has no groups, and is therefore out of scope for any
    held-out-group experiment -- while drawing its groups from OzFish. Both
    figures are emitted so the size of that inflation is visible rather than
    silently removed.

*   The gate is reported as a SWEEP over thresholds, not at one chosen value.
    Picking a single threshold after seeing the data is indistinguishable from
    picking whichever threshold gives the nicest answer (plan section 10.5).

Usage:

    python dataset_summary.py ^
        --coco "N:/marineai/dataset/collated/seavision.json" ^
        --coco "N:/marineai/dataset/collated/seavision_fathomnet.json" ^
        --out-dir "N:/marineai/dataset/collated/logs"
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: conda install numpy")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

GROUP_LEVELS = ["deployment", "site", "survey", "region"]   # finest to coarsest


def genus_of(cat, stats):
    """Genus name for a category, or None.

    Primary route is the lineage dict. Fallback, for species-rank categories
    only, is the first token of the binomial. Counts both so the caller can
    see whether the fallback is doing real work.
    """
    lin = cat.get("lineage") or {}
    if isinstance(lin, dict):
        for k, v in lin.items():
            if str(k).strip().lower() == "genus" and v:
                stats["genus_from_lineage"] += 1
                return str(v).strip()
    # Deliberately NO fallback to splitting the binomial. WoRMS can accept a
    # species whose name implies a genus it is not classified in (Turrum
    # gymnostethus is accepted; the genus Turrum resolves to Carangoides), so
    # the first token of a name is not a genus. A species-rank category with
    # no lineage genus is a data fault and is reported as one.
    if str(cat.get("rank", "")).strip().lower() == "species":
        stats["species_without_lineage_genus"] += 1
    stats["genus_unresolved"] += 1
    return None


def short_side(img, ann):
    """Crop short side in pixels, or None if it cannot be determined."""
    if img.get("crop_provenance") == "frame":
        bbox = ann.get("bbox") or []
        if len(bbox) < 4:
            return None
        w, h = bbox[2], bbox[3]
    else:
        w, h = img.get("width"), img.get("height")
    if not w or not h or w <= 0 or h <= 0:
        return None
    return float(min(w, h))


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def fmt(x, dp=1):
    return "n/a" if x != x else f"{x:.{dp}f}"        # x != x catches NaN


# --------------------------------------------------------------------------
# accumulators, shared across all input files
# --------------------------------------------------------------------------

class Collation:
    def __init__(self):
        self.cats = {}                                  # cid -> category record
        self.cat_genus = {}                             # cid -> genus or None
        self.cat_crops = Counter()                      # cid -> n crops
        self.cat_sources = defaultdict(set)             # cid -> {source}
        self.cat_groups = defaultdict(lambda: defaultdict(set))
        self.cat_sizes = defaultdict(list)              # cid -> [short side]
        # per-source, so a source's standalone strength can be stated rather
        # than inferred from collation-wide totals (a species shared between
        # two sources would otherwise be credited its combined crop count)
        self.cat_src_crops = defaultdict(Counter)       # cid -> src -> n
        self.cat_src_groups = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))
        # crops per category AT each grouping level. A crop counts towards a
        # level only if its own image carries that level, so a source with no
        # groups contributes nothing -- which is the point.
        self.cat_crops_lvl = defaultdict(Counter)       # cid -> level -> n
        self.geo = Counter()        # (src, lat_bin, lon_bin) -> n images
        self.sources = {}                               # source -> dict
        self.genus_stats = Counter()
        self.cid_collisions = []

    # ----------------------------------------------------------------------
    def ingest(self, path):
        print(f"  loading {os.path.basename(path)} ...", flush=True)
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)

        ds_name = {d["id"]: d["name"] for d in doc.get("datasets", [])}
        lic = {l["id"]: l for l in doc.get("licenses", [])}
        ds_lic = {d["id"]: d.get("license_id") for d in doc.get("datasets", [])}

        # -- categories ----------------------------------------------------
        for c in doc.get("categories", []):
            cid = c["id"]
            if cid in self.cats and self.cats[cid].get("name") != c.get("name"):
                self.cid_collisions.append(
                    (cid, self.cats[cid].get("name"), c.get("name")))
            self.cats[cid] = c
            if cid not in self.cat_genus:
                self.cat_genus[cid] = genus_of(c, self.genus_stats)

        # -- images --------------------------------------------------------
        imgs = {}
        for im in doc.get("images", []):
            imgs[im["id"]] = im
            src = ds_name.get(im.get("dataset_id"), "unknown")
            s = self.sources.setdefault(src, {
                "images": 0, "annotations": 0, "categories": set(),
                "sizes": [], "provenance": Counter(), "gear": Counter(),
                "group_levels": Counter(), "empty_background": 0,
                "empty_unlabelled": 0, "has_latlon": 0,
                "unlabelled_flag": 0, "pixel_scale_known": Counter(),
                "license": None, "redistributable": None,
            })
            s["images"] += 1
            s["provenance"][im.get("crop_provenance")] += 1
            s["gear"][im.get("gear")] += 1
            s["pixel_scale_known"][bool(im.get("pixel_scale_known"))] += 1
            if im.get("has_unlabelled_animal"):
                s["unlabelled_flag"] += 1
            for lvl in (im.get("groups") or {}):
                s["group_levels"][lvl] += 1
            lat, lon = im.get("lat"), im.get("lon")
            if lat is not None and lon is not None:
                s["has_latlon"] += 1
                # 1-degree bins; plotting half a million raw points is neither
                # readable nor necessary, and binning here keeps R reading CSVs
                self.geo[(src, int(math.floor(lat)), int(math.floor(lon)))] += 1
            if s["license"] is None:
                lid = ds_lic.get(im.get("dataset_id"))
                if lid in lic:
                    s["license"] = lic[lid].get("name")
                    s["redistributable"] = lic[lid].get("redistributable")

        # -- annotations ---------------------------------------------------
        annotated = set()
        for an in doc.get("annotations", []):
            im = imgs.get(an.get("image_id"))
            if im is None:
                continue
            annotated.add(im["id"])
            src = ds_name.get(im.get("dataset_id"), "unknown")
            s = self.sources[src]
            cid = an.get("category_id")

            s["annotations"] += 1
            ss = short_side(im, an)
            if ss is not None:
                s["sizes"].append(ss)

            if cid is None:
                continue
            s["categories"].add(cid)
            self.cat_crops[cid] += 1
            self.cat_sources[cid].add(src)
            if ss is not None:
                self.cat_sizes[cid].append(ss)
            self.cat_src_crops[cid][src] += 1
            for lvl, val in (im.get("groups") or {}).items():
                self.cat_groups[cid][lvl].add((src, str(val)))
                self.cat_src_groups[cid][src][lvl].add(str(val))
                self.cat_crops_lvl[cid][lvl] += 1

        for im_id, im in imgs.items():
            if im_id not in annotated:
                src = ds_name.get(im.get("dataset_id"), "unknown")
                if im.get("has_unlabelled_animal"):
                    self.sources[src]["empty_unlabelled"] += 1
                else:
                    self.sources[src]["empty_background"] += 1

        del doc, imgs


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def group_count(col, cid):
    """(count, level) at the finest level this category has any groups for."""
    g = col.cat_groups.get(cid) or {}
    for lvl in GROUP_LEVELS:
        if g.get(lvl):
            return len(g[lvl]), lvl
    return 0, None


def group_count_src(col, cid, src):
    """(count, level) for one category within ONE source."""
    g = (col.cat_src_groups.get(cid) or {}).get(src) or {}
    for lvl in GROUP_LEVELS:
        if g.get(lvl):
            return len(g[lvl]), lvl
    return 0, None


def write_category_by_source(col, out_dir):
    """Long format: one row per (category, source)."""
    path = os.path.join(out_dir, "wp6_category_by_source.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category_id", "name", "rank", "genus", "source",
                    "n_crops", "n_groups", "group_level"])
        for cid, per in col.cat_src_crops.items():
            c = col.cats.get(cid, {})
            for src, n in sorted(per.items()):
                ng, lvl = group_count_src(col, cid, src)
                w.writerow([cid, c.get("name"), c.get("rank"),
                            col.cat_genus.get(cid), src, n, ng, lvl or ""])
    return path


def per_source_strength(col, crops=100, groups=3, gen_crops=20):
    """What each source supports ON ITS OWN, ignoring the other sources."""
    out = {}
    for src in sorted(col.sources):
        gen, strong = defaultdict(list), 0
        for cid, per in col.cat_src_crops.items():
            n = per.get(src, 0)
            if not n:
                continue
            c = col.cats.get(cid, {})
            if str(c.get("rank", "")).strip().lower() != "species":
                continue
            ng, _ = group_count_src(col, cid, src)
            if n >= crops and ng >= groups:
                strong += 1
            if n >= gen_crops:
                g = col.cat_genus.get(cid)
                if g:
                    gen[g].append(cid)
        multi = {g: v for g, v in gen.items() if len(v) >= 2}
        out[src] = {
            "strong_species": strong,
            "genera_2plus": len(multi),
            "species_in_them": sum(len(v) for v in multi.values()),
        }
    return out


def write_categories(col, out_dir):
    path = os.path.join(out_dir, "wp6_categories.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["category_id", "aphia_id", "name", "rank", "genus",
                    "kingdom", "phylum", "class", "order", "family",
                    "n_crops", "n_groups", "group_level", "n_sources",
                    "sources", "size_p10", "size_median", "size_p90"])
        for cid in sorted(col.cat_crops, key=lambda c: -col.cat_crops[c]):
            c = col.cats.get(cid, {})
            lin = c.get("lineage") or {}
            lin = {str(k).strip().lower(): v for k, v in lin.items()} \
                if isinstance(lin, dict) else {}
            sz = col.cat_sizes.get(cid) or []
            n, lvl = group_count(col, cid)
            w.writerow([
                cid, c.get("aphia_id"), c.get("name"), c.get("rank"),
                col.cat_genus.get(cid),
                lin.get("kingdom", ""), lin.get("phylum", ""),
                lin.get("class", ""), lin.get("order", ""),
                lin.get("family", ""),
                col.cat_crops[cid], n, lvl or "",
                len(col.cat_sources[cid]),
                "|".join(sorted(col.cat_sources[cid])),
                fmt(pct(sz, 10)), fmt(pct(sz, 50)), fmt(pct(sz, 90)),
            ])
    return path


def write_geo(col, out_dir):
    """1-degree binned image positions, for the map. Sources with no
    per-image coordinates simply contribute no rows -- the figure must say
    so rather than letting them vanish."""
    path = os.path.join(out_dir, "wp6_geo.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "lat_bin", "lon_bin", "n_images"])
        for (src, la, lo), n in sorted(col.geo.items()):
            w.writerow([src, la, lo, n])
    return path


def write_sources(col, out_dir):
    path = os.path.join(out_dir, "wp6_sources.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "images", "annotations", "categories",
                    "crop_provenance", "gear", "group_levels",
                    "size_p10", "size_median", "size_p90",
                    "pct_under_64px", "pct_under_32px",
                    "empty_background", "empty_unlabelled",
                    "has_unlabelled_animal", "images_with_latlon",
                    "pct_with_latlon",
                    "license", "redistributable"])
        for src in sorted(col.sources):
            s = col.sources[src]
            sz = np.array(s["sizes"]) if s["sizes"] else np.array([])
            u64 = 100.0 * float((sz < 64).sum()) / len(sz) if len(sz) else float("nan")
            u32 = 100.0 * float((sz < 32).sum()) / len(sz) if len(sz) else float("nan")
            w.writerow([
                src, s["images"], s["annotations"], len(s["categories"]),
                "|".join(f"{k}:{v}" for k, v in s["provenance"].most_common()),
                "|".join(f"{k}:{v}" for k, v in s["gear"].most_common()),
                "|".join(f"{k}:{v}" for k, v in s["group_levels"].most_common()),
                fmt(pct(sz, 10)), fmt(pct(sz, 50)), fmt(pct(sz, 90)),
                fmt(u64), fmt(u32),
                s["empty_background"], s["empty_unlabelled"],
                s["unlabelled_flag"], s["has_latlon"],
                fmt(100.0 * s["has_latlon"] / s["images"] if s["images"] else 0),
                s["license"], s["redistributable"],
            ])
    return path


def write_ranks(col, out_dir):
    path = os.path.join(out_dir, "wp6_ranks.csv")
    by_rank = Counter()
    crops_by_rank = Counter()
    for cid, n in col.cat_crops.items():
        r = (col.cats.get(cid, {}).get("rank") or "unknown").strip()
        by_rank[r] += 1
        crops_by_rank[r] += n
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "n_categories", "n_crops"])
        for r, n in by_rank.most_common():
            w.writerow([r, n, crops_by_rank[r]])
    return path, by_rank, crops_by_rank


def gate_sweep(col, out_dir, crop_thresholds, group_thresholds):
    """Genera holding >=2 species that each clear (min_crops, min_groups).

    The crop bar is applied to GROUPED crops -- those carrying the grouping
    level the species is being counted at -- not to its collation-wide total.
    The two differ whenever a species appears both in a grouped source and in
    an ungrouped one, and the difference is not cosmetic: the ungrouped crops
    cannot take part in the held-out-group experiment the gate exists to
    justify, so counting them inflates the answer to the question actually
    being asked.

    Both are computed and both are written out. Removing a figure silently is
    how a correction becomes indistinguishable from a mistake.
    """
    species = []
    for cid, n in col.cat_crops.items():
        c = col.cats.get(cid, {})
        if str(c.get("rank", "")).strip().lower() != "species":
            continue
        g = col.cat_genus.get(cid)
        if not g:
            continue
        ng, lvl = group_count(col, cid)
        n_grouped = col.cat_crops_lvl[cid][lvl] if lvl else 0
        species.append((g, cid, n, n_grouped, ng, lvl))

    def genera(keep):
        per = Counter(s[0] for s in keep)
        g2 = [g for g, k in per.items() if k >= 2]
        g3 = [g for g, k in per.items() if k >= 3]
        return len(g2), len(g3), sum(per[g] for g in g2)

    path = os.path.join(out_dir, "wp6_gate.csv")
    table = []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["min_crops", "min_groups", "qualifying_species",
                    "genera_with_2plus", "genera_with_3plus",
                    "species_in_those_genera",
                    "species_lost_to_missing_groups",
                    "qualifying_species_allcrops",
                    "genera_with_2plus_allcrops",
                    "species_inflated_by_ungrouped_crops"])
        for mc in crop_thresholds:
            for mg in group_thresholds:
                # min_groups == 0 IS the grouping-ignored baseline -- the row
                # WP5b's "grouping ignored" column is read from. Applying the
                # grouped-crop bar there would make the row contradict its own
                # purpose, so at mg == 0 the crop bar ignores grouping too.
                bar = 2 if mg == 0 else 3
                keep = [s for s in species if s[bar] >= mc and s[4] >= mg]
                loose = [s for s in species if s[2] >= mc and s[4] >= mg]
                # species excluded ONLY because their source carries no
                # grouping variable at all
                lost = (sum(1 for s in species if s[2] >= mc and s[4] == 0)
                        if mg >= 1 else 0)
                g2, g3, sp_in_g2 = genera(keep)
                lg2, _lg3, _ = genera(loose)
                row = [mc, mg, len(keep), g2, g3, sp_in_g2, lost,
                       len(loose), lg2, len(loose) - len(keep)]
                w.writerow(row)
                table.append(row)
    return path, table, species


def write_genera(col, species, out_dir, min_crops, min_groups):
    """The actual genera behind one reference cell, so they can be eyeballed.

    Qualification is on grouped crops; the collation-wide count is written
    alongside so a species carried largely by ungrouped crops is visible at a
    glance rather than having to be inferred.
    """
    path = os.path.join(out_dir, "wp6_genera.csv")
    keep = [s for s in species if s[3] >= min_crops and s[4] >= min_groups]
    per_genus = defaultdict(list)
    for g, cid, n_all, n_grp, ng, lvl in keep:
        per_genus[g].append((cid, n_all, n_grp, ng, lvl))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["genus", "n_species", "species", "grouped_crops_each",
                    "all_crops_each", "groups_each", "group_level"])
        rows = [(g, v) for g, v in per_genus.items() if len(v) >= 2]
        rows.sort(key=lambda r: (-len(r[1]), r[0]))
        for g, v in rows:
            v.sort(key=lambda t: -t[2])
            w.writerow([
                g, len(v),
                "|".join(str(col.cats.get(cid, {}).get("name")) for cid, *_ in v),
                "|".join(str(t[2]) for t in v),
                "|".join(str(t[1]) for t in v),
                "|".join(str(t[3]) for t in v),
                "|".join(str(t[4] or "") for t in v),
            ])
    return path, len(rows)


def cross_source_overlap(col):
    by_src = defaultdict(set)
    for cid, srcs in col.cat_sources.items():
        c = col.cats.get(cid, {})
        if str(c.get("rank", "")).strip().lower() != "species":
            continue
        for s in srcs:
            by_src[s].add(cid)
    out = []
    names = sorted(by_src)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = by_src[a] & by_src[b]
            if shared:
                out.append((a, b, len(shared)))
    return sorted(out, key=lambda t: -t[2])


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="WP6 dataset summary")
    ap.add_argument("--coco", action="append", required=True,
                    help="COCO file; repeat for each (order does not matter)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ref-crops", type=int, default=50,
                    help="reference threshold for wp6_genera.csv")
    ap.add_argument("--ref-groups", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    col = Collation()
    print("reading collation")
    for p in args.coco:
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
        col.ingest(p)

    crop_thresholds = [10, 20, 50, 100]
    group_thresholds = [0, 1, 2, 3]      # 0 = ignore grouping entirely

    p_cat = write_categories(col, args.out_dir)
    p_bysrc = write_category_by_source(col, args.out_dir)
    p_geo = write_geo(col, args.out_dir)
    p_src = write_sources(col, args.out_dir)
    p_rank, by_rank, crops_by_rank = write_ranks(col, args.out_dir)
    p_gate, gate, species = gate_sweep(col, args.out_dir,
                                       crop_thresholds, group_thresholds)
    p_gen, n_gen = write_genera(col, species, args.out_dir,
                                args.ref_crops, args.ref_groups)

    # ---- printed summary -------------------------------------------------
    tot_img = sum(s["images"] for s in col.sources.values())
    tot_ann = sum(s["annotations"] for s in col.sources.values())

    print()
    print("=" * 72)
    print("COLLATION")
    print("=" * 72)
    print(f"  images      {tot_img:,}")
    print(f"  annotations {tot_ann:,}")
    print(f"  categories  {len(col.cat_crops):,} with at least one crop "
          f"({len(col.cats):,} defined)")
    print()
    print(f"{'source':<14}{'images':>12}{'annots':>12}{'cats':>8}"
          f"{'med px':>9}{'<64px':>8}")
    for src in sorted(col.sources):
        s = col.sources[src]
        sz = np.array(s["sizes"]) if s["sizes"] else np.array([])
        u64 = 100.0 * float((sz < 64).sum()) / len(sz) if len(sz) else float("nan")
        print(f"{src:<14}{s['images']:>12,}{s['annotations']:>12,}"
              f"{len(s['categories']):>8,}{fmt(pct(sz,50),0):>9}{fmt(u64):>8}")

    print()
    print("RANKS")
    for r, n in by_rank.most_common():
        print(f"  {r:<16}{n:>6,} categories{crops_by_rank[r]:>12,} crops")

    print()
    print("PER-SOURCE STRENGTH  (each source counted ON ITS OWN --")
    print("                      crops in other sources do not count)")
    strength = per_source_strength(col)
    print(f"{'source':<14}{'sp >=100 crops':>16}{'genera >=2 sp':>15}"
          f"{'species in them':>17}")
    print(f"{'':<14}{'in >=3 groups':>16}{'(>=20 crops)':>15}{'':>17}")
    for src in sorted(strength):
        s = strength[src]
        print(f"{src:<14}{s['strong_species']:>16,}{s['genera_2plus']:>15,}"
              f"{s['species_in_them']:>17,}")

    print()
    print("GENUS RESOLUTION")
    print(f"  from lineage      {col.genus_stats['genus_from_lineage']:>6,}")
    print(f"  unresolved        {col.genus_stats['genus_unresolved']:>6,}")
    swg = col.genus_stats["species_without_lineage_genus"]
    if swg:
        print(f"  !! {swg:,} species-rank categories have NO genus in their "
              f"lineage -- a data fault, not a naming quirk")

    if col.cid_collisions:
        print()
        print(f"  !! {len(col.cid_collisions)} category id collisions across "
              f"files -- first few:")
        for cid, a, b in col.cid_collisions[:5]:
            print(f"     {cid}: {a!r} vs {b!r}")

    print()
    print("=" * 72)
    print("THE GATE -- genera holding 2+ species, each clearing a threshold")
    print("=" * 72)
    print(f"{'crops':>7}{'groups':>8}{'species':>10}{'genera>=2':>11}"
          f"{'genera>=3':>11}{'lost:nogrp':>12}{'was':>8}{'infl':>7}")
    for (mc, mg, nsp, g2, g3, _, lost, nsp_all, g2_all, infl) in gate:
        print(f"{mc:>7}{mg:>8}{nsp:>10,}{g2:>11,}{g3:>11,}{lost:>12,}"
              f"{g2_all:>8,}{infl:>7,}")
    print()
    print("  groups=0 ignores grouping entirely. 'lost:nogrp' counts species")
    print("  that clear the crop bar but are excluded because their source")
    print("  carries NO grouping variable -- for FathomNet that is every")
    print("  species, until the deferred image-set-upload pass runs.")
    print()
    print("  The crop bar counts only crops carrying the grouping level the")
    print("  species is counted at. 'was' is genera>=2 under the old rule,")
    print("  which counted crops across the whole collation; 'infl' is how")
    print("  many species that rule admitted on crops they cannot use in a")
    print("  held-out-group experiment. If 'infl' is 0 the two rules agree")
    print("  and the earlier figures stand unchanged.")

    print()
    print(f"  R1 asked for three genera with two or more well-populated "
          f"species.")
    print(f"  At >={args.ref_crops} crops in >={args.ref_groups} groups: "
          f"{n_gen} genera qualify.")

    ov = cross_source_overlap(col)
    if ov:
        print()
        print("SHARED SPECIES BETWEEN SOURCES")
        for a, b, n in ov:
            print(f"  {a} & {b}: {n:,}")

    print()
    print("written:")
    for p in (p_src, p_rank, p_cat, p_bysrc, p_geo, p_gate, p_gen):
        print(f"  {p}")


if __name__ == "__main__":
    main()

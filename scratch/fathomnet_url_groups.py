#!/usr/bin/env python3
"""Derive FathomNet dive / deployment groups from frame_url. READ-ONLY.

WP5b, reopened cheaply: deployment identity was thought to need ~500k API
calls, but the frame URLs carry it for most of the collection. This script
proposes groups and reports on them. It writes NOTHING into the collation --
that is a later, separate migration step, and only after validation.

HOW GROUPS ARE FORMED: union, not assignment
    Each image yields tokens (a dive, a deployment, a rig-day ...). Images
    sharing a token share a group, transitively. Where we are unsure we
    MERGE: a too-coarse group costs a little efficiency, a too-fine one leaks
    across a split, which is the one error this exists to prevent.

DATES ARE A SECOND-PASS LINK, NEVER A PRIMARY TOKEN
    Dives cross midnight UTC. If every image carried a date token, dive N
    (days d, d+1) and dive N+1 (day d+1) would merge, and so on down the
    cruise until one group swallowed it. So dates are used only to attach an
    image that has NO reliable dive token (MBARI stills and unpadded
    folders, NOAA OE images on the s3 host) to the reliable dives observed
    on the same vehicle/cruise on the same date. Dive-coded images never
    link to each other through dates.

UNMATCHED IS REPORTED, NOT GUESSED
    Only the top 15 URL templates per institution have been seen. Anything
    no rule recognises is counted and shown with examples, and left without
    a group, rather than bent to the nearest rule.

LEVELS, coarse to fine, so a consumer can refuse the weak ones
    institution > project > cruise > unverified > date > rig_day > deployment/dive

    python scratch/fathomnet_url_groups.py ^
        --coco D:\\marineai\\dataset\\collated\\seavision_fathomnet.json ^
        --out  D:\\marineai\\scratch\\fathomnet_groups
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from urllib.parse import unquote, urlparse

LEVEL_ORDER = ["dive", "deployment", "rig_day", "date", "unverified",
               "cruise", "project", "institution"]

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
                  r"[0-9a-f]{12}", re.I)
DIGITS = re.compile(r"\d+")
FILE_DATE = re.compile(r"(?<!\d)((?:19|20)\d{6})T\d{4,6}")

INSTITUTION_ONLY = {"Joost Daniels", "CENCOOS", "POSCO", "UW NSF-OOI CSSF",
                    "UW NSF-OOI WHOI", "UniOfPlym"}
SOI = {"Schmidt Ocean Institute",
       "Universidad de Costa Rica and Schmidt Ocean Institute"}
ONC = {"Ocean Networks Canada", "ONCS"}


def template(path):
    return DIGITS.sub("#", UUID.sub("U", path))


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def R(tokens, rule, level, reliable, link=None, dive_date=None):
    """Parse result. `link` = (namespace, yyyymmdd) for second-pass
    attachment; `dive_date` = (namespace, yyyymmdd) recorded for a reliable
    dive so others can attach to it."""
    return {"tokens": tokens, "rule": rule, "level": level,
            "reliable": reliable, "link": link, "dive_date": dive_date}


# --------------------------------------------------------------- parsers

def p_sefsc(segs, fname):
    folder = segs[-2] if len(segs) >= 2 else ""
    m = re.match(r"^(\d{9})_", folder)
    if m:                                  # 761901008_cam1_4, _cam4A, _clip,
        return R([f"SEFSC:ID:{m[1]}"],     # _SC2_Cam2, _Cam3 -- cameras and
                 "sefsc_id", "deployment", True)   # parts of one drop
    m = re.match(r"^(B?SC\d+)-camera\d+_(\d\d)-(\d\d)-(\d\d)_", folder)
    if m:                                  # rig + day: several drops a day
        d = f"20{m[4]}{m[2]}{m[3]}"        # merge -- safe direction
        return R([f"SEFSC:RIG:{m[1]}:{d}"], "sefsc_rig_day", "rig_day", True)
    m = re.match(r"^(\d{4}_NC[A-Z]+[-_]\d+)(?:[a-z]\d*)?$", folder)
    if m:                                  # 2021_NCO-002b, -004e2, NCN_042a2
                                           # -> station; drops merge (safe)
        return R([f"SEFSC:NC:{m[1]}"], "sefsc_nc", "deployment", True)
    return None


def p_mbari(host, segs, fname):
    fdate = FILE_DATE.search(fname)
    fdate = fdate[1] if fdate else None
    if host == "database.fathomnet.org":
        # /static/m3/framegrabs/<Vehicle>/(images|stills)/...
        if len(segs) < 6 or segs[2] != "framegrabs":
            return None
        veh = segs[3].replace(" ", "")
        ns = f"MBARI:{veh}"
        kind, rest = segs[4], segs[5:]
        if kind == "images" and len(rest) >= 2:
            folder = rest[0]
            m = re.fullmatch(r"Mini ROV (\d+)", folder)     # 'Mini ROV 0152'
            if m:
                folder = m[1].zfill(4)
            if re.fullmatch(r"\d{8}", folder):                  # i2MAP day
                return R([f"{ns}:DATE:{folder}"], "mbari_dated_folder",
                         "date", False, link=(ns, folder))
            if re.fullmatch(r"\d{4}", folder):                  # dive 0339
                return R([f"{ns}:D{int(folder)}"], "mbari_dive", "dive",
                         True, dive_date=(ns, fdate) if fdate else None)
            if re.fullmatch(r"\d{1,5}", folder):
                # Unpadded: '46' holding 2019 images cannot be Doc Ricketts
                # dive 46 (~2009). Kept as its own token, and attached to any
                # dive seen on the same date.
                return R([f"{ns}:F{folder}"], "mbari_unpadded", "unverified",
                         False, link=(ns, fdate) if fdate else None)
            return None
        if kind == "stills" and len(rest) >= 3 and \
                re.fullmatch(r"\d{4}", rest[0]) and \
                re.fullmatch(r"\d{1,3}", rest[1]):
            d = (dt.date(int(rest[0]), 1, 1)
                 + dt.timedelta(days=int(rest[1]) - 1)).strftime("%Y%m%d")
            return R([f"{ns}:DATE:{d}"], "mbari_stills_day", "date", False,
                     link=(ns, d))
        return None
    folder = segs[-2] if len(segs) >= 2 else ""
    # The SAME dives re-hosted: MBARI_D1032 is Doc Ricketts dive 1032 and must
    # land on the token the database.fathomnet.org images of that dive use,
    # or one dive becomes two groups -- the leaking direction.
    m = re.match(r"^MBARI_([DVT])(\d{3,4})(?:_|$)", folder)
    if m:
        veh = VEHICLE_LETTER[m[1]]
        ns = f"MBARI:{veh}"
        return R([f"{ns}:D{int(m[2])}"], "mbari_dive_rehosted", "dive", True,
                 dive_date=(ns, fdate) if fdate else None)
    m = re.search(r"BIL_Kona_(\d{8})", folder)
    if m:                                  # snorkel + scuba same day merge
        return R([f"MBARI:BIL:{m[1]}"], "mbari_bil_day", "date", True)
    m = re.search(r"BIL_Kona(\d{4})_", folder)
    if m:                                  # year only: every 2024 Kona set
        return R([f"MBARI:BIL:Y{m[1]}"], "mbari_bil_year", "cruise", True)
    return None


VEHICLE_LETTER = {"D": "DocRicketts", "V": "Ventana", "T": "Tiburon"}


def p_noaa(host, path, segs, fname):
    toks = set()
    for m in re.finditer(r"EX(\d{4}(?:L\d)?)_DIVE(\d+)((?:_\d+)*)", path):
        toks.add(f"NOAA:EX{m[1]}:D{int(m[2])}")
        for extra in re.findall(r"_(\d+)", m[3]):    # DIVE04_05 -> 4 and 5
            toks.add(f"NOAA:EX{m[1]}:D{int(extra)}")
    for m in re.finditer(r"D\d-EX(\d{4}(?:L\d)?)-(\d+)", path):  # hurlimage
        toks.add(f"NOAA:EX{m[1]}:D{int(m[2])}")
    vid = re.search(r"EX(\d{4}(?:L\d)?)_VID_(\d{8})", fname)
    if toks:
        cruises = {t.split(":")[1] for t in toks}
        dd = None
        if vid and len(cruises) == 1:
            dd = (f"NOAA:{next(iter(cruises))}", vid[2])
        rule = "noaa_dive_multi" if len(toks) > 1 else "noaa_dive"
        return R(sorted(toks), rule, "dive", True, dive_date=dd)
    if vid:                                # s3 host: cruise + date only
        ns = f"NOAA:EX{vid[1]}"
        return R([f"{ns}:DATE:{vid[2]}"], "noaa_cruise_day", "date", False,
                 link=(ns, vid[2]))
    return None


def p_soi(path):
    m = re.search(r"(?:^|[_/])S(\d{4})(?=[_/.]|$)", path)
    if m:
        return R([f"SOI:S{int(m[1])}"], "soi_dive", "dive", True)
    return None


def p_onc(segs, path):
    m = re.search(r"_H(\d{4})(?=[_/]|$)", path)
    if m:                                  # Hercules dive
        return R([f"ONC:H{int(m[1])}"], "onc_dive", "dive", True)
    if len(segs) >= 3 and UUID.fullmatch(segs[0]):
        # /<uuid>/<Project>/<sub>/... -- fixed cameras and inspection sets at
        # one site; the whole project-subfolder is one group.
        return R([f"ONC:{segs[1]}:{segs[2]}"], "onc_project", "project",
                 True)
    return None


def p_oet(path):
    m = re.search(r"OET_NA(\d+)", path)
    if m:
        return R([f"OET:NA{m[1]}"], "oet_cruise", "cruise", True)
    return None


def parse(inst, url):
    p = urlparse(url)
    path = unquote(p.path)
    segs = [s for s in path.split("/") if s]
    fname = segs[-1] if segs else ""
    if inst in INSTITUTION_ONLY:
        return R([f"INST:{inst}"], "institution_only", "institution", True)
    if inst == "NOAA NMFS SEFSC":
        return p_sefsc(segs, fname)
    if inst == "MBARI":
        return p_mbari(p.netloc, segs, fname)
    if inst == "NOAA Ocean Exploration":
        return p_noaa(p.netloc, path, segs, fname)
    if inst in SOI:
        return p_soi(path)
    if inst in ONC:
        return p_onc(segs, path)
    if inst == "OET":
        return p_oet(path)
    return None


# ------------------------------------------------------------------- main

POS_KM = 5.0          # NOAA position link radius; merging is the safe side


def km(a, b):
    """Great-circle distance in km between (lat, lon) pairs."""
    import math
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = (math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2)
         * math.sin((lo2 - lo1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def median_pos(pts):
    return (statistics.median(p[0] for p in pts),
            statistics.median(p[1] for p in pts))


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))] if s else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", required=True, help="output DIRECTORY")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists; move it aside rather than overwrite")

    print(f"loading {os.path.basename(args.coco)} ...", flush=True)
    with open(args.coco, encoding="utf-8") as fh:
        doc = json.load(fh)
    cats = {c["id"]: c for c in doc["categories"]}
    cat_of = {}
    for a in doc["annotations"]:
        cat_of.setdefault(a["image_id"], a.get("category_id"))

    rows, unmatched = [], defaultdict(Counter)
    unmatched_eg = defaultdict(dict)
    dsu = DSU()
    dive_dates = defaultdict(lambda: defaultdict(set))  # ns -> date -> toks
    mbari_dive_dates = defaultdict(list)                # (veh, n) -> dates

    for im in doc["images"]:
        sm = im.get("source_meta") or {}
        inst = str(sm.get("owner_institution") or "UNKNOWN")
        url = sm.get("frame_url") or ""
        cat = cats.get(cat_of.get(im["id"])) or {}
        teleost = (cat.get("lineage") or {}).get("class") == "Teleostei"
        res = parse(inst, url) if url else None
        row = {"uid": os.path.splitext(im["file_name"])[0],
               "uuid": sm.get("fathomnet_image_uuid"), "inst": inst,
               "teleost": teleost, "res": res,
               "pos": ((float(im["lat"]), float(im["lon"]))
                       if im.get("lat") is not None and
                       im.get("lon") is not None else None)}
        rows.append(row)
        if res is None:
            t = template(unquote(urlparse(url).path)) if url else "<no url>"
            unmatched[inst][t] += 1
            unmatched_eg[inst].setdefault(t, url)
            continue
        first = res["tokens"][0]
        for t in res["tokens"]:
            dsu.union(first, t)
        if res["reliable"] and res["dive_date"] and res["dive_date"][1]:
            ns, d = res["dive_date"]
            for t in res["tokens"]:
                dive_dates[ns][d].add(t)
            if res["rule"] in ("mbari_dive", "mbari_dive_rehosted"):
                veh = ns.split(":", 1)[1]
                mbari_dive_dates[(veh, int(first.rsplit(":D", 1)[1]))] \
                    .append(d)
    del doc

    # ---- second pass: attach date-only images to dives on the same date
    link_outcome = Counter()
    for r in rows:
        res = r["res"]
        if not res or res["reliable"] or not res["link"]:
            r["link"] = ""
            continue
        ns, d = res["link"]
        cands = dive_dates.get(ns, {}).get(d, set()) if d else set()
        for c in cands:
            dsu.union(res["tokens"][0], c)
        r["link"] = ("none" if not cands else
                     "one_dive" if len(cands) == 1 else "several_dives")
        link_outcome[(r["inst"], res["rule"], r["link"])] += 1

    # ---- third pass, NOAA OE only: the s3 images carry cruise + date but
    # the dive-coded images of the same cruise mostly carry no date, so the
    # date link cannot fire. Many do carry lat/lon, and a dive is compact.
    # Attach a still-unlinked image to every dive of its cruise whose median
    # position is within POS_KM. Two dives at one site on different days
    # both match and merge -- the safe direction.
    dive_pts = defaultdict(list)                  # token -> positions
    for r in rows:
        res = r["res"]
        if res and res["reliable"] and r["pos"] and \
                r["inst"] == "NOAA Ocean Exploration":
            for t in res["tokens"]:
                dive_pts[t].append(r["pos"])
    dive_centre = {t: median_pos(p) for t, p in dive_pts.items()}
    for r in rows:
        res = r["res"]
        if not res or res["rule"] != "noaa_cruise_day" or r["link"] != "none":
            continue
        cruise = res["tokens"][0].rsplit(":DATE:", 1)[0] + ":D"
        if not r["pos"]:
            r["link"] = "none_no_position"
        else:
            near = [t for t, c in dive_centre.items()
                    if t.startswith(cruise) and km(r["pos"], c) <= POS_KM]
            for t in near:
                dsu.union(res["tokens"][0], t)
            r["link"] = ("none" if not near else
                         "pos_one_dive" if len(near) == 1 else
                         "pos_several_dives")
        link_outcome[(r["inst"], res["rule"], "none")] -= 1
        link_outcome[(r["inst"], res["rule"], r["link"])] += 1
    link_outcome = +link_outcome                   # drop zero counts

    # ---- MBARI order check, BEFORE naming: a padded folder whose dates
    # contradict its number is not trusted as a dive and is demoted.
    by_veh = defaultdict(list)
    for (veh, num), ds in mbari_dive_dates.items():
        med = sorted(ds)[len(ds) // 2]
        by_veh[veh].append((num, dt.datetime.strptime(med, "%Y%m%d").date()))
    order_bad, demote = [], set()
    order_summary = []
    for veh, lst in sorted(by_veh.items()):
        lst.sort()
        running, n_bad = None, 0
        for num, d in lst:
            if running and (running - d).days > 180:
                n_bad += 1
                order_bad.append([veh, num, str(d), str(running)])
                demote.add(f"MBARI:{veh}:D{num}")
            running = max(running, d) if running else d
        order_summary.append((veh, len(lst), n_bad, lst[0][0], lst[-1][0]))

    # ---- name components deterministically
    comp_tokens = defaultdict(set)
    for r in rows:
        if r["res"]:
            for t in r["res"]["tokens"]:
                comp_tokens[dsu.find(t)].add(t)
    name = {root: (min(ts) + (f"+{len(ts) - 1}" if len(ts) > 1 else ""))
            for root, ts in comp_tokens.items()}

    # A group's level is set by its RELIABLE members where it has any: an
    # image attached to a real dive by date does not downgrade that dive,
    # it joins it. Only groups with no reliable member take the level of
    # their weak members (a date-only group, an unpadded MBARI folder).
    comp_levels = defaultdict(Counter)
    comp_rel_levels = defaultdict(Counter)
    comp_size = Counter()
    for r in rows:
        if r["res"]:
            g = dsu.find(r["res"]["tokens"][0])
            r["group"] = name[g]
            comp_levels[r["group"]][r["res"]["level"]] += 1
            if r["res"]["reliable"]:
                comp_rel_levels[r["group"]][r["res"]["level"]] += 1
            comp_size[r["group"]] += 1
        else:
            r["group"] = ""

    demoted_groups = {name[dsu.find(t)] for t in demote if t in dsu.p}

    def comp_level(g):
        if g in demoted_groups:
            return "unverified"
        src = comp_rel_levels[g] or comp_levels[g]
        return max(src, key=LEVEL_ORDER.index)

    # ---- write per-image CSV
    os.makedirs(args.out)
    csv_path = os.path.join(args.out, "fathomnet_groups.csv.gz")
    with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "fathomnet_image_uuid", "owner_institution",
                    "rule", "image_level", "group", "group_level",
                    "date_link", "teleost"])
        for r in rows:
            res = r["res"]
            w.writerow([r["uid"], r["uuid"], r["inst"],
                        res["rule"] if res else "UNMATCHED",
                        res["level"] if res else "",
                        r["group"],
                        comp_level(r["group"]) if r["group"] else "",
                        r["link"] if res else "", int(r["teleost"])])

    # ---- report
    report = {}
    by_inst = defaultdict(list)
    for r in rows:
        by_inst[r["inst"]].append(r)
    for inst, rs in sorted(by_inst.items(), key=lambda kv: -len(kv[1])):
        n = len(rs)
        um = sum(1 for r in rs if not r["res"])
        groups = Counter(r["group"] for r in rs if r["group"])
        sizes = list(groups.values())
        lv = Counter(comp_level(r["group"]) for r in rs if r["group"])
        tel = sum(r["teleost"] for r in rs)
        tel_fine = sum(1 for r in rs if r["teleost"] and r["group"]
                       and comp_level(r["group"]) in
                       ("dive", "deployment", "rig_day"))
        rules = Counter(r["res"]["rule"] for r in rs if r["res"])
        big = groups.most_common(5)
        print(f"\n{'=' * 78}\n{inst}: {n:,} crops")
        print(f"  unmatched {um:,} ({100 * um / n:.2f}%)   groups "
              f"{len(groups):,}   crops/group p50 {pct(sizes, .5):,} "
              f"p90 {pct(sizes, .9):,} max {max(sizes, default=0):,}")
        print(f"  largest group holds {100 * big[0][1] / n:.1f}% "
              if big else "", end="")
        print(f"  teleost crops {tel:,}, of which in dive/deployment/rig-day "
              f"groups {tel_fine:,}")
        print(f"  rules:  {dict(rules)}")
        print(f"  group levels (crops): {dict(lv)}")
        for g, k in big:
            print(f"    {k:>8,}  {g}  ({len(comp_tokens[dsu.find(g.split('+')[0])]):,} tokens)")
        for t, k in unmatched[inst].most_common(10):
            print(f"  UNMATCHED {k:>7,}  {t}\n              e.g. "
                  f"{unmatched_eg[inst][t]}")
        report[inst] = {
            "crops": n, "unmatched": um, "groups": len(groups),
            "group_size_p50_p90_max": [pct(sizes, .5), pct(sizes, .9),
                                       max(sizes, default=0)],
            "teleost": tel, "teleost_fine_grouped": tel_fine,
            "rules": dict(rules), "group_levels": dict(lv),
            "largest": big,
            "unmatched_templates": [[t, k, unmatched_eg[inst][t]] for t, k
                                    in unmatched[inst].most_common(30)]}

    print(f"\n{'=' * 78}\nDATE LINKS (images with no reliable dive token)")
    for (inst, rule, out), k in sorted(link_outcome.items()):
        print(f"  {inst:<24} {rule:<20} {out:<14} {k:>8,}")
    report["_date_links"] = [[*k, v] for k, v in sorted(link_outcome.items())]

    print(f"\n{'=' * 78}\nMBARI: dives out of date order (median date more "
          f"than 180 days before an EARLIER-numbered dive) -- DEMOTED")
    for veh, n, nb, lo, hi in order_summary:
        print(f"  {veh:<14} {n:>5} dated dives, {nb} out of order  "
              f"range D{lo}..D{hi}")
    for b in order_bad[:15]:
        print(f"    {b[0]} D{b[1]} median {b[2]} but an earlier dive "
              f"reached {b[3]}")
    bad = order_bad

    # Spatial coherence: a real dive or deployment is compact. A group whose
    # geolocated images spread over tens of km has merged things that should
    # not be one group -- or its positions are wrong. Either way, look.
    print(f"\n{'=' * 78}\nSPATIAL COHERENCE (groups with >=10 geolocated "
          f"images; radius = p95 distance from the group median)")
    spatial = {}
    gpts = defaultdict(list)
    for r in rows:
        if r["group"] and r["pos"]:
            gpts[(r["inst"], r["group"])].append(r["pos"])
    per_inst = defaultdict(list)
    for (inst, g), pts in gpts.items():
        if len(pts) < 10:
            continue
        c = median_pos(pts)
        rad = pct([km(p, c) for p in pts], 0.95)
        per_inst[inst].append((rad, g, len(pts)))
    for inst, lst in sorted(per_inst.items()):
        lst.sort(reverse=True)
        rads = [x[0] for x in lst]
        n_wide = sum(1 for x in rads if x > 10)
        print(f"  {inst:<26} {len(lst):>4} groups  radius p50 "
              f"{pct(rads, .5):.2f} km  max {rads[0]:.1f} km  "
              f">10 km: {n_wide}")
        for rad, g, n in lst[:3]:
            if rad > 10:
                print(f"      {rad:>8.1f} km  {g}  ({n} geolocated)")
        spatial[inst] = [[round(r_, 2), g, n] for r_, g, n in lst[:20]]
    report["_spatial"] = spatial
    report["_mbari_order_violations"] = bad

    with open(os.path.join(args.out, "report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    total_um = sum(v["unmatched"] for k, v in report.items()
                   if not k.startswith("_"))
    print(f"\nwrote {csv_path}\n      {os.path.join(args.out, 'report.json')}")
    print(f"unmatched overall: {total_um:,}")
    print("Nothing in the collation was changed.")


if __name__ == "__main__":
    main()
